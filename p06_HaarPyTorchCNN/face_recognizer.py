"""
face_recognizer.py

PyTorch 小型 CNN 定義、訓練（Method B fine-tune）與推論。

架構：
    Input 96×96×1
    → Conv(32) + BN + ReLU + MaxPool(2)   → 48×48×32
    → Conv(64) + BN + ReLU + MaxPool(2)   → 24×24×64
    → Conv(128) + BN + ReLU + GAP         → 128-dim 嵌入向量
    → FC(128 → N)  → Softmax

Method B 策略：
    新增第 N 人時，載入舊模型 backbone，擴展 FC 至 N 類，
    用所有人的儲存資料 fine-tune 20 epochs（避免遺忘舊人）。
"""

import os
import random
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from model_store import UNKNOWN_DIR_NAME

# --- 常數 ---
CNN_INPUT_SIZE        = 96     # 輸入影像邊長（px）
BATCH_SIZE            = 32
TRAIN_EPOCHS_FULL     = 50    # 從頭訓練最大 epoch 數
TRAIN_EPOCHS_FINETUNE = 20    # Fine-tune 最大 epoch 數
UNKNOWN_THRESHOLD     = 0.60  # 信心值低於此 → Unknown
UNKNOWN_SAMPLE_COUNT  = 60    # 每次訓練產生的合成 unknown 樣本數
EARLY_STOP_PATIENCE   = 5     # 連續幾個 epoch 沒進步則提早停止
EARLY_STOP_MIN_DELTA  = 0.001 # 視為「有進步」的最小 Loss 改善量


# ==============================================================================
# CNN 模型
# ==============================================================================
class FaceCNN(nn.Module):
    """小型人臉辨識 CNN，約 200K 參數。"""

    def __init__(self, NumClasses: int):
        super().__init__()
        # 特徵提取層（backbone）
        self._Features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 48×48×32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 24×24×64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                      # 1×1×128
        )
        # 分類層（動態擴展）
        self._Classifier = nn.Linear(128, NumClasses)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self._Features(X)
        X = X.view(X.size(0), -1)   # flatten → (B, 128)
        X = self._Classifier(X)
        return X


# ==============================================================================
# 訓練資料集
# ==============================================================================
class FaceDataset(Dataset):
    """從磁碟讀取臉部 ROI 圖片的 PyTorch Dataset。"""

    def __init__(self, DataDir: str, LabelMap: dict, Transform=None):
        """
        DataDir : data/faces/ 根目錄
        LabelMap: {人名: class_index}
        """
        self._Samples  = []   # list of (img_path, class_index)
        self._Transform = Transform

        for Name, Idx in LabelMap.items():
            PersonDir = os.path.join(DataDir, Name)
            if not os.path.isdir(PersonDir):
                continue
            for FileName in sorted(os.listdir(PersonDir)):
                if FileName.lower().endswith((".jpg", ".png")):
                    self._Samples.append(
                        (os.path.join(PersonDir, FileName), Idx)
                    )

    def __len__(self) -> int:
        return len(self._Samples)

    def __getitem__(self, Idx: int):
        ImgPath, Label = self._Samples[Idx]
        try:
            # 讀取灰階圖片
            Img = cv2.imread(ImgPath, cv2.IMREAD_GRAYSCALE)
            if Img is None:
                raise ValueError(f"無法讀取：{ImgPath}")
            Img = cv2.resize(Img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            # 轉 float32 [0, 1]，shape (1, H, W)
            Tensor = torch.tensor(Img.astype(np.float32) / 255.0).unsqueeze(0)
            if self._Transform:
                Tensor = self._Transform(Tensor)
                Tensor = Tensor.clamp(0.0, 1.0)  # 防止增強後超出範圍
            return Tensor, Label
        except Exception as Error:
            print(f"[FaceDataset] 讀取失敗：{Error}")
            return torch.zeros(1, CNN_INPUT_SIZE, CNN_INPUT_SIZE), Label


# ==============================================================================
# 辨識器
# ==============================================================================
class FaceRecognizer:
    """整合 CNN 模型的訓練與推論。"""

    def __init__(self):
        self._Model      = None
        self._LabelToIdx = {}   # {人名: class_index}
        self._IdxToLabel = {}   # {class_index: 人名}
        self._Device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------
    # 查詢
    # --------------------------------------------------------------------------
    def isModelLoaded(self) -> bool:
        return self._Model is not None

    def getPersonList(self) -> list:
        """回傳目前模型已知的人員清單。"""
        return list(self._LabelToIdx.keys())

    def clearModel(self) -> None:
        """清除模型與標籤（移除人員後重置用）。"""
        self._Model      = None
        self._LabelToIdx = {}
        self._IdxToLabel = {}

    # --------------------------------------------------------------------------
    # 合成 Unknown 樣本產生
    # --------------------------------------------------------------------------
    def _generateAndSaveUnknownSamples(self, DataDir: str) -> None:
        """
        產生合成的 unknown 樣本並存入 DataDir/__unknown__/。
        每次訓練前重新產生，確保多樣性。

        樣本組成：
            50% 對已知人臉做極端旋轉 + 大量雜訊（教 CNN「不是這個人的臉」）
            50% 隨機雜訊 + 幾何圖形（教 CNN「完全陌生的特徵」）
        """
        UnknownDir = os.path.join(DataDir, UNKNOWN_DIR_NAME)

        # 若已有足夠樣本，直接沿用，不重新產生
        if os.path.isdir(UnknownDir):
            ExistingCount = len([
                F for F in os.listdir(UnknownDir)
                if F.lower().endswith(".jpg")
            ])
            if ExistingCount >= UNKNOWN_SAMPLE_COUNT:
                return

        # 第一次，或樣本不足時才產生
        if os.path.isdir(UnknownDir):
            shutil.rmtree(UnknownDir)
        os.makedirs(UnknownDir)

        # 收集所有已知人臉圖片路徑
        KnownImgPaths = []
        for PersonName in os.listdir(DataDir):
            if PersonName == UNKNOWN_DIR_NAME:
                continue
            PersonDir = os.path.join(DataDir, PersonName)
            if not os.path.isdir(PersonDir):
                continue
            for F in os.listdir(PersonDir):
                if F.lower().endswith(".jpg"):
                    KnownImgPaths.append(os.path.join(PersonDir, F))

        HalfCount = UNKNOWN_SAMPLE_COUNT // 2

        for i in range(UNKNOWN_SAMPLE_COUNT):
            if KnownImgPaths and i < HalfCount:
                # --- 類型 A：極端變形已知人臉 ---
                ImgPath = random.choice(KnownImgPaths)
                Img = cv2.imread(ImgPath, cv2.IMREAD_GRAYSCALE)
                if Img is None:
                    Img = np.random.randint(0, 255, (CNN_INPUT_SIZE, CNN_INPUT_SIZE), dtype=np.uint8)
                else:
                    Img = cv2.resize(Img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                    # 大角度旋轉（45° ~ 135°，正負隨機）
                    Angle = random.uniform(45, 135) * random.choice([-1, 1])
                    M = cv2.getRotationMatrix2D((CNN_INPUT_SIZE // 2, CNN_INPUT_SIZE // 2), Angle, 1.0)
                    Img = cv2.warpAffine(Img, M, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                    # 加入大量 Gaussian 雜訊
                    Noise = np.random.normal(0, 60, Img.shape).astype(np.int16)
                    Img = np.clip(Img.astype(np.int16) + Noise, 0, 255).astype(np.uint8)
                    # 水平 + 垂直翻轉
                    Img = cv2.flip(Img, -1)
                    # 隨機裁切再縮回（破壞空間結構）
                    CropX = random.randint(0, 20)
                    CropY = random.randint(0, 20)
                    Crop  = Img[CropY:CropY + CNN_INPUT_SIZE - CropY,
                                CropX:CropX + CNN_INPUT_SIZE - CropX]
                    if Crop.size > 0:
                        Img = cv2.resize(Crop, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            else:
                # --- 類型 B：隨機雜訊 + 幾何圖形 ---
                Img = np.random.randint(30, 220, (CNN_INPUT_SIZE, CNN_INPUT_SIZE), dtype=np.uint8)
                # 隨機疊加 2~4 個橢圓
                for _ in range(random.randint(2, 4)):
                    CX    = random.randint(10, CNN_INPUT_SIZE - 10)
                    CY    = random.randint(10, CNN_INPUT_SIZE - 10)
                    RX    = random.randint(5, 35)
                    RY    = random.randint(5, 35)
                    Color = random.randint(20, 230)
                    Thick = random.choice([-1, 1, 2])   # -1 = 實心
                    cv2.ellipse(Img, (CX, CY), (RX, RY),
                                random.uniform(0, 360), 0, 360, Color, Thick)
                # 局部模糊（破壞紋理）
                if random.random() > 0.5:
                    BlurK = random.choice([3, 5, 7])
                    Img   = cv2.GaussianBlur(Img, (BlurK, BlurK), 0)

            FilePath = os.path.join(UnknownDir, f"img_{i + 1:04d}.jpg")
            cv2.imwrite(FilePath, Img)

    # --------------------------------------------------------------------------
    # 訓練
    # --------------------------------------------------------------------------
    def train(self, DataDir: str, ProgressCallback=None) -> bool:
        """
        訓練或 fine-tune 模型（Method B）。

        DataDir         : data/faces/ 根目錄
        ProgressCallback: fn(Epoch, TotalEpochs, Loss) 進度回呼

        回傳 True 表示成功，False 表示失敗（無真實人員資料）。
        """
        try:
            # 產生合成 unknown 樣本（每次訓練前重新產生）
            self._generateAndSaveUnknownSamples(DataDir)

            # 掃描所有資料夾（含 __unknown__）
            Persons = sorted([
                D for D in os.listdir(DataDir)
                if os.path.isdir(os.path.join(DataDir, D))
            ])
            # 確認至少有 1 位真實人員
            RealPersons = [P for P in Persons if P != UNKNOWN_DIR_NAME]
            if len(RealPersons) < 1:
                print("[FaceRecognizer] 沒有真實人員資料，無法訓練。")
                return False

            NewLabelMap = {Name: Idx for Idx, Name in enumerate(Persons)}
            NumClasses  = len(Persons)

            # 判斷是否為 fine-tune（舊模型存在且類別數增加）
            IsFinetuning = (
                self._Model is not None and
                len(self._LabelToIdx) > 0 and
                NumClasses > len(self._LabelToIdx)
            )

            if IsFinetuning:
                # --- Method B：擴展舊模型 FC 層 ---
                NewModel = FaceCNN(NumClasses).to(self._Device)
                # 複製 backbone 權重
                NewModel._Features.load_state_dict(
                    self._Model._Features.state_dict()
                )
                # 複製舊 FC 權重到對應位置（保留舊人知識）
                OldWeight = self._Model._Classifier.weight.data
                OldBias   = self._Model._Classifier.bias.data
                with torch.no_grad():
                    for OldName, OldIdx in self._LabelToIdx.items():
                        NewIdx = NewLabelMap.get(OldName)
                        if NewIdx is not None:
                            NewModel._Classifier.weight.data[NewIdx] = OldWeight[OldIdx]
                            NewModel._Classifier.bias.data[NewIdx]   = OldBias[OldIdx]
                self._Model = NewModel
                Epochs = TRAIN_EPOCHS_FINETUNE
                print(f"[FaceRecognizer] Fine-tune 模式：{len(self._LabelToIdx)} → {NumClasses} 類")
            else:
                # --- 從頭訓練 ---
                self._Model = FaceCNN(NumClasses).to(self._Device)
                Epochs = TRAIN_EPOCHS_FULL
                print(f"[FaceRecognizer] 從頭訓練：{NumClasses} 類")

            # 更新標籤對應表
            self._LabelToIdx = NewLabelMap
            self._IdxToLabel = {Idx: Name for Name, Idx in NewLabelMap.items()}

            # 資料增強（適用灰階單通道 tensor）
            AugTransform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.RandomRotation(10),
                T.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            ])

            TrainDataset = FaceDataset(DataDir, NewLabelMap, Transform=AugTransform)
            TrainLoader  = DataLoader(
                TrainDataset, batch_size=BATCH_SIZE,
                shuffle=True, drop_last=False
            )

            Criterion = nn.CrossEntropyLoss()
            Optimizer = optim.Adam(self._Model.parameters(), lr=0.001)
            Scheduler = optim.lr_scheduler.StepLR(Optimizer, step_size=15, gamma=0.5)

            self._Model.train()
            BestLoss     = float("inf")
            NoImprovCount = 0

            for Epoch in range(1, Epochs + 1):
                TotalLoss = 0.0
                Batches   = 0
                for Imgs, Labels in TrainLoader:
                    Imgs   = Imgs.to(self._Device)
                    Labels = Labels.to(self._Device)
                    Optimizer.zero_grad()
                    Outputs = self._Model(Imgs)
                    Loss    = Criterion(Outputs, Labels)
                    Loss.backward()
                    Optimizer.step()
                    TotalLoss += Loss.item()
                    Batches   += 1
                AvgLoss = TotalLoss / max(Batches, 1)
                Scheduler.step()

                if ProgressCallback:
                    ProgressCallback(Epoch, Epochs, AvgLoss)

                # Early Stopping：Loss 連續 N 個 epoch 沒有顯著改善則提早停止
                if BestLoss - AvgLoss > EARLY_STOP_MIN_DELTA:
                    BestLoss      = AvgLoss
                    NoImprovCount = 0
                else:
                    NoImprovCount += 1
                    if NoImprovCount >= EARLY_STOP_PATIENCE:
                        print(f"[FaceRecognizer] Early stopping at epoch {Epoch}/{Epochs}")
                        break

            self._Model.eval()
            return True

        except Exception as Error:
            print(f"[FaceRecognizer] 訓練失敗：{Error}")
            import traceback
            traceback.print_exc()
            return False

    # --------------------------------------------------------------------------
    # 推論
    # --------------------------------------------------------------------------
    def predict(self, Roi: np.ndarray) -> tuple:
        """
        推論單張臉部 ROI。

        Roi: 96×96 灰階 numpy array（uint8）

        回傳 (name, confidence)
        信心值 < UNKNOWN_THRESHOLD 時，name = 'Unknown'
        """
        try:
            if self._Model is None or not self._IdxToLabel:
                return "Unknown", 0.0

            # 前處理
            Img    = cv2.resize(Roi, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            Tensor = torch.tensor(
                Img.astype(np.float32) / 255.0
            ).unsqueeze(0).unsqueeze(0)          # (1, 1, H, W)
            Tensor = Tensor.to(self._Device)

            self._Model.eval()
            with torch.no_grad():
                Outputs  = self._Model(Tensor)
                Probs    = torch.softmax(Outputs, dim=1)
                MaxProb, MaxIdx = torch.max(Probs, dim=1)

            Confidence = MaxProb.item()
            Name       = self._IdxToLabel.get(MaxIdx.item(), "Unknown")

            # CNN 判定為 unknown 類別，或信心值低於門檻
            if Name == UNKNOWN_DIR_NAME or Confidence < UNKNOWN_THRESHOLD:
                return "Unknown", Confidence
            return Name, Confidence

        except Exception as Error:
            print(f"[FaceRecognizer] 推論失敗：{Error}")
            return "Unknown", 0.0

    # --------------------------------------------------------------------------
    # 模型 I/O
    # --------------------------------------------------------------------------
    def saveModel(self, ModelPath: str) -> bool:
        """儲存模型權重與標籤對應表至指定路徑。"""
        try:
            os.makedirs(os.path.dirname(ModelPath), exist_ok=True)
            torch.save({
                "StateDict":  self._Model.state_dict(),
                "LabelToIdx": self._LabelToIdx,
                "NumClasses": len(self._LabelToIdx),
            }, ModelPath)
            return True
        except Exception as Error:
            print(f"[FaceRecognizer] 儲存模型失敗：{Error}")
            return False

    def loadModel(self, ModelPath: str) -> bool:
        """從指定路徑載入模型權重與標籤對應表。"""
        try:
            if not os.path.exists(ModelPath):
                return False
            Checkpoint       = torch.load(ModelPath, map_location=self._Device)
            NumClasses       = Checkpoint["NumClasses"]
            self._LabelToIdx = Checkpoint["LabelToIdx"]
            self._IdxToLabel = {Idx: Name for Name, Idx in self._LabelToIdx.items()}
            self._Model      = FaceCNN(NumClasses).to(self._Device)
            self._Model.load_state_dict(Checkpoint["StateDict"])
            self._Model.eval()
            return True
        except Exception as Error:
            print(f"[FaceRecognizer] 載入模型失敗：{Error}")
            return False
