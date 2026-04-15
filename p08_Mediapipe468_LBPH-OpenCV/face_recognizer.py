"""
face_recognizer.py

FaceRecognizer：整合 MpFaceLandmarker 與 LbphRecognizer，
提供學習（AddSample）、辨識（Predict）、儲存/載入（SaveModel/LoadModel）等功能。

與 p07 的差異：
  - 以 LbphRecognizer 取代 RandomForest
  - 從 468 個 3D landmark 萃取 5 個像素關鍵點，對齊人臉影像後交給 LBPH
  - 訓練資料儲存對齊後的 100×100 灰階影像（而非特徵向量）
  - 模型儲存為 face_model_lbph.yml（LBPH 本體）
             + face_model_lbph_meta.npz（人名對應 + 各人臉部影像）

辨識策略：
  LBPH 距離 ≤ Threshold → 已知人物（信心度由距離轉換為 0~1）
  LBPH 距離 > Threshold → Unknown

對外 API 與 p07 face_recognizer.py 保持一致，main.py 可無縫替換。
"""

import os
import numpy as np

from mp_face_landmarker import MpFaceLandmarker
from lbph_recognizer import LbphRecognizer, alignFace

# 模型儲存路徑
DEFAULT_MODEL_PATH = "face_model_lbph.yml"
DEFAULT_META_PATH  = "face_model_lbph_meta.npz"

# MediaPipe 468 點中，用於計算 5 個關鍵點的 landmark 索引
_LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
_NOSE_TIP_INDEX    = 1
_LEFT_MOUTH_INDEX  = 61    # 左嘴角
_RIGHT_MOUTH_INDEX = 291   # 右嘴角


class FaceRecognizer:
    """
    人臉辨識器。
    以 MpFaceLandmarker 取得 468 個 3D landmark，
    萃取 5 個像素關鍵點後對齊人臉影像，交由 LBPH 進行學習與辨識。
    """

    def __init__(self, ModelPath: str = DEFAULT_MODEL_PATH,
                 MetaPath: str = DEFAULT_META_PATH):
        self._ModelPath  = ModelPath
        self._MetaPath   = MetaPath
        # MediaPipe 偵測器（直接取得 468 個 3D 點）
        self._Detector   = MpFaceLandmarker()
        # LBPH 辨識器
        self._Recognizer = LbphRecognizer()
        # 訓練資料：{人名: [對齊後灰階人臉影像 (np.ndarray), ...]}
        self._Samples: dict = {}
        # 人名 ↔ 整數標籤對應（LBPH 使用整數 label）
        self._NameToLabel: dict = {}   # {人名: int}
        self._LabelToName: dict = {}   # {int: 人名}

    # ──────────────────────────────────────────────────────────────────────────
    # 公開 API（與 p07 face_recognizer.py 保持一致）
    # ──────────────────────────────────────────────────────────────────────────

    def LoadModel(self) -> bool:
        """
        載入先前儲存的 LBPH 模型與人名對應表。

        Returns
        -------
        True 表示成功，False 表示失敗或檔案不存在。
        """
        try:
            if not os.path.exists(self._ModelPath) or not os.path.exists(self._MetaPath):
                return False

            # 載入人名對應與各人臉部影像（meta.npz）
            Meta    = np.load(self._MetaPath, allow_pickle=True)
            Persons = list(Meta['persons'])
            Labels  = list(Meta['labels'].astype(int))

            self._NameToLabel = {Name: Lbl for Name, Lbl in zip(Persons, Labels)}
            self._LabelToName = {Lbl: Name for Name, Lbl in self._NameToLabel.items()}

            # 還原 _Samples（各人對齊後臉部影像，供 RemovePerson 重訓使用）
            self._Samples = {}
            for Name in Persons:
                Key = f"imgs_{Name}"
                if Key in Meta:
                    self._Samples[Name] = list(Meta[Key])
                else:
                    self._Samples[Name] = []

            # 直接載入已訓練的 LBPH 模型（不重訓）
            Ok = self._Recognizer.read(self._ModelPath)
            print(f"[FaceRecognizer] LoadModel 完成，已登錄人物：{Persons}")
            return Ok

        except Exception as Error:
            print(f"[FaceRecognizer] LoadModel 失敗：{Error}")
            return False

    def SaveModel(self) -> bool:
        """
        儲存 LBPH 模型（.yml）與人名對應 + 各人臉部影像（.npz）。

        Returns
        -------
        True 表示儲存成功，False 表示失敗或無訓練資料。
        """
        try:
            ValidPersons = {Name: Imgs for Name, Imgs in self._Samples.items() if Imgs}
            if not ValidPersons:
                return False

            # 儲存 LBPH 模型
            ModelOk = self._Recognizer.write(self._ModelPath)

            # 儲存人名對應與各人臉部影像到 npz
            Persons = list(self._NameToLabel.keys())
            Labels  = [self._NameToLabel[Name] for Name in Persons]

            SaveDict = {
                'persons': np.array(Persons, dtype=object),
                'labels':  np.array(Labels, dtype=np.int32),
            }
            for Name in Persons:
                Key = f"imgs_{Name}"
                Imgs = self._Samples.get(Name, [])
                if Imgs:
                    SaveDict[Key] = np.array(Imgs, dtype=np.uint8)

            np.savez_compressed(self._MetaPath, **SaveDict)
            print(f"[FaceRecognizer] SaveModel 完成：{self._ModelPath}, {self._MetaPath}")
            return ModelOk

        except Exception as Error:
            print(f"[FaceRecognizer] SaveModel 失敗：{Error}")
            return False

    def AddSample(self, Frame: np.ndarray, PersonName: str,
                  Retrain: bool = False) -> tuple:
        """
        從 BGR 影像偵測人臉，5 點對齊後加入訓練樣本。

        Parameters
        ----------
        Frame      : BGR 格式的 numpy 影像（來自 OpenCV）
        PersonName : 要學習的人名
        Retrain    : 加入後是否立即重訓（預設 False；學習結束後呼叫 FinishLearning）

        Returns
        -------
        (Added: bool, KeyPoints: list)
          Added     : True 表示至少成功加入一個有效樣本
          KeyPoints : 每張臉的關鍵點中心座標列表（dict 格式，供 UI 疊加顯示）
        """
        try:
            Detections = self._Detector.detect(Frame)
            if not Detections:
                return False, []

            if PersonName not in self._Samples:
                self._Samples[PersonName] = []

            H, W   = Frame.shape[:2]
            Added  = False
            KpList = []

            for _, Landmarks3D, KeyPoints in Detections:
                FivePts     = self._extractFivePts(Landmarks3D, W, H)
                AlignedFace = alignFace(Frame, FivePts)
                if AlignedFace is not None:
                    self._Samples[PersonName].append(AlignedFace)
                    Added = True
                    KpList.append(KeyPoints)

            if Added and Retrain:
                self._trainClassifier()
            return Added, KpList

        except Exception as Error:
            print(f"[FaceRecognizer] AddSample 失敗：{Error}")
            return False, []

    def FinishLearning(self) -> None:
        """
        批次學習結束後呼叫，執行一次完整的 LBPH 重訓。
        搭配 AddSample(Retrain=False) 使用，避免每幀重訓。
        """
        try:
            self._trainClassifier()
        except Exception as Error:
            print(f"[FaceRecognizer] FinishLearning 失敗：{Error}")

    def Predict(self, Frame: np.ndarray) -> list:
        """
        從 BGR 影像偵測並辨識人臉。

        Returns
        -------
        list of (Top, Right, Bottom, Left, Name, Confidence)
          Top/Right/Bottom/Left : 人臉邊界框像素座標
          Name       : 辨識結果人名，或 "Unknown"
          Confidence : 0.0~1.0 的信心度
        """
        Results = []
        try:
            if not self._Recognizer.IsTrained:
                return Results

            H, W       = Frame.shape[:2]
            Detections = self._Detector.detect(Frame)
            if not Detections:
                return Results

            for BoundingBox, Landmarks3D, _ in Detections:
                FivePts     = self._extractFivePts(Landmarks3D, W, H)
                AlignedFace = alignFace(Frame, FivePts)
                if AlignedFace is None:
                    continue

                LabelIdx, Conf = self._Recognizer.predict(AlignedFace)
                Name = self._LabelToName.get(LabelIdx, "Unknown")
                if LabelIdx == -1:
                    Name = "Unknown"

                Top, Right, Bottom, Left = BoundingBox
                Results.append((Top, Right, Bottom, Left, Name, float(Conf)))

        except Exception as Error:
            print(f"[FaceRecognizer] Predict 失敗：{Error}")
        return Results

    def CanDetect(self) -> bool:
        """若已有訓練資料且 LBPH 完成訓練，回傳 True。"""
        return self._Recognizer.IsTrained

    def GetKnownPersons(self) -> list:
        """回傳目前有訓練樣本的人名列表。"""
        return [Name for Name, Imgs in self._Samples.items() if Imgs]

    def GetAccumulatedPersons(self) -> list:
        """同 GetKnownPersons（供 main.py 呼叫）。"""
        return self.GetKnownPersons()

    def GetSampleCounts(self) -> dict:
        """回傳各人名的訓練樣本數量 {人名: 數量}。"""
        return {Name: len(Imgs) for Name, Imgs in self._Samples.items()}

    def RemovePerson(self, PersonName: str) -> bool:
        """
        移除指定人物的所有訓練樣本並重新訓練 LBPH。

        Returns
        -------
        True 表示移除成功，False 表示找不到該人名。
        """
        try:
            if PersonName not in self._Samples:
                return False

            del self._Samples[PersonName]
            # 移除標籤對應
            if PersonName in self._NameToLabel:
                Lbl = self._NameToLabel.pop(PersonName)
                self._LabelToName.pop(Lbl, None)

            if self._Samples:
                # 重建連續整數標籤後重訓
                self._rebuildLabelMap()
                self._trainClassifier()
            else:
                # 無剩餘資料，重置辨識器
                self._Recognizer = LbphRecognizer(self._Recognizer.Threshold)
            return True

        except Exception as Error:
            print(f"[FaceRecognizer] RemovePerson 失敗：{Error}")
            return False

    def SetThresholds(self, LbphThresh: float = None) -> None:
        """
        動態更新 LBPH 距離閾值，無需重訓模型。

        Parameters
        ----------
        LbphThresh : LBPH 距離閾值（越低越嚴格；預設 80.0）
        """
        try:
            if LbphThresh is not None:
                self._Recognizer.Threshold = LbphThresh
        except Exception as Error:
            print(f"[FaceRecognizer] SetThresholds 失敗：{Error}")

    # ──────────────────────────────────────────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────────────────────────────────────────

    def _extractFivePts(self, Landmarks3D: np.ndarray,
                        W: int, H: int) -> np.ndarray:
        """
        從 468 個 3D 歸一化 landmark 計算 5 個像素座標關鍵點。

        Returns
        -------
        np.ndarray, shape=(5, 2), dtype=float32
          順序：[左眼中心, 右眼中心, 鼻尖, 左嘴角, 右嘴角]
        """
        LeftEye   = Landmarks3D[_LEFT_EYE_INDICES,  :2].mean(axis=0) * [W, H]
        RightEye  = Landmarks3D[_RIGHT_EYE_INDICES, :2].mean(axis=0) * [W, H]
        NoseTip   = Landmarks3D[_NOSE_TIP_INDEX,    :2]               * [W, H]
        LeftMouth  = Landmarks3D[_LEFT_MOUTH_INDEX,  :2]              * [W, H]
        RightMouth = Landmarks3D[_RIGHT_MOUTH_INDEX, :2]              * [W, H]
        return np.array(
            [LeftEye, RightEye, NoseTip, LeftMouth, RightMouth],
            dtype=np.float32
        )

    def _trainClassifier(self) -> None:
        """
        根據目前的 _Samples 從頭訓練 LBPH。
        同時重建 _NameToLabel / _LabelToName 對應表。
        """
        try:
            ValidPersons = {Name: Imgs for Name, Imgs in self._Samples.items() if Imgs}
            if not ValidPersons:
                self._Recognizer = LbphRecognizer(self._Recognizer.Threshold)
                return

            self._rebuildLabelMap(list(ValidPersons.keys()))

            AllImages = []
            AllLabels = []
            for Name, Imgs in ValidPersons.items():
                Lbl = self._NameToLabel[Name]
                for Img in Imgs:
                    AllImages.append(Img)
                    AllLabels.append(Lbl)

            self._Recognizer.fit(AllImages, AllLabels)

        except Exception as Error:
            print(f"[FaceRecognizer] _trainClassifier 失敗：{Error}")

    def _rebuildLabelMap(self, Persons: list = None) -> None:
        """
        重新建立連續整數的 label 對應表。

        Parameters
        ----------
        Persons : 人名列表；若為 None，使用目前 _Samples 中有資料的人名。
        """
        if Persons is None:
            Persons = [Name for Name, Imgs in self._Samples.items() if Imgs]
        self._NameToLabel = {Name: Idx for Idx, Name in enumerate(Persons)}
        self._LabelToName = {Idx: Name for Name, Idx in self._NameToLabel.items()}
