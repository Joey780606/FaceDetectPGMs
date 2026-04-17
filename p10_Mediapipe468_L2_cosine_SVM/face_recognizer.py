"""
face_recognizer.py

FaceRecognizer 類別：整合 MpFaceLandmarker、face_feature_3d 與 svm_classifier_np，
提供學習（AddSample）、辨識（Predict）、儲存/載入（SaveModel/LoadModel）等功能。

辨識策略（OvR 線性 SVM）：
  - 訓練時：Z-Score 標準化 + L2 正規化後，以 SGD hinge loss 訓練 OvR 線性 SVM。
  - 推論時：計算各分類器的原始分數，sigmoid 轉換後與閾值比較決定 Known / Unknown。
  - 單人模式：自動生成合成負樣本，1 人即可訓練並偵測陌生人。

模型資料儲存至 face_model.npz（純 NumPy 壓縮格式，無 sklearn/scipy 依賴）。
"""

import os
import numpy as np

from mp_face_landmarker import MpFaceLandmarker
from face_feature_3d import extractFeatures3D
from svm_classifier_np import SvmClassifier, SVM_UNKNOWN_THRESH

# 模型儲存路徑（相對於執行目錄）
DEFAULT_MODEL_PATH = "face_model.npz"


class FaceRecognizer:
    """
    人臉辨識器。
    以 MpFaceLandmarker（MediaPipe FaceLandmarker）取得 468 個 3D landmark，
    萃取 351 維臉部比例特徵向量後交由 SvmClassifier 進行線性 SVM 分類。
    """

    def __init__(self, ModelPath: str = DEFAULT_MODEL_PATH):
        self._ModelPath  = ModelPath
        # MediaPipe 偵測器（直接取得 468 3D 點）
        self._Detector   = MpFaceLandmarker()
        # 訓練資料：{人名: [特徵向量 (np.ndarray), ...]}
        self._Samples: dict = {}
        # Cosine 比對器，未訓練時為 None
        self._Matcher    = None
        self._IsTrained  = False

    # ──────────────────────────────────────────────────────────────────────────
    # 公開 API
    # ──────────────────────────────────────────────────────────────────────────

    def LoadModel(self) -> bool:
        """
        載入先前儲存的訓練資料（face_model.npz）並重新計算 Cosine 比對器。

        Returns
        -------
        True 表示成功載入並完成訓練，False 表示失敗或檔案不存在。
        """
        try:
            if not os.path.exists(self._ModelPath):
                return False
            Data    = np.load(self._ModelPath, allow_pickle=True)
            Persons = list(Data['persons'])
            X       = Data['X']
            Y       = Data['Y']

            # 重建 _Samples dict
            self._Samples = {}
            for Idx, Name in enumerate(Persons):
                Mask = Y == Idx
                self._Samples[str(Name)] = list(X[Mask])

            # 重新訓練（計算各人平均向量，速度很快）
            self._trainMatcher()
            return self._IsTrained

        except Exception as Error:
            print(f"[FaceRecognizer] LoadModel 失敗：{Error}")
            return False

    def SaveModel(self) -> bool:
        """
        將目前的訓練樣本儲存至 face_model.npz。

        Returns
        -------
        True 表示儲存成功，False 表示失敗或無資料可儲存。
        """
        try:
            ValidPersons = {Name: Vecs for Name, Vecs in self._Samples.items() if Vecs}
            if not ValidPersons:
                return False

            Persons = list(ValidPersons.keys())
            XList, YList = [], []
            for Idx, Name in enumerate(Persons):
                for Vec in ValidPersons[Name]:
                    XList.append(Vec)
                    YList.append(Idx)

            X = np.array(XList, dtype=float)
            Y = np.array(YList, dtype=int)
            np.savez_compressed(
                self._ModelPath,
                X=X,
                Y=Y,
                persons=np.array(Persons, dtype=object),
            )
            return True

        except Exception as Error:
            print(f"[FaceRecognizer] SaveModel 失敗：{Error}")
            return False

    def AddSample(self, Frame: np.ndarray, PersonName: str,
                  Retrain: bool = False) -> tuple:
        """
        從 BGR 影像中偵測人臉，萃取 1404 維特徵向量並加入訓練樣本。

        Parameters
        ----------
        Frame      : BGR 格式的 numpy 影像（來自 OpenCV）
        PersonName : 要學習的人名
        Retrain    : 加入樣本後是否立即重新計算比對器。
                     批次學習時傳 False（預設），學習結束後再呼叫 FinishLearning()。

        Returns
        -------
        (Added: bool, KeyPoints: list)
          Added     : True 表示至少成功加入一個有效樣本
          KeyPoints : 每張臉的關鍵點中心座標列表
        """
        try:
            Detections = self._Detector.detect(Frame)
            if not Detections:
                return False, []

            if PersonName not in self._Samples:
                self._Samples[PersonName] = []

            Added = False
            KeyPointsList = []
            for _, Landmarks3D, KeyPoints in Detections:
                Vec = extractFeatures3D(Landmarks3D)    # 1404 維特徵向量
                if Vec is not None:
                    self._Samples[PersonName].append(Vec)
                    Added = True
                    KeyPointsList.append(KeyPoints)

            if Added and Retrain:
                self._trainMatcher()
            return Added, KeyPointsList

        except Exception as Error:
            print(f"[FaceRecognizer] AddSample 失敗：{Error}")
            return False, []

    def FinishLearning(self) -> None:
        """
        批次學習結束後呼叫，執行一次比對器重新計算。
        搭配 AddSample(Retrain=False) 使用。
        """
        try:
            self._trainMatcher()
        except Exception as Error:
            print(f"[FaceRecognizer] FinishLearning 失敗：{Error}")

    def Predict(self, Frame: np.ndarray) -> list:
        """
        從 BGR 影像中偵測並辨識人臉。

        Parameters
        ----------
        Frame : BGR 格式的 numpy 影像（來自 OpenCV）

        Returns
        -------
        list of (Top, Right, Bottom, Left, Name, Confidence)
          Top/Right/Bottom/Left : 人臉邊界框座標（像素）
          Name       : 辨識結果人名，或 "Unknown"
          Confidence : Cosine 相似度（0.0 ~ 1.0）
        """
        Results = []
        try:
            if not self._IsTrained or self._Matcher is None:
                return Results

            Detections = self._Detector.detect(Frame)
            if not Detections:
                return Results

            ValidBoxes = []
            Vecs       = []
            for BoundingBox, Landmarks3D, _ in Detections:
                Vec = extractFeatures3D(Landmarks3D)
                if Vec is not None:
                    ValidBoxes.append(BoundingBox)
                    Vecs.append(Vec)

            if not Vecs:
                return Results

            X = np.array(Vecs, dtype=float)
            Names, Confs = self._Matcher.predict(X)

            for j, (Top, Right, Bottom, Left) in enumerate(ValidBoxes):
                Results.append((Top, Right, Bottom, Left, Names[j], float(Confs[j])))

        except Exception as Error:
            print(f"[FaceRecognizer] Predict 失敗：{Error}")
        return Results

    def CanDetect(self) -> bool:
        """若已有訓練資料且比對器完成計算，回傳 True。"""
        return self._IsTrained and self._Matcher is not None

    def SetThresholds(self, CosineThresh: float = None) -> None:
        """
        動態更新 Cosine 相似度閾值，無需重新計算模型。

        Parameters
        ----------
        CosineThresh : Cosine 相似度閾值（低於此值 → Unknown）
        """
        try:
            if CosineThresh is not None and self._Matcher is not None:
                self._Matcher._Threshold = CosineThresh
        except Exception as Error:
            print(f"[FaceRecognizer] SetThresholds 失敗：{Error}")

    def GetKnownPersons(self) -> list:
        """回傳目前有訓練樣本的人名列表。"""
        return [Name for Name, Vecs in self._Samples.items() if Vecs]

    def GetAccumulatedPersons(self) -> list:
        """同 GetKnownPersons（供 main.py 呼叫）。"""
        return self.GetKnownPersons()

    def GetSampleCounts(self) -> dict:
        """回傳各人名的訓練樣本數量 {人名: 數量}。"""
        return {Name: len(Vecs) for Name, Vecs in self._Samples.items()}

    def RemovePerson(self, PersonName: str) -> bool:
        """
        移除指定人物的所有訓練樣本並重新計算比對器。

        Returns
        -------
        True 表示移除成功，False 表示找不到該人名。
        """
        try:
            if PersonName not in self._Samples:
                return False
            del self._Samples[PersonName]
            if self._Samples:
                self._trainMatcher()
            else:
                self._Matcher   = None
                self._IsTrained = False
            return True

        except Exception as Error:
            print(f"[FaceRecognizer] RemovePerson 失敗：{Error}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────────────────────────────────────────

    def _trainMatcher(self) -> None:
        """
        根據目前的 _Samples 重新訓練 SvmClassifier（OvR 線性 SVM）。
        """
        try:
            ValidPersons = {Name: Vecs for Name, Vecs in self._Samples.items() if Vecs}
            if not ValidPersons:
                self._Matcher   = None
                self._IsTrained = False
                return

            # 保留現有閾值（如果 Matcher 已存在）
            Threshold = (self._Matcher._Threshold
                         if self._Matcher is not None
                         else SVM_UNKNOWN_THRESH)

            Matcher = SvmClassifier(Threshold=Threshold)
            Matcher.fit(ValidPersons)
            self._Matcher   = Matcher
            self._IsTrained = Matcher.IsTrained

        except Exception as Error:
            print(f"[FaceRecognizer] 訓練 SvmClassifier 失敗：{Error}")
            self._Matcher   = None
            self._IsTrained = False
