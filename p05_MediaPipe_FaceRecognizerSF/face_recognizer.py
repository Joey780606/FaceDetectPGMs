"""
face_recognizer.py

FaceRecognizer 類別：整合 FaceAligner（MediaPipe 偵測對齊）與
OpenCV FaceRecognizerSF（128D 嵌入向量提取），提供：
  - AddSample()   : 學習人臉（收集 128D 嵌入向量）
  - Predict()     : 辨識人臉（Cosine Similarity 比對）
  - SaveModel()   : 儲存嵌入向量至 face_model.npz
  - LoadModel()   : 從 face_model.npz 載入嵌入向量

辨識策略：
  For each detected face:
    1. MediaPipe → 對齊 112×112
    2. FaceRecognizerSF.feature() → 128D 向量
    3. 對每位已登錄人物：平均 Cosine Similarity（所有樣本）
    4. 最高相似度 ≥ 0.363（OpenCV 官方建議閾值）→ 認出；否則 Unknown

授權：
  - face_recognition_sface_2021dec.onnx：Apache 2.0（商用安全）
  - face_model.npz：用戶自行收集的人臉嵌入，用戶自有資料
"""

import os
import numpy as np
import cv2

from face_aligner import FaceAligner

# 模型檔案路徑
_SFNET_MODEL_PATH = "face_recognition_sface_2021dec.onnx"
_DATA_MODEL_PATH  = "face_model.npz"

# Cosine Similarity 辨識閾值（OpenCV 官方建議值）
# 高於此值才視為「認識」，否則回傳 Unknown
_COSINE_THRESHOLD = 0.363


class FaceRecognizer:
    """
    人臉辨識器。

    以 FaceAligner（MediaPipe）偵測並對齊臉部，
    交由 OpenCV FaceRecognizerSF 提取 128D 特徵向量，
    再以 Cosine Similarity 進行身份比對。
    """

    def __init__(self,
                 ModelPath: str = _DATA_MODEL_PATH,
                 SfnetPath: str = _SFNET_MODEL_PATH):
        """
        Parameters
        ----------
        ModelPath : 儲存訓練嵌入向量的 .npz 路徑
        SfnetPath : FaceRecognizerSF .onnx 模型路徑
        """
        self._ModelPath  = ModelPath
        self._SfnetPath  = SfnetPath

        # 臉部對齊器（MediaPipe FaceLandmarker）
        self._Aligner    = FaceAligner()

        # OpenCV FaceRecognizerSF（提取 128D 嵌入向量）
        self._SfNet      = None
        self._loadSfNet()

        # 訓練資料：{人名: [128D np.ndarray, ...]}
        self._Samples: dict = {}

    # ──────────────────────────────────────────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────────────────────────────────────────

    def _loadSfNet(self) -> None:
        """載入 OpenCV FaceRecognizerSF 模型。"""
        try:
            if not os.path.exists(self._SfnetPath):
                print(f"[FaceRecognizer] 找不到模型檔：{self._SfnetPath}")
                return
            self._SfNet = cv2.FaceRecognizerSF.create(self._SfnetPath, "")
            print("[FaceRecognizer] FaceRecognizerSF 載入成功")
        except Exception as Error:
            print(f"[FaceRecognizer] FaceRecognizerSF 載入失敗：{Error}")
            self._SfNet = None

    def _extractFeature(self, AlignedFace: np.ndarray) -> np.ndarray:
        """
        從 112×112 對齊臉部提取 128D 嵌入向量。

        Parameters
        ----------
        AlignedFace : np.ndarray, shape (112, 112, 3), BGR

        Returns
        -------
        np.ndarray shape (128,) 或 None（失敗時）
        """
        try:
            if self._SfNet is None:
                return None
            Feature = self._SfNet.feature(AlignedFace)
            # feature() 回傳 shape (1, 128)，壓平為 (128,)
            return Feature.flatten()
        except Exception as Error:
            print(f"[FaceRecognizer] 特徵提取失敗：{Error}")
            return None

    def _cosineSimilarity(self, Vec1: np.ndarray, Vec2: np.ndarray) -> float:
        """
        計算兩個 128D 向量的 Cosine Similarity。
        回傳值範圍 [0, 1]，值愈大表示愈相似。
        """
        try:
            # FaceRecognizerSF 提供內建的 match 方法（更精確）
            if self._SfNet is not None:
                Score = self._SfNet.match(
                    Vec1.reshape(1, -1),
                    Vec2.reshape(1, -1),
                    cv2.FaceRecognizerSF_FR_COSINE
                )
                return float(Score)
            # 回退至手動計算
            N1 = np.linalg.norm(Vec1)
            N2 = np.linalg.norm(Vec2)
            if N1 == 0 or N2 == 0:
                return 0.0
            return float(np.dot(Vec1, Vec2) / (N1 * N2))
        except Exception:
            return 0.0

    def _predictForEmbedding(self, QueryVec: np.ndarray) -> tuple:
        """
        以單一嵌入向量在所有已登錄人物中搜尋最佳匹配。

        Returns
        -------
        (Name, Confidence)
          Name       : 最佳匹配人名，或 "Unknown"
          Confidence : Cosine Similarity 分數（float）
        """
        if not self._Samples:
            return "Unknown", 0.0

        BestName  = "Unknown"
        BestScore = -1.0

        for PersonName, Embeddings in self._Samples.items():
            if not Embeddings:
                continue
            # 對此人所有樣本計算相似度，取平均
            Scores = [self._cosineSimilarity(QueryVec, Emb) for Emb in Embeddings]
            AvgScore = float(np.mean(Scores))
            if AvgScore > BestScore:
                BestScore = AvgScore
                BestName  = PersonName

        if BestScore >= _COSINE_THRESHOLD:
            return BestName, BestScore
        return "Unknown", BestScore

    # ──────────────────────────────────────────────────────────────────────────
    # 公開 API
    # ──────────────────────────────────────────────────────────────────────────

    def LoadModel(self) -> bool:
        """
        從 face_model.npz 載入先前儲存的嵌入向量。

        Returns
        -------
        True 表示成功，False 表示失敗或檔案不存在。
        """
        try:
            if not os.path.exists(self._ModelPath):
                return False

            Data    = np.load(self._ModelPath, allow_pickle=True)
            Persons = list(Data["persons"])
            X       = Data["X"]    # shape (N, 128)
            Y       = Data["Y"]    # shape (N,)

            self._Samples = {}
            for Idx, Name in enumerate(Persons):
                Mask = Y == Idx
                self._Samples[str(Name)] = list(X[Mask].astype(np.float32))

            print(f"[FaceRecognizer] LoadModel 成功，已登錄人物：{list(self._Samples.keys())}")
            return True

        except Exception as Error:
            print(f"[FaceRecognizer] LoadModel 失敗：{Error}")
            return False

    def SaveModel(self) -> bool:
        """
        將目前所有嵌入向量儲存至 face_model.npz。

        Returns
        -------
        True 表示成功，False 表示失敗或無資料可儲存。
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

            X = np.array(XList, dtype=np.float32)
            Y = np.array(YList, dtype=np.int32)

            np.savez_compressed(
                self._ModelPath,
                X=X,
                Y=Y,
                persons=np.array(Persons, dtype=object),
            )
            print(f"[FaceRecognizer] SaveModel 成功：{self._ModelPath}")
            return True

        except Exception as Error:
            print(f"[FaceRecognizer] SaveModel 失敗：{Error}")
            return False

    def AddSample(self, Frame: np.ndarray, PersonName: str,
                  Retrain: bool = False) -> tuple:
        """
        從 BGR 影像中偵測人臉，提取 128D 嵌入向量並加入訓練樣本。

        Parameters
        ----------
        Frame      : BGR 格式的 numpy 影像（來自 OpenCV）
        PersonName : 要學習的人名
        Retrain    : 保留參數（與 p04 介面相容，FaceRecognizerSF 不需重訓）

        Returns
        -------
        (Added: bool, KeyPoints: list)
          Added     : True 表示至少成功加入一個有效樣本
          KeyPoints : 每張臉的關鍵點中心座標列表（供 UI 疊加顯示）
        """
        try:
            # MediaPipe 偵測並對齊
            Detections = self._Aligner.Detect(Frame)
            if not Detections:
                return False, []

            if PersonName not in self._Samples:
                self._Samples[PersonName] = []

            Added     = False
            KeyPoints = []

            for AlignedFace, BoundingBox, KP in Detections:
                try:
                    Feature = self._extractFeature(AlignedFace)
                    if Feature is not None:
                        self._Samples[PersonName].append(Feature)
                        Added = True
                        KeyPoints.append(KP)
                except Exception as FaceError:
                    print(f"[FaceRecognizer] 樣本加入失敗：{FaceError}")

            return Added, KeyPoints

        except Exception as Error:
            print(f"[FaceRecognizer] AddSample 失敗：{Error}")
            return False, []

    def Predict(self, Frame: np.ndarray) -> list:
        """
        從 BGR 影像中偵測並辨識所有人臉。

        Returns
        -------
        list of (Top, Right, Bottom, Left, Name, Confidence)
          Top/Right/Bottom/Left : int，像素座標
          Name                  : str，人名或 "Unknown"
          Confidence            : float，Cosine Similarity 分數
        """
        try:
            Detections = self._Aligner.Detect(Frame)
            if not Detections:
                return []

            Results = []
            for AlignedFace, BoundingBox, KP in Detections:
                try:
                    Feature = self._extractFeature(AlignedFace)
                    if Feature is None:
                        continue
                    Name, Confidence = self._predictForEmbedding(Feature)
                    Top, Right, Bottom, Left = BoundingBox
                    Results.append((Top, Right, Bottom, Left, Name, Confidence))
                except Exception as FaceError:
                    print(f"[FaceRecognizer] 單臉辨識失敗：{FaceError}")

            return Results

        except Exception as Error:
            print(f"[FaceRecognizer] Predict 失敗：{Error}")
            return []

    def CanDetect(self) -> bool:
        """回傳是否有任何已登錄的人臉樣本（可以開始辨識）。"""
        return bool(self._Samples)

    def GetSampleCounts(self) -> dict:
        """
        回傳每位人物的樣本數量。

        Returns
        -------
        dict {人名: 樣本數量}
        """
        return {Name: len(Vecs) for Name, Vecs in self._Samples.items() if Vecs}

    def GetKnownPersons(self) -> list:
        """回傳所有已登錄人物的姓名清單。"""
        return [Name for Name, Vecs in self._Samples.items() if Vecs]

    def GetAccumulatedPersons(self) -> list:
        """回傳所有已登錄人物的姓名清單（與 GetKnownPersons 相同，保留介面相容性）。"""
        return self.GetKnownPersons()

    def RemovePerson(self, PersonName: str) -> bool:
        """
        移除指定人物的所有訓練樣本，並更新 face_model.npz。

        Returns
        -------
        True 表示移除並儲存成功，False 表示人物不存在或儲存失敗。
        """
        try:
            if PersonName not in self._Samples:
                return False
            del self._Samples[PersonName]

            # 若移除後還有其他人的資料，重新儲存；否則刪除檔案
            if self._Samples:
                return self.SaveModel()
            else:
                if os.path.exists(self._ModelPath):
                    os.remove(self._ModelPath)
                return True

        except Exception as Error:
            print(f"[FaceRecognizer] RemovePerson 失敗：{Error}")
            return False

    def FinishLearning(self) -> None:
        """
        學習結束後呼叫（與 p04 介面相容）。
        FaceRecognizerSF 是比對式方法，不需重訓，此函數為空操作。
        """
        pass
