"""
face_recognizer.py

FaceRecognizer 類別：整合 MpFaceLandmarker、face_feature_3d 與 random_forest_np，
提供學習（AddSample）、辨識（Predict）、儲存/載入（SaveModel/LoadModel）等功能。

與 p04 的差異：
  - 以 MpFaceLandmarker 取代 MpFaceDetector，直接取得 468 個 3D 歸一化 landmark
  - 以 extractFeatures3D 取代 extractFeatures，特徵向量從 23 維升至 1404 維
  - 其餘辨識策略、儲存格式與 p04 完全相同

辨識策略（混合方案）：
  1 人已訓練 → OnePerson（馬氏距離閾值）
  2+ 人已訓練 → RandomForest 初步分類 + 馬氏距離二次驗證
    RF 認出人名 且 該人的馬氏距離在閾值內 → 確認為此人
    RF 認出人名 但 馬氏距離過大（臉型不像此人）→ 改判 Unknown
    RF 直接判為 Unknown → Unknown

模型資料儲存至 face_model.npz（純 NumPy 壓縮格式，無 sklearn/scipy 依賴）。
"""

import os
import numpy as np

from mp_face_landmarker import MpFaceLandmarker
from face_feature_3d import extractFeatures3D
from random_forest_np import RandomForest, OnePerson

# 模型儲存路徑（相對於執行目錄）
DEFAULT_MODEL_PATH = "face_model.npz"


class FaceRecognizer:
    """
    人臉辨識器。
    以 MpFaceLandmarker（MediaPipe FaceLandmarker）取得 468 個 3D landmark，
    萃取 1404 維特徵向量後交由純 NumPy 分類器進行學習與辨識。
    """

    def __init__(self, ModelPath: str = DEFAULT_MODEL_PATH):
        self._ModelPath  = ModelPath
        # MediaPipe 偵測器（直接取得 468 3D 點）
        self._Detector   = MpFaceLandmarker()
        # 訓練資料：{人名: [特徵向量 (np.ndarray), ...]}
        self._Samples: dict = {}
        # 主分類器：OnePerson（1人）或 RandomForest（2+人），未訓練時為 None
        self._Classifier     = None
        # 馬氏距離二次驗證器：{人名: OnePerson}，多人模式才使用
        self._Validators: dict = {}
        self._IsTrained      = False

    # ──────────────────────────────────────────────────────────────────────────
    # 公開 API
    # ──────────────────────────────────────────────────────────────────────────

    def LoadModel(self) -> bool:
        """
        載入先前儲存的訓練資料（face_model.npz）並重新訓練分類器。

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

            self._trainClassifier()
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
        Retrain    : 加入樣本後是否立即重訓分類器。
                     批次學習時傳 False（預設），學習結束後再呼叫 FinishLearning()。
                     單次加入時傳 True 可立即更新分類器。

        Returns
        -------
        (Added: bool, KeyPoints: list)
          Added     : True 表示至少成功加入一個有效樣本
          KeyPoints : 每張臉的關鍵點中心座標列表，每個元素為 dict：
                      {"left_eye": (cx,cy), "right_eye": (cx,cy),
                       "nose": (cx,cy), "mouth": (cx,cy)}
        """
        try:
            # MpFaceLandmarker.detect() 回傳 [(BoundingBox, Landmarks3D, KeyPoints), ...]
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

            # Retrain=False 時跳過重訓（批次學習期間），避免每幀都訓練 100 棵樹
            if Added and Retrain:
                self._trainClassifier()
            return Added, KeyPointsList

        except Exception as Error:
            print(f"[FaceRecognizer] AddSample 失敗：{Error}")
            return False, []

    def FinishLearning(self) -> None:
        """
        批次學習結束後呼叫，執行一次完整的分類器重訓。
        搭配 AddSample(Retrain=False) 使用，避免每幀重訓的效能問題。
        """
        try:
            self._trainClassifier()
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
          Confidence : 0.0～1.0 的信心度
        """
        Results = []
        try:
            if not self._IsTrained or self._Classifier is None:
                return Results

            # 一次偵測：取得 BoundingBox、Landmarks3D、KeyPoints
            Detections = self._Detector.detect(Frame)
            if not Detections:
                return Results

            # 萃取各張臉的特徵向量（過濾側臉 / IOD 不足的情況）
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
            Names, Confs = self._Classifier.predict(X)

            # 多人模式：RF 結果再經馬氏距離驗證（混合方案）
            if self._Validators:
                Names, Confs = self._hybridValidate(Vecs, Names, Confs)

            for j, (Top, Right, Bottom, Left) in enumerate(ValidBoxes):
                Results.append((Top, Right, Bottom, Left, Names[j], float(Confs[j])))

        except Exception as Error:
            print(f"[FaceRecognizer] Predict 失敗：{Error}")
        return Results

    def CanDetect(self) -> bool:
        """若已有訓練資料且分類器完成訓練，回傳 True。"""
        return self._IsTrained and self._Classifier is not None

    def SetThresholds(self, MahalThresh: float = None, RfThresh: float = None) -> None:
        """
        動態更新辨識閾值，無需重訓模型。

        Parameters
        ----------
        MahalThresh : 馬氏距離閾值（OnePerson 模式 + 多人模式的馬氏驗證器）
        RfThresh    : Random Forest 信心度閾值（多人模式主分類器）
        """
        try:
            if MahalThresh is not None:
                # 單人模式：直接更新主分類器
                if isinstance(self._Classifier, OnePerson):
                    self._Classifier._UnknownThreshold = MahalThresh
                # 多人模式：更新各人的馬氏距離驗證器
                for Val in self._Validators.values():
                    Val._UnknownThreshold = MahalThresh
            if RfThresh is not None:
                if isinstance(self._Classifier, RandomForest):
                    self._Classifier._UnknownThreshold = RfThresh
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
        移除指定人物的所有訓練樣本並重新訓練分類器。

        Returns
        -------
        True 表示移除成功，False 表示找不到該人名。
        """
        try:
            if PersonName not in self._Samples:
                return False
            del self._Samples[PersonName]
            if self._Samples:
                self._trainClassifier()
            else:
                self._Classifier = None
                self._IsTrained  = False
            return True

        except Exception as Error:
            print(f"[FaceRecognizer] RemovePerson 失敗：{Error}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────────────────────────────────────────

    def _trainClassifier(self) -> None:
        """
        根據目前的 _Samples 重新訓練分類器。

        1 人 → OnePerson（馬氏距離）
        2+ 人 → RandomForest（主分類）+ 各人 OnePerson（二次驗證）
        """
        try:
            # 過濾掉沒有樣本的人名
            ValidPersons = {Name: Vecs for Name, Vecs in self._Samples.items() if Vecs}
            if not ValidPersons:
                self._Classifier = None
                self._Validators = {}
                self._IsTrained  = False
                return

            Persons = list(ValidPersons.keys())
            NPerson = len(Persons)

            if NPerson == 1:
                # 單人模式：馬氏距離分類器，不需要 Validators
                Name = Persons[0]
                X    = np.array(ValidPersons[Name], dtype=float)
                Clf  = OnePerson(PersonName=Name)
                Clf.fit(X)
                self._Classifier = Clf
                self._Validators = {}
                self._IsTrained  = Clf.IsTrained

            else:
                # 多人模式：隨機森林 + 各人馬氏距離驗證器
                XList, YList = [], []
                for Idx, Name in enumerate(Persons):
                    for Vec in ValidPersons[Name]:
                        XList.append(Vec)
                        YList.append(Idx)

                X      = np.array(XList, dtype=float)
                Y      = np.array(YList, dtype=int)
                Forest = RandomForest()
                Forest.fit(X, Y, ClassNames=Persons)
                self._Classifier = Forest
                self._IsTrained  = Forest.IsTrained

                # 為每位已訓練的人建立各自的馬氏距離驗證器
                self._Validators = {}
                for Name in Persons:
                    Xp  = np.array(ValidPersons[Name], dtype=float)
                    Val = OnePerson(PersonName=Name)
                    Val.fit(Xp)
                    self._Validators[Name] = Val

        except Exception as Error:
            print(f"[FaceRecognizer] 訓練分類器失敗：{Error}")
            self._Classifier = None
            self._Validators = {}
            self._IsTrained  = False

    def _hybridValidate(self, Vecs: list, Names: list,
                        Confs: np.ndarray) -> tuple:
        """
        混合驗證：對 RF 已辨識出人名的結果，用該人的馬氏距離驗證器再次確認。

        若馬氏距離超過閾值（臉型不像 RF 選出的那個人），改判為 Unknown。
        RF 已判為 Unknown 的結果維持不變。

        Parameters
        ----------
        Vecs  : 各張臉的特徵向量列表
        Names : RF 預測的人名列表
        Confs : RF 預測的信心度陣列

        Returns
        -------
        (FinalNames: list, FinalConfs: np.ndarray)
        """
        FinalNames = []
        FinalConfs = []
        RfThresh   = self._Classifier._UnknownThreshold

        for Vec, Name, Conf in zip(Vecs, Names, Confs):
            if Name == "Unknown":
                # RF 已判為 Unknown，不再額外驗證
                print(f"[辨識] RF信心度={Conf:.2f}(閾{RfThresh:.2f}) ✗Unknown")
                FinalNames.append("Unknown")
                FinalConfs.append(float(Conf))
                continue

            Validator = self._Validators.get(Name)
            if Validator is None or not Validator.IsTrained:
                # 找不到對應的驗證器，保留 RF 結果
                print(f"[辨識] RF信心度={Conf:.2f}(閾{RfThresh:.2f}) ✓{Name}  馬氏驗證器不存在")
                FinalNames.append(Name)
                FinalConfs.append(float(Conf))
                continue

            # 取馬氏距離後以 Silent=True 呼叫 predict（避免重複列印）
            MahalDist   = Validator.getMahalDist(Vec)
            MahalThresh = Validator._UnknownThreshold
            ValNames, _ = Validator.predict(Vec.reshape(1, -1), Silent=True)
            FinalName   = ValNames[0]

            MahalMark = "✓通過" if FinalName != "Unknown" else "✗Unknown"
            print(f"[辨識] RF信心度={Conf:.2f}(閾{RfThresh:.2f}) ✓{Name}"
                  f"  馬氏距離={MahalDist:.2f}(閾{MahalThresh:.1f}) {MahalMark}"
                  f"  → {FinalName}")

            if FinalName == "Unknown":
                FinalNames.append("Unknown")
                FinalConfs.append(0.0)
            else:
                FinalNames.append(Name)
                FinalConfs.append(float(Conf))

        return FinalNames, np.array(FinalConfs, dtype=float)
