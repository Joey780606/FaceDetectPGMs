"""
face_recognizer.py

FaceRecognizer 類別：整合 MpFaceDetector、face_feature 與 random_forest_np，
提供學習（AddSample）、辨識（Predict）、儲存/載入（SaveModel/LoadModel）等功能。

與 p03 的差異：
  - 以 MpFaceDetector（MediaPipe）取代 face_recognition / dlib
  - detect() 一次回傳 (BoundingBox, LandmarkDict)，不再分兩步驟
  - 其餘辨識策略、儲存格式與 p03 完全相同

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

from mp_face_detector import MpFaceDetector
from face_feature import extractFeatures
from random_forest_np import RandomForest, OnePerson

# 模型儲存路徑（相對於執行目錄）
DEFAULT_MODEL_PATH = "face_model.npz"


class FaceRecognizer:
    """
    人臉辨識器。
    以 MpFaceDetector（MediaPipe FaceMesh）偵測 68 個等效 landmark，
    萃取幾何特徵後交由純 NumPy 分類器進行學習與辨識。
    """

    def __init__(self, ModelPath: str = DEFAULT_MODEL_PATH):
        self._ModelPath  = ModelPath
        # MediaPipe 偵測器（取代 face_recognition）
        self._Detector   = MpFaceDetector()
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

    def AddSample(self, Frame: np.ndarray, PersonName: str) -> bool:
        """
        從 BGR 影像中偵測人臉，萃取 68 個等效 landmark 特徵並加入訓練樣本。
        每次成功加入後自動重新訓練分類器。

        Parameters
        ----------
        Frame      : BGR 格式的 numpy 影像（來自 OpenCV）
        PersonName : 要學習的人名

        Returns
        -------
        True 表示至少成功加入一個有效樣本，False 表示未偵測到人臉或特徵無效。
        """
        try:
            # MpFaceDetector.detect() 一次回傳 [(BoundingBox, LandmarkDict), ...]
            Detections = self._Detector.detect(Frame)
            if not Detections:
                return False

            if PersonName not in self._Samples:
                self._Samples[PersonName] = []

            Added = False
            for _, LandmarkDict in Detections:
                Vec = extractFeatures(LandmarkDict) #Joey: 從臉部提取特徵,共23維
                if Vec is not None:
                    self._Samples[PersonName].append(Vec)
                    Added = True

            if Added:
                self._trainClassifier()
            return Added

        except Exception as Error:
            print(f"[FaceRecognizer] AddSample 失敗：{Error}")
            return False

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

            # 一次偵測：同時取得 BoundingBox 與 LandmarkDict
            Detections = self._Detector.detect(Frame)
            if not Detections:
                return Results

            # 萃取各張臉的特徵向量（過濾側臉 / IOD 不足的情況）
            ValidBoxes = []
            Vecs       = []
            for BoundingBox, LandmarkDict in Detections:
                Vec = extractFeatures(LandmarkDict)
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

            for j, (Top, Right, Bottom, Left) in enumerate(ValidBoxes): #一個影像可能有多個臉,所以j 是第幾個人,和每個人的範圍
                Results.append((Top, Right, Bottom, Left, Names[j], float(Confs[j])))

        except Exception as Error:
            print(f"[FaceRecognizer] Predict 失敗：{Error}")
        return Results

    def CanDetect(self) -> bool:
        """若已有訓練資料且分類器完成訓練，回傳 True。"""
        return self._IsTrained and self._Classifier is not None

    def GetKnownPersons(self) -> list:
        """回傳目前有訓練樣本的人名列表。"""
        return [Name for Name, Vecs in self._Samples.items() if Vecs]

    def GetAccumulatedPersons(self) -> list:
        """同 GetKnownPersons（供 main.py 呼叫）。"""
        return self.GetKnownPersons()

    def GetSampleCounts(self) -> dict:
        """回傳各人名的訓練樣本數量 {人名: 數量}。"""
        return {Name: len(Vecs) for Name, Vecs in self._Samples.items()}    #Joey: 一個人可能有多個訓練樣本

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
            ValidPersons = {Name: Vecs for Name, Vecs in self._Samples.items() if Vecs} # Joey: 從_Samples這個 dict,過濾出有樣本的人,並建立新的dict
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

    def _hybridValidate(self, Vecs: list, Names: list, Confs: np.ndarray) -> tuple:
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
        for Vec, Name, Conf in zip(Vecs, Names, Confs):
            if Name == "Unknown":
                # RF 已判為 Unknown，不再額外驗證
                FinalNames.append("Unknown")
                FinalConfs.append(float(Conf))
                continue

            Validator = self._Validators.get(Name)
            if Validator is None or not Validator.IsTrained:
                # 找不到對應的驗證器，保留 RF 結果
                FinalNames.append(Name)
                FinalConfs.append(float(Conf))
                continue

            # 用馬氏距離驗證：若距離過大則改判 Unknown
            ValNames, _ = Validator.predict(Vec.reshape(1, -1))
            if ValNames[0] == "Unknown":
                FinalNames.append("Unknown")
                FinalConfs.append(0.0)
            else:
                FinalNames.append(Name)
                FinalConfs.append(float(Conf))

        return FinalNames, np.array(FinalConfs, dtype=float)
