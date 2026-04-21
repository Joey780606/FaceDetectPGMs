"""
face_recognizer.py  (p11)

FaceRecognizer 類別：整合 MpFaceLandmarker、face_feature_3d、
face_pose_classifier 與 svm_classifier_np，實作五類姿態 SVM 人臉辨識。

【與 p10 的差異】
  - 訓練資料按姿態類別（0~4）分類儲存
  - 訓練時依各姿態分別建立一個 SvmClassifier（共 5 個）
  - 推論時先判斷當前姿態，再選對應分類器；若無則備援使用正臉分類器
  - NPZ 格式新增 P 欄位（姿態標籤 0~4）
  - AddSample 回傳值增加 PoseCat（供 UI 即時顯示姿態）
  - Predict 回傳值增加 PoseCat（供 UI 即時顯示姿態）

模型儲存至 face_model.npz（純 NumPy，無 sklearn/scipy 依賴）。
"""

import os
import numpy as np

from mp_face_landmarker import MpFaceLandmarker
from face_feature_3d import extractFeatures3D
from face_pose_classifier import (classifyPose, classifyPoseWithValues,
                                  POSE_FRONTAL, POSE_NAMES)
from svm_classifier_np import SvmClassifier, SVM_UNKNOWN_THRESH

DEFAULT_MODEL_PATH = "face_model.npz"
N_POSES = 5


class FaceRecognizer:
    """
    五類姿態 SVM 人臉辨識器。
    以 MpFaceLandmarker 取得 468 個 3D landmark，
    萃取 325 維向量夾角特徵，依姿態類別交由對應的 SvmClassifier 分類。
    """

    def __init__(self, ModelPath: str = DEFAULT_MODEL_PATH):
        self._ModelPath   = ModelPath
        self._Detector    = MpFaceLandmarker()
        # 訓練資料：{人名: {姿態類別(0~4): [特徵向量, ...]}}
        self._Samples: dict = {}
        # 五個姿態各一個分類器（未有訓練資料時為 None）
        self._Classifiers = [None] * N_POSES
        # 共用信心度閾值
        self._Threshold   = SVM_UNKNOWN_THRESH
        self._IsTrained   = False

    # ──────────────────────────────────────────────────────────────────────────
    # 公開 API
    # ──────────────────────────────────────────────────────────────────────────

    def LoadModel(self) -> bool:
        """
        載入 face_model.npz 並重新訓練五個姿態分類器。

        Returns
        -------
        True 表示成功載入，False 表示失敗或檔案不存在。
        """
        try:
            if not os.path.exists(self._ModelPath):
                return False
            Data    = np.load(self._ModelPath, allow_pickle=True)
            Persons = list(Data['persons'])
            X       = Data['X']
            Y       = Data['Y']
            P       = Data['P']

            self._Samples = {}
            for Idx, Name in enumerate(Persons):
                PersonMask = (Y == Idx)
                self._Samples[str(Name)] = {}
                for PoseCat in range(N_POSES):
                    PoseMask = PersonMask & (P == PoseCat)
                    if PoseMask.any():
                        self._Samples[str(Name)][PoseCat] = list(X[PoseMask])

            self._trainMatcher()
            return self._IsTrained

        except Exception as Error:
            print(f"[FaceRecognizer] LoadModel 失敗：{Error}")
            return False

    def SaveModel(self) -> bool:
        """
        將訓練樣本儲存至 face_model.npz。

        Returns
        -------
        True 表示儲存成功，False 表示失敗或無資料。
        """
        try:
            ValidPersons = {
                Name: PoseDict
                for Name, PoseDict in self._Samples.items()
                if any(Vecs for Vecs in PoseDict.values())
            }
            if not ValidPersons:
                return False

            Persons = list(ValidPersons.keys())
            XList, YList, PList = [], [], []
            for Idx, Name in enumerate(Persons):
                for PoseCat, Vecs in ValidPersons[Name].items():
                    for Vec in Vecs:
                        XList.append(Vec)
                        YList.append(Idx)
                        PList.append(PoseCat)

            X = np.array(XList, dtype=float)
            Y = np.array(YList, dtype=int)
            P = np.array(PList, dtype=int)
            np.savez_compressed(
                self._ModelPath,
                X=X, Y=Y, P=P,
                persons=np.array(Persons, dtype=object),
            )
            return True

        except Exception as Error:
            print(f"[FaceRecognizer] SaveModel 失敗：{Error}")
            return False

    def AddSample(self, Frame: np.ndarray, PersonName: str,
                  Retrain: bool = False) -> tuple:
        """
        從 BGR 影像偵測人臉，萃取特徵向量並依姿態類別加入訓練樣本。

        Parameters
        ----------
        Frame      : BGR 格式的 numpy 影像
        PersonName : 要學習的人名
        Retrain    : 加入後是否立即重訓（批次學習傳 False，結束後呼叫 FinishLearning）

        Returns
        -------
        (Added: bool, KeyPoints: list, PoseCat: int, Yaw: float, Pitch: float)
          Added     : 是否成功加入至少一個有效樣本
          KeyPoints : 偵測到的臉部關鍵點列表（供 UI 疊加顯示）
          PoseCat   : 本幀的姿態類別（供 UI 顯示目前姿態）
          Yaw       : 水平轉角原始值（負=左，正=右）
          Pitch     : 垂直傾角原始值（負=上，正=下）
        """
        try:
            Detections = self._Detector.detect(Frame)
            if not Detections:
                return False, [], POSE_FRONTAL, 0.0, 0.0

            if PersonName not in self._Samples:
                self._Samples[PersonName] = {}

            Added       = False
            KeyPointsList = []
            LastPoseCat = POSE_FRONTAL
            LastYaw     = 0.0
            LastPitch   = 0.0

            for _, Landmarks3D, KeyPoints in Detections:
                Vec = extractFeatures3D(Landmarks3D)
                if Vec is None:
                    continue
                PoseCat, Yaw, Pitch = classifyPoseWithValues(Landmarks3D)
                if PoseCat not in self._Samples[PersonName]:
                    self._Samples[PersonName][PoseCat] = []
                self._Samples[PersonName][PoseCat].append(Vec)
                KeyPointsList.append(KeyPoints)
                LastPoseCat = PoseCat
                LastYaw     = Yaw
                LastPitch   = Pitch
                Added       = True

            if Added and Retrain:
                self._trainMatcher()
            return Added, KeyPointsList, LastPoseCat, LastYaw, LastPitch

        except Exception as Error:
            print(f"[FaceRecognizer] AddSample 失敗：{Error}")
            return False, [], POSE_FRONTAL, 0.0, 0.0

    def FinishLearning(self) -> None:
        """批次學習結束後呼叫，執行一次分類器重訓。"""
        try:
            self._trainMatcher()
        except Exception as Error:
            print(f"[FaceRecognizer] FinishLearning 失敗：{Error}")

    def Predict(self, Frame: np.ndarray) -> list:
        """
        從 BGR 影像偵測並辨識人臉。

        Returns
        -------
        list of (Top, Right, Bottom, Left, Name, Confidence, PoseCat)
          Top/Right/Bottom/Left : 人臉邊界框（像素）
          Name       : 辨識結果或 "Unknown"
          Confidence : sigmoid 信心度（0.0 ~ 1.0）
          PoseCat    : 本幀姿態類別（0~4）
        """
        Results = []
        try:
            if not self._IsTrained:
                return Results

            Detections = self._Detector.detect(Frame)
            if not Detections:
                return Results

            for BoundingBox, Landmarks3D, _ in Detections:
                Vec = extractFeatures3D(Landmarks3D)
                if Vec is None:
                    continue

                PoseCat, Yaw, Pitch = classifyPoseWithValues(Landmarks3D)

                # 選對應姿態的分類器；若無則備援使用正臉分類器
                Clf = self._Classifiers[PoseCat]
                if Clf is None or not Clf.IsTrained:
                    Clf = self._Classifiers[POSE_FRONTAL]
                if Clf is None or not Clf.IsTrained:
                    continue

                Names, Confs = Clf.predict(np.array([Vec]), Thresholds=None)
                Top, Right, Bottom, Left = BoundingBox
                Results.append((Top, Right, Bottom, Left,
                                Names[0], float(Confs[0]), PoseCat, Yaw, Pitch))

        except Exception as Error:
            print(f"[FaceRecognizer] Predict 失敗：{Error}")
        return Results

    def CanDetect(self) -> bool:
        """若已有訓練資料且至少一個分類器完成訓練，回傳 True。"""
        return self._IsTrained

    def SetThresholds(self, CosineThresh: float = None) -> None:
        """動態更新所有分類器的信心度閾值，無需重訓。"""
        try:
            if CosineThresh is not None:
                self._Threshold = CosineThresh
                for Clf in self._Classifiers:
                    if Clf is not None:
                        Clf._Threshold = CosineThresh
        except Exception as Error:
            print(f"[FaceRecognizer] SetThresholds 失敗：{Error}")

    def GetKnownPersons(self) -> list:
        """回傳有訓練樣本的人名列表。"""
        return [
            Name for Name, PoseDict in self._Samples.items()
            if any(Vecs for Vecs in PoseDict.values())
        ]

    def GetAccumulatedPersons(self) -> list:
        """同 GetKnownPersons（供 main.py 呼叫）。"""
        return self.GetKnownPersons()

    def GetSampleCounts(self) -> dict:
        """回傳各人名的訓練樣本總數 {人名: 總數}。"""
        return {
            Name: sum(len(Vecs) for Vecs in PoseDict.values())
            for Name, PoseDict in self._Samples.items()
            if any(Vecs for Vecs in PoseDict.values())
        }

    def GetPersonPoseCounts(self, PersonName: str) -> dict:
        """
        回傳指定人名各姿態的樣本數量 {姿態類別: 數量}。
        供學習 UI 顯示各姿態收集進度。
        """
        if PersonName not in self._Samples:
            return {}
        return {
            PoseCat: len(Vecs)
            for PoseCat, Vecs in self._Samples[PersonName].items()
        }

    def RemovePerson(self, PersonName: str) -> bool:
        """移除指定人物的所有訓練樣本並重新訓練分類器。"""
        try:
            if PersonName not in self._Samples:
                return False
            del self._Samples[PersonName]
            if self._Samples:
                self._trainMatcher()
            else:
                self._Classifiers = [None] * N_POSES
                self._IsTrained   = False
            return True
        except Exception as Error:
            print(f"[FaceRecognizer] RemovePerson 失敗：{Error}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────────────────────────────────────────

    def _trainMatcher(self) -> None:
        """
        依當前 _Samples，對五個姿態類別各別訓練一個 SvmClassifier。
        若某姿態無任何人的樣本，該分類器設為 None。
        """
        try:
            AnyTrained = False

            for PoseCat in range(N_POSES):
                # 收集所有人在此姿態的樣本
                PoseSamples = {}
                for Name, PoseDict in self._Samples.items():
                    Vecs = PoseDict.get(PoseCat, [])
                    if Vecs:
                        PoseSamples[Name] = Vecs

                if PoseSamples:
                    Clf = SvmClassifier(Threshold=self._Threshold)
                    Clf.fit(PoseSamples)
                    self._Classifiers[PoseCat] = Clf
                    if Clf.IsTrained:
                        AnyTrained = True
                        TotalVecs  = sum(len(v) for v in PoseSamples.values())
                        print(f"[FaceRecognizer] 姿態{PoseCat}({POSE_NAMES[PoseCat]}) "
                              f"訓練完成：{len(PoseSamples)}人 / {TotalVecs}筆")
                else:
                    self._Classifiers[PoseCat] = None

            self._IsTrained = AnyTrained

        except Exception as Error:
            print(f"[FaceRecognizer] _trainMatcher 失敗：{Error}")
            self._IsTrained = False
