"""
face_recognizer.py  (p12 OneClassSVM 版)

FaceRecognizer 類別：整合 MpFaceLandmarker、face_feature_3d、
face_pose_classifier 與 svm_classifier_np，實作姿態不變人臉辨識。

【架構說明】
  face_feature_3d 在取特徵前先將 landmark 旋轉至正臉座標系（姿態正規化），
  使各角度的同一張臉產生接近一致的特徵向量。
  每人獨立一個 OneClassSVM，推論時取 decision score 最高者為候選，
  再依四層 Unknown 偵測機制決定最終結果。

  訓練資料依姿態類別分開儲存（供 UI 顯示各角度蒐集進度），
  訓練時合併所有姿態的樣本給各人的 OneClassSVM。

模型儲存至 face_model.npz（純 NumPy，無 sklearn/scipy 依賴）。
"""

import os
import numpy as np

from mp_face_landmarker import MpFaceLandmarker
from face_feature_3d import extractFeatures3D
from face_pose_classifier import (classifyPoseWithValues,
                                  POSE_FRONTAL, POSE_NAMES, ROLL_THRESH)
from svm_classifier_np import (SvmClassifier, SVM_CONF_THRESH,
                               SVM_MARGIN_THRESH, COSINE_VERIFY_THRESH)

DEFAULT_MODEL_PATH = "face_model.npz"
N_POSES = 5


class FaceRecognizer:
    """
    姿態不變 OneClassSVM 人臉辨識器。
    以 MpFaceLandmarker 取得 468 個 3D landmark，
    經姿態正規化後萃取 325 維特徵，交由各人 OneClassSVM 分類。
    """

    def __init__(self, ModelPath: str = DEFAULT_MODEL_PATH):
        self._ModelPath  = ModelPath
        self._Detector   = MpFaceLandmarker()
        # 訓練資料：{人名: {姿態類別(0~4): [特徵向量, ...]}}
        self._Samples: dict = {}
        # 單一分類器物件（內含各人 OneClassSVM）
        self._Classifier = None
        # 共用閾值
        self._Threshold          = SVM_CONF_THRESH
        self._MarginThresh       = SVM_MARGIN_THRESH
        self._CosineVerifyThresh = COSINE_VERIFY_THRESH
        self._IsTrained          = False

    # ──────────────────────────────────────────────────────────────────────────
    # 公開 API
    # ──────────────────────────────────────────────────────────────────────────

    def LoadModel(self) -> bool:
        """
        載入 face_model.npz 並重新訓練分類器。

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

    def ExportPerson(self, PersonName: str, FilePath: str) -> bool:
        """
        將指定人物的訓練資料匯出至獨立 .npz 檔案（格式與主模型相同）。

        Returns
        -------
        True 表示匯出成功，False 表示失敗或人名不存在。
        """
        try:
            if PersonName not in self._Samples:
                return False
            PoseDict = self._Samples[PersonName]
            if not any(Vecs for Vecs in PoseDict.values()):
                return False

            XList, YList, PList = [], [], []
            for PoseCat, Vecs in PoseDict.items():
                for Vec in Vecs:
                    XList.append(Vec)
                    YList.append(0)
                    PList.append(PoseCat)

            X = np.array(XList, dtype=float)
            Y = np.array(YList, dtype=int)
            P = np.array(PList, dtype=int)
            np.savez_compressed(
                FilePath,
                X=X, Y=Y, P=P,
                persons=np.array([PersonName], dtype=object),
            )
            return True

        except Exception as Error:
            print(f"[FaceRecognizer] ExportPerson 失敗：{Error}")
            return False

    def ImportPersonFiles(self, FilePaths: list) -> tuple:
        """
        讀取多個個人 .npz 檔，將所有人物資料合併進目前模型並重新訓練。
        若人名已存在，則追加樣本（不覆蓋）。

        Returns
        -------
        (Success: bool, ImportedPersons: list[str])
        """
        try:
            ImportedPersons = []
            for FilePath in FilePaths:
                if not os.path.exists(FilePath):
                    print(f"[FaceRecognizer] 檔案不存在：{FilePath}")
                    continue
                Data    = np.load(FilePath, allow_pickle=True)
                Persons = list(Data['persons'])
                X       = Data['X']
                Y       = Data['Y']
                P       = Data['P']

                for Idx, Name in enumerate(Persons):
                    Name       = str(Name)
                    PersonMask = (Y == Idx)
                    if Name not in self._Samples:
                        self._Samples[Name] = {}
                    for PoseCat in range(N_POSES):
                        PoseMask = PersonMask & (P == PoseCat)
                        if PoseMask.any():
                            if PoseCat not in self._Samples[Name]:
                                self._Samples[Name][PoseCat] = []
                            self._Samples[Name][PoseCat].extend(list(X[PoseMask]))
                    if Name not in ImportedPersons:
                        ImportedPersons.append(Name)

            if ImportedPersons:
                self._trainMatcher()
                return True, ImportedPersons
            return False, []

        except Exception as Error:
            print(f"[FaceRecognizer] ImportPersonFiles 失敗：{Error}")
            return False, []

    def AddSample(self, Frame: np.ndarray, PersonName: str,
                  Retrain: bool = False, FrontalOnly: bool = False) -> tuple:
        """
        從 BGR 影像偵測人臉，萃取特徵向量並依姿態類別加入訓練樣本。

        Parameters
        ----------
        FrontalOnly : True 時只接受正臉（POSE_FRONTAL）樣本，側臉偵測到但不加入。

        Returns
        -------
        (Added: bool, KeyPoints: list, PoseCat: int, Yaw: float, Pitch: float)
        """
        try:
            Detections = self._Detector.detect(Frame)
            if not Detections:
                return False, [], POSE_FRONTAL, 0.0, 0.0

            if PersonName not in self._Samples:
                self._Samples[PersonName] = {}

            Added         = False
            KeyPointsList = []
            LastPoseCat   = POSE_FRONTAL
            LastYaw       = 0.0
            LastPitch     = 0.0

            for _, Landmarks3D, KeyPoints in Detections:
                Vec = extractFeatures3D(Landmarks3D)
                if Vec is None:
                    continue
                PoseCat, Yaw, Pitch, _Roll = classifyPoseWithValues(Landmarks3D)
                LastPoseCat = PoseCat
                LastYaw     = Yaw
                LastPitch   = Pitch

                if FrontalOnly and PoseCat != POSE_FRONTAL:
                    continue  # 非正臉不加入訓練集

                if PoseCat not in self._Samples[PersonName]:
                    self._Samples[PersonName][PoseCat] = []
                self._Samples[PersonName][PoseCat].append(Vec)
                KeyPointsList.append(KeyPoints)
                Added = True

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
        list of (Top, Right, Bottom, Left, Name, Confidence, PoseCat, Yaw, Pitch)
          Confidence 為 OneClassSVM raw decision score（可能為負值）
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

                PoseCat, Yaw, Pitch, Roll = classifyPoseWithValues(Landmarks3D)
                # 側臉或歪頭：信心度閾值 -0.1；margin 閾值設 0.0（停用分差檢查）
                IsNonFrontal    = (PoseCat != POSE_FRONTAL) or (abs(Roll) > ROLL_THRESH)
                AdjThresh       = (self._Threshold - 0.1
                                   if IsNonFrontal else self._Threshold)
                AdjMarginThresh = (0.0 if IsNonFrontal else self._MarginThresh)

                # 組合姿態標籤供 debug print 辨識（正臉/歪頭/側臉）
                if not IsNonFrontal:
                    PoseLabel = f"正臉 Y:{Yaw:+.2f}"
                elif abs(Roll) > ROLL_THRESH:
                    PoseLabel = f"歪頭 R:{Roll:+.2f}"
                else:
                    PoseLabel = f"{POSE_NAMES[PoseCat]} Y:{Yaw:+.2f}"

                Names, Confs = self._Classifier.predict(
                    np.array([Vec]),
                    Thresholds=np.array([AdjThresh]),
                    MarginThresholds=np.array([AdjMarginThresh]),
                    PoseLabels=[PoseLabel]
                )
                Name = Names[0]
                Conf = float(Confs[0])

                Top, Right, Bottom, Left = BoundingBox
                Results.append((Top, Right, Bottom, Left,
                                Name, Conf, PoseCat, Yaw, Pitch, Roll))

        except Exception as Error:
            print(f"[FaceRecognizer] Predict 失敗：{Error}")
        return Results

    def CanDetect(self) -> bool:
        """若分類器已完成訓練，回傳 True。"""
        return self._IsTrained

    def SetThresholds(self, CosineThresh: float = None,
                      MarginThresh: float = None,
                      CosineVerifyThresh: float = None) -> None:
        """動態更新分類器的信心度閾值、分差閾值與餘弦驗證閾值，無需重訓。"""
        try:
            if CosineThresh is not None:
                self._Threshold = CosineThresh
                if self._Classifier is not None:
                    self._Classifier._Threshold = CosineThresh
            if MarginThresh is not None:
                self._MarginThresh = MarginThresh
                if self._Classifier is not None:
                    self._Classifier._MarginThresh = MarginThresh
            if CosineVerifyThresh is not None:
                self._CosineVerifyThresh = CosineVerifyThresh
                if self._Classifier is not None:
                    self._Classifier._CosineVerifyThresh = CosineVerifyThresh
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
        """回傳指定人名各姿態的樣本數量 {姿態類別: 數量}。"""
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
                self._Classifier = None
                self._IsTrained  = False
            return True
        except Exception as Error:
            print(f"[FaceRecognizer] RemovePerson 失敗：{Error}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────────────────────────────────────────

    def _trainMatcher(self) -> None:
        """
        將所有人、所有姿態的樣本合併，訓練各人 OneClassSVM。
        """
        try:
            AllSamples = {}
            for Name, PoseDict in self._Samples.items():
                AllVecs = [v for Vecs in PoseDict.values() for v in Vecs]
                if AllVecs:
                    AllSamples[Name] = AllVecs

            if not AllSamples:
                self._Classifier = None
                self._IsTrained  = False
                return

            Clf = SvmClassifier(Threshold=self._Threshold,
                               MarginThresh=self._MarginThresh,
                               CosineVerifyThresh=self._CosineVerifyThresh,
                               Label="全角度")
            Clf.fit(AllSamples)
            self._Classifier = Clf
            self._IsTrained  = Clf.IsTrained

            if Clf.IsTrained:
                TotalVecs  = sum(len(V) for V in AllSamples.values())
                CountsInfo = "  ".join(f"{N}:{len(V)}" for N, V in AllSamples.items())
                print(f"[FaceRecognizer] 全角度 OneClassSVM 訓練完成："
                      f"{len(AllSamples)}人 / {TotalVecs}筆  [{CountsInfo}]")

        except Exception as Error:
            print(f"[FaceRecognizer] _trainMatcher 失敗：{Error}")
            self._IsTrained = False
