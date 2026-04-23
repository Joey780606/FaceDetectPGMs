"""
face_recognizer.py

MediaPipe FaceLandmarker + OneClassSVM 人臉辨識後端。
五個臉部角度象限（正臉、左上、右上、左下、右下），每人各象限一個 OneClassSVM。
"""

import os
import cv2
import numpy as np
import joblib
import mediapipe as mp
from sklearn.svm import OneClassSVM

# --- 路徑 ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_SCRIPT_DIR, "face_landmarker.task")
_SAVE_PATH  = os.path.join(_SCRIPT_DIR, "face_model.npz")

# --- 常數（供 main.py import）---
POSE_TARGET          = 20     # 每象限每人目標幀數
POSE_NAMES           = ["正臉", "左上", "右上", "左下", "右下"]
MIN_TRAIN_SAMPLES    = 5      # 訓練 OneClassSVM 的最低樣本數
UNKNOWN_THRESHOLD    = 0.0    # 高於此值才認定為已知人物（decision_function 分數）
YAW_THRESHOLD        = 0.05   # Yaw 閾值：鼻尖偏離眼中點超過 0.05 IOD 才算側臉
PITCH_THRESHOLD      = 0.06   # Pitch 閾值：鼻尖偏離臉高中點超過 6% 才算仰/俯臉
NEUTRAL_NOSE_FRAC    = 0.50   # 正臉時鼻尖約位於臉高（額頭→下巴）50% 處

# --- MediaPipe Landmark 索引 ---
_IDX_LE_OUTER  = 33    # 左眼外角（camera 視角）
_IDX_LE_INNER  = 133   # 左眼內角
_IDX_RE_INNER  = 362   # 右眼內角
_IDX_RE_OUTER  = 263   # 右眼外角
_IDX_NOSE_TIP  = 1     # 鼻尖
_IDX_FOREHEAD  = 10    # 額頭（臉高上端基準點）
_IDX_CHIN      = 152   # 下巴（臉高下端基準點）
_IDX_MOUTH_TOP = 13    # 上唇中心
_IDX_MOUTH_BOT = 14    # 下唇中心


class FaceRecognizer:
    """
    人臉辨識器：
    - MediaPipe FaceLandmarker 取得 468 個 3D landmarks
    - IOD 歸一化特徵
    - 五象限 OneClassSVM 辨識，每人各象限獨立模型
    """

    def __init__(self):
        self._Detector  = None
        # {pose_idx: {name: [feature_vector, ...]}}
        self._TrainData = {i: {} for i in range(5)}
        # {pose_idx: {name: OneClassSVM}}
        self._SVMs      = {i: {} for i in range(5)}

    # --------------------------------------------------------------------------
    # 公開：初始化
    # --------------------------------------------------------------------------
    def LoadModel(self, Path: str = _SAVE_PATH) -> None:
        """初始化 MediaPipe，並嘗試載入已存的人臉模型。"""
        try:
            Options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1
            )
            self._Detector = mp.tasks.vision.FaceLandmarker.create_from_options(Options)
        except Exception as Error:
            raise RuntimeError(f"MediaPipe 初始化失敗：{Error}")

        if os.path.exists(Path):
            try:
                Data = joblib.load(Path)
                self._TrainData = Data.get('TrainData', {i: {} for i in range(5)})
                self._SVMs      = Data.get('SVMs',      {i: {} for i in range(5)})
            except Exception as Error:
                print(f"[FaceRecognizer] 載入模型失敗，使用空模型：{Error}")

    # --------------------------------------------------------------------------
    # 公開：學習
    # --------------------------------------------------------------------------
    def AddSample(self, Frame, Name: str, Retrain: bool = False):
        """
        抽取特徵並存入對應象限。
        若該象限該人已達 POSE_TARGET 則跳過（仍回傳 KeyPoints 供 UI 顯示）。
        回傳 (Added: bool, KeyPoints: list)
        """
        try:
            Result = self._extractLandmarks(Frame)
            if Result is None:
                return False, []
            Features, PoseIdx, _, KeyPoints = Result

            CurrentCount = len(self._TrainData[PoseIdx].get(Name, []))
            if CurrentCount >= POSE_TARGET:
                return False, KeyPoints  # 此象限已滿，跳過

            if Name not in self._TrainData[PoseIdx]:
                self._TrainData[PoseIdx][Name] = []
            self._TrainData[PoseIdx][Name].append(Features)
            return True, KeyPoints

        except Exception as Error:
            print(f"[FaceRecognizer] AddSample 失敗：{Error}")
            return False, []

    def GetLearnPoseCounts(self, Name: str) -> dict:
        """回傳指定人名在各象限的已收集幀數：{pose_idx: count}。"""
        return {i: len(self._TrainData[i].get(Name, [])) for i in range(5)}

    def FinishLearning(self) -> None:
        """對各象限各人訓練 OneClassSVM（樣本數不足時跳過）。"""
        for PoseIdx in range(5):
            for Name, Features in self._TrainData[PoseIdx].items():
                if len(Features) >= MIN_TRAIN_SAMPLES:
                    try:
                        Ocsvm = OneClassSVM(kernel='rbf', nu=0.1)
                        Ocsvm.fit(np.array(Features, dtype=np.float32))
                        self._SVMs[PoseIdx][Name] = Ocsvm
                    except Exception as Error:
                        print(f"[FaceRecognizer] 訓練 pose{PoseIdx} [{Name}] 失敗：{Error}")

    # --------------------------------------------------------------------------
    # 公開：推論
    # --------------------------------------------------------------------------
    def Predict(self, Frame) -> list:
        """
        對一幀影像進行人臉辨識。
        回傳 [(top, right, bottom, left, name, confidence)]
        """
        try:
            Result = self._extractLandmarks(Frame)
            if Result is None:
                return []
            Features, PoseIdx, BBox, _ = Result
            Top, Right, Bottom, Left = BBox

            # 取對應象限的 SVMs；若無資料則 fallback 至任一有資料的象限
            UsedPoseIdx = PoseIdx
            PoseSVMs = self._SVMs[PoseIdx]
            if not PoseSVMs:
                for i in range(5):
                    if self._SVMs[i]:
                        PoseSVMs = self._SVMs[i]
                        UsedPoseIdx = i
                        break

            if not PoseSVMs:
                print(f"[Debug] 象限={POSE_NAMES[PoseIdx]} | 無任何已訓練模型 → Unknown")
                return [(Top, Right, Bottom, Left, "Unknown", 0.0)]

            # 計算各人 decision_function 分數
            Scores = {}
            for Name, Ocsvm in PoseSVMs.items():
                try:
                    Scores[Name] = float(Ocsvm.decision_function([Features])[0])
                except Exception:
                    Scores[Name] = -999.0

            BestName  = max(Scores, key=Scores.get)
            BestScore = Scores[BestName]

            # --- Debug 輸出 ---
            FallbackNote = f"(fallback→{POSE_NAMES[UsedPoseIdx]})" if UsedPoseIdx != PoseIdx else ""
            ScoreStr = "  ".join(f"{N}:{S:+.3f}" for N, S in sorted(Scores.items()))
            Result   = BestName if BestScore > UNKNOWN_THRESHOLD else "Unknown"
            print(f"[Debug] 象限={POSE_NAMES[PoseIdx]}{FallbackNote} | {ScoreStr} | → {Result}")

            if BestScore > UNKNOWN_THRESHOLD:
                return [(Top, Right, Bottom, Left, BestName, BestScore)]
            else:
                return [(Top, Right, Bottom, Left, "Unknown", 0.0)]

        except Exception as Error:
            print(f"[FaceRecognizer] Predict 失敗：{Error}")
            return []

    # --------------------------------------------------------------------------
    # 公開：資料管理
    # --------------------------------------------------------------------------
    def SaveModel(self, Path: str = _SAVE_PATH) -> bool:
        """儲存訓練資料與 SVM 模型。"""
        try:
            joblib.dump({'TrainData': self._TrainData, 'SVMs': self._SVMs}, Path)
            return True
        except Exception as Error:
            print(f"[FaceRecognizer] SaveModel 失敗：{Error}")
            return False

    def GetSampleCounts(self) -> dict:
        """回傳各人的總訓練幀數：{name: total_count}。"""
        Counts = {}
        for PoseIdx in range(5):
            for Name, Features in self._TrainData[PoseIdx].items():
                Counts[Name] = Counts.get(Name, 0) + len(Features)
        return Counts

    def GetKnownPersons(self) -> list:
        """回傳已訓練（有 SVM）的人名列表。"""
        Names = set()
        for i in range(5):
            Names.update(self._SVMs[i].keys())
        return list(Names)

    def GetAccumulatedPersons(self) -> list:
        """回傳所有已收集資料的人名（含未滿訓練量的人）。"""
        Names = set()
        for i in range(5):
            Names.update(self._TrainData[i].keys())
        return list(Names)

    def CanDetect(self) -> bool:
        """是否有任一象限已訓練好 SVM。"""
        return any(bool(self._SVMs[i]) for i in range(5))

    def RemovePerson(self, Name: str) -> bool:
        """移除指定人物的所有訓練資料與 SVM，並重新存檔。"""
        try:
            for i in range(5):
                self._TrainData[i].pop(Name, None)
                self._SVMs[i].pop(Name, None)
            self.SaveModel()
            return True
        except Exception as Error:
            print(f"[FaceRecognizer] RemovePerson 失敗：{Error}")
            return False

    def Close(self) -> None:
        """釋放 MediaPipe 資源。"""
        try:
            if self._Detector is not None:
                self._Detector.close()
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # 私有：特徵抽取
    # --------------------------------------------------------------------------
    def _extractLandmarks(self, Frame):
        """
        對一幀影像執行 MediaPipe 偵測，回傳：
        (feature_vector, pose_idx, (top, right, bottom, left), key_points)
        未偵測到人臉時回傳 None。
        """
        try:
            H, W = Frame.shape[:2]
            RgbFrame = cv2.cvtColor(Frame, cv2.COLOR_BGR2RGB)
            MpImage  = mp.Image(image_format=mp.ImageFormat.SRGB, data=RgbFrame)
            DetResult = self._Detector.detect(MpImage)

            if not DetResult.face_landmarks:
                return None

            Lms = DetResult.face_landmarks[0]  # 第一張臉，共 468 個

            # 雙眼中心（歸一化座標）
            LeX = (Lms[_IDX_LE_INNER].x + Lms[_IDX_LE_OUTER].x) / 2
            LeY = (Lms[_IDX_LE_INNER].y + Lms[_IDX_LE_OUTER].y) / 2
            LeZ = (Lms[_IDX_LE_INNER].z + Lms[_IDX_LE_OUTER].z) / 2
            ReX = (Lms[_IDX_RE_INNER].x + Lms[_IDX_RE_OUTER].x) / 2
            ReY = (Lms[_IDX_RE_INNER].y + Lms[_IDX_RE_OUTER].y) / 2
            ReZ = (Lms[_IDX_RE_INNER].z + Lms[_IDX_RE_OUTER].z) / 2

            # IOD（瞳距，2D）
            IOD = np.sqrt((LeX - ReX) ** 2 + (LeY - ReY) ** 2)
            if IOD < 1e-6:
                return None

            # 臉部中心（兩眼中點）
            EyeMidX = (LeX + ReX) / 2
            EyeMidY = (LeY + ReY) / 2
            EyeMidZ = (LeZ + ReZ) / 2

            # IOD 歸一化特徵向量（468 × 3 = 1404-dim）
            Features = []
            for Lm in Lms:
                Features.extend([
                    (Lm.x - EyeMidX) / IOD,
                    (Lm.y - EyeMidY) / IOD,
                    (Lm.z - EyeMidZ) / IOD,
                ])
            Features = np.array(Features, dtype=np.float32)

            # 臉部角度分類
            # Yaw（左右）：鼻尖相對眼中點的水平偏移，IOD 歸一化
            #   正臉時 ≈ 0；往左轉為負，往右轉為正
            Yaw = (Lms[_IDX_NOSE_TIP].x - EyeMidX) / IOD

            # Pitch（上下）：鼻尖在臉高（額頭→下巴）中的比例位置，減去正臉基準值
            #   正臉時鼻尖約在臉高 50% 處 → Pitch ≈ 0
            #   往上看：鼻尖相對偏高 → Pitch < 0
            #   往下看：鼻尖相對偏低 → Pitch > 0
            FaceHeight = Lms[_IDX_CHIN].y - Lms[_IDX_FOREHEAD].y
            if FaceHeight < 1e-4:
                return None
            NoseFrac = (Lms[_IDX_NOSE_TIP].y - Lms[_IDX_FOREHEAD].y) / FaceHeight
            Pitch = NoseFrac - NEUTRAL_NOSE_FRAC

            PoseIdx = self._classifyPose(Yaw, Pitch)

            # 像素座標 bounding box
            AllX = [Lm.x * W for Lm in Lms]
            AllY = [Lm.y * H for Lm in Lms]
            Left   = max(0, int(min(AllX)))
            Right  = min(W, int(max(AllX)))
            Top    = max(0, int(min(AllY)))
            Bottom = min(H, int(max(AllY)))

            # 關鍵點（供 UI 疊加顯示）
            MouthX = int((Lms[_IDX_MOUTH_TOP].x + Lms[_IDX_MOUTH_BOT].x) / 2 * W)
            MouthY = int((Lms[_IDX_MOUTH_TOP].y + Lms[_IDX_MOUTH_BOT].y) / 2 * H)
            KeyPoints = [{
                'left_eye':  (int(LeX * W), int(LeY * H)),
                'right_eye': (int(ReX * W), int(ReY * H)),
                'nose':      (int(Lms[_IDX_NOSE_TIP].x * W), int(Lms[_IDX_NOSE_TIP].y * H)),
                'mouth':     (MouthX, MouthY),
            }]

            return Features, PoseIdx, (Top, Right, Bottom, Left), KeyPoints

        except Exception as Error:
            print(f"[FaceRecognizer] _extractLandmarks 失敗：{Error}")
            return None

    def _classifyPose(self, Yaw: float, Pitch: float) -> int:
        """依據 Yaw/Pitch 分類成五象限（0=正臉, 1=左上, 2=右上, 3=左下, 4=右下）。"""
        Ty = YAW_THRESHOLD
        Tp = PITCH_THRESHOLD
        if abs(Yaw) < Ty and abs(Pitch) < Tp:
            return 0  # 正臉
        elif Yaw <= 0 and Pitch <= 0:
            return 1  # 左上
        elif Yaw > 0 and Pitch <= 0:
            return 2  # 右上
        elif Yaw <= 0 and Pitch > 0:
            return 3  # 左下
        else:
            return 4  # 右下
