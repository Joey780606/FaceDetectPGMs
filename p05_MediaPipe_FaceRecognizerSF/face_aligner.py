"""
face_aligner.py

FaceAligner 類別：使用 MediaPipe FaceLandmarker 偵測 478 個臉部特徵點，
並以 ArcFace 標準 5 點對齊法將臉部裁切至 112×112 像素，
供 OpenCV FaceRecognizerSF 進行特徵提取。

對齊目標座標（ArcFace 112×112 標準）：
    左眼  (38.29, 51.70)  右眼  (73.53, 51.50)
    鼻尖  (56.03, 71.74)
    左嘴角(41.55, 92.37)  右嘴角(70.73, 92.20)

授權：MediaPipe face_landmarker.task 為 Apache 2.0，商用安全。
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as MpPython
from mediapipe.tasks.python import vision as MpVision

# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe 478 點中對齊所需的 5 個關鍵點索引
# ──────────────────────────────────────────────────────────────────────────────
# 左眼：外眼角(33) 與 內眼角(133) 的中點
_LEFT_EYE_IDXS  = [33, 133]
# 右眼：外眼角(362) 與 內眼角(263) 的中點  （MediaPipe 以鏡像方向定義）
_RIGHT_EYE_IDXS = [362, 263]
# 鼻尖
_NOSE_IDX       = 4
# 左嘴角
_MOUTH_L_IDX    = 61
# 右嘴角
_MOUTH_R_IDX    = 291

# ArcFace 112×112 標準 5 點目標座標（對應上方 5 個關鍵點的順序）
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],   # 左眼中心
    [73.5318, 51.5014],   # 右眼中心
    [56.0252, 71.7366],   # 鼻尖
    [41.5493, 92.3655],   # 左嘴角
    [70.7299, 92.2041],   # 右嘴角
], dtype=np.float32)

# 對齊後輸出尺寸
_ALIGN_SIZE = 112


class FaceAligner:
    """
    使用 MediaPipe FaceLandmarker 偵測人臉並對齊至 112×112 像素。

    Detect(Frame) 回傳每張臉的：
        - 對齊後的 112×112 BGR 影像（供 FaceRecognizerSF 使用）
        - 邊界框 (Top, Right, Bottom, Left)
        - 關鍵點字典（供 UI 疊加顯示）
    """

    def __init__(self, ModelPath: str = "face_landmarker.task"):
        """
        初始化 MediaPipe FaceLandmarker（靜態影像模式）。

        Parameters
        ----------
        ModelPath : MediaPipe FaceLandmarker 模型檔案路徑
        """
        self._ModelPath = ModelPath
        self._Detector  = None
        self._loadDetector()

    def _loadDetector(self) -> None:
        """載入 MediaPipe FaceLandmarker 模型。"""
        try:
            BaseOptions    = MpPython.BaseOptions(model_asset_path=self._ModelPath)
            Options        = MpVision.FaceLandmarkerOptions(
                base_options   = BaseOptions,
                running_mode   = MpVision.RunningMode.IMAGE,
                num_faces      = 10,       # 最多同時偵測 10 張臉
                min_face_detection_confidence = 0.5,
                min_face_presence_confidence  = 0.5,
                min_tracking_confidence       = 0.5,
            )
            self._Detector = MpVision.FaceLandmarker.create_from_options(Options)
            print("[FaceAligner] MediaPipe FaceLandmarker 載入成功")
        except Exception as Error:
            print(f"[FaceAligner] FaceLandmarker 載入失敗：{Error}")
            self._Detector = None

    def close(self) -> None:
        """釋放 MediaPipe 資源。"""
        if self._Detector is not None:
            try:
                self._Detector.close()
            except Exception:
                pass
            self._Detector = None

    def Detect(self, Frame: np.ndarray) -> list:
        """
        偵測影像中所有人臉，並將每張臉對齊至 112×112。

        Parameters
        ----------
        Frame : BGR 格式的 numpy 影像（來自 OpenCV）

        Returns
        -------
        list of (AlignedFace, BoundingBox, KeyPoints)
            AlignedFace : np.ndarray, shape (112, 112, 3), dtype uint8, BGR
            BoundingBox : (Top, Right, Bottom, Left) 像素座標（int）
            KeyPoints   : dict {"left_eye":(cx,cy), "right_eye":(cx,cy),
                                "nose":(cx,cy), "mouth":(cx,cy)}
        """
        if self._Detector is None:
            return []

        try:
            H, W = Frame.shape[:2]

            # BGR → RGB，轉為 MediaPipe Image
            FrameRgb  = cv2.cvtColor(Frame, cv2.COLOR_BGR2RGB)
            MpImage   = mp.Image(image_format=mp.ImageFormat.SRGB, data=FrameRgb)

            # MediaPipe 偵測
            Result = self._Detector.detect(MpImage)
            if not Result.face_landmarks:
                return []

            Detections = []
            for FaceLandmarks in Result.face_landmarks:
                try:
                    Detection = self._processFace(Frame, FaceLandmarks, W, H)
                    if Detection is not None:
                        Detections.append(Detection)
                except Exception as FaceError:
                    print(f"[FaceAligner] 單臉處理失敗：{FaceError}")

            return Detections

        except Exception as Error:
            print(f"[FaceAligner] Detect 失敗：{Error}")
            return []

    def _processFace(self, Frame: np.ndarray, Landmarks, W: int, H: int):
        """
        處理單張臉：提取 5 個關鍵點，計算相似性變換，裁切對齊。

        Returns
        -------
        (AlignedFace, BoundingBox, KeyPoints) 或 None（對齊失敗時）
        """
        # 將歸一化座標轉為像素座標
        Pts = np.array(
            [[Lm.x * W, Lm.y * H] for Lm in Landmarks],
            dtype=np.float32
        )

        if len(Pts) < 468:
            return None

        # 提取 5 個對齊用關鍵點（像素座標）
        LeftEye  = Pts[_LEFT_EYE_IDXS].mean(axis=0)
        RightEye = Pts[_RIGHT_EYE_IDXS].mean(axis=0)
        Nose     = Pts[_NOSE_IDX]
        MouthL   = Pts[_MOUTH_L_IDX]
        MouthR   = Pts[_MOUTH_R_IDX]

        SrcPts = np.array([LeftEye, RightEye, Nose, MouthL, MouthR], dtype=np.float32)

        # 計算相似性仿射變換（旋轉 + 縮放 + 平移，不含剪切）
        TransMat, _ = cv2.estimateAffinePartial2D(
            SrcPts, _ARCFACE_DST,
            method=cv2.LMEDS
        )
        if TransMat is None:
            return None

        # 仿射變換到 112×112
        AlignedFace = cv2.warpAffine(
            Frame, TransMat, (_ALIGN_SIZE, _ALIGN_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        # 計算邊界框（由所有 landmark 的最大外框）
        XMin = max(0, int(Pts[:, 0].min()))
        YMin = max(0, int(Pts[:, 1].min()))
        XMax = min(W - 1, int(Pts[:, 0].max()))
        YMax = min(H - 1, int(Pts[:, 1].max()))
        BoundingBox = (YMin, XMax, YMax, XMin)  # (Top, Right, Bottom, Left)

        # 關鍵點字典（供 UI 疊加顯示，取嘴巴中心）
        MouthCenter = ((MouthL + MouthR) / 2).astype(int)
        KeyPoints = {
            "left_eye":  (int(LeftEye[0]),  int(LeftEye[1])),
            "right_eye": (int(RightEye[0]), int(RightEye[1])),
            "nose":      (int(Nose[0]),      int(Nose[1])),
            "mouth":     (int(MouthCenter[0]), int(MouthCenter[1])),
        }

        return AlignedFace, BoundingBox, KeyPoints
