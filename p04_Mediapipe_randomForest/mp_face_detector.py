"""
mp_face_detector.py

使用 MediaPipe Tasks API（FaceLandmarker）偵測人臉並萃取關鍵點。
適用於 mediapipe >= 0.10（舊版 solutions API 已移除）。

核心職責：
  1. 首次執行時自動下載 face_landmarker.task 模型檔（約 6 MB）
  2. 以 FaceLandmarker 取得 478 個 3D 歸一化座標（前 468 個為 FaceMesh 點）
  3. 依照 DLIB68_MEDIAPIPE_MAP 從 468 點中選出 68 個等效點
  4. 將結果轉換為與 face_recognition.face_landmarks() 相同的 dict 格式，
     使 face_feature.py 可直接複用，無需修改。

輸出格式（LandmarkDict）：
  {
    'chin'          : [(x,y), ...],   # 17 點（下顎線）
    'left_eyebrow'  : [(x,y), ...],   #  5 點
    'right_eyebrow' : [(x,y), ...],   #  5 點
    'nose_bridge'   : [(x,y), ...],   #  4 點
    'nose_tip'      : [(x,y), ...],   #  5 點
    'left_eye'      : [(x,y), ...],   #  6 點
    'right_eye'     : [(x,y), ...],   #  6 點
    'top_lip'       : [(x,y), ...],   # 12 點（index 0=左嘴角，index 6=右嘴角）
    'bottom_lip'    : [(x,y), ...],   #  8 點（內唇）
  }

授權：MediaPipe (Apache 2.0) — 商用免費，含模型權重
"""

import os
import urllib.request
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError as _E:
    raise ImportError("請先執行：pip install mediapipe>=0.10") from _E


# ==============================================================================
# 模型下載設定
# ==============================================================================
# Google 官方 FaceLandmarker 模型（Apache 2.0 授權，商用免費）
_MODEL_URL      = ("https://storage.googleapis.com/mediapipe-models/"
                   "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
_MODEL_FILENAME = "face_landmarker.task"   # 儲存在執行目錄下


def _ensureModel(ModelPath: str) -> None:
    """若模型檔不存在則自動從 Google 下載（約 6 MB）。"""
    if os.path.exists(ModelPath):
        return
    print(f"[MpFaceDetector] 下載 FaceLandmarker 模型至 {ModelPath} ...")
    try:
        urllib.request.urlretrieve(_MODEL_URL, ModelPath)
        print("[MpFaceDetector] 模型下載完成。")
    except Exception as Error:
        raise RuntimeError(
            f"[MpFaceDetector] 模型下載失敗：{Error}\n"
            f"請手動下載後放至：{ModelPath}\n"
            f"下載網址：{_MODEL_URL}"
        ) from Error


# ==============================================================================
# MediaPipe 478 點 → dlib 68 等效點映射
#
# FaceLandmarker 回傳 478 個點（前 468 為 FaceMesh，後 10 為虹膜），
# 以下索引皆在前 468 點範圍內。
#
# 每個 key 對應 face_recognition.face_landmarks() 的部位名稱，
# value 是 MediaPipe 的 landmark 索引列表。
# ==============================================================================
DLIB68_MEDIAPIPE_MAP: dict[str, list[int]] = {
    # dlib 0–16：下顎線（左端 → 下巴中心 → 右端），共 17 點
    'chin': [
        234, 93, 132, 58, 172, 136, 150, 149,
        176, 148, 152, 377, 400, 378, 379, 365, 454,
    ],
    # dlib 17–21：左眉毛（左端 → 右端），共 5 點
    'left_eyebrow': [70, 63, 105, 66, 107],
    # dlib 22–26：右眉毛（左端 → 右端），共 5 點
    'right_eyebrow': [336, 296, 334, 293, 300],
    # dlib 27–30：鼻梁（眉心 → 鼻根），共 4 點
    'nose_bridge': [168, 6, 197, 195],
    # dlib 31–35：鼻翼（左端 → 鼻尖 → 右端），共 5 點
    'nose_tip': [209, 198, 1, 422, 429],
    # dlib 36–41：左眼（順時針），共 6 點
    'left_eye': [33, 160, 158, 133, 153, 144],
    # dlib 42–47：右眼（順時針），共 6 點
    'right_eye': [362, 385, 387, 263, 373, 380],
    # dlib 48–59：外唇（index 0=左嘴角，index 6=右嘴角），共 12 點
    'top_lip': [61, 40, 37, 0, 267, 270, 291, 321, 405, 17, 181, 78],
    # dlib 60–67：內唇，共 8 點
    'bottom_lip': [78, 81, 13, 311, 308, 402, 14, 178],
}


# ==============================================================================
# Class: MpFaceDetector
# ==============================================================================
class MpFaceDetector:
    """
    MediaPipe FaceLandmarker 人臉偵測器（Tasks API，mediapipe >= 0.10）。

    以 detect(Frame) 一次完成偵測與關鍵點轉換，
    輸出格式與 face_recognition.face_landmarks() 相容。
    """

    def __init__(self, MaxFaces: int = 5, MinDetectConf: float = 0.5,
                 MinPresenceConf: float = 0.5, MinTrackConf: float = 0.5,
                 ModelPath: str = _MODEL_FILENAME):
        """
        Parameters
        ----------
        MaxFaces       : 同時偵測的最大人臉數
        MinDetectConf  : 人臉偵測最低信心度（0~1）
        MinPresenceConf: 人臉存在最低信心度（0~1）
        MinTrackConf   : 追蹤最低信心度（0~1）
        ModelPath      : face_landmarker.task 模型檔路徑
        """
        try:
            # 確保模型檔存在（首次執行自動下載）
            _ensureModel(ModelPath)

            BaseOptions = mp_python.BaseOptions(model_asset_path=ModelPath)
            Options     = mp_vision.FaceLandmarkerOptions(
                base_options=BaseOptions,
                running_mode=mp_vision.RunningMode.IMAGE,   # 同步逐幀模式
                num_faces=MaxFaces,
                min_face_detection_confidence=MinDetectConf,
                min_face_presence_confidence=MinPresenceConf,
                min_tracking_confidence=MinTrackConf,
            )
            self._Landmarker = mp_vision.FaceLandmarker.create_from_options(Options)
        except Exception as Error:
            raise RuntimeError(f"[MpFaceDetector] 初始化 FaceLandmarker 失敗：{Error}") from Error

    def detect(self, Frame: np.ndarray) -> list:
        """
        偵測 BGR 影像中的所有人臉，回傳邊界框與 dlib 相容的 landmark dict。

        Parameters
        ----------
        Frame : OpenCV BGR 格式的 numpy 陣列，shape=(H, W, 3)

        Returns
        -------
        list of (BoundingBox, LandmarkDict)
          BoundingBox  = (Top, Right, Bottom, Left)（像素座標，同 face_recognition）
          LandmarkDict = {'chin': [...], 'left_eye': [...], ...}（像素 (x,y) tuple）
        """
        Results = []
        try:
            H, W = Frame.shape[:2]

            # MediaPipe Tasks API 使用 RGB 輸入，以 mp.Image 包裝
            RgbFrame = Frame[:, :, ::-1].copy()
            MpImage  = mp.Image(image_format=mp.ImageFormat.SRGB, data=RgbFrame)

            Detection = self._Landmarker.detect(MpImage)

            if not Detection.face_landmarks:
                return Results

            for FaceLms in Detection.face_landmarks:
                try:
                    # 建立 68 點 dict（像素座標）
                    LandmarkDict = self._buildLandmarkDict(FaceLms, W, H)   # Joey: 此程式的 function

                    # 計算邊界框（從全部 landmark 的 min/max）
                    BoundingBox = self._buildBoundingBox(FaceLms, W, H)   # Joey: 此程式的 function

                    Results.append((BoundingBox, LandmarkDict))
                except Exception as FaceError:
                    print(f"[MpFaceDetector] 單張臉處理失敗：{FaceError}")
                    continue

        except Exception as Error:
            print(f"[MpFaceDetector] detect 失敗：{Error}")
        return Results

    def close(self) -> None:
        """釋放 FaceLandmarker 資源。"""
        try:
            self._Landmarker.close()
        except Exception as Error:
            print(f"[MpFaceDetector] 釋放資源失敗：{Error}")

    # ──────────────────────────────────────────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────────────────────────────────────────

    def _landmarkToPixel(self, Lm, W: int, H: int) -> tuple[int, int]:
        """MediaPipe 歸一化座標（0~1）→ 像素座標 (x, y)。"""
        return (int(Lm.x * W), int(Lm.y * H))

    def _buildLandmarkDict(self, FaceLms, W: int, H: int) -> dict:
        """
        依照 DLIB68_MEDIAPIPE_MAP 從 landmark 列表中選出 68 個，
        組成與 face_recognition 相容的 dict。
        """
        LandmarkDict = {}
        for PartName, Indices in DLIB68_MEDIAPIPE_MAP.items():
            try:
                Points = [self._landmarkToPixel(FaceLms[Idx], W, H) for Idx in Indices]
                LandmarkDict[PartName] = Points
            except Exception as PartError:
                print(f"[MpFaceDetector] 部位 '{PartName}' 轉換失敗：{PartError}")
                LandmarkDict[PartName] = []
        return LandmarkDict

    def _buildBoundingBox(self, FaceLms, W: int, H: int) -> tuple[int, int, int, int]:
        """
        從所有 landmark 計算人臉邊界框。

        Returns
        -------
        (Top, Right, Bottom, Left)（像素座標，同 face_recognition 格式）
        """
        XList = [int(Lm.x * W) for Lm in FaceLms]
        YList = [int(Lm.y * H) for Lm in FaceLms]
        Left   = max(0,     min(XList))
        Right  = min(W - 1, max(XList))
        Top    = max(0,     min(YList))
        Bottom = min(H - 1, max(YList))
        return (Top, Right, Bottom, Left)
