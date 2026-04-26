"""
mp_face_landmarker.py

使用 MediaPipe Tasks API（FaceLandmarker）偵測人臉，
直接回傳全部 468 個 3D 歸一化座標（x, y, z），
不轉換為 dlib 68 點格式（與 p04 的 mp_face_detector.py 不同）。

核心職責：
  1. 首次執行時自動下載 face_landmarker.task 模型檔（約 6 MB）
  2. 以 FaceLandmarker 取得 478 個 3D 歸一化座標（前 468 個為 FaceMesh 點）
  3. 直接回傳前 468 點的 (x, y, z) 歸一化座標陣列
  4. 同時回傳像素座標的關鍵點中心，供 UI 顯示學習時的眼鼻嘴點

輸出格式（每個人臉）：
  (BoundingBox, Landmarks3D, KeyPoints)
    BoundingBox  = (Top, Right, Bottom, Left)       # 像素座標
    Landmarks3D  = np.ndarray, shape=(468, 3)       # 歸一化 (x, y, z)
    KeyPoints    = {
        "left_eye":  (cx, cy),   # 像素座標
        "right_eye": (cx, cy),
        "nose":      (cx, cy),
        "mouth":     (cx, cy),
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
_MODEL_URL      = ("https://storage.googleapis.com/mediapipe-models/"
                   "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
_MODEL_FILENAME = "face_landmarker.task"   # 儲存在執行目錄下


def _ensureModel(ModelPath: str) -> None:
    """若模型檔不存在則自動從 Google 下載（約 6 MB）。"""
    if os.path.exists(ModelPath):
        return
    print(f"[MpFaceLandmarker] 下載 FaceLandmarker 模型至 {ModelPath} ...")
    try:
        urllib.request.urlretrieve(_MODEL_URL, ModelPath)
        print("[MpFaceLandmarker] 模型下載完成。")
    except Exception as Error:
        raise RuntimeError(
            f"[MpFaceLandmarker] 模型下載失敗：{Error}\n"
            f"請手動下載後放至：{ModelPath}\n"
            f"下載網址：{_MODEL_URL}"
        ) from Error


# ==============================================================================
# 關鍵 Landmark 索引（用於 KeyPoints 計算與 IOD 歸一化）
# ==============================================================================
# 左眼（同 p04 DLIB68_MEDIAPIPE_MAP）
_LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
# 右眼
_RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
# 嘴巴（左角、右角、上唇中、下唇中）
_MOUTH_INDICES     = [61, 291, 0, 17]
# 鼻尖
_NOSE_INDEX        = 1


# ==============================================================================
# Class: MpFaceLandmarker
# ==============================================================================
class MpFaceLandmarker:
    """
    MediaPipe FaceLandmarker 人臉偵測器（Tasks API，mediapipe >= 0.10）。

    以 detect(Frame) 一次完成偵測，直接回傳 468 個 3D 歸一化 landmark，
    不做 dlib 68 點轉換（p07 直接使用全部 468 點作為特徵）。
    """

    def __init__(self, MaxFaces: int = 5, MinDetectConf: float = 0.5,
                 MinPresenceConf: float = 0.5, MinTrackConf: float = 0.5,
                 ModelPath: str = _MODEL_FILENAME):
        """
        Parameters
        ----------
        MaxFaces        : 同時偵測的最大人臉數
        MinDetectConf   : 人臉偵測最低信心度（0~1）
        MinPresenceConf : 人臉存在最低信心度（0~1）
        MinTrackConf    : 追蹤最低信心度（0~1）
        ModelPath       : face_landmarker.task 模型檔路徑
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
            raise RuntimeError(f"[MpFaceLandmarker] 初始化失敗：{Error}") from Error

    def detect(self, Frame: np.ndarray) -> list:
        """
        偵測 BGR 影像中的所有人臉，回傳邊界框、468 個 3D landmark 及關鍵點。

        Parameters
        ----------
        Frame : OpenCV BGR 格式的 numpy 陣列，shape=(H, W, 3)

        Returns
        -------
        list of (BoundingBox, Landmarks3D, KeyPoints)
          BoundingBox  = (Top, Right, Bottom, Left)       # 像素座標
          Landmarks3D  = np.ndarray, shape=(468, 3)       # 歸一化 (x, y, z)
          KeyPoints    = {"left_eye": (cx,cy), "right_eye": (cx,cy),
                          "nose": (cx,cy), "mouth": (cx,cy)}  # 像素座標
        """
        Results = []
        try:
            H, W = Frame.shape[:2]

            # MediaPipe Tasks API 使用 RGB 輸入，以 mp.Image 包裝
            RgbFrame = Frame[:, :, ::-1].copy()
            MpImage  = mp.Image(image_format=mp.ImageFormat.SRGB, data=RgbFrame)

            Detection = self._Landmarker.detect(MpImage)    # Mediapipe的函式,回傳 FaceDetection 物件,包含 face_landmarks 的物件

            if not Detection.face_landmarks:
                return Results

            for FaceLms in Detection.face_landmarks:    # 一張圖裡每個人臉的 landmark 列表
                try:
                    # 只取前 468 個點（後 10 個為虹膜，不使用）
                    LmsCount = min(len(FaceLms), 468)
                    if LmsCount < 468:
                        print(f"[MpFaceLandmarker] 偵測到 {LmsCount} 個點（預期 468），跳過。")
                        continue

                    # 建立 (468, 3) 的歸一化座標陣列
                    Landmarks3D = np.array(
                        [[FaceLms[i].x, FaceLms[i].y, FaceLms[i].z]
                         for i in range(468)],
                        dtype=float
                    )

                    # 計算邊界框（從 x, y 座標的 min/max）
                    BoundingBox = self._buildBoundingBox(Landmarks3D, W, H)

                    # 計算關鍵點像素座標（雙眼、鼻子、嘴巴中心）
                    KeyPoints = self._buildKeyPoints(Landmarks3D, W, H)

                    Results.append((BoundingBox, Landmarks3D, KeyPoints))

                except Exception as FaceError:
                    print(f"[MpFaceLandmarker] 單張臉處理失敗：{FaceError}")
                    continue

        except Exception as Error:
            print(f"[MpFaceLandmarker] detect 失敗：{Error}")
        return Results

    def close(self) -> None:
        """釋放 FaceLandmarker 資源。"""
        try:
            self._Landmarker.close()
        except Exception as Error:
            print(f"[MpFaceLandmarker] 釋放資源失敗：{Error}")

    # ──────────────────────────────────────────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────────────────────────────────────────

    def _buildBoundingBox(self, Landmarks3D: np.ndarray,
                          W: int, H: int) -> tuple:
        """
        從所有 468 個歸一化 landmark 計算人臉邊界框（像素座標）。

        Returns
        -------
        (Top, Right, Bottom, Left)
        """
        XPixels = (Landmarks3D[:, 0] * W).astype(int)
        YPixels = (Landmarks3D[:, 1] * H).astype(int)
        Left   = int(max(0,     XPixels.min()))
        Right  = int(min(W - 1, XPixels.max()))
        Top    = int(max(0,     YPixels.min()))
        Bottom = int(min(H - 1, YPixels.max()))
        return (Top, Right, Bottom, Left)

    def _buildKeyPoints(self, Landmarks3D: np.ndarray,
                        W: int, H: int) -> dict:
        """
        計算雙眼、鼻子、嘴巴的中心像素座標，供 UI 學習時疊加顯示。

        Returns
        -------
        {"left_eye": (cx,cy), "right_eye": (cx,cy),
         "nose": (cx,cy), "mouth": (cx,cy)}
        """
        def _centerPixel(Indices: list) -> tuple:
            """取指定索引群的平均，轉換為像素座標。"""
            Points = Landmarks3D[Indices]   # shape=(N, 3)
            Cx = int(Points[:, 0].mean() * W)
            Cy = int(Points[:, 1].mean() * H)
            return (Cx, Cy)

        return {
            'left_eye':  _centerPixel(_LEFT_EYE_INDICES),
            'right_eye': _centerPixel(_RIGHT_EYE_INDICES),
            'nose':      (int(Landmarks3D[_NOSE_INDEX, 0] * W),
                          int(Landmarks3D[_NOSE_INDEX, 1] * H)),
            'mouth':     _centerPixel(_MOUTH_INDICES),
        }
