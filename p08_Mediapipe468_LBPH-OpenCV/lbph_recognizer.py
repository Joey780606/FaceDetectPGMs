"""
lbph_recognizer.py

使用 OpenCV LBPH（Local Binary Patterns Histograms）進行人臉辨識。

核心職責：
  1. alignFace()：以 5 個關鍵點對齊人臉至正臉模板（Similarity Transform）
     - 5 點：左眼中心、右眼中心、鼻尖、左嘴角、右嘴角
     - 輸出：FACE_SIZE×FACE_SIZE 灰階影像（uint8）
  2. LbphRecognizer：封裝 cv2.face.LBPHFaceRecognizer
     - fit()：從頭訓練
     - predict()：預測標籤與信心度
     - write() / read()：模型儲存 / 載入

信心度轉換（對外介面）：
  Confidence = max(0.0, 1.0 - LbphDist / (2 × Threshold))
  保持 0~1 範圍（越高越確定），與 p07 Random Forest 介面一致。

注意：需安裝 opencv-contrib-python（含 cv2.face 模組），
      不要同時安裝 opencv-python 與 opencv-contrib-python。
"""

import cv2
import numpy as np

# 確認 cv2.face 模組存在（需 opencv-contrib-python）
try:
    _ = cv2.face.LBPHFaceRecognizer_create
except AttributeError:
    raise ImportError(
        "找不到 cv2.face 模組。\n"
        "請執行：pip install opencv-contrib-python\n"
        "（若已安裝 opencv-python，請先移除：pip uninstall opencv-python）"
    )


# ==============================================================================
# 常數
# ==============================================================================

# 對齊後人臉影像邊長（正方形，單位：像素）
FACE_SIZE = 100

# 眼距最小值（像素）；低於此值視為側臉過度或偵測異常，跳過該幀
MIN_EYE_DIST_PX = 20

# 正臉模板的 5 個關鍵點座標（FACE_SIZE×FACE_SIZE 空間）
# 順序：左眼中心, 右眼中心, 鼻尖, 左嘴角, 右嘴角
_CANONICAL_5PT = np.array([
    [35.0, 42.0],   # 左眼中心
    [65.0, 42.0],   # 右眼中心
    [50.0, 60.0],   # 鼻尖
    [35.0, 78.0],   # 左嘴角
    [65.0, 78.0],   # 右嘴角
], dtype=np.float32)

# 預設 LBPH 距離閾值（超過此值判為 Unknown）
DEFAULT_THRESHOLD = 80.0

# CLAHE 實例（模組層級共用，避免重複建立）
# clipLimit=2.0：限制對比度放大倍率，防止過度增強雜訊
# tileGridSize=(8,8)：對 100×100 影像適中，提供局部自適應光線補償
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ==============================================================================
# 公開函式：人臉對齊
# ==============================================================================

def alignFace(Frame: np.ndarray, FivePts: np.ndarray) -> np.ndarray | None:
    """
    以 5 個關鍵點對齊人臉至正臉模板，輸出 FACE_SIZE×FACE_SIZE 灰階影像。

    學習與偵測兩個階段必須呼叫同一函式，確保前處理一致。

    Parameters
    ----------
    Frame   : BGR 格式的 OpenCV 影像，shape=(H, W, 3)
    FivePts : shape=(5, 2) 的像素座標陣列（float32）
              順序：[左眼中心, 右眼中心, 鼻尖, 左嘴角, 右嘴角]

    Returns
    -------
    np.ndarray, shape=(FACE_SIZE, FACE_SIZE), dtype=uint8  或  None（對齊失敗）

    側臉過濾：
      兩眼像素距離 < MIN_EYE_DIST_PX 時視為側臉/偵測異常，回傳 None。
    """
    try:
        SrcPts = np.array(FivePts, dtype=np.float32)

        # 側臉過濾：左右眼距離過小則跳過
        EyeDist = float(np.linalg.norm(SrcPts[0] - SrcPts[1]))
        if EyeDist < MIN_EYE_DIST_PX:
            return None

        # 計算相似度變換（Similarity Transform：旋轉 + 縮放 + 平移，4 DOF）
        # LMEDS 方法對少量遮擋 landmark 更穩健
        M, _ = cv2.estimateAffinePartial2D(
            SrcPts, _CANONICAL_5PT, method=cv2.LMEDS
        )
        if M is None:
            return None

        # 應用仿射變換，輸出 FACE_SIZE×FACE_SIZE BGR 影像
        Warped = cv2.warpAffine(
            Frame, M, (FACE_SIZE, FACE_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE   # 邊界填充用鄰近像素，減少黑邊干擾
        )

        # 轉灰階
        if Warped.ndim == 3:
            Gray = cv2.cvtColor(Warped, cv2.COLOR_BGR2GRAY)
        else:
            Gray = Warped

        # CLAHE 光線歸一化：減少光源方向與亮度變化對 LBPH 的干擾
        # 學習與偵測兩階段均套用，確保前處理一致
        Gray = _CLAHE.apply(Gray.astype(np.uint8))

        return Gray.astype(np.uint8)

    except Exception as Error:
        print(f"[lbph_recognizer] alignFace 失敗：{Error}")
        return None


# ==============================================================================
# Class: LbphRecognizer
# ==============================================================================

class LbphRecognizer:
    """
    OpenCV LBPHFaceRecognizer 封裝。

    提供 fit（從頭訓練）、predict（預測）、write / read（儲存 / 載入）。
    對外信心度以 0~1 呈現（越高越確定）；內部 LBPH 距離越低越相似。

    LBPH 參數說明：
      radius    = 1  : LBP 取樣半徑（鄰域大小）
      neighbors = 8  : LBP 取樣點數
      grid_x    = 8  : 橫向分割格數（格越多特徵越細，但過多容易過擬合）
      grid_y    = 8  : 縱向分割格數
      threshold      : 距離閾值，超過視為 Unknown
    """

    def __init__(self, Threshold: float = DEFAULT_THRESHOLD):
        """
        Parameters
        ----------
        Threshold : LBPH 距離閾值（距離 > Threshold 判為 Unknown，預設 80.0）
        """
        self._Threshold  = Threshold
        self._IsTrained  = False
        self._Recognizer = self._createRecognizer(Threshold)

    # ──────────────────────────────────────────────────────────────────────────
    # 屬性
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def IsTrained(self) -> bool:
        """是否已完成訓練。"""
        return self._IsTrained

    @property
    def Threshold(self) -> float:
        """目前 LBPH 距離閾值。"""
        return self._Threshold

    @Threshold.setter
    def Threshold(self, Value: float) -> None:
        """動態更新閾值，無需重訓。"""
        self._Threshold = Value
        try:
            self._Recognizer.setThreshold(Value)
        except Exception as Error:
            print(f"[LbphRecognizer] 更新閾值失敗：{Error}")

    # ──────────────────────────────────────────────────────────────────────────
    # 公開方法
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, Images: list, Labels: list) -> None:
        """
        從頭訓練 LBPH 模型。

        Parameters
        ----------
        Images : list of np.ndarray
                 每個元素為對齊後灰階人臉影像，shape=(FACE_SIZE, FACE_SIZE), uint8
        Labels : list of int
                 與 Images 一一對應的整數標籤（0-based）
        """
        try:
            if not Images:
                print("[LbphRecognizer] fit：無訓練樣本，略過")
                return

            ImgList = [np.array(Img, dtype=np.uint8) for Img in Images]
            LblArr  = np.array(Labels, dtype=np.int32)

            # 重建辨識器以確保從頭訓練（train 不會清除舊資料）
            self._Recognizer = self._createRecognizer(self._Threshold)
            self._Recognizer.train(ImgList, LblArr)
            self._IsTrained = True
            print(f"[LbphRecognizer] 訓練完成，共 {len(Images)} 張樣本")

        except Exception as Error:
            print(f"[LbphRecognizer] fit 失敗：{Error}")
            self._IsTrained = False

    def predict(self, Image: np.ndarray) -> tuple:
        """
        預測單張人臉影像的身份。

        Parameters
        ----------
        Image : 對齊後灰階人臉影像，shape=(FACE_SIZE, FACE_SIZE), uint8

        Returns
        -------
        (LabelIdx: int, Confidence: float)
          LabelIdx   : 預測的整數標籤；Unknown 時回傳 -1
          Confidence : 0.0~1.0（由 LBPH 距離轉換，越高越確定）
        """
        try:
            Img = np.array(Image, dtype=np.uint8)
            LabelIdx, LbphDist = self._Recognizer.predict(Img)

            # 距離超過閾值 → Unknown
            if LbphDist > self._Threshold:
                Conf = max(0.0, 1.0 - LbphDist / (2.0 * self._Threshold))
                print(f"[LbphRecognizer] 距離={LbphDist:.1f}(閾{self._Threshold:.0f}) → Unknown")
                return -1, Conf

            # 距離轉換為 0~1 信心度
            Conf = max(0.0, 1.0 - LbphDist / (2.0 * self._Threshold))
            print(f"[LbphRecognizer] 距離={LbphDist:.1f}(閾{self._Threshold:.0f}) → Label={LabelIdx} Conf={Conf:.2f}")
            return int(LabelIdx), Conf

        except Exception as Error:
            print(f"[LbphRecognizer] predict 失敗：{Error}")
            return -1, 0.0

    def write(self, ModelPath: str) -> bool:
        """
        將已訓練的 LBPH 模型儲存至檔案（OpenCV XML/YAML 格式）。

        Returns
        -------
        True 表示儲存成功，False 表示失敗。
        """
        try:
            if not self._IsTrained:
                print("[LbphRecognizer] write：模型尚未訓練，略過")
                return False
            self._Recognizer.write(ModelPath)
            print(f"[LbphRecognizer] 模型已儲存：{ModelPath}")
            return True
        except Exception as Error:
            print(f"[LbphRecognizer] write 失敗：{Error}")
            return False

    def read(self, ModelPath: str) -> bool:
        """
        從檔案載入已訓練的 LBPH 模型。

        Returns
        -------
        True 表示載入成功，False 表示失敗。
        """
        try:
            self._Recognizer.read(ModelPath)
            self._IsTrained = True
            print(f"[LbphRecognizer] 模型已載入：{ModelPath}")
            return True
        except Exception as Error:
            print(f"[LbphRecognizer] read 失敗：{Error}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────────────────────────────────────────

    def _createRecognizer(self, Threshold: float):
        """建立新的 LBPHFaceRecognizer 實例。"""
        return cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8,
            threshold=Threshold
        )
