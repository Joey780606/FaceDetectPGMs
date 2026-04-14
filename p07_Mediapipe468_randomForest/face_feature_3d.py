"""
face_feature_3d.py

從 MpFaceLandmarker.detect() 回傳的 468 個 3D 歸一化 landmark 中，
萃取 1404 維特徵向量，供 Random Forest 訓練與辨識使用。

特徵設計（1404 維）：
  1. 以鼻尖（Index 1）為原點，計算所有 468 點的相對位移 (dx, dy, dz)
  2. 計算 3D 瞳距（IOD）：左眼中心與右眼中心的歐氏距離（歸一化座標）
  3. 所有相對位移除以 IOD → 消除距離鏡頭遠近造成的縮放干擾
  4. 攤平為 468 × 3 = 1404 維向量

側臉判斷：
  若 IOD < MIN_IOD_NORM（歸一化空間閾值），視為側臉或偵測異常，回傳 None。
"""

import numpy as np

# 左眼 landmark 索引（同 p04 DLIB68_MEDIAPIPE_MAP）
_LEFT_EYE_INDICES  = np.array([33, 160, 158, 133, 153, 144])
# 右眼 landmark 索引
_RIGHT_EYE_INDICES = np.array([362, 385, 387, 263, 373, 380])
# 鼻尖索引（原點）
_NOSE_TIP_INDEX    = 1

# 歸一化空間瞳距最小合理值；小於此值視為側臉或偵測異常
MIN_IOD_NORM = 1e-5


def extractFeatures3D(Landmarks3D: np.ndarray) -> np.ndarray | None:
    """
    從 468 個 3D 歸一化 landmark 中萃取 1404 維特徵向量。

    Parameters
    ----------
    Landmarks3D : np.ndarray, shape=(468, 3)
        MediaPipe FaceLandmarker 回傳的歸一化座標 (x, y, z)，
        x, y 在 0~1 範圍，z 為相對深度。

    Returns
    -------
    np.ndarray, shape=(1404,)  或  None（側臉 / IOD 過小 / 萃取失敗）
    """
    try:
        if Landmarks3D.shape != (468, 3):
            print(f"[face_feature_3d] 輸入維度錯誤：{Landmarks3D.shape}，預期 (468, 3)")
            return None

        # ── 計算 3D 瞳距（IOD）────────────────────────────────────────────────
        LeftEyeCenter  = Landmarks3D[_LEFT_EYE_INDICES].mean(axis=0)   # shape=(3,)
        RightEyeCenter = Landmarks3D[_RIGHT_EYE_INDICES].mean(axis=0)  # shape=(3,)
        Iod = float(np.linalg.norm(LeftEyeCenter - RightEyeCenter))

        if Iod < MIN_IOD_NORM:
            # 側臉或偵測異常，跳過此幀
            return None

        # ── 相對座標（鼻尖為原點）────────────────────────────────────────────
        NoseTip  = Landmarks3D[_NOSE_TIP_INDEX]          # shape=(3,)
        Relative = Landmarks3D - NoseTip                  # shape=(468, 3)

        # ── IOD 歸一化（消除縮放干擾）────────────────────────────────────────
        Normalized = Relative / Iod                        # shape=(468, 3)

        # ── 攤平為 1404 維向量 ────────────────────────────────────────────────
        return Normalized.flatten().astype(float)          # shape=(1404,)

    except Exception as Error:
        print(f"[face_feature_3d] 特徵萃取失敗：{Error}")
        return None
