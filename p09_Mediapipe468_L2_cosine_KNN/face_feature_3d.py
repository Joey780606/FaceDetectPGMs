"""
face_feature_3d.py

臉部比例特徵萃取（Facial Anthropometry）。

【設計動機】
原本的 1404 維向量使用全部 468 點座標，但人類臉部拓撲高度相似，
導致任何兩人的 Cosine 相似度都落在 0.99 以上，無法有效區分個人。

改用「臉部比例特徵」：計算關鍵點之間的成對距離比值，
這才是人與人之間真正不同的生物特徵（仿生物特徵量測學 / Facial Anthropometry）。

【特徵向量組成（351 維）】
  Part A：26 個關鍵點之間的所有成對 3D 距離，除以 IOD → C(26,2) = 325 維
  Part B：26 個關鍵點的 z 深度（相對鼻尖，除以 IOD） → 26 維
  合計：325 + 26 = 351 維

【26 個關鍵點涵蓋】
  眼睛四角、眼瞼上下緣、眉毛三點×2、鼻翼、鼻基底、
  嘴角、上下唇、下巴、顴骨、額頭

【IOD 歸一化】
  所有距離 / IOD（瞳距）→ 消除臉離鏡頭遠近造成的縮放干擾

【退化防護】
  IOD < MIN_IOD_NORM → 視為 MediaPipe 偵測退化，回傳 None
"""

import numpy as np
from itertools import combinations

# ── 26 個關鍵 landmark 索引（MediaPipe Face Mesh 468 點）────────────────────
_KEY_INDICES = np.array([
    1,    # 鼻尖（相對座標原點，z 深度恆為 0）
    10,   # 額頭中心
    33,   # 左眼外角
    159,  # 左眼上緣
    145,  # 左眼下緣
    133,  # 左眼內角
    362,  # 右眼內角
    386,  # 右眼上緣
    374,  # 右眼下緣
    263,  # 右眼外角
    46,   # 左眉外緣
    105,  # 左眉頂點
    107,  # 左眉內緣
    336,  # 右眉內緣
    334,  # 右眉頂點
    276,  # 右眉外緣
    129,  # 左鼻翼
    358,  # 右鼻翼
    94,   # 鼻基底
    61,   # 左嘴角
    291,  # 右嘴角
    13,   # 上唇中心
    14,   # 下唇中心
    152,  # 下巴
    234,  # 左顴骨
    454,  # 右顴骨
])

# IOD 計算用的左右眼索引（在 _KEY_INDICES 中的位置）
_LEFT_EYE_IN_KEY  = np.array([2, 3, 4, 5])   # index→33,159,145,133
_RIGHT_EYE_IN_KEY = np.array([6, 7, 8, 9])   # index→362,386,374,263

# 預先計算所有成對索引 C(26,2) = 325 對
_PAIRS = np.array(list(combinations(range(len(_KEY_INDICES)), 2)), dtype=int)

# IOD 最小合理值（小於此值視為 MediaPipe 偵測退化）
MIN_IOD_NORM = 1e-5


def extractFeatures3D(Landmarks3D: np.ndarray) -> np.ndarray | None:
    """
    從 468 個 3D 歸一化 landmark 萃取 351 維臉部比例特徵向量。

    Parameters
    ----------
    Landmarks3D : np.ndarray, shape=(468, 3)
        MediaPipe FaceLandmarker 回傳的歸一化座標 (x, y, z)

    Returns
    -------
    np.ndarray, shape=(351,)  或  None（IOD 退化 / 萃取失敗）
    """
    try:
        if Landmarks3D.shape != (468, 3):
            print(f"[face_feature_3d] 輸入維度錯誤：{Landmarks3D.shape}，預期 (468, 3)")
            return None

        # ── 取出 26 個關鍵點 ────────────────────────────────────────────────────
        KeyPts = Landmarks3D[_KEY_INDICES]   # shape=(26, 3)

        # ── 計算 IOD（左右眼中心的 3D 距離）────────────────────────────────────
        LeftEyeCenter  = KeyPts[_LEFT_EYE_IN_KEY].mean(axis=0)
        RightEyeCenter = KeyPts[_RIGHT_EYE_IN_KEY].mean(axis=0)
        Iod = float(np.linalg.norm(LeftEyeCenter - RightEyeCenter))

        if Iod < MIN_IOD_NORM:
            return None

        # ── 以鼻尖為原點（相對座標）────────────────────────────────────────────
        NoseTip = Landmarks3D[1]
        KeyPts  = KeyPts - NoseTip   # shape=(26, 3)

        # ── Part A：成對 3D 距離 / IOD（325 維）────────────────────────────────
        PtA   = KeyPts[_PAIRS[:, 0]]                          # shape=(325, 3)
        PtB   = KeyPts[_PAIRS[:, 1]]                          # shape=(325, 3)
        Dists = np.linalg.norm(PtA - PtB, axis=1) / Iod      # shape=(325,)

        # ── Part B：各關鍵點 z 深度 / IOD（26 維）──────────────────────────────
        ZDepths = KeyPts[:, 2] / Iod                          # shape=(26,)

        return np.concatenate([Dists, ZDepths]).astype(float)  # shape=(351,)

    except Exception as Error:
        print(f"[face_feature_3d] 特徵萃取失敗：{Error}")
        return None
