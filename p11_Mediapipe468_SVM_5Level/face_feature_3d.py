"""
face_feature_3d.py  (p11 改版)

臉部特徵萃取 — 以主管建議的向量夾角法實作。

【設計原則（主管建議）】
  Step 1 重心向量化：v_i = P_i − P_鼻尖
          以鼻尖為參考原點，解決平移問題。
  Step 2 單位化：    unit_i = v_i / ‖v_i‖
          方向向量化，解決個人頭型大小的縮放問題。
  Step 3 點對點向量夾角：θ_ij = arccos(unit_i · unit_j)
          夾角為 3D 幾何不變量，不受頭部遠近影響。

【特徵向量組成（325 維）】
  Part A：C(25, 2) = 300 個向量夾角（弧度 0 ~ π）
  Part B：25 個正規化距離 ‖v_i‖ / 臉部寬度（尺度比例）
  合計：300 + 25 = 325 維

【選取 25 個非鼻尖關鍵點】
  眼睛四角、眼瞼、眉毛×2、鼻翼、鼻基底、嘴角、上下唇、
  下巴、顴骨×2、額頭（共 25 點，鼻尖僅當原點，不入特徵）

【退化防護】
  臉部寬度（顴骨間距）< MIN_FACE_WIDTH → 視為偵測退化，回傳 None
"""

import numpy as np
from itertools import combinations

# ── 25 個非鼻尖關鍵 landmark 索引（MediaPipe Face Mesh 468 點）────────────────
_KEY_INDICES = np.array([
    10,   # index  0：額頭中心
    33,   # index  1：左眼外角
    159,  # index  2：左眼上緣
    145,  # index  3：左眼下緣
    133,  # index  4：左眼內角
    362,  # index  5：右眼內角
    386,  # index  6：右眼上緣
    374,  # index  7：右眼下緣
    263,  # index  8：右眼外角
    46,   # index  9：左眉外緣
    105,  # index 10：左眉頂點
    107,  # index 11：左眉內緣
    336,  # index 12：右眉內緣
    334,  # index 13：右眉頂點
    276,  # index 14：右眉外緣
    129,  # index 15：左鼻翼
    358,  # index 16：右鼻翼
    94,   # index 17：鼻基底
    61,   # index 18：左嘴角
    291,  # index 19：右嘴角
    13,   # index 20：上唇中心
    14,   # index 21：下唇中心
    152,  # index 22：下巴
    234,  # index 23：左顴骨 ← 臉部寬度左端
    454,  # index 24：右顴骨 ← 臉部寬度右端
])  # 共 25 個

# 鼻尖 landmark 索引（MediaPipe），僅當平移原點，不加入特徵
_NOSE_TIP_LM_IDX = 1

# 顴骨在 _KEY_INDICES 中的位置索引（用於計算臉部寬度）
_CHEEK_LEFT_IDX  = 23
_CHEEK_RIGHT_IDX = 24

# 預先計算 C(25, 2) = 300 對索引
_PAIRS = np.array(list(combinations(range(len(_KEY_INDICES)), 2)), dtype=int)

# 臉部寬度最小合理值（小於此值視為 MediaPipe 偵測退化）
MIN_FACE_WIDTH = 1e-5


def extractFeatures3D(Landmarks3D: np.ndarray) -> np.ndarray | None:
    """
    從 468 個 3D 歸一化 landmark 萃取 325 維臉部幾何特徵向量。

    Parameters
    ----------
    Landmarks3D : np.ndarray, shape=(468, 3)
        MediaPipe FaceLandmarker 回傳的歸一化座標 (x, y, z)

    Returns
    -------
    np.ndarray, shape=(325,)  或  None（退化 / 萃取失敗）
    """
    try:
        if Landmarks3D.shape != (468, 3):
            print(f"[face_feature_3d] 輸入維度錯誤：{Landmarks3D.shape}，預期 (468, 3)")
            return None

        # Step 1：以鼻尖為原點，計算 25 個關鍵點的位移向量
        NoseTip = Landmarks3D[_NOSE_TIP_LM_IDX]        # shape=(3,)
        KeyPts  = Landmarks3D[_KEY_INDICES]              # shape=(25, 3)
        Vecs    = KeyPts - NoseTip                        # shape=(25, 3)

        # 計算臉部寬度（左右顴骨間距）作為尺度基準
        FaceWidth = float(
            np.linalg.norm(Vecs[_CHEEK_LEFT_IDX] - Vecs[_CHEEK_RIGHT_IDX])
        )
        if FaceWidth < MIN_FACE_WIDTH:
            return None

        # Part B：各關鍵點到鼻尖的正規化距離（‖v_i‖ / 臉部寬度）
        NormDists = np.linalg.norm(Vecs, axis=1) / FaceWidth    # shape=(25,)

        # Step 2：將各位移向量單位化（方向化）
        VecNorms = np.linalg.norm(Vecs, axis=1, keepdims=True)  # shape=(25, 1)
        VecNorms[VecNorms < 1e-8] = 1.0                          # 防零向量除法
        UnitVecs = Vecs / VecNorms                                # shape=(25, 3)

        # Step 3：Part A — C(25,2)=300 對向量夾角（弧度 0 ~ π）
        # cos θ_ij = unit_i · unit_j；arccos 轉換為實際夾角
        CosMat  = np.clip(UnitVecs @ UnitVecs.T, -1.0, 1.0)     # shape=(25, 25)
        AngMat  = np.arccos(CosMat)                               # shape=(25, 25)
        Angles  = AngMat[_PAIRS[:, 0], _PAIRS[:, 1]]             # shape=(300,)

        return np.concatenate([Angles, NormDists]).astype(float)  # shape=(325,)

    except Exception as Error:
        print(f"[face_feature_3d] 特徵萃取失敗：{Error}")
        return None
