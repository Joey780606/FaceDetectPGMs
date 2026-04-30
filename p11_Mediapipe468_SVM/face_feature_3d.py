"""
face_feature_3d.py  (p11 改版)

臉部特徵萃取 — 以主管建議的向量夾角法實作，加入頭部姿態正規化。

【設計原則】
  Step 0 姿態正規化：從臉部幾何建立旋轉矩陣 R，
          套用 R^T 將所有 landmark 轉回正臉座標系，
          消除頭部左右轉、上下仰的影響。
  Step 1 重心向量化：v_i = P_i − P_鼻尖
          以鼻尖為參考原點，解決平移問題。
  Step 2 單位化：    unit_i = v_i / ‖v_i‖
          方向向量化，解決個人頭型大小的縮放問題。
  Step 3 點對點向量夾角：θ_ij = arccos(unit_i · unit_j)
          在正規化座標系下，夾角真正成為姿態不變量。

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
    10,   # index  0：額頭中心        ← 額頭
    33,   # index  1：左眼外角        ← 眼睛四角（左眼）
    159,  # index  2：左眼上緣        ← 眼瞼（左眼）
    145,  # index  3：左眼下緣        ← 眼瞼（左眼）
    133,  # index  4：左眼內角        ← 眼睛四角（左眼）
    362,  # index  5：右眼內角        ← 眼睛四角（右眼）
    386,  # index  6：右眼上緣        ← 眼瞼（右眼）
    374,  # index  7：右眼下緣        ← 眼瞼（右眼）
    263,  # index  8：右眼外角        ← 眼睛四角（右眼）
    46,   # index  9：左眉外緣        ← 眉毛（左）
    105,  # index 10：左眉頂點        ← 眉毛（左）
    107,  # index 11：左眉內緣        ← 眉毛（左）
    336,  # index 12：右眉內緣        ← 眉毛（右）
    334,  # index 13：右眉頂點        ← 眉毛（右）
    276,  # index 14：右眉外緣        ← 眉毛（右）
    129,  # index 15：左鼻翼          ← 鼻翼
    358,  # index 16：右鼻翼          ← 鼻翼
    94,   # index 17：鼻基底          ← 鼻基底
    61,   # index 18：左嘴角          ← 嘴角
    291,  # index 19：右嘴角          ← 嘴角
    13,   # index 20：上唇中心        ← 上唇
    14,   # index 21：下唇中心        ← 下唇
    152,  # index 22：下巴            ← 下巴
    234,  # index 23：左顴骨          ← 顴骨×2（臉部寬度左端）
    454,  # index 24：右顴骨          ← 顴骨×2（臉部寬度右端）
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


def _buildFaceRotationMatrix(Landmarks3D: np.ndarray) -> np.ndarray:
    """
    從左右顴骨（234, 454）與額頭/下巴（10, 152）建立臉部座標系旋轉矩陣 R。

    臉部座標系定義：
      X 軸：左顴骨 → 右顴骨（臉部左右方向）
      Y 軸：下巴 → 額頭（臉部上下方向），與 X 正交化
      Z 軸：X × Y（臉部法向量，正臉時指向攝影機）

    R 的三列為上述三軸在攝影機空間的方向向量。
    R^T（= R^-1）將攝影機空間座標轉回臉部正規化座標系。

    Returns
    -------
    np.ndarray, shape=(3, 3)，失敗時回傳單位矩陣。
    """
    try:
        # X 軸：左顴骨 → 右顴骨
        XRaw  = Landmarks3D[454] - Landmarks3D[234]
        XNorm = np.linalg.norm(XRaw)
        if XNorm < 1e-8:
            return np.eye(3)
        XAxis = XRaw / XNorm

        # Y 軸：下巴 → 額頭（Gram-Schmidt 對 X 正交化）
        YRaw  = Landmarks3D[10] - Landmarks3D[152]
        YAxis = YRaw - np.dot(YRaw, XAxis) * XAxis
        YNorm = np.linalg.norm(YAxis)
        if YNorm < 1e-8:
            return np.eye(3)
        YAxis = YAxis / YNorm

        # Z 軸：X × Y
        ZAxis = np.cross(XAxis, YAxis)
        ZNorm = np.linalg.norm(ZAxis)
        if ZNorm < 1e-8:
            return np.eye(3)
        ZAxis = ZAxis / ZNorm

        return np.column_stack([XAxis, YAxis, ZAxis])   # shape (3, 3)

    except Exception:
        return np.eye(3)


def extractFeatures3D(Landmarks3D: np.ndarray) -> np.ndarray | None:
    """
    從 468 個 3D 歸一化 landmark 萃取 325 維臉部幾何特徵向量。
    先將 landmark 旋轉至正臉座標系，再取夾角特徵，使特徵對頭部姿態不變。

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

        # Step 0：建立旋轉矩陣，將所有 landmark 轉回正臉座標系
        R        = _buildFaceRotationMatrix(Landmarks3D)    # 建旋轉矩陣
        NoseTip  = Landmarks3D[_NOSE_TIP_LM_IDX]           # shape=(3,)
        Centered = Landmarks3D - NoseTip                    # shape=(468, 3),平移到鼻尖為原點
        Canonical = (R.T @ Centered.T).T                    # shape=(468, 3),旋轉至正臉座標系

        # Step 1：取 25 個關鍵點位移向量（已在正規化座標系，鼻尖為原點）
        Vecs = Canonical[_KEY_INDICES]                      # shape=(25, 3)

        # 計算臉部寬度（顴骨間距，正規化座標系下）
        FaceWidth = float(
            np.linalg.norm(Vecs[_CHEEK_LEFT_IDX] - Vecs[_CHEEK_RIGHT_IDX])
        )
        if FaceWidth < MIN_FACE_WIDTH:
            return None

        # Part B：各關鍵點到鼻尖的正規化距離（‖v_i‖ / 臉部寬度）
        NormDists = np.linalg.norm(Vecs, axis=1) / FaceWidth    # shape=(25,)

        # Step 2：將各位移向量單位化
        VecNorms = np.linalg.norm(Vecs, axis=1, keepdims=True)  # shape=(25, 1)
        VecNorms[VecNorms < 1e-8] = 1.0
        UnitVecs = Vecs / VecNorms                               # shape=(25, 3)

        # Step 3：Part A — C(25,2)=300 對向量夾角（弧度 0 ~ π）
        CosMat = np.clip(UnitVecs @ UnitVecs.T, -1.0, 1.0)      # shape=(25, 25)
        AngMat = np.arccos(CosMat)                                # shape=(25, 25)
        Angles = AngMat[_PAIRS[:, 0], _PAIRS[:, 1]]              # shape=(300,)

        return np.concatenate([Angles, NormDists]).astype(float)  # shape=(325,)

    except Exception as Error:
        print(f"[face_feature_3d] 特徵萃取失敗：{Error}")
        return None
