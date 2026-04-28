"""
face_pose_classifier.py

依 MediaPipe 468 個 3D landmark 判斷頭部姿態，分為五類（從觀察者視角）：
  0 = 置中（正臉）
  1 = 左上（臉朝左上方）
  2 = 右上（臉朝右上方）
  3 = 左下（臉朝左下方）
  4 = 右下（臉朝右下方）

【Signed Yaw（水平轉角）】
  SignedYaw = (DistA − DistB) / FaceWidth
    DistA = NoseTipX − min(LeftCheekX, RightCheekX)
    DistB = max(LeftCheekX, RightCheekX) − NoseTipX
  負 = 鼻尖偏左（臉朝左），正 = 鼻尖偏右（臉朝右），正臉 ≈ 0

【Signed Pitch（垂直傾角）】
  SignedPitch = (ChinZ − ForeheadZ) / FaceHeight
    FaceHeight = ChinY − ForeheadY（像素高度，作為縮放基準）
  MediaPipe Z 軸：越小表示越靠近鏡頭
  低頭（DOWN）：下巴遠離、額頭靠近 → ChinZ↑、ForeheadZ↓ → 正值
  仰頭（UP）  ：下巴靠近、額頭遠離 → ChinZ↓、ForeheadZ↑ → 負值
  正臉 ≈ 0
"""

import numpy as np

# ── 姿態常數 ─────────────────────────────────────────────────────────────────
POSE_FRONTAL    = 0   # 置中（正臉）
POSE_LEFT_UP    = 1   # 臉朝左上方
POSE_RIGHT_UP   = 2   # 臉朝右上方
POSE_LEFT_DOWN  = 3   # 臉朝左下方
POSE_RIGHT_DOWN = 4   # 臉朝右下方

POSE_NAMES    = ['置中', '左上', '右上', '左下', '右下']
POSE_NAMES_EN = ['Ctr',  'LU',  'RU',   'LD',   'RD']   # OpenCV putText 用

# ── 判斷閾值 ──────────────────────────────────────────────────────────────────
# 正臉範圍刻意設寬：只有頭部明顯轉動才歸入四個側臉象限
YAW_THRESH   = 0.30   # 水平轉角絕對值超過此值才算偏左/偏右
PITCH_THRESH = 0.10   # Z 軸縱傾絕對值超過此值才算偏上/偏下
ROLL_THRESH  = 0.15   # 歪頭角度閾值（弧度，≈ 8.6°）；超過視為歪頭，套用穩定臉快取


def _computeSignedYaw(Landmarks3D: np.ndarray) -> float:
    """
    計算有符號的水平轉角比例（Signed Yaw）。

    利用左右顴骨（index 234, 454）與鼻尖（index 1）的 x 軸不對稱度：
      正臉 → YAW ≈ 0；臉朝左 → YAW < 0；臉朝右 → YAW > 0

    Returns
    -------
    float，正臉 ≈ 0，負 = 臉朝左，正 = 臉朝右
    """
    try:
        LeftCheekX  = float(Landmarks3D[234, 0])
        RightCheekX = float(Landmarks3D[454, 0])
        NoseTipX    = float(Landmarks3D[1,   0])
        MinX      = min(LeftCheekX, RightCheekX)
        MaxX      = max(LeftCheekX, RightCheekX)
        FaceWidth = MaxX - MinX
        if FaceWidth < 1e-5:
            return 0.0
        DistA = NoseTipX - MinX
        DistB = MaxX     - NoseTipX
        return (DistA - DistB) / FaceWidth
    except Exception:
        return 0.0


def _computeSignedPitch(Landmarks3D: np.ndarray) -> float:
    """
    計算有符號的垂直傾角（Signed Pitch），使用 Z 軸深度差。

    MediaPipe Z 軸：越小越靠近鏡頭。
      低頭（DOWN）：下巴遠離鏡頭（ChinZ↑）、額頭靠近（ForeheadZ↓）→ 正值
      仰頭（UP）  ：下巴靠近鏡頭（ChinZ↓）、額頭遠離（ForeheadZ↑）→ 負值
      正臉 ≈ 0

    Returns
    -------
    float，正臉 ≈ 0，負 = 臉朝上，正 = 臉朝下
    """
    try:
        ForeheadY = float(Landmarks3D[10,  1])
        ChinY     = float(Landmarks3D[152, 1])
        ForeheadZ = float(Landmarks3D[10,  2])
        ChinZ     = float(Landmarks3D[152, 2])
        FaceHeight = ChinY - ForeheadY
        if abs(FaceHeight) < 1e-5:  # 避免圖的資料有錯(因為實際上不可能下巴和額頭在相同Y的位置
            return 0.0  # 這種情況下無法計算傾角，直接回傳 0（當作正臉）, 也避免下行除以零造成錯誤
        return (ChinZ - ForeheadZ) / FaceHeight
    except Exception:
        return 0.0


def _computeRoll(Landmarks3D: np.ndarray) -> float:
    """
    計算有符號的橫滾角（Roll），使用左右顴骨連線相對水平的弧度。

    利用左顴骨（index 234）→ 右顴骨（index 454）連線在螢幕 x-y 平面的傾角：
      Roll ≈ 0  → 頭部直立
      Roll > 0  → 頭向右傾（右耳朝下）
      Roll < 0  → 頭向左傾（左耳朝下）

    Returns
    -------
    float，弧度值
    """
    try:
        Dx = float(Landmarks3D[454, 0] - Landmarks3D[234, 0])
        Dy = float(Landmarks3D[454, 1] - Landmarks3D[234, 1])
        if abs(Dx) < 1e-8 and abs(Dy) < 1e-8:
            return 0.0
        return float(np.arctan2(Dy, Dx))
    except Exception:
        return 0.0


def classifyPose(Landmarks3D: np.ndarray) -> int:
    """
    依 468 個 3D landmark 判斷頭部姿態類別（0~4）。

    Parameters
    ----------
    Landmarks3D : np.ndarray, shape=(468, 3)

    Returns
    -------
    int — POSE_FRONTAL=0 / POSE_LEFT_UP=1 / POSE_RIGHT_UP=2 /
           POSE_LEFT_DOWN=3 / POSE_RIGHT_DOWN=4
    """
    try:
        Yaw   = _computeSignedYaw(Landmarks3D)
        Pitch = _computeSignedPitch(Landmarks3D)

        # 正臉範圍：兩軸均未超過閾值
        if abs(Yaw) < YAW_THRESH and abs(Pitch) < PITCH_THRESH:
            return POSE_FRONTAL

        # 非正臉：依符號落入四象限
        if   Yaw <= 0 and Pitch <= 0:
            return POSE_LEFT_UP
        elif Yaw >  0 and Pitch <= 0:
            return POSE_RIGHT_UP
        elif Yaw <= 0 and Pitch >  0:
            return POSE_LEFT_DOWN
        else:
            return POSE_RIGHT_DOWN

    except Exception:
        return POSE_FRONTAL


def classifyPoseWithValues(Landmarks3D: np.ndarray) -> tuple:
    """
    判斷姿態類別，同時回傳原始 Yaw / Pitch / Roll 數值（供 UI 顯示與閾值除錯）。

    Returns
    -------
    (PoseCat: int, SignedYaw: float, SignedPitch: float, Roll: float)
    """
    try:
        Yaw   = _computeSignedYaw(Landmarks3D)
        Pitch = _computeSignedPitch(Landmarks3D)
        Roll  = _computeRoll(Landmarks3D)

        if abs(Yaw) < YAW_THRESH and abs(Pitch) < PITCH_THRESH:
            Cat = POSE_FRONTAL
        elif Yaw <= 0 and Pitch <= 0:
            Cat = POSE_LEFT_UP
        elif Yaw >  0 and Pitch <= 0:
            Cat = POSE_RIGHT_UP
        elif Yaw <= 0 and Pitch >  0:
            Cat = POSE_LEFT_DOWN
        else:
            Cat = POSE_RIGHT_DOWN

        return Cat, Yaw, Pitch, Roll
    except Exception:
        return POSE_FRONTAL, 0.0, 0.0, 0.0


def getSignedYawPitch(Landmarks3D: np.ndarray) -> tuple:
    """回傳 (SignedYaw, SignedPitch)，供外部顯示或除錯使用。"""
    return _computeSignedYaw(Landmarks3D), _computeSignedPitch(Landmarks3D)
