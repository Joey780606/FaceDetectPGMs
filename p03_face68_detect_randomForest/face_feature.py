"""
face_feature.py

從 face_recognition.face_landmarks() 回傳的 68 個 landmark 座標中，
萃取具代表性的臉部幾何特徵向量，供 Random Forest 訓練與辨識使用。

特徵向量共 23 維：
  距離特徵 15 個（皆除以瞳距 IOD 歸一化）
  角度特徵  3 個
  比例特徵  5 個

缺失值以 -1.0 填充。
若瞳距小於 MIN_IOD_PIXELS（側臉或偵測異常），回傳 None。
"""

import numpy as np

# 瞳距最小合理值（像素）；小於此值視為側臉或偵測異常
MIN_IOD_PIXELS = 30


# ==============================================================================
# 私有輔助函式
# ==============================================================================

def _pt(Points: list, Index: int) -> np.ndarray:
    """取得指定索引的座標點，回傳 numpy [x, y]。"""
    return np.array(Points[Index], dtype=float)


def _center(Points: list) -> np.ndarray:
    """計算一組座標點的幾何中心。"""
    return np.mean(np.array(Points, dtype=float), axis=0)


def _dist(A: np.ndarray, B: np.ndarray) -> float:
    """計算兩點歐氏距離。"""
    return float(np.linalg.norm(A - B))


def _angle_deg(A: np.ndarray, Vertex: np.ndarray, B: np.ndarray) -> float:
    """計算以 Vertex 為頂點，向量 A→Vertex 與 Vertex→B 的夾角（度）。"""
    Va = A - Vertex
    Vb = B - Vertex
    NormA = np.linalg.norm(Va)
    NormB = np.linalg.norm(Vb)
    if NormA < 1e-9 or NormB < 1e-9:
        return 0.0
    Cos = float(np.clip(np.dot(Va, Vb) / (NormA * NormB), -1.0, 1.0))
    return float(np.degrees(np.arccos(Cos)))


def _xspan_left(Pts: np.ndarray) -> np.ndarray:
    """回傳 x 座標最小的點（最左端）。"""
    return Pts[np.argmin(Pts[:, 0])]


def _xspan_right(Pts: np.ndarray) -> np.ndarray:
    """回傳 x 座標最大的點（最右端）。"""
    return Pts[np.argmax(Pts[:, 0])]


def _yspan_top(Pts: np.ndarray) -> np.ndarray:
    """回傳 y 座標最小的點（最上端，螢幕座標系）。"""
    return Pts[np.argmin(Pts[:, 1])]


def _yspan_bot(Pts: np.ndarray) -> np.ndarray:
    """回傳 y 座標最大的點（最下端，螢幕座標系）。"""
    return Pts[np.argmax(Pts[:, 1])]


# ==============================================================================
# 公開函式
# ==============================================================================

def extractFeatures(Landmarks: dict) -> np.ndarray | None:
    """
    從 face_recognition.face_landmarks() 回傳的單張臉 landmark dict 中，
    萃取 23 維特徵向量。

    Parameters
    ----------
    Landmarks : dict
        key 為部位名稱，value 為 (x, y) tuple 的列表。

    Returns
    -------
    np.ndarray, shape (23,)  或  None（側臉 / IOD 過小 / 萃取失敗）
    """
    try:
        # ── 取出各部位座標 ────────────────────────────────────────────────
        Chin       = Landmarks.get('chin',         [])
        LeftEye    = Landmarks.get('left_eye',     [])
        RightEye   = Landmarks.get('right_eye',    [])
        LeftBrow   = Landmarks.get('left_eyebrow', [])
        RightBrow  = Landmarks.get('right_eyebrow',[])
        NoseBridge = Landmarks.get('nose_bridge',  [])
        NoseTip    = Landmarks.get('nose_tip',     [])
        TopLip     = Landmarks.get('top_lip',      [])

        # 雙眼為必要部位，缺失則無法繼續
        if not LeftEye or not RightEye:
            return None

        # ── 瞳距（IOD）計算 ───────────────────────────────────────────────
        LeftEyePts   = np.array(LeftEye,  dtype=float)
        RightEyePts  = np.array(RightEye, dtype=float)
        LeftEyeCenter  = _center(LeftEye)
        RightEyeCenter = _center(RightEye)
        Iod = _dist(LeftEyeCenter, RightEyeCenter)

        if Iod < MIN_IOD_PIXELS:
            # 側臉或偵測異常，跳過此幀
            return None

        def D(A: np.ndarray, B: np.ndarray) -> float:
            """距離除以 IOD 做歸一化。"""
            return _dist(A, B) / Iod

        # ── 各部位端點 ────────────────────────────────────────────────────

        # 眼睛
        LEyeLeft   = _xspan_left(LeftEyePts)
        LEyeRight  = _xspan_right(LeftEyePts)
        LEyeTop    = _yspan_top(LeftEyePts)
        LEyeBot    = _yspan_bot(LeftEyePts)
        REyeLeft   = _xspan_left(RightEyePts)
        REyeRight  = _xspan_right(RightEyePts)
        REyeTop    = _yspan_top(RightEyePts)
        REyeBot    = _yspan_bot(RightEyePts)

        # 眉毛中心
        LBrowCenter = _center(LeftBrow)  if LeftBrow  else LeftEyeCenter  - np.array([0.0, 15.0])
        RBrowCenter = _center(RightBrow) if RightBrow else RightEyeCenter - np.array([0.0, 15.0])
        LBrowPts    = np.array(LeftBrow,  dtype=float) if LeftBrow  else None
        RBrowPts    = np.array(RightBrow, dtype=float) if RightBrow else None
        LBrowLeft   = _xspan_left(LBrowPts)  if LBrowPts is not None else LEyeLeft  - np.array([0.0, 10.0])
        LBrowRight  = _xspan_right(LBrowPts) if LBrowPts is not None else LEyeRight - np.array([0.0, 10.0])
        RBrowLeft   = _xspan_left(RBrowPts)  if RBrowPts is not None else REyeLeft  - np.array([0.0, 10.0])
        RBrowRight  = _xspan_right(RBrowPts) if RBrowPts is not None else REyeRight - np.array([0.0, 10.0])

        # 鼻子
        NoseTipCenter  = _center(NoseTip) if NoseTip else (LeftEyeCenter + RightEyeCenter) / 2 + np.array([0.0, 40.0])
        NoseBridgeTop  = np.array(NoseBridge[0], dtype=float) if NoseBridge else (LeftEyeCenter + RightEyeCenter) / 2
        NoseTipPts     = np.array(NoseTip, dtype=float) if NoseTip else None
        NoseTipLeft    = _xspan_left(NoseTipPts)  if NoseTipPts is not None else NoseTipCenter - np.array([8.0, 0.0])
        NoseTipRight   = _xspan_right(NoseTipPts) if NoseTipPts is not None else NoseTipCenter + np.array([8.0, 0.0])

        # 嘴巴（top_lip 共 12 個點，索引 0=左角，6=右角）
        TopLipPts   = np.array(TopLip, dtype=float) if TopLip else None
        MouthLeft   = TopLipPts[0]   if TopLipPts is not None else NoseTipCenter + np.array([-20.0, 18.0])
        MouthRight  = TopLipPts[6]   if TopLipPts is not None else NoseTipCenter + np.array([20.0,  18.0])
        MouthCenter = _center(TopLip) if TopLip else NoseTipCenter + np.array([0.0, 18.0])

        # 下巴（chin 共 17 個點，索引 0=左端，8=最低點，16=右端）
        ChinPts   = np.array(Chin, dtype=float) if Chin else None
        ChinLeft  = ChinPts[0]  if ChinPts is not None else LeftEyeCenter  + np.array([-15.0, 55.0])
        ChinRight = ChinPts[-1] if ChinPts is not None else RightEyeCenter + np.array([15.0,  55.0])
        ChinBot   = ChinPts[8]  if ChinPts is not None else (ChinLeft + ChinRight) / 2 + np.array([0.0, 15.0])

        # ── 距離特徵 15 個（除以 IOD）────────────────────────────────────
        F01 = D(LEyeLeft,    LEyeRight)                                         # 左眼寬度
        F02 = D(REyeLeft,    REyeRight)                                         # 右眼寬度
        F03 = D(LBrowLeft,   LBrowRight) if LeftBrow  else -1.0                 # 左眉寬度
        F04 = D(RBrowLeft,   RBrowRight) if RightBrow else -1.0                 # 右眉寬度
        if len(NoseBridge) >= 2:
            F05 = _dist(np.array(NoseBridge[0], dtype=float),
                        np.array(NoseBridge[-1], dtype=float)) / Iod            # 鼻梁長度
        else:
            F05 = -1.0
        F06 = D(NoseTipLeft, NoseTipRight) if NoseTip else -1.0                 # 鼻翼寬度
        F07 = D(MouthLeft,   MouthRight)                                         # 嘴角寬度
        F08 = D(LeftEyeCenter,  LBrowCenter) if LeftBrow  else -1.0             # 左眼→左眉距離
        F09 = D(RightEyeCenter, RBrowCenter) if RightBrow else -1.0             # 右眼→右眉距離
        F10 = D(NoseTipCenter, MouthCenter)  if NoseTip and TopLip else -1.0    # 鼻尖→嘴中心距離
        F11 = D(ChinLeft, ChinRight)         if Chin else -1.0                  # 下巴寬度
        F12 = D(ChinBot,  NoseBridgeTop)     if Chin and NoseBridge else -1.0   # 臉高
        F13 = D(LeftEyeCenter,  NoseTipCenter) if NoseTip else -1.0             # 左眼中心→鼻尖
        F14 = D(RightEyeCenter, NoseTipCenter) if NoseTip else -1.0             # 右眼中心→鼻尖
        LEyeW = _dist(LEyeLeft, LEyeRight)
        REyeW = _dist(REyeLeft, REyeRight)
        F15 = (LEyeW - REyeW) / Iod                                             # 眼寬不對稱率

        # ── 角度特徵 3 個（度）───────────────────────────────────────────
        # F16：左眼外角—鼻尖—右眼外角 夾角
        F16 = _angle_deg(LEyeRight, NoseTipCenter, REyeLeft) if NoseTip else -1.0
        # F17：左嘴角—嘴中心—右嘴角 夾角
        F17 = _angle_deg(MouthLeft, MouthCenter, MouthRight) if TopLip else -1.0
        # F18：兩眉連線與水平的仰角（度），仰頭為正
        if LeftBrow and RightBrow:
            BrowVec = RBrowCenter - LBrowCenter
            F18 = float(np.degrees(np.arctan2(-BrowVec[1], max(abs(BrowVec[0]), 1e-9))))
        else:
            F18 = -1.0

        # ── 比例特徵 5 個 ─────────────────────────────────────────────────
        NoseW  = _dist(NoseTipLeft, NoseTipRight) if NoseTip else 0.0
        MouthW = _dist(MouthLeft, MouthRight)
        F19 = (NoseW / MouthW)  if MouthW > 1e-9 and NoseTip else -1.0         # 鼻翼寬 / 嘴角寬
        F20 = MouthW / Iod                                                       # 嘴角寬 / 瞳距
        F21 = (_dist(LEyeTop, LEyeBot) / LEyeW) if LEyeW > 1e-9 else -1.0      # 左眼高 / 左眼寬
        F22 = (_dist(REyeTop, REyeBot) / REyeW) if REyeW > 1e-9 else -1.0      # 右眼高 / 右眼寬
        ChinW = _dist(ChinLeft, ChinRight) if Chin else 0.0
        FaceH = _dist(ChinBot, NoseBridgeTop) if Chin and NoseBridge else 0.0
        F23 = (ChinW / FaceH) if FaceH > 1e-9 and Chin and NoseBridge else -1.0 # 下巴寬 / 臉高

        return np.array([
            F01, F02, F03, F04, F05,
            F06, F07, F08, F09, F10,
            F11, F12, F13, F14, F15,
            F16, F17, F18,
            F19, F20, F21, F22, F23,
        ], dtype=float)

    except Exception as Error:
        print(f"[face_feature] 特徵萃取失敗：{Error}")
        return None
