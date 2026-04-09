"""
random_forest_np.py

純 NumPy 實作的隨機森林分類器，不依賴 sklearn / scipy，
避免 Windows Application Control 封鎖 scipy DLL 的問題。

包含：
  DecisionTree  - 單棵決策樹（Gini 分裂準則）
  RandomForest  - 多棵樹的集成分類器（Bootstrap 抽樣）
  OnePerson     - 單人模式：馬氏距離閾值判斷 Known / Unknown
"""

import numpy as np

# ── 超參數預設值 ───────────────────────────────────────────────────────────────
DEFAULT_N_TREES       = 100    # 隨機森林樹的數量
DEFAULT_MAX_DEPTH     = 12     # 決策樹最大深度
DEFAULT_MIN_SAMPLES   = 2      # 節點分裂最小樣本數
UNKNOWN_THRESHOLD     = 0.45   # 最高類別機率低於此值 → 視為 Unknown
MAHAL_UNKNOWN_THRESH  = 8.0    # 單人模式馬氏距離閾值（> 此值 → Unknown）


# ==============================================================================
# 私有輔助函式
# ==============================================================================

def _gini(Counts: np.ndarray) -> float:
    """計算 Gini 不純度。"""
    Total = Counts.sum()
    if Total == 0:
        return 0.0
    P = Counts / Total
    return float(1.0 - np.sum(P ** 2))


def _bestSplit(X: np.ndarray, Y: np.ndarray, FeatureIndices: np.ndarray,
               AllClasses: np.ndarray) -> tuple:
    """
    在指定特徵子集中尋找使 Gini 增益最大的分裂點。

    Returns
    -------
    (BestFeat, BestThresh, BestGain)：最佳特徵索引、閾值、Gini 增益
    若找不到有效分裂，BestFeat = -1。
    """
    NSamples = X.shape[0]
    NClasses = len(AllClasses)

    # 父節點 Gini
    ParentCounts = np.array([np.sum(Y == c) for c in AllClasses], dtype=float)
    ParentGini   = _gini(ParentCounts)

    BestGain   = -1.0
    BestFeat   = -1
    BestThresh = 0.0

    for FeatIdx in FeatureIndices:
        Values     = X[:, FeatIdx]
        UniqueVals = np.unique(Values)
        if len(UniqueVals) < 2:
            continue

        # 候選閾值：相鄰唯一值的中點
        Thresholds = (UniqueVals[:-1] + UniqueVals[1:]) / 2.0

        for Thresh in Thresholds:
            LeftMask  = Values <= Thresh
            RightMask = ~LeftMask
            NLeft  = int(LeftMask.sum())
            NRight = int(RightMask.sum())
            if NLeft == 0 or NRight == 0:
                continue

            LeftCounts  = np.array([np.sum(Y[LeftMask]  == c) for c in AllClasses], dtype=float)
            RightCounts = np.array([np.sum(Y[RightMask] == c) for c in AllClasses], dtype=float)
            ChildGini   = (NLeft * _gini(LeftCounts) + NRight * _gini(RightCounts)) / NSamples
            Gain        = ParentGini - ChildGini

            if Gain > BestGain:
                BestGain   = Gain
                BestFeat   = FeatIdx
                BestThresh = float(Thresh)

    return BestFeat, BestThresh, BestGain


# ==============================================================================
# Class: DecisionTree
# ==============================================================================
class DecisionTree:
    """
    單棵決策樹（純 NumPy，Gini 分裂準則）。
    葉節點儲存各類別的機率分佈，長度固定為訓練時的類別數。
    """

    def __init__(self, MaxDepth: int = DEFAULT_MAX_DEPTH,
                 MinSamples: int = DEFAULT_MIN_SAMPLES):
        self._MaxDepth   = MaxDepth
        self._MinSamples = MinSamples
        self._Root       = None
        self._Classes    = None   # 整數類別陣列（由 RandomForest 設定）

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        訓練決策樹。
        self._Classes 須在呼叫前由外部設定（RandomForest 負責設定）。

        Parameters
        ----------
        X : shape (n_samples, n_features)
        Y : shape (n_samples,) 整數標籤，與 self._Classes 對應
        """
        NFeatures = X.shape[1]
        NFeatSub  = max(1, int(np.sqrt(NFeatures)))   # 每節點隨機選取的特徵數
        self._Root = self._buildNode(X, Y, Depth=0, NFeatSub=NFeatSub)

    def _buildNode(self, X: np.ndarray, Y: np.ndarray, Depth: int, NFeatSub: int) -> dict:
        """遞迴建立節點，回傳 dict 表示的樹結構。"""
        NSamples = len(Y)

        # 葉節點條件：只剩一類、深度達上限、樣本數不足
        if len(np.unique(Y)) == 1 or Depth >= self._MaxDepth or NSamples < self._MinSamples:
            return self._makeLeaf(Y)

        # 隨機選擇特徵子集
        NFeat = X.shape[1]
        FeatureIndices = np.random.choice(NFeat, size=min(NFeatSub, NFeat), replace=False)

        BestFeat, BestThresh, BestGain = _bestSplit(X, Y, FeatureIndices, self._Classes)
        if BestFeat == -1 or BestGain <= 0:
            return self._makeLeaf(Y)

        LeftMask  = X[:, BestFeat] <= BestThresh
        RightMask = ~LeftMask
        if LeftMask.sum() == 0 or RightMask.sum() == 0:
            return self._makeLeaf(Y)

        return {
            'feat':   int(BestFeat),
            'thresh': BestThresh,
            'left':   self._buildNode(X[LeftMask],  Y[LeftMask],  Depth + 1, NFeatSub),
            'right':  self._buildNode(X[RightMask], Y[RightMask], Depth + 1, NFeatSub),
        }

    def _makeLeaf(self, Y: np.ndarray) -> dict:
        """建立葉節點，儲存各類別的機率分佈（長度 = 類別總數）。"""
        NClasses = len(self._Classes)
        Total    = len(Y)
        Proba    = np.zeros(NClasses, dtype=float)
        if Total > 0:
            for i, c in enumerate(self._Classes):
                Proba[i] = float(np.sum(Y == c)) / Total
        return {'leaf': True, 'proba': Proba}

    def predictProba(self, X: np.ndarray) -> np.ndarray:
        """
        預測機率矩陣。

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        return np.array([self._traverse(x, self._Root) for x in X])

    def _traverse(self, x: np.ndarray, Node: dict) -> np.ndarray:
        """遞迴走訪樹，回傳葉節點的機率分佈向量。"""
        if 'leaf' in Node:
            return Node['proba']
        if x[Node['feat']] <= Node['thresh']:
            return self._traverse(x, Node['left'])
        else:
            return self._traverse(x, Node['right'])


# ==============================================================================
# Class: RandomForest
# ==============================================================================
class RandomForest:
    """
    純 NumPy 隨機森林分類器（多類別，Bootstrap 抽樣）。
    最高類別機率低於 UnknownThreshold 時判斷為 "Unknown"。
    """

    def __init__(self, NTrees: int = DEFAULT_N_TREES,
                 MaxDepth: int = DEFAULT_MAX_DEPTH,
                 MinSamples: int = DEFAULT_MIN_SAMPLES,
                 UnknownThreshold: float = UNKNOWN_THRESHOLD):
        self._NTrees           = NTrees
        self._MaxDepth         = MaxDepth
        self._MinSamples       = MinSamples
        self._UnknownThreshold = UnknownThreshold
        self._Trees            = []
        self._ClassNames       = []    # 人名列表，索引對應整數標籤
        self._IsTrained        = False

    def fit(self, X: np.ndarray, Y: np.ndarray, ClassNames: list) -> None:
        """
        訓練隨機森林。

        Parameters
        ----------
        X          : shape (n_samples, n_features)
        Y          : shape (n_samples,) 整數標籤，範圍 0..len(ClassNames)-1
        ClassNames : 各標籤對應的人名列表
        """
        self._ClassNames = list(ClassNames)
        NClasses         = len(ClassNames)
        AllClasses       = np.arange(NClasses, dtype=int)
        NSamples         = X.shape[0]

        self._Trees = []
        for _ in range(self._NTrees):
            # Bootstrap 抽樣
            Indices = np.random.choice(NSamples, size=NSamples, replace=True)
            Xb = X[Indices]
            Yb = Y[Indices]

            Tree = DecisionTree(MaxDepth=self._MaxDepth, MinSamples=self._MinSamples)
            Tree._Classes = AllClasses   # 設定完整類別集合
            Tree.fit(Xb, Yb)
            self._Trees.append(Tree)

        self._IsTrained = True

    def predictProba(self, X: np.ndarray) -> np.ndarray:
        """
        回傳各樣本對各類別的平均機率。

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        if not self._IsTrained or not self._Trees:
            NClasses = len(self._ClassNames) if self._ClassNames else 1
            return np.zeros((len(X), NClasses))

        # 各樹預測後平均
        AllProba = np.stack([Tree.predictProba(X) for Tree in self._Trees], axis=0)
        return np.mean(AllProba, axis=0)

    def predict(self, X: np.ndarray) -> tuple:
        """
        回傳預測結果。

        Returns
        -------
        (Names: list[str], Confidences: np.ndarray)
        Names : 各樣本的預測人名，最高信心度低於閾值時回傳 "Unknown"
        Confidences : 各樣本最高類別的信心度
        """
        Proba  = self.predictProba(X)
        Names  = []
        Confs  = []
        for Row in Proba:
            MaxProb = float(Row.max())
            Confs.append(MaxProb)
            if MaxProb < self._UnknownThreshold:
                Names.append("Unknown")
            else:
                Names.append(self._ClassNames[int(Row.argmax())])
        return Names, np.array(Confs, dtype=float)

    @property
    def IsTrained(self) -> bool:
        return self._IsTrained


# ==============================================================================
# Class: OnePerson（單人馬氏距離分類器）
# ==============================================================================
class OnePerson:
    """
    只有一個人的訓練資料時，使用馬氏距離判斷 Known / Unknown。

    距離 ≤ UnknownThreshold → 回傳該人，信心度由距離轉換（exp 函數）。
    距離 >  UnknownThreshold → 回傳 "Unknown"。
    """

    def __init__(self, PersonName: str,
                 UnknownThreshold: float = MAHAL_UNKNOWN_THRESH):
        self._PersonName       = PersonName
        self._UnknownThreshold = UnknownThreshold
        self._Mean             = None
        self._InvCov           = None
        self._IsTrained        = False

    def fit(self, X: np.ndarray) -> None:
        """以樣本矩陣計算均值與偽逆共變異數矩陣。"""
        try:
            self._Mean = np.mean(X, axis=0)
            if X.shape[0] < 2:
                # 只有一個樣本，共變異數無法計算，使用單位矩陣
                self._InvCov = np.eye(X.shape[1])
            else:
                Cov = np.cov(X, rowvar=False)
                # 若矩陣奇異則用偽逆避免計算失敗
                self._InvCov = np.linalg.pinv(Cov)
            self._IsTrained = True
        except Exception as Error:
            print(f"[OnePerson] 訓練失敗：{Error}")
            self._InvCov    = np.eye(X.shape[1]) if X.ndim == 2 else None
            self._IsTrained = False

    def predict(self, X: np.ndarray) -> tuple:
        """
        以馬氏距離判斷各樣本，回傳 (人名列表, 信心度陣列)。
        """
        if not self._IsTrained or self._Mean is None:
            return ["Unknown"] * len(X), np.zeros(len(X))

        Names = []
        Confs = []
        for x in X:
            try:
                Diff      = x - self._Mean
                MahalSq   = float(Diff @ self._InvCov @ Diff)
                MahalDist = float(np.sqrt(max(0.0, MahalSq)))
                # 距離轉信心度：距離越小 → 信心度越高（最大 1.0）
                Conf = float(np.exp(-0.5 * MahalDist))
                if MahalDist > self._UnknownThreshold:
                    Names.append("Unknown")
                else:
                    Names.append(self._PersonName)
                Confs.append(Conf)
            except Exception as Error:
                print(f"[OnePerson] 單筆預測失敗：{Error}")
                Names.append("Unknown")
                Confs.append(0.0)
        return Names, np.array(Confs, dtype=float)

    @property
    def IsTrained(self) -> bool:
        return self._IsTrained
