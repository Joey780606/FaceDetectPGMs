"""
cosine_matcher_np.py

純 NumPy 實作的 k-NN 人臉比對器（Z-Score 標準化 + Cosine 相似度）。
不依賴 sklearn / scipy，商用授權完全安全（僅使用 NumPy，BSD License）。

【為何需要 Z-Score 標準化】
  臉部比例特徵（325 個成對距離 + 26 個 z 深度）雖然描述臉型，
  但對任何人類而言這些值都在同一個範圍內（因為人臉拓撲相似），
  導致所有人的 351 維向量幾乎指向同一方向 → Cosine 相似度 ~0.997，無法區分。

  解法：Z-Score 標準化
    每個特徵維度：x' = (x - μ) / σ（μ、σ 來自全部訓練樣本）
    → 向量不再描述「你的臉型值是多少」
    → 而是描述「你的臉型與所有訓練樣本的平均差了多少個標準差」
    → 不同人的偏差方向不同 → Cosine 相似度真正能區分個人

【流程】
  訓練：收集所有樣本 → 計算全局 μ/σ → 標準化 → L2 正規化後存入矩陣
  推論：同樣標準化 + 正規化 → dot product 得 Cosine 相似度 → k-NN 投票
"""

import numpy as np
from collections import Counter

# 相似度閾值預設值（標準化後，低於此值 → Unknown）
COSINE_UNKNOWN_THRESH = 0.75

# k-NN 的 K 值預設
KNN_K = 7


class CosineMatcher:
    """
    Z-Score 標準化 + k-NN Cosine 相似度人臉比對器。

    訓練時：
      1. 收集所有人的所有樣本，計算各維度的全局均值與標準差
      2. 對所有樣本做 Z-Score 標準化後 L2 正規化，存入矩陣

    推論時：
      1. 對 query 向量做相同的 Z-Score 標準化 + L2 正規化
      2. 矩陣乘法一次求得所有樣本的 Cosine 相似度
      3. 取前 K 個最近鄰居多數決，平均相似度超過閾值 → 此人
    """

    def __init__(self, Threshold: float = COSINE_UNKNOWN_THRESH, K: int = KNN_K):
        self._Threshold   = Threshold
        self._K           = K
        self._Vectors     = None    # shape (N, D)，已標準化 + L2 正規化的樣本矩陣
        self._Labels      = None    # shape (N,)，對應人名
        self._GlobalMean  = None    # shape (D,)，各維度均值
        self._GlobalStd   = None    # shape (D,)，各維度標準差（已消除零值）
        self._IsTrained   = False

    def fit(self, Samples: dict) -> None:
        """
        從各人的樣本字典建立標準化向量矩陣。

        Parameters
        ----------
        Samples : dict  {人名: [特徵向量 (np.ndarray), ...]}
        """
        AllVecs   = []
        AllLabels = []

        for Name, Vecs in Samples.items():
            if not Vecs:
                continue
            for Vec in Vecs:
                AllVecs.append(Vec)
                AllLabels.append(Name)

        if not AllVecs:
            self._IsTrained = False
            return

        RawMatrix = np.array(AllVecs, dtype=float)   # shape (N, D)

        # ── 計算全局 Z-Score 統計量 ──────────────────────────────────────────
        self._GlobalMean = np.mean(RawMatrix, axis=0)                 # shape (D,)
        self._GlobalStd  = np.std(RawMatrix,  axis=0)                 # shape (D,)
        # 標準差為 0 的維度（所有人都一樣）沒有辨別力，設為 1 避免除以零
        self._GlobalStd[self._GlobalStd < 1e-10] = 1.0

        # ── Z-Score 標準化 + L2 正規化 ───────────────────────────────────────
        Standardized = (RawMatrix - self._GlobalMean) / self._GlobalStd  # shape (N, D)
        Norms        = np.linalg.norm(Standardized, axis=1, keepdims=True)
        Norms[Norms < 1e-10] = 1.0
        self._Vectors  = Standardized / Norms    # shape (N, D)，已正規化
        self._Labels   = np.array(AllLabels)
        self._IsTrained = True

    def predict(self, X: np.ndarray) -> tuple:
        """
        以 Z-Score 標準化 + k-NN Cosine 相似度比對各樣本。

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        (Names: list[str], Confs: np.ndarray)
        """
        Names = []
        Confs = []

        for x in X:
            # ── 同樣的標準化流程 ─────────────────────────────────────────────
            xStd  = (x - self._GlobalMean) / self._GlobalStd
            Norm  = np.linalg.norm(xStd)
            if Norm < 1e-10:
                Names.append("Unknown")
                Confs.append(0.0)
                continue
            xNorm = xStd / Norm

            # ── 矩陣乘法一次求所有 Cosine 相似度 ────────────────────────────
            Sims = self._Vectors @ xNorm   # shape (N,)

            # ── 取前 K 個最近鄰居 ────────────────────────────────────────────
            K       = min(self._K, len(Sims))
            TopKIdx = np.argpartition(Sims, -K)[-K:]
            TopKIdx = TopKIdx[np.argsort(Sims[TopKIdx])[::-1]]

            TopKSims   = Sims[TopKIdx]
            TopKLabels = self._Labels[TopKIdx]

            # ── 多數決 ───────────────────────────────────────────────────────
            Vote           = Counter(TopKLabels.tolist())
            BestName, _    = Vote.most_common(1)[0]
            WinnerMask     = TopKLabels == BestName
            BestSim        = float(np.mean(TopKSims[WinnerMask]))

            FinalName = BestName if BestSim >= self._Threshold else "Unknown"

            VoteStr = "  ".join(f"{n}×{c}" for n, c in Vote.most_common())
            print(f"[k-NN] Top{K}投票=[{VoteStr}]"
                  f"  獲勝={BestName} 相似度={BestSim:.3f}(閾{self._Threshold:.2f})"
                  f"  → {FinalName}")

            Names.append(FinalName)
            Confs.append(max(0.0, BestSim))

        return Names, np.array(Confs, dtype=float)

    @property
    def IsTrained(self) -> bool:
        return self._IsTrained
