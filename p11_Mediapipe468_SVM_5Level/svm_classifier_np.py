"""
svm_classifier_np.py

人臉分類器：以 sklearn LinearSVC（liblinear 座標下降，保證收斂至全域最優）
取代原手寫 SGD SVM。Z-Score + L2 正規化管線維持不變。

【多人模式（≥ 2 人）：LinearSVC OvR】
  sklearn LinearSVC 以 liblinear 精確求解，每次結果相同，邊距最大化有保證。
  推論時取 decision_function 最高分類別；sigmoid 轉換後與閾值比較。

【單人模式（1 人）：最大 Cosine 相似度】
  只有一人時無法提供負類樣本，LinearSVC 無法訓練，改用最大 Cosine 相似度比對。

【Z-Score + L2 正規化】
  臉部比例特徵對所有人類原始值都落在相近範圍，
  Z-Score 轉換為「與訓練樣本平均的偏差方向」，使不同人的特徵向量真正可分。
"""

import numpy as np
import warnings
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning

# 信心度閾值預設值（sigmoid 值，低於此值 → Unknown）
SVM_UNKNOWN_THRESH = 0.20

# 分差閾值預設值（多人模式：top-1 與 top-2 原始分差低於此值 → Unknown）
SVM_MARGIN_THRESH = 0.50

# 餘弦驗證閾值預設值（預設關閉，-1.0 表示永不觸發）
COSINE_VERIFY_THRESH = -1.0

# KNN 驗證參數
KNN_K              = 5     # 比對最近 K 個訓練樣本
KNN_PERCENTILE     = 80    # 閾值 = 訓練樣本 KNN 距離的第 P 百分位數
KNN_VERIFY_ENABLED = False # True = 開啟 KNN 驗證；False = 關閉

# LinearSVC 超參數
SVM_C_PARAM  = 500   # 正則化強度（C = 1/(2λ)，λ=0.001 → C=500，大邊距）
SVM_MAX_ITER = 2000  # liblinear 最大迭代次數


class SvmClassifier:
    """
    線性 SVM 人臉分類器。
    多人模式：sklearn LinearSVC OvR（liblinear，精確解）。
    單人模式：最大 Cosine 相似度比對。
    兩種模式均在 predict 後可選餘弦驗證與 KNN 驗證。
    """

    def __init__(self, Threshold: float = SVM_UNKNOWN_THRESH,
                 Label: str = "",
                 MarginThresh: float = SVM_MARGIN_THRESH,
                 CosineVerifyThresh: float = COSINE_VERIFY_THRESH,
                 C: float = SVM_C_PARAM,
                 MaxIter: int = SVM_MAX_ITER):
        self._Threshold          = Threshold
        self._Label              = Label
        self._MarginThresh       = MarginThresh
        self._CosineVerifyThresh = CosineVerifyThresh
        self._C                  = C
        self._MaxIter            = MaxIter

        self._GlobalMean     = None   # shape (D,)，Z-Score 均值
        self._GlobalStd      = None   # shape (D,)，Z-Score 標準差
        self._Clf            = None   # LinearSVC 物件（多人模式）
        self._SingleVecs     = None   # shape (N, D)，單人模式訓練向量
        self._ClassNames     = []     # 人名列表，索引對應整數標籤
        self._ClassMeans     = {}     # {人名: 正規化平均向量 shape (D,)}
        self._ClassVecs      = {}     # {人名: 正規化訓練向量矩陣 shape (N, D)}
        self._ClassKnnThresh = {}     # {人名: KNN 距離閾值}
        self._SinglePersonMode = False
        self._IsTrained        = False

    # ──────────────────────────────────────────────────────────────────────────
    # 公開方法
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, Samples: dict) -> None:
        """
        從各人的樣本字典建立分類器。

        Parameters
        ----------
        Samples : dict  {人名: [特徵向量 (np.ndarray), ...]}
        """
        try:
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

            # ── Z-Score 統計量 ────────────────────────────────────────────────
            self._GlobalMean = np.mean(RawMatrix, axis=0)
            self._GlobalStd  = np.std(RawMatrix,  axis=0)
            self._GlobalStd[self._GlobalStd < 1e-10] = 1.0

            # ── Z-Score 標準化 + L2 正規化 ────────────────────────────────────
            Zmat  = (RawMatrix - self._GlobalMean) / self._GlobalStd
            Norms = np.linalg.norm(Zmat, axis=1, keepdims=True)
            Norms[Norms < 1e-10] = 1.0
            Xnorm = Zmat / Norms    # shape (N, D)

            # ── 計算各人正規化平均向量與 KNN 閾值 ─────────────────────────────
            self._ClassNames     = list(dict.fromkeys(AllLabels))
            self._ClassMeans     = {}
            self._ClassVecs      = {}
            self._ClassKnnThresh = {}
            AllLabelsArr = np.array(AllLabels)
            for Name in self._ClassNames:
                Mask      = AllLabelsArr == Name
                ClassVecs = Xnorm[Mask]
                Mean      = np.mean(ClassVecs, axis=0)
                MeanNorm  = np.linalg.norm(Mean)
                self._ClassMeans[Name] = Mean / MeanNorm if MeanNorm > 1e-8 else Mean
                self._ClassVecs[Name]  = ClassVecs
                KnnDists = self._computeWithinClassKnnDists(ClassVecs)
                Thresh   = float(np.percentile(KnnDists, KNN_PERCENTILE))
                self._ClassKnnThresh[Name] = Thresh
                print(f"  KNN閾值[{Name}]: {Thresh:.3f}"
                      f"  (p{KNN_PERCENTILE} of {len(KnnDists)} 筆距離)")

            NClasses = len(self._ClassNames)
            D        = Xnorm.shape[1]

            # ── 單人模式：儲存訓練向量，推論時用最大 Cosine 相似度 ──────────────
            if NClasses == 1:
                self._SingleVecs       = Xnorm.copy()
                self._SinglePersonMode = True
                self._IsTrained        = True
                print(f"[SvmClassifier] 單人模式完成："
                      f"{len(Xnorm)} 筆訓練向量，{D} 維特徵")
                return

            # ── 多人模式：sklearn LinearSVC ───────────────────────────────────
            self._SinglePersonMode = False
            LabelMap = {Name: Idx for Idx, Name in enumerate(self._ClassNames)}
            Yint     = np.array([LabelMap[n] for n in AllLabels], dtype=int)

            Clf = LinearSVC(C=self._C, max_iter=self._MaxIter,
                            multi_class='ovr', dual='auto',
                            class_weight='balanced')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                Clf.fit(Xnorm, Yint)

            self._Clf       = Clf
            self._IsTrained = True
            print(f"[SvmClassifier] LinearSVC 多人模式完成："
                  f"{NClasses} 類別，{len(Xnorm)} 筆樣本，{D} 維特徵")

        except Exception as Error:
            print(f"[SvmClassifier] fit 失敗：{Error}")
            self._IsTrained = False

    def predict(self, X: np.ndarray,
                Thresholds: np.ndarray = None,
                MarginThresholds: np.ndarray = None) -> tuple:
        """
        預測人名與信心度。

        Parameters
        ----------
        X               : shape (n_samples, n_features)
        Thresholds      : shape (n_samples,)，可選
        MarginThresholds: shape (n_samples,)，可選

        Returns
        -------
        (Names: list[str], Confs: np.ndarray)
        """
        Names = []
        Confs = []

        for i, x in enumerate(X):
            Thresh       = float(Thresholds[i])       if Thresholds       is not None else self._Threshold
            MarginThresh = float(MarginThresholds[i]) if MarginThresholds is not None else self._MarginThresh

            # ── 前處理（與訓練相同）──────────────────────────────────────────
            xz   = (x - self._GlobalMean) / self._GlobalStd
            Norm = np.linalg.norm(xz)
            if Norm < 1e-10:
                Names.append("Unknown")
                Confs.append(0.0)
                continue
            xn = xz / Norm

            if self._SinglePersonMode:
                # ── 單人模式：最大 Cosine 相似度 ─────────────────────────────
                Sims   = self._SingleVecs @ xn
                MaxSim = float(np.max(Sims))
                Conf   = float(1.0 / (1.0 + np.exp(-MaxSim)))
                Name   = self._ClassNames[0] if Conf >= Thresh else "Unknown"

                Tag = f"[SVM-1P/{self._Label}]" if self._Label else "[SVM-1P]"
                print(f"{Tag} 最大cosine={MaxSim:.3f}"
                      f"  sigmoid={Conf:.3f}(閾{Thresh:.2f})"
                      f"  → {Name}", end="")
            else:
                # ── 多人模式：LinearSVC decision_function ─────────────────────
                RawOut = self._Clf.decision_function(xn.reshape(1, -1))
                if RawOut.ndim == 1:
                    # 2 類別時 sklearn 回傳 shape (1,)，轉為 [−score, +score]
                    Scores = np.array([-RawOut[0], RawOut[0]])
                else:
                    Scores = RawOut[0]

                BestIdx = int(np.argmax(Scores))
                BestRaw = float(Scores[BestIdx])
                Conf    = float(1.0 / (1.0 + np.exp(-BestRaw)))

                if len(Scores) >= 2:
                    SecondRaw = float(np.sort(Scores)[-2])
                    Margin    = BestRaw - SecondRaw
                else:
                    Margin = BestRaw

                if Conf < Thresh:
                    Name   = "Unknown"
                    Reject = f"低信心({Conf:.3f}<{Thresh:.2f})"
                elif Margin < MarginThresh:
                    Name   = "Unknown"
                    Reject = f"分差小({Margin:.2f}<{MarginThresh:.2f})"
                else:
                    Name   = self._ClassNames[BestIdx]
                    Reject = ""

                ScoreStr  = "  ".join(
                    f"{n}:{s:.2f}" for n, s in zip(self._ClassNames, Scores)
                )
                Tag       = f"[SVM/{self._Label}]" if self._Label else "[SVM]"
                RejectStr = f"  ✗{Reject}" if Reject else ""
                print(f"{Tag} Scores=[{ScoreStr}]"
                      f"  margin={Margin:.2f}(閾{MarginThresh:.2f})"
                      f"  sigmoid={Conf:.3f}(閾{Thresh:.2f})"
                      f"  → {Name}{RejectStr}", end="")

            # ── 餘弦驗證（預設關閉）──────────────────────────────────────────
            if Name != "Unknown" and Name in self._ClassMeans:
                VerifyCos = float(np.dot(xn, self._ClassMeans[Name]))
                if VerifyCos < self._CosineVerifyThresh:
                    print(f"  ✗cos({VerifyCos:.3f}) → Unknown", end="")
                    Name = "Unknown"
                else:
                    print(f"  cos={VerifyCos:.3f}", end="")

            # ── KNN 驗證（預設關閉）──────────────────────────────────────────
            if KNN_VERIFY_ENABLED and Name != "Unknown" and Name in self._ClassVecs:
                KnnDist   = self._knnVerify(xn, Name)
                KnnThresh = self._ClassKnnThresh.get(Name, float('inf'))
                if KnnDist > KnnThresh:
                    print(f"  ✗knn({KnnDist:.3f}>{KnnThresh:.3f}) → Unknown")
                    Name = "Unknown"
                else:
                    print(f"  ✓knn({KnnDist:.3f}<{KnnThresh:.3f})")
            else:
                print()

            Names.append(Name)
            Confs.append(max(0.0, Conf))

        return Names, np.array(Confs, dtype=float)

    @property
    def IsTrained(self) -> bool:
        return self._IsTrained

    # ──────────────────────────────────────────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────────────────────────────────────────

    def _computeWithinClassKnnDists(self, Vecs: np.ndarray) -> np.ndarray:
        """計算每個樣本到其 K 個最近類別內鄰居的平均 L2 距離（排除自身）。"""
        N = len(Vecs)
        K = min(KNN_K, N - 1)
        if K <= 0:
            return np.full(N, 0.1)
        Dists = []
        for i in range(N):
            Diff  = Vecs - Vecs[i]
            L2    = np.sqrt(np.sum(Diff ** 2, axis=1))
            L2[i] = np.inf
            Dists.append(float(np.mean(np.sort(L2)[:K])))
        return np.array(Dists)

    def _knnVerify(self, xn: np.ndarray, Name: str) -> float:
        """計算 query 到指定類別 K 個最近訓練樣本的平均 L2 距離。"""
        Vecs = self._ClassVecs[Name]
        K    = min(KNN_K, len(Vecs))
        Diff = Vecs - xn
        L2   = np.sqrt(np.sum(Diff ** 2, axis=1))
        return float(np.mean(np.sort(L2)[:K]))
