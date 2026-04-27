"""
svm_classifier_np.py  (p12 OneClassSVM 版)

人臉分類器：每人獨立一個 sklearn OneClassSVM。
  - 訓練：Z-Score 標準化 + L2 正規化 → 各人獨立 OneClassSVM
  - 推論：各人 SVM.decision_function() → 取最高分 → 四層 Unknown 偵測
  - Unknown 偵測（四層）：
    第一層：top-1 score < SVM_CONF_THRESH → Unknown
    第二層：score[top-1] − score[top-2] < MARGIN_THRESH → Unknown（正臉時啟用）
    第三層：餘弦驗證（COSINE_VERIFY_THRESH，預設 -1.0 關閉）
    第四層：KNN 驗證（KNN_VERIFY_ENABLED，預設關閉）

【OneClassSVM decision_function 語意】
  > 0：樣本落在 inlier 超球面內部（認識）
  < 0：樣本落在 outlier 超球面外部（不認識）
"""

import numpy as np
from sklearn.svm import OneClassSVM

# 信心度閾值預設值（OneClassSVM raw decision score，低於此值 → Unknown）
# slider 範圍：-1.0 ~ 1.0
SVM_CONF_THRESH = 0.0

# 分差閾值預設值（top-1 與 top-2 分差低於此值 → Unknown）
# slider 範圍：0.0 ~ 3.0
SVM_MARGIN_THRESH = 0.3

# 餘弦驗證閾值預設值（預設關閉，-1.0 表示永不觸發）
# slider 範圍：-1.0 ~ 0.8
COSINE_VERIFY_THRESH = -1.0

# KNN 驗證參數
KNN_K              = 5     # 比對最近 K 個訓練樣本
KNN_PERCENTILE     = 80    # 閾值 = 訓練樣本 KNN 距離的第 P 百分位數
KNN_VERIFY_ENABLED = False # True = 開啟 KNN 驗證；False = 關閉

# OneClassSVM 超參數
SVM_NU     = 0.2       # 訓練集中異常點比例上限（0 < nu ≤ 1）
SVM_KERNEL = 'rbf'     # RBF kernel
SVM_GAMMA  = 'scale'   # gamma = 1 / (n_features * X.var())


class SvmClassifier:
    """
    OneClassSVM 人臉分類器。
    每人獨立一個 OneClassSVM，推論時取 decision score 最高者為候選，
    再依四層 Unknown 偵測機制決定最終結果。
    """

    def __init__(self, Threshold: float = SVM_CONF_THRESH,
                 Label: str = "",
                 MarginThresh: float = SVM_MARGIN_THRESH,
                 CosineVerifyThresh: float = COSINE_VERIFY_THRESH):
        self._Threshold          = Threshold
        self._Label              = Label
        self._MarginThresh       = MarginThresh
        self._CosineVerifyThresh = CosineVerifyThresh

        self._GlobalMean     = None   # shape (D,)，Z-Score 均值
        self._GlobalStd      = None   # shape (D,)，Z-Score 標準差
        self._Svms           = {}     # {人名: OneClassSVM}
        self._ClassNames     = []     # 人名列表
        self._ClassMeans     = {}     # {人名: 正規化平均向量 shape (D,)}
        self._ClassVecs      = {}     # {人名: 正規化訓練向量矩陣 shape (N, D)}
        self._ClassKnnThresh = {}     # {人名: KNN 距離閾值}
        self._IsTrained      = False

    # ──────────────────────────────────────────────────────────────────────────
    # 公開方法
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, Samples: dict) -> None:
        """
        從各人樣本字典建立各人獨立的 OneClassSVM。

        Parameters
        ----------
        Samples : dict  {人名: [特徵向量 (np.ndarray shape (325,)), ...]}
        """
        try:
            # 收集所有人的向量，用於計算全局 Z-Score 統計量
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

            # ── 全局 Z-Score 統計量 ───────────────────────────────────────────
            self._GlobalMean = np.mean(RawMatrix, axis=0)
            self._GlobalStd  = np.std(RawMatrix,  axis=0)
            self._GlobalStd[self._GlobalStd < 1e-10] = 1.0

            # ── Z-Score 標準化 + L2 正規化 ────────────────────────────────────
            Zmat  = (RawMatrix - self._GlobalMean) / self._GlobalStd
            Norms = np.linalg.norm(Zmat, axis=1, keepdims=True)
            Norms[Norms < 1e-10] = 1.0
            Xnorm = Zmat / Norms    # shape (N, D)

            # ── 各人獨立訓練 OneClassSVM ──────────────────────────────────────
            self._ClassNames     = list(dict.fromkeys(AllLabels))
            self._Svms           = {}
            self._ClassMeans     = {}
            self._ClassVecs      = {}
            self._ClassKnnThresh = {}

            AllLabelsArr = np.array(AllLabels)
            for Name in self._ClassNames:
                Mask      = AllLabelsArr == Name
                ClassVecs = Xnorm[Mask]   # shape (Count, D)
                Count     = int(Mask.sum())

                # 計算平均向量（供餘弦驗證）
                Mean     = np.mean(ClassVecs, axis=0)
                MeanNorm = np.linalg.norm(Mean)
                self._ClassMeans[Name] = Mean / MeanNorm if MeanNorm > 1e-8 else Mean
                self._ClassVecs[Name]  = ClassVecs

                # KNN 距離閾值（供 KNN 驗證）
                KnnDists = self._computeWithinClassKnnDists(ClassVecs)
                Thresh   = float(np.percentile(KnnDists, KNN_PERCENTILE))
                self._ClassKnnThresh[Name] = Thresh

                # 訓練各人 OneClassSVM
                Svm = OneClassSVM(kernel=SVM_KERNEL, nu=SVM_NU, gamma=SVM_GAMMA)
                Svm.fit(ClassVecs)
                self._Svms[Name] = Svm

                print(f"  OneClassSVM[{Name}]: {Count} 筆樣本"
                      f"  KNN閾值={Thresh:.3f}")

            self._IsTrained = True
            print(f"[SvmClassifier] OneClassSVM 訓練完成："
                  f"{len(self._ClassNames)} 人，{len(Xnorm)} 筆樣本，"
                  f"{Xnorm.shape[1]} 維特徵")

        except Exception as Error:
            print(f"[SvmClassifier] fit 失敗：{Error}")
            self._IsTrained = False

    def predict(self, X: np.ndarray,
                Thresholds: np.ndarray = None,
                MarginThresholds: np.ndarray = None) -> tuple:
        """
        預測人名與信心度（raw decision score）。

        Parameters
        ----------
        X               : shape (n_samples, n_features)
        Thresholds      : shape (n_samples,)，可選（None 則使用 self._Threshold）
        MarginThresholds: shape (n_samples,)，可選（None 則使用 self._MarginThresh）

        Returns
        -------
        (Names: list[str], Confs: np.ndarray)
          Confs 為 top-1 的 raw decision score（可能為負值）
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
                Confs.append(-999.0)
                continue
            xn = xz / Norm   # shape (D,)

            # ── 各人 OneClassSVM decision_function ───────────────────────────
            # decision_function > 0：inlier；< 0：outlier
            Scores = {
                Name: float(Svm.decision_function(xn.reshape(1, -1))[0])
                for Name, Svm in self._Svms.items()
            }

            # 依分數由高到低排序
            SortedNames = sorted(Scores.keys(), key=lambda n: Scores[n], reverse=True)
            TopName  = SortedNames[0]
            TopScore = Scores[TopName]

            Reject = ""

            # ── 第一層：信心度閾值 ────────────────────────────────────────────
            if TopScore < Thresh:
                Name   = "Unknown"
                Reject = f"低信心({TopScore:.3f}<{Thresh:.2f})"
            else:
                # ── 第二層：分差閾值（只有多人且 MarginThresh > 0 時啟用）───────
                if len(SortedNames) >= 2 and MarginThresh > 0.0:
                    SecondScore = Scores[SortedNames[1]]
                    Margin      = TopScore - SecondScore
                    if Margin < MarginThresh:
                        Name   = "Unknown"
                        Reject = f"分差小({Margin:.2f}<{MarginThresh:.2f})"
                    else:
                        Name = TopName
                else:
                    Name = TopName

            # ── 調試輸出 ──────────────────────────────────────────────────────
            ScoreStr  = "  ".join(f"{n}:{Scores[n]:.3f}" for n in SortedNames)
            Tag       = f"[OCSVM/{self._Label}]" if self._Label else "[OCSVM]"
            RejectStr = f"  ✗{Reject}" if Reject else ""
            print(f"{Tag} Scores=[{ScoreStr}]"
                  f"  thresh={Thresh:.2f}"
                  f"  → {Name}{RejectStr}", end="")

            # ── 第三層：餘弦驗證（預設關閉）──────────────────────────────────
            if Name != "Unknown" and Name in self._ClassMeans:
                VerifyCos = float(np.dot(xn, self._ClassMeans[Name]))
                if VerifyCos < self._CosineVerifyThresh:
                    print(f"  ✗cos({VerifyCos:.3f}) → Unknown", end="")
                    Name = "Unknown"
                else:
                    print(f"  cos={VerifyCos:.3f}", end="")

            # ── 第四層：KNN 驗證（預設關閉）──────────────────────────────────
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
            Confs.append(TopScore)

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
