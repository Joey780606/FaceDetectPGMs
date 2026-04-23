"""
svm_classifier_np.py

純 NumPy 實作的線性 SVM 人臉分類器。
不依賴 sklearn / scipy，商用授權完全安全（僅使用 NumPy，BSD License）。

【為何仍需 Z-Score 標準化】
  與 cosine_matcher_np.py 同樣原因：臉部比例特徵對所有人類的原始值都落在相近範圍，
  Z-Score 將特徵轉換為「與訓練樣本平均的偏差方向」，使不同人的特徵向量真正可分。

【多人模式（≥ 2 人）：OvR 線性 SVM】
  對每個人訓練一個二元 SVM（此人 vs 其他所有人），以 SGD hinge loss 最佳化。
  推論時取原始分數最高的類別；sigmoid 轉換後與閾值比較決定 Known / Unknown。

【單人模式（僅 1 人）：最大 Cosine 相似度】
  只有一人時，使用者無法提供負類樣本。
  改為儲存所有正規化訓練向量，推論時計算 query 與各訓練樣本的 cosine 相似度，
  取最大值；sigmoid 轉換後與閾值比較。

【餘弦驗證（兩種模式共用）】
  SVM / cosine 得出預測人名後，額外計算 query 與該人正規化平均向量的餘弦相似度。
  若低於 CosineVerifyThresh → 判為 Unknown（防止陌生人被 SVM 誤認）。

【流程】
  多人訓練：收集所有樣本 → Z-Score + L2 正規化 → OvR SGD hinge loss 訓練 + 儲存各人平均向量
  單人訓練：Z-Score + L2 正規化 → 儲存訓練向量矩陣 + 儲存平均向量
  推論：同樣前處理 → 分類器計分 → sigmoid → 閾值判斷 → 餘弦驗證
"""

import numpy as np

# 信心度閾值預設值（sigmoid 值，低於此值 → Unknown）
SVM_UNKNOWN_THRESH = 0.60

# 分差閾值預設值（多人模式：top-1 與 top-2 原始分差低於此值 → Unknown）
# 已知者：SVM 對最佳類別信心高，分差大；陌生人：各類分數相近，分差小
SVM_MARGIN_THRESH = 0.50

# 餘弦驗證閾值預設值（query 與預測人的平均向量餘弦相似度，低於此值 → Unknown）
COSINE_VERIFY_THRESH = -1.0   # 預設關閉（-1.0 表示永不觸發）

# KNN 驗證參數
KNN_K              = 5     # 比對最近 K 個訓練樣本
KNN_PERCENTILE     = 80    # 閾值 = 訓練樣本 KNN 距離的第 P 百分位數（降低 → 更嚴格）
KNN_VERIFY_ENABLED = False # True = 開啟 KNN 驗證；False = 關閉（純 sigmoid 模式）

# SGD 超參數（多人模式）
SGD_N_EPOCHS      = 200
SGD_LEARNING_RATE = 0.01
SGD_LR_DECAY      = 0.99     # 每 epoch 後衰減學習率
SGD_LAMBDA_REG    = 0.001    # L2 正則化係數（C = 1/(2λ) = 500，大邊距）
SGD_BATCH_SIZE    = 16


class SvmClassifier:
    """
    線性 SVM 人臉分類器。
    多人模式：OvR SGD hinge loss。
    單人模式：最大 Cosine 相似度比對。
    兩種模式均在 predict 後加餘弦驗證，降低陌生人誤認率。
    """

    def __init__(self, Threshold: float = SVM_UNKNOWN_THRESH,
                 NEpochs: int = SGD_N_EPOCHS,
                 LearningRate: float = SGD_LEARNING_RATE,
                 LrDecay: float = SGD_LR_DECAY,
                 LambdaReg: float = SGD_LAMBDA_REG,
                 BatchSize: int = SGD_BATCH_SIZE,
                 Label: str = "",
                 MarginThresh: float = SVM_MARGIN_THRESH,
                 CosineVerifyThresh: float = COSINE_VERIFY_THRESH):
        self._Threshold          = Threshold
        self._NEpochs            = NEpochs
        self._LearningRate       = LearningRate
        self._LrDecay            = LrDecay
        self._LambdaReg          = LambdaReg
        self._BatchSize          = BatchSize
        self._Label              = Label
        self._MarginThresh       = MarginThresh
        self._CosineVerifyThresh = CosineVerifyThresh

        self._GlobalMean        = None    # shape (D,)，Z-Score 均值
        self._GlobalStd         = None    # shape (D,)，Z-Score 標準差
        self._W                 = None    # shape (n_classes, D) 或 (N_samples, D)（單人）
        self._b                 = None    # shape (n_classes,) 或 (N_samples,)（單人）
        self._ClassNames        = []      # 人名列表，索引對應整數標籤
        self._ClassMeans        = {}      # {人名: 正規化平均向量 shape (D,)}
        self._ClassVecs         = {}      # {人名: 正規化訓練向量矩陣 shape (N, D)}
        self._ClassKnnThresh    = {}      # {人名: KNN 距離閾值（訓練時自動校準）}
        self._SinglePersonMode  = False   # True = 使用最大 Cosine 模式
        self._IsTrained         = False

    # ──────────────────────────────────────────────────────────────────────────
    # 公開方法
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, Samples: dict) -> None:
        """
        從各人的樣本字典建立分類器並計算各人正規化平均向量。

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

            # ── 計算各人正規化平均向量（供餘弦驗證使用）─────────────────────
            self._ClassNames     = list(dict.fromkeys(AllLabels))
            self._ClassMeans     = {}
            self._ClassVecs      = {}
            self._ClassKnnThresh = {}
            AllLabelsArr = np.array(AllLabels)
            for Name in self._ClassNames:
                Mask      = AllLabelsArr == Name
                ClassVecs = Xnorm[Mask]                   # shape (K, D)
                Mean      = np.mean(ClassVecs, axis=0)    # shape (D,)
                MeanNorm  = np.linalg.norm(Mean)
                self._ClassMeans[Name] = Mean / MeanNorm if MeanNorm > 1e-8 else Mean

                # 儲存訓練向量並自動校準 KNN 閾值
                self._ClassVecs[Name] = ClassVecs
                KnnDists = self._computeWithinClassKnnDists(ClassVecs)
                Thresh   = float(np.percentile(KnnDists, KNN_PERCENTILE))
                self._ClassKnnThresh[Name] = Thresh
                print(f"  KNN閾值[{Name}]: {Thresh:.3f}"
                      f"  (p{KNN_PERCENTILE} of {len(KnnDists)} 筆距離)")

            NClasses = len(self._ClassNames)
            D        = Xnorm.shape[1]

            # ── 單人模式：儲存訓練向量矩陣，推論時用最大 Cosine 相似度 ─────────
            if NClasses == 1:
                self._W                = Xnorm.copy()
                self._b                = np.zeros(len(Xnorm), dtype=float)
                self._SinglePersonMode = True
                self._IsTrained        = True
                print(f"[SvmClassifier] 單人模式完成："
                      f"{len(Xnorm)} 筆訓練向量，{D} 維特徵")
                return

            # ── 多人模式：OvR SGD SVM ─────────────────────────────────────────
            self._SinglePersonMode = False
            LabelMap = {Name: Idx for Idx, Name in enumerate(self._ClassNames)}
            Yint     = np.array([LabelMap[n] for n in AllLabels], dtype=int)

            self._W = np.zeros((NClasses, D), dtype=float)
            self._b = np.zeros(NClasses,      dtype=float)

            for i in range(NClasses):
                Ybinary = np.where(Yint == i, 1.0, -1.0)
                Wi, Bi  = self._trainOneBinary(Xnorm, Ybinary)
                self._W[i] = Wi
                self._b[i] = Bi

            self._IsTrained = True
            print(f"[SvmClassifier] 多人模式完成：{NClasses} 類別，"
                  f"{len(Xnorm)} 筆樣本，{D} 維特徵")

        except Exception as Error:
            print(f"[SvmClassifier] fit 失敗：{Error}")
            self._IsTrained = False

    def predict(self, X: np.ndarray,
                Thresholds: np.ndarray = None,
                MarginThresholds: np.ndarray = None) -> tuple:
        """
        預測人名與信心度。SVM / cosine 預測後加餘弦驗證。

        Parameters
        ----------
        X          : np.ndarray, shape (n_samples, n_features)
        Thresholds : np.ndarray, shape (n_samples,)，可選。

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
                Sims   = self._W @ xn
                MaxSim = float(np.max(Sims))
                Conf   = float(1.0 / (1.0 + np.exp(-MaxSim)))
                Name   = self._ClassNames[0] if Conf >= Thresh else "Unknown"

                Tag = f"[SVM-1P/{self._Label}]" if self._Label else "[SVM-1P]"
                print(f"{Tag} 最大cosine={MaxSim:.3f}"
                      f"  sigmoid={Conf:.3f}(閾{Thresh:.2f})"
                      f"  → {Name}", end="")
            else:
                # ── 多人模式：OvR 線性分類器 ──────────────────────────────────
                Scores  = self._W @ xn + self._b
                BestIdx = int(np.argmax(Scores))
                BestRaw = float(Scores[BestIdx])
                Conf    = float(1.0 / (1.0 + np.exp(-BestRaw)))

                # 分差檢查：top-1 與 top-2 分差小 → 模型搖擺不定 → Unknown
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

                ScoreStr = "  ".join(
                    f"{n}:{s:.2f}" for n, s in zip(self._ClassNames, Scores)
                )
                Tag = f"[SVM/{self._Label}]" if self._Label else "[SVM]"
                RejectStr = f"  ✗{Reject}" if Reject else ""
                print(f"{Tag} Scores=[{ScoreStr}]"
                      f"  margin={Margin:.2f}(閾{MarginThresh:.2f})"
                      f"  sigmoid={Conf:.3f}(閾{Thresh:.2f})"
                      f"  → {Name}{RejectStr}", end="")

            # ── 餘弦驗證（可選，預設關閉）────────────────────────────────────
            if Name != "Unknown" and Name in self._ClassMeans:
                VerifyCos = float(np.dot(xn, self._ClassMeans[Name]))
                if VerifyCos < self._CosineVerifyThresh:
                    print(f"  ✗cos({VerifyCos:.3f}) → Unknown", end="")
                    Name = "Unknown"
                else:
                    print(f"  cos={VerifyCos:.3f}", end="")

            # ── KNN 驗證：確認 query 是否落在該人訓練樣本的鄰近範圍內 ──────────
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
            Diff = Vecs - Vecs[i]
            L2   = np.sqrt(np.sum(Diff ** 2, axis=1))
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

    def _trainOneBinary(self, X: np.ndarray, Y: np.ndarray) -> tuple:
        """
        以 mini-batch SGD 訓練單個二元線性 SVM（hinge loss + L2 正則化）。
        """
        N  = X.shape[0]
        D  = X.shape[1]
        w  = np.zeros(D, dtype=float)
        b  = 0.0
        Lr = self._LearningRate

        for _ in range(self._NEpochs):
            Indices   = np.random.permutation(N)
            BatchSize = min(self._BatchSize, N)

            for Start in range(0, N, BatchSize):
                Batch = Indices[Start: Start + BatchSize]
                Xb    = X[Batch]
                Yb    = Y[Batch]

                Scores = Xb @ w + b
                Margin = Yb * Scores
                Mask   = Margin < 1.0

                if Mask.any():
                    dw = -np.mean(Yb[Mask, np.newaxis] * Xb[Mask], axis=0)
                    db = -float(np.mean(Yb[Mask]))
                else:
                    dw = np.zeros(D)
                    db = 0.0

                dw += self._LambdaReg * w
                w -= Lr * dw
                b -= Lr * db

            Lr *= self._LrDecay

        return w, b
