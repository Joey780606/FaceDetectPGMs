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
  此方式與 p09 k-NN 的陌生人偵測邏輯一致，確保 1 人亦能有效判斷陌生人。

【流程】
  多人訓練：收集所有樣本 → Z-Score + L2 正規化 → OvR SGD hinge loss 訓練
  單人訓練：Z-Score + L2 正規化 → 儲存訓練向量矩陣
  推論：同樣前處理 → 分類器計分 → sigmoid → 閾值判斷
"""

import numpy as np

# 信心度閾值預設值（sigmoid 值，低於此值 → Unknown）
SVM_UNKNOWN_THRESH = 0.60

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

    兩種模式均使用相同的 Z-Score + L2 前處理，介面完全一致。
    """

    def __init__(self, Threshold: float = SVM_UNKNOWN_THRESH,
                 NEpochs: int = SGD_N_EPOCHS,
                 LearningRate: float = SGD_LEARNING_RATE,
                 LrDecay: float = SGD_LR_DECAY,
                 LambdaReg: float = SGD_LAMBDA_REG,
                 BatchSize: int = SGD_BATCH_SIZE):
        self._Threshold         = Threshold
        self._NEpochs           = NEpochs
        self._LearningRate      = LearningRate
        self._LrDecay           = LrDecay
        self._LambdaReg         = LambdaReg
        self._BatchSize         = BatchSize

        self._GlobalMean        = None    # shape (D,)，Z-Score 均值
        self._GlobalStd         = None    # shape (D,)，Z-Score 標準差
        self._W                 = None    # shape (n_classes, D) 或 (N_samples, D)（單人）
        self._b                 = None    # shape (n_classes,) 或 (N_samples,)（單人）
        self._ClassNames        = []      # 人名列表，索引對應整數標籤
        self._SinglePersonMode  = False   # True = 使用最大 Cosine 模式
        self._IsTrained         = False

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

            # ── 建立整數標籤 ──────────────────────────────────────────────────
            self._ClassNames = list(dict.fromkeys(AllLabels))   # 保持插入順序去重
            NClasses = len(self._ClassNames)
            D        = Xnorm.shape[1]

            # ── 單人模式：儲存訓練向量矩陣，推論時用最大 Cosine 相似度 ─────────
            if NClasses == 1:
                self._W                = Xnorm.copy()              # (N_samples, D)
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

    def predict(self, X: np.ndarray) -> tuple:
        """
        預測人名與信心度。

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
            # ── 前處理（與訓練相同）──────────────────────────────────────────
            xz   = (x - self._GlobalMean) / self._GlobalStd
            Norm = np.linalg.norm(xz)
            if Norm < 1e-10:
                Names.append("Unknown")
                Confs.append(0.0)
                continue
            xn = xz / Norm

            if self._SinglePersonMode:
                # ── 單人模式：與所有訓練向量的最大 Cosine 相似度 ─────────────
                Sims    = self._W @ xn      # cosine 相似度，shape (N_samples,)
                MaxSim  = float(np.max(Sims))
                Conf    = float(1.0 / (1.0 + np.exp(-MaxSim)))
                Name    = self._ClassNames[0] if Conf >= self._Threshold else "Unknown"

                print(f"[SVM-1P] 最大cosine={MaxSim:.3f}"
                      f"  sigmoid={Conf:.3f}(閾{self._Threshold:.2f})"
                      f"  → {Name}")
            else:
                # ── 多人模式：OvR 線性分類器 ──────────────────────────────────
                Scores  = self._W @ xn + self._b   # shape (n_classes,)
                BestIdx = int(np.argmax(Scores))
                BestRaw = float(Scores[BestIdx])
                Conf    = float(1.0 / (1.0 + np.exp(-BestRaw)))
                Name    = self._ClassNames[BestIdx] if Conf >= self._Threshold else "Unknown"

                ScoreStr = "  ".join(
                    f"{n}:{s:.2f}" for n, s in zip(self._ClassNames, Scores)
                )
                print(f"[SVM] Scores=[{ScoreStr}]"
                      f"  獲勝={self._ClassNames[BestIdx]} sigmoid={Conf:.3f}(閾{self._Threshold:.2f})"
                      f"  → {Name}")

            Names.append(Name)
            Confs.append(max(0.0, Conf))

        return Names, np.array(Confs, dtype=float)

    @property
    def IsTrained(self) -> bool:
        return self._IsTrained

    # ──────────────────────────────────────────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────────────────────────────────────────

    def _trainOneBinary(self, X: np.ndarray, Y: np.ndarray) -> tuple:
        """
        以 mini-batch SGD 訓練單個二元線性 SVM（hinge loss + L2 正則化）。

        Parameters
        ----------
        X : shape (N, D)，已 Z-Score + L2 正規化的樣本矩陣
        Y : shape (N,)，±1 二元標籤

        Returns
        -------
        (w: ndarray(D,), b: float)
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
                Xb    = X[Batch]    # (B, D)
                Yb    = Y[Batch]    # (B,)

                Scores = Xb @ w + b
                Margin = Yb * Scores
                Mask   = Margin < 1.0   # hinge 激活條件

                if Mask.any():
                    # hinge 梯度：−mean(y_i × x_i) for violated samples
                    dw = -np.mean(Yb[Mask, np.newaxis] * Xb[Mask], axis=0)
                    db = -float(np.mean(Yb[Mask]))
                else:
                    dw = np.zeros(D)
                    db = 0.0

                # L2 正則化梯度（只作用於 w，不正則化 b）
                dw += self._LambdaReg * w

                w -= Lr * dw
                b -= Lr * db

            Lr *= self._LrDecay

        return w, b
