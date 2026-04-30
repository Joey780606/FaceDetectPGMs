# p12_Mediapipe468_OneClassSVM 實作計畫

## 狀態：✅ 核心實作完成（持續改進中）

---

## 已完成的檔案

| 檔案 | 狀態 | 說明 |
|------|------|------|
| `mp_face_landmarker.py` | ✅ 完成 | MediaPipe 468 點偵測，移植自 p11 |
| `face_feature_3d.py` | ✅ 完成 | 325 維特徵萃取，移植自 p11 |
| `face_pose_classifier.py` | ✅ 完成 | 5 級姿態分類 + Roll 偵測（新增） |
| `svm_classifier_np.py` | ✅ 完成 | 完整重寫為 OneClassSVM，含 PoseLabels debug |
| `face_recognizer.py` | ✅ 完成 | 移除 Unknown class，適配 OneClassSVM + Roll |
| `main.py` | ✅ 完成 | 移除 TrainUnknown，加入 StealEatStep + Roll 延伸 |

---

## OneClassSVM 核心設計

### svm_classifier_np.py

**訓練（fit）：**
```
samples = {人名: [325維向量, ...]}
→ 合併所有人的向量做全局 Z-Score 標準化
→ 每筆向量做 L2 正規化
→ 每人獨立訓練一個 OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
→ 記錄每人的平均向量（供 cosine 驗證用）
→ 記錄所有訓練向量（供 KNN 驗證用）
```

**推論（predict）：**
```
x（新樣本）→ 同樣標準化 + L2 正規化
→ 對每人 SVM 呼叫 decision_function(x) → score
→ 四層 Unknown 偵測 → 回傳 (Name, Conf)
```

**debug print 格式：**
```
[OCSVM/全角度][正臉 Y:+0.02] Scores=[Joey:0.821 ...] thresh=0.00 → Joey cos=0.895
[OCSVM/全角度][歪頭 R:+0.21] Scores=[Joey:-0.12 ...] thresh=-0.10 → Unknown ✗低信心
[OCSVM/全角度][左上 Y:-0.38] Scores=[Joey:0.45  ...] thresh=-0.10 → Joey cos=0.82
```

### Unknown 偵測四層

**第一層（信心度）：**
top-1 score < `SVM_CONF_THRESH`（預設 0.0）→ Unknown
側臉或歪頭時降低閾值至 −0.1

**第二層（分差 margin）：**
score[top-1] − score[top-2] < `MARGIN_THRESH`（預設 0.3）→ Unknown
側臉或歪頭時強制設為 0.0（停用）

**第三層（餘弦驗證）：**
query 與該人平均向量 cosine < `COSINE_VERIFY_THRESH`（預設 −1.0 關閉）→ Unknown

**第四層（KNN 驗證）：**
`KNN_VERIFY_ENABLED` 預設關閉

---

## 常數設定

### svm_classifier_np.py
```python
SVM_CONF_THRESH      = 0.0    # raw score 閾值（slider: -1.0~1.0）
SVM_MARGIN_THRESH    = 0.3    # 分差閾值（slider: 0.0~3.0）
COSINE_VERIFY_THRESH = -1.0   # 餘弦驗證（-1.0 = 關閉）
KNN_VERIFY_ENABLED   = False
SVM_NU               = 0.1    # 越低邊界越寬鬆，越高越嚴格
SVM_KERNEL           = 'rbf'
SVM_GAMMA            = 'scale'
```

### face_pose_classifier.py
```python
YAW_THRESH   = 0.30   # 水平轉角閾值
PITCH_THRESH = 0.15   # 垂直傾角閾值
ROLL_THRESH  = 0.15   # 歪頭角度閾值（弧度，≈ 8.6°）
```

### main.py
```python
STABLE_FACE_IOU_THRESH    = 0.35   # 穩定臉 IoU 閾值
STABLE_FACE_CENTER_THRESH = 0.50   # 中心點距離 fallback 閾值（歪頭用）
STABLE_FACE_MAX_MISS      = 10     # 連續 miss 幾 tick 後清快取
STABLE_FACE_CLEAR_THRESH  = -0.30  # 分數低於此值才真正清快取
StealEatStep              = True
```

---

## 後續改進紀錄

### Roll（歪頭）偵測與處理
- `face_pose_classifier.py` 新增 `_computeRoll()` 與 `ROLL_THRESH`
- `classifyPoseWithValues()` 回傳 4-tuple：`(PoseCat, Yaw, Pitch, Roll)`
- `face_recognizer.py`：歪頭時同側臉，觸發 −0.1 閾值調整
- result tuple 擴充為 10 值：`(Top, Right, Bottom, Left, Name, Conf, PoseCat, Yaw, Pitch, Roll)`

### StealEatStep 時序穩定追蹤
- 正臉認識 → 更新快取 `_StableFace`
- 正臉 Unknown 且分數 < `STABLE_FACE_CLEAR_THRESH(−0.30)` → 清快取
- 正臉 Unknown 但分數介於 −0.30~0（邊界模糊）→ 沿用快取
- 側臉 OR 歪頭 → 以 `_isSameFace()` 判斷是否沿用快取

### _isSameFace() IoU + 中心點距離
- IoU ≥ 0.35 → 同一張臉（原判斷）
- IoU 不足時：中心距離 / 臉寬 < 0.50 → 同一張臉（歪頭 fallback）
- 歪頭時 bounding box 形狀改變，IoU 易低估，中心點距離更穩定

### PoseLabels debug 輸出
- `svm_classifier_np.py predict()` 新增 `PoseLabels` 參數
- `face_recognizer.py` 組合姿態字串傳入，輸出含姿態標籤的 debug log

---

## 已知限制與調優建議

- `SVM_NU=0.1` 邊界較緊，score 易在 0 附近跳動；可調低至 0.05 讓邊界更寬鬆
- `STABLE_FACE_CLEAR_THRESH=-0.30` 可視場景調整（−0.20 更保守；−0.40 更寬鬆）
- 訓練樣本數量不均衡（如 A:1000、B:100）會讓 GlobalMean 偏向 A，影響 B/C/D 的歸一化；訓練時盡量各人樣本數接近
- Roll 的 R^T 補正在 z 軸深度估計誤差較大時仍有殘差，訓練時可適度加入歪頭樣本
