# p12_Mediapipe468_OneClassSVM 實作計畫

## Context

p12 是 p11（LinearSVC 多類別分類）的升級版，改用 **OneClassSVM** 架構：
每人獨立一個 SVM，更適合「辨識陌生人」場景。  
目前 p12 目錄只有 Refmain.py（UI 參考）與 CLAUDE.md，所有核心模組均需建立。

---

## 需建立的檔案（6 個）

| 檔案 | 動作 | 說明 |
|------|------|------|
| `mp_face_landmarker.py` | 直接移植 p11 | MediaPipe 468 點偵測，介面不變 |
| `face_feature_3d.py` | 直接移植 p11 | 325 維特徵萃取，演算法不變 |
| `face_pose_classifier.py` | 直接移植 p11 | 5 級姿態分類，邏輯不變 |
| `svm_classifier_np.py` | **完整重寫** | LinearSVC → OneClassSVM（見下方設計） |
| `face_recognizer.py` | 修改 p11 | 移除 Unknown 類別訓練，適配 OneClassSVM API |
| `main.py` | 基於 Refmain.py | 移除 TrainUnknown 按鈕，其餘 UI 保留 |

---

## OneClassSVM 核心設計

### svm_classifier_np.py 重寫方案

**類別：** `SvmClassifier`

**訓練（fit）：**
```
samples = {人名: [325維向量, ...]}
→ 合併所有人的向量做 Z-Score 標準化（StandardScaler）
→ 每筆向量做 L2 正規化
→ 每人獨立訓練一個 OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
→ 記錄每人的平均向量（供 cosine 驗證用）
→ 記錄所有訓練向量（供 KNN 驗證用）
```

**推論（predict）：**
```
x（新樣本）→ 同樣標準化 + L2 正規化
→ 對每人 SVM 呼叫 decision_function(x) → score
→ 取得 {人名: score} 字典
```

### OneClassSVM Unknown 偵測四層（重新設計）✅ 使用者確認

OneClassSVM 的 `decision_function()` 回傳 raw score：
- `> 0` → 判定為 inlier（屬於此人）
- `< 0` → 判定為 outlier（不屬於此人）

**前置判斷：** 若所有人的 SVM 都回傳負值 → 直接 Unknown

**第一層（信心度）：**  
top-1 的 raw decision score < `SVM_CONF_THRESH`（預設 0.0）→ Unknown  
slider 範圍：-1.0 ~ 1.0（使用者確認使用 raw score，不做 sigmoid 轉換）  
側臉時降低閾值至 -0.1（同 p11 邏輯）

**第二層（分差 margin）：**  
score[top-1] - score[top-2] < `MARGIN_THRESH`（預設 0.3）→ Unknown  
slider 範圍：0.0 ~ 3.0  
側臉時強制設為 0.0（停用）

**第三層（餘弦驗證）：**  
query 與該人平均向量 cosine < `COSINE_VERIFY_THRESH`（預設 -1.0 關閉）→ Unknown  
slider 範圍：-1.0 ~ 0.8

**第四層（KNN 驗證）：**  
`KNN_VERIFY_ENABLED` 預設關閉，邏輯與 p11 相同

---

## face_recognizer.py 修改重點

- 移除：`AddSamplesFromFolder`（Unknown class 圖檔批量導入，不再需要）
- 保留：所有公開 API（`AddSample`, `FinishLearning`, `Predict`, `LoadModel`, `SaveModel`, `ExportPerson`, `ImportPersonFiles`, `SetThresholds`, `GetKnownPersons`, `RemovePerson` 等）
- 訓練資料存儲格式不變（.npz 存 X/Y/persons，載入時重訓各人 SVM）

---

## main.py UI 修改重點

基於 Refmain.py，以下差異：
- **移除** TrainUnknown 按鈕（✅ 使用者確認：OneClassSVM 本身能偵測 Unknown，不需要額外匯入陌生人圖檔）
- **保留** Learning / Detect / Remove / Export / Import & Merge 按鈕
- **保留** 三個 slider：信心度閾值（-1.0~1.0）、分差閾值（0.0~3.0）、餘弦驗證閾值（-1.0~0.8）
- Slider 標籤和預設值更新以反映 OneClassSVM 的 score 語意

---

## 常數設定（svm_classifier_np.py）

```python
SVM_CONF_THRESH      = 0.0    # OneClassSVM raw decision score 閾值（slider: -1.0~1.0）
SVM_MARGIN_THRESH    = 0.3    # 分差閾值（slider: 0.0~3.0）
COSINE_VERIFY_THRESH = -1.0   # 餘弦驗證（-1.0 = 關閉；slider: -1.0~0.8）
KNN_VERIFY_ENABLED   = False  # KNN 驗證（預設關閉）
SVM_NU               = 0.1    # OneClassSVM 異常比例上限
SVM_KERNEL           = 'rbf'  # RBF kernel
SVM_GAMMA            = 'scale'
```

---

## 實作順序

1. `mp_face_landmarker.py` — 移植（不改動）
2. `face_feature_3d.py` — 移植（不改動）
3. `face_pose_classifier.py` — 移植（不改動）
4. `svm_classifier_np.py` — 完整重寫（OneClassSVM）
5. `face_recognizer.py` — 修改（移除 Unknown class，適配新 SVM）
6. `main.py` — 基於 Refmain.py 建立（移除 TrainUnknown）

---

## 驗證方式

1. 執行 `main.py`，確認 Webcam 開啟
2. 輸入姓名 → Learning → 收集 100 幀 → 訓練完成
3. 按 Detect → 確認正確辨識
4. 不輸入已知人臉 → 確認顯示 Unknown
5. 側臉測試 → 確認閾值自動調整
6. Export / Import & Merge 功能驗證
