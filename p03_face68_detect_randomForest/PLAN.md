# PLAN.md — p03 face68 RandomForest

## 專案結構

```
p03_face68_detect_randomForest/
├── main.py             # 主程式（CustomTkinter UI + Webcam）
├── face_recognizer.py  # FaceRecognizer 類別（辨識核心）
├── face_feature.py     # 68 landmark → 23 維特徵向量
├── random_forest_np.py # 純 NumPy 實作：DecisionTree / RandomForest / OnePerson
├── face_model.npz      # 訓練資料儲存（執行後自動產生）
├── PLAN.md             # 本文件
├── CLAUDE.md           # Claude Code 專案指引
└── requirements.txt    # 套件依賴（若有）
```

---

## 辨識流程（Predict）

```
Webcam frame
    │
    ▼
face_recognition.face_locations()   ← HOG 模型（純 CPU，跨環境）
    │  偵測到人臉位置
    ▼
face_recognition.face_landmarks()   ← 取得 68 個 landmark 座標
    │
    ▼
face_feature.extractFeatures()      ← 轉換為 23 維特徵向量
    │  若 IOD < 30px（側臉）→ 回傳 None，跳過此幀
    ▼
分類器（依訓練人數切換）
    │
    ├─ 1 人 → OnePerson（馬氏距離）
    │         距離 ≤ 4.0 → 顯示人名
    │         距離 > 4.0 → Unknown
    │
    └─ 2+ 人 → 混合方案
               ① RandomForest 初步分類（取機率最高類別）
                   max_prob < 0.45 → Unknown（直接結束）
               ② 對 RF 選出的人，做馬氏距離二次驗證
                   距離 ≤ 4.0 → 確認，顯示人名
                   距離 > 4.0 → 改判 Unknown
    │
    ▼
回傳 [(Top, Right, Bottom, Left, Name, Confidence), ...]
    │
    ▼
main.py 繪製人臉框
    已知人 → 綠框 + 姓名
    Unknown → 紅框 + "您好,我不認識你,我可以認識你嗎?"
```

---

## 學習流程（AddSample）

```
使用者輸入姓名 → 點 "Learning"
    │
    ▼
main.py 收集 30 frames（最多 60 秒，500ms/frame）
    顯示提示：「請慢慢左右、上下轉動頭部，以提升辨識多樣性」
    │
    ▼（每 frame）
face_recognizer.AddSample(frame, name)
    │
    ├─ face_locations + face_landmarks (這二個是 face_recognition library內建的二個函式,不是我們自建的) 
    ├─ extractFeatures → 特徵向量
    ├─ 加入 _Samples[name]
    └─ 自動重訓分類器（_trainClassifier）
    │
    ▼
30 frames 完成 → SaveModel() → 儲存 face_model.npz
```

---

## 模組說明

### `face_feature.py`

**函式：** `extractFeatures(Landmarks: dict) -> np.ndarray | None`

**輸入：** `face_recognition.face_landmarks()` 回傳的單張臉字典

**輸出：** 23 維 float 向量（側臉 / 異常回傳 None）

| 維度 | 類型 | 說明 |
|------|------|------|
| F01–F02 | 距離 | 左/右眼寬度 ÷ IOD |
| F03–F04 | 距離 | 左/右眉寬度 ÷ IOD |
| F05 | 距離 | 鼻梁長度 ÷ IOD |
| F06 | 距離 | 鼻翼寬度 ÷ IOD |
| F07 | 距離 | 嘴角寬度 ÷ IOD |
| F08–F09 | 距離 | 眼中心→眉中心（左/右）÷ IOD |
| F10 | 距離 | 鼻尖→嘴中心 ÷ IOD |
| F11 | 距離 | 下巴寬度 ÷ IOD |
| F12 | 距離 | 臉高（下巴底→鼻梁頂）÷ IOD |
| F13–F14 | 距離 | 眼中心→鼻尖（左/右）÷ IOD |
| F15 | 距離 | 眼寬不對稱率（左-右）÷ IOD |
| F16 | 角度 | 左眼外角—鼻尖—右眼外角 夾角（度） |
| F17 | 角度 | 左嘴角—嘴中心—右嘴角 夾角（度） |
| F18 | 角度 | 兩眉連線仰角（度） |
| F19 | 比例 | 鼻翼寬 / 嘴角寬 |
| F20 | 比例 | 嘴角寬 / IOD |
| F21–F22 | 比例 | 眼高 / 眼寬（左/右） |
| F23 | 比例 | 下巴寬 / 臉高 |

**缺失值：** -1.0（部位偵測失敗時填入）  
**側臉保護：** IOD < 30 像素 → 回傳 None

---

### `random_forest_np.py`

#### `DecisionTree`
- Gini 不純度分裂準則
- 每節點隨機選 `sqrt(n_features)` 個候選特徵
- 最大深度 12，最小分裂樣本 2
- 葉節點儲存完整類別機率分佈（長度固定 = NClasses）

#### `RandomForest`
- 100 棵樹，Bootstrap 抽樣（可重複取樣）
- 預測：各樹機率平均後取 argmax
- Unknown 判定：`max(mean_prob) < 0.45`

#### `OnePerson`（單人 & 多人二次驗證）
- 以樣本計算均值與偽逆共變異數矩陣
- 馬氏距離 ≤ 4.0 → 確認為此人；> 4.0 → Unknown
- 樣本 < 2 筆時，以單位矩陣代替共變異數

---

### `face_recognizer.py`

#### 內部狀態

| 屬性 | 型別 | 說明 |
|------|------|------|
| `_Samples` | `dict[str, list[ndarray]]` | 各人的訓練特徵向量列表 |
| `_Classifier` | `OnePerson \| RandomForest \| None` | 主分類器 |
| `_Validators` | `dict[str, OnePerson]` | 多人模式的馬氏距離驗證器 |
| `_IsTrained` | `bool` | 分類器是否已完成訓練 |

#### 分類器切換邏輯

```
訓練人數 = 1 → _Classifier = OnePerson
                _Validators = {}（空）

訓練人數 ≥ 2 → _Classifier = RandomForest
                _Validators = {人名: OnePerson, ...}（每人一個）
```

#### 公開 API

| 方法 | 說明 |
|------|------|
| `LoadModel()` | 從 face_model.npz 載入並重新訓練 |
| `SaveModel()` | 儲存所有樣本至 face_model.npz |
| `AddSample(frame, name)` | 新增一張臉的訓練樣本，自動重訓 |
| `Predict(frame)` | 辨識影像中所有人臉，回傳結果列表 |
| `CanDetect()` | 是否已有足夠訓練資料可辨識 |
| `GetKnownPersons()` | 已訓練的人名列表 |
| `GetAccumulatedPersons()` | 同上（供 main.py 呼叫） |
| `GetSampleCounts()` | 各人的訓練樣本數 |
| `RemovePerson(name)` | 移除指定人物並重新訓練 |

---

### `main.py`（UI）

4 列版面（CustomTkinter）：

| Row | 元件 | 功能 |
|-----|------|------|
| 0 | Detect 按鈕、辨識結果標籤 | 啟動/停止持續辨識 |
| 1 | 姓名輸入、Learning / Remove 按鈕、進度條 | 學習與管理 |
| 2 | Webcam 畫面（640×480） | 即時影像 + 人臉框覆蓋 |
| 3 | Log 文字框 | 已登錄人物與樣本數統計 |

**偵測機制：** 滑動窗口多數決（5 frames），避免單幀誤判。

---

## 資料儲存格式（face_model.npz）

| Key | 型別 | 說明 |
|-----|------|------|
| `X` | `float64 (N, 23)` | 所有人的特徵向量 |
| `Y` | `int (N,)` | 對應的人名索引 |
| `persons` | `object array` | 人名字串陣列 |

---

## 授權（商用安全）

| 套件 | 授權 |
|------|------|
| face_recognition | MIT |
| dlib | Boost |
| numpy | BSD |
| OpenCV | Apache 2.0 |
| customtkinter | MIT |

> **注意：** 臉部辨識屬於生物特徵資料，商用前須確認符合當地隱私法規（台灣個資法第6條、GDPR 等）。建議在使用前明確告知使用者並取得同意。

---

## 驗證步驟

1. `python main.py` 啟動，Webcam 畫面正常顯示
2. 輸入姓名 → 點 "Learning" → 進度條更新、顯示轉頭提示，收集 30 frames 後自動結束
3. Row 3 Log 顯示樣本數
4. 點 "Detect" → 已訓練人臉出現綠框 + 姓名
5. 未訓練的人臉出現紅框 + "您好,我不認識你,我可以認識你嗎?"
6. 訓練第二個人 → 兩人均能正確辨識，第三人仍顯示 Unknown
7. "Remove" 後該人消失，`GetSampleCounts` 不再包含此人
