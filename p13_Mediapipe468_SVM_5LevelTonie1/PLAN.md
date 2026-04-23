# PLAN: p13_Mediapipe468_SVM_5LevelTonie1

## 專案概述

MediaPipe FaceLandmarker + OneClassSVM 人臉辨識系統。
依臉部角度分五象限訓練，每人每象限各一個 OneClassSVM。

---

## 檔案結構

```
p13_Mediapipe468_SVM_5LevelTonie1/
├── main.py               # UI（CustomTkinter，以 Refmain.py 為藍圖）
├── face_recognizer.py    # 後端全部邏輯
├── face_landmarker.task  # MediaPipe 模型檔（手動放入）
├── face_model.npz        # 訓練後自動產生的模型檔
├── requirements.txt
├── CLAUDE.md
└── PLAN.md
```

---

## main.py（UI）

**4-Row 佈局（與 Refmain.py 相同）：**
- Row 0：Detect 按鈕 + 辨識結果 Label
- Row 1：姓名輸入 + Learning 按鈕 + Remove 按鈕 + 五象限進度區（隱藏/顯示）
- Row 2：Webcam 畫面（垂直延伸）
- Row 3：Log textbox

**學習流程：**
- 按 Learning → 開始收集，按鈕 disable
- 每 500ms 抓一幀 → 判斷象限 → 若該象限未滿 20 張則存入
- 五象限各達 20 張（共 100 張）→ 自動呼叫 FinishLearning() + SaveModel() + 顯示完成 dialog
- 進度區顯示：`正臉: x/20  左上: x/20  右上: x/20  左下: x/20  右下: x/20`

**Detect 流程：**
- 按 Detect → 開始推論，按鈕切換為 Stop
- 每 300ms 推論一幀，累積 5 次多數決後顯示結果

**主要類別與方法：**

| 方法 | 說明 |
|------|------|
| `WebcamManager` | daemon 執行緒持續讀取 webcam frame |
| `MainApp._BuildUI()` | 4 Row 佈局 |
| `MainApp._UpdateWebcamView()` | 每 30ms 更新畫面（含人臉框、學習關鍵點疊加）|
| `MainApp._OnBtnDetectNone()` | Detect/Stop 切換 |
| `MainApp._DetectNoneTick()` | 推論 tick（背景執行緒） |
| `MainApp._OnBtnLearn()` | 學習啟動，已全滿時提示 |
| `MainApp._LearningTick()` | 學習 tick（背景執行緒） |
| `MainApp._StopLearning()` | 學習結束，訓練 + 儲存 |
| `MainApp._OnBtnRemove()` | 移除指定人物資料 |

---

## face_recognizer.py（後端）

### 資料結構

```python
_TrainData = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}}
# {pose_idx: {name: [feature_vector, ...]}}

_SVMs = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}}
# {pose_idx: {name: OneClassSVM}}
```

### 公開方法

| 方法 | 回傳 | 說明 |
|------|------|------|
| `LoadModel(Path)` | — | 初始化 MediaPipe + 載入模型 |
| `AddSample(Frame, Name, Retrain)` | `(Added, KeyPoints)` | 抽特徵→判象限→存入 |
| `GetLearnPoseCounts(Name)` | `{pose_idx: count}` | 各象限已收集幀數 |
| `FinishLearning()` | — | 訓練各象限 OneClassSVM |
| `Predict(Frame)` | `[(top,right,bottom,left,name,conf)]` | 推論 |
| `SaveModel(Path)` | `bool` | 儲存模型 |
| `GetSampleCounts()` | `{name: total}` | 各人總幀數 |
| `GetKnownPersons()` | `list` | 已訓練人名 |
| `GetAccumulatedPersons()` | `list` | 已收集人名（含未訓練） |
| `CanDetect()` | `bool` | 是否可推論 |
| `RemovePerson(Name)` | `bool` | 移除該人全部資料 |
| `Close()` | — | 釋放 MediaPipe 資源 |

### 特徵提取（IOD 歸一化）

```
使用 Landmark 索引：
  左眼中心 = avg(lm[33], lm[133])
  右眼中心 = avg(lm[362], lm[263])
  IOD      = 2D distance(左眼中心, 右眼中心)
  臉部中心 = avg(左眼中心, 右眼中心)

特徵向量（1404-dim）：
  feature[i] = (lm[i].xyz - 臉部中心) / IOD
  flatten → np.float32 array
```

### 臉部角度分類（五象限）

```
# Yaw（左右）：IOD 歸一化，正臉時 ≈ 0
Yaw = (lm[1].x - eye_mid_x) / IOD

# Pitch（上下）：臉高歸一化，正臉時 ≈ 0
#   lm[10]=額頭基準點, lm[152]=下巴基準點
FaceHeight = lm[152].y - lm[10].y
NoseFrac   = (lm[1].y - lm[10].y) / FaceHeight   # 正臉時 ≈ 0.50
Pitch      = NoseFrac - 0.50                       # 正臉時 ≈ 0

# 注意：Pitch 使用臉高歸一化（非 IOD），因為鼻尖天生低於眼睛
# IOD 歸一化會讓 Pitch 永遠為正值，導致所有幀落入「下」象限

YAW_THRESHOLD   = 0.05
PITCH_THRESHOLD = 0.06

象限分類：
  |Yaw| < 0.05 且 |Pitch| < 0.06 → 0 正臉
  Yaw ≤ 0 且 Pitch ≤ 0           → 1 左上
  Yaw > 0 且 Pitch ≤ 0           → 2 右上
  Yaw ≤ 0 且 Pitch > 0           → 3 左下
  Yaw > 0 且 Pitch > 0           → 4 右下
```

### Unknown 偵測

```python
# 每人各象限各一個 OneClassSVM
OneClassSVM(kernel='rbf', nu=0.1).fit(features)

# 推論：取所有已訓練人的 decision_function 最高分
BestScore > 0.0 → 該人（已知）
BestScore ≤ 0.0 → Unknown
```

### Debug 輸出（Predict 時）

```
[Debug] 象限=正臉 | Joey:+1.234  Ben:-0.456 | → Joey
[Debug] 象限=左上(fallback→正臉) | Joey:-0.210 | → Unknown
```

---

## 常數摘要

| 常數 | 值 | 說明 |
|------|----|------|
| `POSE_TARGET` | 20 | 每象限每人目標幀數 |
| `MIN_TRAIN_SAMPLES` | 5 | 訓練 OneClassSVM 最低樣本數 |
| `UNKNOWN_THRESHOLD` | 0.0 | decision_function 閾值 |
| `YAW_THRESHOLD` | 0.05 | Yaw 分類閾值（IOD 單位） |
| `PITCH_THRESHOLD` | 0.06 | Pitch 分類閾值（臉高比例單位） |
| `NEUTRAL_NOSE_FRAC` | 0.50 | 正臉鼻尖在臉高的預估位置 |

---

## 驗證方式

1. `pip install -r requirements.txt`
2. 確認 `face_landmarker.task` 已放在目錄下
3. `python main.py`
4. 輸入姓名 → Learning → 對鏡頭慢慢做各角度 → 五象限各滿 20 張自動完成
5. 再訓練第二人
6. Detect → 確認正確辨識
7. 陌生臉測試 → 應顯示 Unknown
8. console 觀察 `[Debug]` 輸出確認分數與象限

---

## 已知調整點

- `NEUTRAL_NOSE_FRAC`：若正臉仍偏入上/下象限，微調此值（±0.02）
- `PITCH_THRESHOLD`：若正臉容易跑進其他象限，可調大（0.08）；若角度偵測不靈敏，調小（0.04）
- `UNKNOWN_THRESHOLD`：若誤認率高，可調高（0.3）；若 Unknown 太多，調低（-0.2）
