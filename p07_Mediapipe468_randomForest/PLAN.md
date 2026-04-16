# PLAN.md — p07 MediaPipe 468 + Random Forest 人臉辨識系統

## 專案目標

開啟電腦 Webcam，輸入人名後，以 MediaPipe FaceLandmarker 找出 468 個 3D 臉部特徵點（x, y, z 軸），
將特徵向量交給 Random Forest 做分類訓練，再即時偵測並辨識人臉是誰。

參考資料：https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

---

## 架構總覽

```
main.py
  └── WebcamManager          # 背景執行緒持續抓 frame
  └── MainApp (CTk)          # CustomTkinter 主視窗
        └── FaceRecognizer   # 整合所有辨識邏輯
              ├── MpFaceLandmarker    # MediaPipe 468 點偵測
              ├── face_feature_3d    # 1404 維特徵萃取
              └── RandomForest / OnePerson  # 純 NumPy 分類器
```

---

## 模組說明

### mp_face_landmarker.py — MpFaceLandmarker

- 使用 MediaPipe Tasks API（`mediapipe >= 0.10`）的 FaceLandmarker
- 首次執行自動下載 `face_landmarker.task`（約 6 MB）
- 輸入：BGR frame（OpenCV ndarray）
- 輸出：每個人臉的 `(BoundingBox, Landmarks3D, KeyPoints)`
  - `BoundingBox = (Top, Right, Bottom, Left)`（像素座標）
  - `Landmarks3D = np.ndarray shape=(468, 3)`（歸一化 x, y, z）
  - `KeyPoints = {"left_eye", "right_eye", "nose", "mouth"}`（像素中心點）
- 只取前 468 點（後 10 個為虹膜，不使用）

### face_feature_3d.py — extractFeatures3D

特徵設計（1404 維）：
1. 以鼻尖（Index 1）為原點，計算所有 468 點的相對位移 (dx, dy, dz)
2. 計算 3D 瞳距（IOD）：左眼中心與右眼中心的歐氏距離（歸一化座標）
3. 所有相對位移 ÷ IOD → 消除臉離鏡頭遠近造成的縮放干擾
4. 攤平為 468 × 3 = 1404 維向量
5. 若 IOD < MIN_IOD_NORM（1e-5）→ 視為 MediaPipe 偵測退化（兩眼幾乎重疊），回傳 None
   ※ 真實側臉的 3D IOD 遠大於此閾值，不會被過濾；側臉辨識完全由分類器處理

### random_forest_np.py — 純 NumPy 分類器

不依賴 sklearn / scipy（避免 Windows Application Control 封鎖 DLL 問題）。

包含三個類別：

| 類別 | 用途 |
|------|------|
| `DecisionTree` | Gini 分裂準則，每節點隨機選 sqrt(N) 個特徵 |
| `RandomForest` | Bootstrap 抽樣，100 棵樹集成，信心度 < 0.45 → Unknown |
| `OnePerson` | 單人模式，馬氏距離閾值（預設 16.0）判斷 Known / Unknown |

### face_recognizer.py — FaceRecognizer

整合所有模組，提供高階 API：

| 方法 | 說明 |
|------|------|
| `LoadModel()` | 載入 face_model.npz 並重訓分類器 |
| `SaveModel()` | 儲存所有樣本至 face_model.npz |
| `AddSample(Frame, Name, Retrain)` | 收集一幀學習樣本 |
| `FinishLearning()` | 批次學習結束後執行完整重訓 |
| `Predict(Frame)` | 偵測並辨識人臉，回傳 `(Top,Right,Bottom,Left,Name,Confidence)` |
| `RemovePerson(Name)` | 刪除指定人物的所有樣本並重訓 |
| `CanDetect()` | 是否有已訓練好的分類器 |
| `GetSampleCounts()` | 各人名的樣本數量 |

**辨識策略（混合方案）：**
- **1 人已訓練** → OnePerson（馬氏距離閾值）
- **2+ 人已訓練** → RandomForest 初步分類 + 馬氏距離二次驗證
  - RF 認出人名 且 馬氏距離在閾值內 → 確認為此人
  - RF 認出人名 但 馬氏距離過大 → 改判 Unknown
  - RF 直接判為 Unknown → Unknown

### main.py — MainApp（CustomTkinter UI）

UI 佈局（4 Row）：

```
Row 0：Detect 按鈕 + 辨識結果標籤
Row 1：姓名輸入 + Learning 按鈕 + Remove 按鈕
        └── 學習進度區（預設隱藏）
              ├── 剩餘秒數 + 時間進度條
              └── 已學習 Frame 數 + Frame 進度條
Row 2：Webcam 畫面（垂直延伸）
Row 3：學習資料摘要 Log（最新訊息在最上方）
```

**主要常數：**

| 常數 | 值 | 說明 |
|------|----|------|
| `LEARN_TARGET_FRAMES` | 30 | 學習目標 Frame 數 |
| `LEARN_TIMEOUT_SECONDS` | 60 | 學習最長等待秒數 |
| `UI_REFRESH_MS` | 30 | Webcam 畫面更新間隔 |
| `LEARN_TICK_MS` | 500 | 學習抓 Frame 間隔（每秒 2 個樣本） |
| `DETECT_TICK_MS` | 300 | 辨識推論間隔 |
| `DETECT_NONE_DETECT_TARGET` | 5 | 多數決累積 Frame 數 |

**執行緒架構：**
- `WebcamManager` 以 daemon 執行緒持續抓 Frame，存入 `_LatestFrame`（有 Lock 保護）
- 學習 tick、偵測 tick 各自在 daemon 執行緒執行，結果用 `self.after(0, callback)` 回到主執行緒
- `_InferenceActive` 旗標防止推論重入

**Detect 流程（多數決）：**
1. 每 300ms 抓一幀 → 背景執行緒跑 `Predict()`
2. 結果累積入滑動窗口（最多 5 筆）
3. 窗口滿後取多數決人名 → 顯示問候語

**Learning 流程：**
1. 每 500ms 抓一幀 → 背景執行緒跑 `AddSample(Retrain=False)`
2. 累積 30 Frame 或超過 60 秒 → 呼叫 `FinishLearning()` 完整重訓
3. 學習中疊加顯示雙眼、鼻子、嘴巴中心點（藍/綠/紅色圓點）

---

## 資料存儲格式

`face_model.npz`（純 NumPy 壓縮格式）：

| 欄位 | 型別 | 說明 |
|------|------|------|
| `X` | float64 (N, 1404) | 所有樣本特徵向量 |
| `Y` | int (N,) | 對應人名的整數標籤 |
| `persons` | object array | 人名列表，索引與 Y 對應 |

---

## 設計決策

1. **相對座標化**：以鼻尖（Index 1）為原點，計算其他點與鼻尖的相對位移。
2. **IOD 歸一化**：所有位移除以瞳距，消除臉離鏡頭遠近造成的縮放干擾。
3. **純 NumPy 分類器**：不用 sklearn/scipy，避免 Windows 環境 DLL 封鎖問題。
4. **批次學習模式**：`AddSample(Retrain=False)` + `FinishLearning()` 分離，避免每幀都重訓 100 棵樹。
5. **混合辨識策略**：RF 做初步分類，馬氏距離做二次驗證，提升 Unknown 判斷精度。
6. **UI / 推論分離**：所有耗時操作（MediaPipe、RF 推論）在 daemon 執行緒執行，結果透過 `after()` 回到主執行緒更新 UI。

---

## 目前完成狀態

- [x] mp_face_landmarker.py — MediaPipe FaceLandmarker 偵測器
- [x] face_feature_3d.py — 1404 維特徵萃取（IOD 歸一化）
- [x] random_forest_np.py — 純 NumPy 決策樹、隨機森林、OnePerson
- [x] face_recognizer.py — 整合辨識邏輯（混合方案）
- [x] main.py — CustomTkinter UI（學習、偵測、移除、Log）
- [x] requirements.txt

---

## 待辦 / 已知問題

- `OnePerson._predict()` 中的 DEBUG 印出行（馬氏距離）確認閾值後可移除
- 可考慮調整 `MAHAL_UNKNOWN_THRESH`（目前 16.0）與 `UNKNOWN_THRESHOLD`（目前 0.45）以改善辨識準確率
