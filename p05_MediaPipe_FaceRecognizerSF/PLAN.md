# p05_MediaPipe_Deepface 實作計劃

## Context

p05 是一個以商用授權（Apache 2.0）為前提的人臉學習與辨識系統。  
架構確認：**MediaPipe FaceLandmarker（偵測 + 對齊）+ OpenCV FaceRecognizerSF（128D 特徵提取）+ Cosine Similarity（辨識）**。  
UI 藍圖為 `Refmain.py`，程式碼風格與模組分工對齊 p04 的作法。

---

## 授權確認（商用安全）

| 元件 | 模型 | 授權 |
|------|------|------|
| MediaPipe FaceLandmarker | face_landmarker.task | Apache 2.0 ✅ |
| OpenCV FaceRecognizerSF | face_recognition_sface_2021dec.onnx | Apache 2.0 ✅ |
| 訓練資料集（WiderFace, MS1M） | 僅用於預訓練，不需再申請授權 | 商用可用 ✅ |
| 用戶人臉嵌入（face_model.npz） | 自行收集，用戶自有資料 | 無爭議 ✅ |

---

## 頭部旋轉角度限制

- **MediaPipe 偵測上限**：Yaw/Pitch/Roll ≈ ±45°（超過則 landmark 不穩）
- **FaceRecognizerSF 辨識上限**：Yaw ≈ ±30°（訓練資料以正臉為主）
- **學習建議**：使用者學習時應緩慢左右各轉 ±25°、上下 ±20°，學習提示已寫入 `_LblRemain`
- **偵測實用角度**：正臉至側臉 ±30° 內辨識率最佳，±30°~±45° 仍可偵測但置信度下降

---

## 檔案結構

```
p05_MediaPipe_Deepface/
├── main.py                              # 主程式 UI（依 Refmain.py 藍圖）
├── face_recognizer.py                   # FaceRecognizer 類別（協調器）
├── face_aligner.py                      # MediaPipe 偵測 + 5點對齊到 112×112
├── model_downloader.py                  # 自動下載模型檔（首次執行）
├── requirements.txt
├── PLAN.md                              # 本計劃文件
├── face_landmarker.task                 # 執行時自動下載（Apache 2.0）
└── face_recognition_sface_2021dec.onnx  # 執行時自動下載（Apache 2.0）
```

---

## 各模組設計

### 1. `model_downloader.py`

```
函數：ensureModels()
- 確認 face_landmarker.task 存在，否則從 MediaPipe 官方下載
- 確認 face_recognition_sface_2021dec.onnx 存在，否則從 OpenCV Zoo 下載
- 所有下載包含 try-except，失敗時印出錯誤並回傳 False
```

### 2. `face_aligner.py`

```
類別：FaceAligner
- __init__(): 載入 MediaPipe FaceLandmarker（靜態模式）

函數：Detect(Frame: np.ndarray) -> list
  返回：[(AlignedFace_112x112, BoundingBox_TRBL, KeyPoints_dict), ...]

對齊演算法：
  1. MediaPipe 偵測 478 個 landmark
  2. 取 5 個關鍵點：
     - 左眼中心 = mean(landmark[33], landmark[133])
     - 右眼中心 = mean(landmark[362], landmark[263])
     - 鼻尖      = landmark[4]
     - 左嘴角    = landmark[61]
     - 右嘴角    = landmark[291]
  3. ArcFace 標準座標（FaceRecognizerSF 訓練時用的 112×112 對齊目標）：
     LEFT_EYE  = (38.29, 51.70)
     RIGHT_EYE = (73.53, 51.50)
     NOSE      = (56.03, 71.74)
     MOUTH_L   = (41.55, 92.37)
     MOUTH_R   = (70.73, 92.20)
  4. cv2.estimateAffinePartial2D + cv2.warpAffine → 112×112
  5. BoundingBox 由 landmark 最大外框計算（min/max x, y）

KeyPoints_dict 格式（供學習時 UI 疊加顯示）：
  {"left_eye": (cx, cy), "right_eye": (cx, cy),
   "nose": (cx, cy), "mouth": (cx, cy)}
```

### 3. `face_recognizer.py`

```
類別：FaceRecognizer

內部狀態：
  _Aligner    : FaceAligner
  _SfNet      : cv2.FaceRecognizerSF（載入 .onnx）
  _Samples    : dict {PersonName: [128D np.ndarray, ...]}

公開 API（與 Refmain.py/p04 相容）：
  LoadModel()               → bool
  SaveModel()               → bool
  AddSample(Frame, PersonName, Retrain=False) → (bool, list)
  Predict(Frame)            → list of (Top, Right, Bottom, Left, Name, Confidence)
  CanDetect()               → bool（有任何樣本即 True）
  GetSampleCounts()         → {name: int}
  GetKnownPersons()         → list[str]
  GetAccumulatedPersons()   → list[str]
  RemovePerson(PersonName)  → bool
  FinishLearning()          → None（保留 API，FaceRecognizerSF 不需重訓）

辨識策略（Predict）：
  For each detected face:
    1. FaceAligner.Detect() → AlignedFace_112x112
    2. FaceRecognizerSF.feature(AlignedFace) → 128D 向量
    3. For each person in _Samples:
         person_score = mean(cosine_similarity(query, each_embedding))
    4. best_person = argmax(person_score)
    5. if person_score[best_person] >= 0.363 → Name = best_person, Confidence = score
       else → Name = "Unknown", Confidence = score

儲存格式（face_model.npz）：
  - 與 p04 相同格式：persons, X（embeddings）, Y（person index）
  - X shape: (N, 128)，dtype=float32
```

### 4. `main.py`

```
完全依照 Refmain.py 的 UI 結構，替換以下部分：
  - import: from face_recognizer import FaceRecognizer
  - 所有常數保持不變（LEARN_TARGET_FRAMES=30, LEARN_TIMEOUT_SECONDS=60 等）
  - _InitComponents(): 加入 model_downloader.ensureModels() 呼叫
  - _OnClose(): 改呼叫 self._Recognizer._Aligner.close()（MediaPipe 資源）
  - 其餘 UI 邏輯、按鈕事件、執行緒結構 100% 沿用 Refmain.py
```

---

## 學習參數

| 參數 | 值 | 說明 |
|------|----|------|
| 目標 frame 數 | 30 張 | 每 500ms 收集 1 張，約 15 秒最快完成 |
| 超時時間 | 60 秒 | 保底避免卡住 |
| Cosine 閾值 | 0.363 | OpenCV 官方建議值 |
| AlignedFace 大小 | 112×112 | FaceRecognizerSF 標準輸入 |
| 嵌入向量維度 | 128D | FaceRecognizerSF 輸出 |

---

## 驗證方式

1. 執行 `python main.py`
2. 首次啟動：確認模型自動下載（face_landmarker.task、face_recognition_sface_2021dec.onnx）
3. 在姓名欄輸入名字 → 按 Learning → 確認進度條更新、30 張後自動結束
4. 確認 `face_model.npz` 已儲存
5. 按 Detect → 確認已登錄人物出現綠框與姓名，陌生人出現紅框 Unknown
6. 關閉視窗後再次開啟 → 確認模型從 .npz 正常 LoadModel

---

## 關鍵參考檔案

| 用途 | 路徑 |
|------|------|
| UI 藍圖 | `p05_MediaPipe_Deepface/Refmain.py` |
| p04 FaceRecognizer 介面參考 | `p04_Mediapipe_randomForest/face_recognizer.py` |
| p04 MediaPipe 偵測器參考 | `p04_Mediapipe_randomForest/mp_face_detector.py` |
| OpenCV FaceRecognizerSF 文件 | https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html |
