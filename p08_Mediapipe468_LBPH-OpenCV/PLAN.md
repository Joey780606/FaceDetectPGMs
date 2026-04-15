# PLAN.md — p08_Mediapipe468_LBPH-OpenCV

## 專案目標

使用 MediaPipe FaceLandmarker 取得 468 個 3D 人臉特徵點，
再以 OpenCV LBPH（Local Binary Patterns Histograms）進行人臉辨識分類。
透過 CustomTkinter 建立 GUI，支援學習（Learning）、偵測（Detect）、移除（Remove）功能。

---

## 模組架構

| 檔案 | 來源 | 職責 |
|------|------|------|
| `mp_face_landmarker.py` | 從 p07 複製 | MediaPipe FaceLandmarker，回傳 468 個 3D landmark + BoundingBox + KeyPoints |
| `face_feature_3d.py` | 從 p07 複製 | IOD 歸一化特徵萃取（供方案 A 或研究用途保留） |
| `lbph_recognizer.py` | 新建 | 封裝 `cv2.face.LBPHFaceRecognizer`，提供 Fit / Update / Predict / Save / Load |
| `face_recognizer.py` | 新建 | 整合 MediaPipe + LBPH，對外 API 與 p07 保持一致 |
| `main.py` | 新建 | 以 Refmain.py 為藍圖的 CustomTkinter 主程式 |
| `requirements.txt` | 新建 | 套件清單 |

---

## 設計決策：LBPH 輸入格式選擇

### 背景

`cv2.face.LBPHFaceRecognizer` 是 OpenCV 的 LBPH 人臉辨識器，
原生接受 **2D 灰階影像（uint8，shape=(H,W)）** 作為輸入。
MediaPipe 回傳的是 **468 個 3D 歸一化座標（float，shape=(468,3)）**，
兩者之間存在格式不相容的問題，需要決定橋接策略。

---

### 方案 A：Landmark 向量 → 2D 偽影像 → LBPH（未採用，供研究參考）

**流程：**
1. 呼叫 `extractFeatures3D(Landmarks3D)` 取得 IOD 歸一化的 1404 維浮點向量
2. 將向量線性映射至 0–255（uint8），例如：
   ```python
   # 假設歸一化後座標大致落在 [-5.0, 5.0]
   CLIP_RANGE = 5.0
   Pseudo = np.clip(Vec, -CLIP_RANGE, CLIP_RANGE)
   Pseudo = ((Pseudo + CLIP_RANGE) / (2 * CLIP_RANGE) * 255).astype(np.uint8)
   ```
3. Reshape 為 2D 陣列，例如 `(36, 39)` = 1404 像素
4. 以此 2D 陣列作為「偽影像」交給 LBPH 訓練與辨識

**優點：**
- 直接使用 MediaPipe 468 點的數值資料
- IOD 歸一化確保縮放不變性（臉距鏡頭遠近不影響結果）
- 不需要 image crop 步驟

**缺點與問題：**
- LBPH 本質上計算「局部二進制模式（LBP）」，是為分析影像**紋理結構**設計，
  並非為數值型特徵向量設計。將 landmark 座標值排列成 2D 陣列後，
  相鄰像素的空間關係與 LBP 的假設不符，理論基礎薄弱。
- 閾值（confidence threshold）難以設定，缺乏參考基準。
- 若更改 landmark 排列順序，模型需完全重新訓練。

**結論：** 不採用。LBPH 套用在非影像數值資料上屬非常規用法，效果難以預期。

---

### 方案 B：MediaPipe 裁切對齊人臉影像 → LBPH（採用）

**流程：**
1. `MpFaceLandmarker.detect(Frame)` 偵測人臉，取得：
   - `BoundingBox`：人臉邊界框（像素座標）
   - `Landmarks3D`：468 個 3D 歸一化座標
   - `KeyPoints`：雙眼、鼻子、嘴巴中心（像素座標）
2. 從 `Landmarks3D` 計算 **5 個像素座標關鍵點**（見下方輔助方法）
3. 以 5 點仿射對齊（Similarity Transform）將人臉 warp 到正臉模板
4. 輸出固定尺寸（100×100）灰階影像
5. 交給 LBPH 訓練（`train`）與辨識（`predict`）

**縮放不變性（Design Decision）：**
- 固定輸出尺寸（100×100）本身即消除因臉部距離鏡頭遠近造成的尺寸差異
- 5 點仿射對齊確保臉部特徵在影像中的絕對位置一致，處理 Roll/Yaw/Pitch 傾斜

---

### 輔助方法：5 點仿射對齊（alignFace）

**採用原因：**
討論時確認希望程式能接受**較大幅度的人臉轉動**。
方案 B 原本只用雙眼做 Roll 校正，改為 5 點對齊可同時改善 Roll、Yaw、Pitch 容忍度。

**5 個關鍵點（像素座標）：**

| 點 | MediaPipe landmark 索引 | 說明 |
|----|------------------------|------|
| 左眼中心 | [33,160,158,133,153,144] 平均 | 左眼 6 點中心 |
| 右眼中心 | [362,385,387,263,373,380] 平均 | 右眼 6 點中心 |
| 鼻尖 | 1 | 鼻尖 |
| 左嘴角 | 61 | 左嘴角 |
| 右嘴角 | 291 | 右嘴角 |

**正臉模板座標（100×100 空間）：**
```python
_CANONICAL_5PT = np.array([
    [35.0, 42.0],   # 左眼中心
    [65.0, 42.0],   # 右眼中心
    [50.0, 60.0],   # 鼻尖
    [35.0, 78.0],   # 左嘴角
    [65.0, 78.0],   # 右嘴角
], dtype=np.float32)
```

**對齊流程：**
```
5 個像素座標關鍵點 (SrcPts)
  → cv2.estimateAffinePartial2D(SrcPts, _CANONICAL_5PT, method=cv2.LMEDS)
  → 取得 2×3 相似度變換矩陣 M（旋轉+縮放+平移，4 DOF）
  → cv2.warpAffine(Frame, M, (100, 100))
  → cv2.cvtColor(GRAY)
  → 100×100 uint8 灰階影像
```

- 使用 **LMEDS（Least Median of Squares）** 方法，對極端角度下部分 landmark 遮擋更穩健
- 眼距（像素）< 20 視為側臉過度，跳過該幀（與 p07 MIN_IOD_NORM 機制相同）
- 學習時鼓勵使用者慢慢左右上下轉頭（±30°），增加訓練多樣性

**LBPH 設定：**
```python
cv2.face.LBPHFaceRecognizer_create(
    radius    = 1,    # LBP 取樣半徑
    neighbors = 8,    # LBP 取樣點數
    grid_x    = 8,    # 橫向分割格數（影響直方圖維度）
    grid_y    = 8,    # 縱向分割格數
    threshold = 80.0  # Unknown 判斷閾值（confidence 大於此值視為陌生人）
)
```
- `predict()` 回傳 `(label_index, confidence)`，confidence 越低表示越相似
- 需維護 `label_index ↔ person_name` 的對應表

**優點：**
- LBPH 最佳使用場景（真實影像紋理），辨識效果可預期且有大量文獻參考
- MediaPipe 468 點在此扮演「精確人臉對齊」的角色，比傳統 Haar cascade 更準確
- Unknown 閾值（confidence threshold）在學術與業界均有參考值
- 模型可直接以 `recognizer.write(path)` 儲存為 OpenCV XML 格式

**缺點：**
- 輸入是裁切影像而非 landmark 座標本身，與 CLAUDE.md 字面敘述有出入
- 對光線變化較 landmark 方法敏感

**結論：採用此方案。**

---

## face_recognizer.py 對外 API（與 p07 保持一致）

| 方法 | 說明 |
|------|------|
| `LoadModel() → bool` | 載入 LBPH 模型（XML）與人名對應表 |
| `SaveModel() → bool` | 儲存 LBPH 模型（XML）與人名對應表 |
| `AddSample(Frame, PersonName, Retrain) → (bool, KeyPoints)` | 收集學習樣本 |
| `FinishLearning() → None` | 批次學習結束後，執行一次完整訓練 |
| `Predict(Frame) → list[(Top,Right,Bottom,Left,Name,Conf)]` | 人臉辨識 |
| `CanDetect() → bool` | 是否已有訓練資料可辨識 |
| `GetKnownPersons() → list` | 回傳已學習的人名清單 |
| `GetSampleCounts() → dict` | 回傳各人名的樣本數量 |
| `RemovePerson(PersonName) → bool` | 移除指定人物並重新訓練 |

---

## 資料儲存

| 檔案 | 格式 | 說明 |
|------|------|------|
| `face_model_lbph.yml` | OpenCV XML/YAML | LBPH 訓練後的模型（由 `recognizer.write()` 產生） |
| `face_model_lbph_meta.npz` | NumPy 壓縮 | label index ↔ person name 對應表 + 各人對齊後臉部影像（供 RemovePerson 重訓） |

**注意：** 對齊後臉部影像（100×100 uint8）儲存在 meta.npz 中，每人 30 張約 300KB，10 人共 3MB，npz 壓縮後更小。

---

## 實作步驟

- [x] 規劃（本文件）
- [x] Step 0：確認 LBPH 輸入格式（方案 A vs B）→ 採用方案 B
- [x] Step 0.5：確認對齊方法（2 點 Roll 校正 vs 5 點 Similarity Transform）→ 採用 5 點對齊
- [x] Step 1：複製 `mp_face_landmarker.py` 與 `face_feature_3d.py` 從 p07
- [x] Step 2：建立 `lbph_recognizer.py`
  - `alignFace()`：5 點仿射對齊 → 100×100 灰階
  - `LbphRecognizer`：fit / predict / write / read
- [x] Step 3：建立 `face_recognizer.py`（整合 MediaPipe + LBPH）
- [x] Step 4：建立 `main.py`（以 Refmain.py 為藍圖）
- [x] Step 5：建立 `requirements.txt`
- [ ] Step 6：整合測試
