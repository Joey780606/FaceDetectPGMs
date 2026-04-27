# p11 姿態正規化 SVM 人臉辨識 — 設計規劃

## 一、與 p10 的核心差異

| 項目 | p10 | p11 |
|------|-----|-----|
| 特徵維度 | 351 維 | 325 維（300 夾角 + 25 距離） |
| 姿態處理 | 動態閾值補償（治標） | 旋轉矩陣正規化至正臉座標系（治本） |
| SVM 分類器 | 1 個（OvR SGD，手寫） | 1 個（sklearn LinearSVC，liblinear 精確解） |
| 訓練樣本 | 各姿態混合 | 僅收正臉（FrontalOnly=True） |
| Unknown 類別 | 無 | 支援自訓練 "Unknown." 負樣本類別 |
| 時序穩定追蹤 | 無 | StealEatStep（IoU bounding box 追蹤） |

---

## 二、整體流程

```
Webcam 影像
    │
    ▼
MediaPipe FaceLandmarker（mp_face_landmarker.py）
    │  468 個 3D 歸一化座標 (468, 3)
    ├──► face_pose_classifier.py → Yaw / Pitch → 姿態類別（0~4，供 UI 顯示）
    └──► face_feature_3d.py
              │ Step 0：_buildFaceRotationMatrix → R.T 旋轉至正臉座標系
              │ Step 1：取 25 個關鍵點，以鼻尖為原點
              │ Step 2：單位化（消除縮放）
              │ Step 3：C(25,2)=300 對夾角 + 25 正規化距離
              └──► 325 維特徵向量

【訓練】（FrontalOnly=True，只收正臉）
    每幀 → extractFeatures3D → Vec → _Samples[人名][PoseCat].append(Vec)
    FinishLearning → _trainMatcher → SvmClassifier.fit（LinearSVC）

【推論】
    當前幀 → extractFeatures3D → Vec
    → SvmClassifier.predict（Threshold / MarginThreshold）
    → sigmoid 判斷 → margin 判斷（正臉）→ KNN 判斷（可選）
    → 人名 + 信心度 + 姿態類別
    → StealEatStep：非正臉時以 IoU 確認是否沿用上次正臉結果
```

---

## 三、特徵萃取（face_feature_3d.py）

**姿態正規化：**
```
X 軸：左顴骨(234) → 右顴骨(454)
Y 軸：下巴(152) → 額頭(10)，Gram-Schmidt 對 X 正交化
Z 軸：X × Y（臉部法向量）
R.T @ (Landmark - NoseTip) → 正臉座標系
```

**特徵向量（325 維）：**
- Part A：C(25,2) = 300 個向量夾角（弧度 0~π）← 姿態不變量
- Part B：25 個正規化距離 ‖v_i‖ / 臉部寬度 ← 臉型比例

**選取的 25 個關鍵點：**
額頭、眼睛四角×2、眼瞼×2、眉毛×2（外/頂/內）、鼻翼×2、鼻基底、嘴角×2、上下唇、下巴、左右顴骨

---

## 四、SVM 分類器（svm_classifier_np.py）

| 項目 | 說明 |
|------|------|
| 多人模式 | sklearn LinearSVC，multi_class='ovr'，C=500，max_iter=2000，class_weight='balanced' |
| 單人模式 | 最大 Cosine 相似度（LinearSVC 需至少 2 類） |
| 前處理 | Z-Score 標準化 + L2 正規化（與訓練相同） |
| Unknown 判斷 | sigmoid < Threshold → Unknown |
| Margin 判斷 | top-1 - top-2 < MarginThresh → Unknown（正臉限定） |
| 餘弦驗證 | COSINE_VERIFY_THRESH=-1.0（預設關閉）；query 與該人平均向量 cosine < 閾值 → Unknown |
| KNN 驗證 | KNN_VERIFY_ENABLED=False（預設關閉） |

---

## 五、Unknown 偵測機制

```
sigmoid 低於閾值 → "Unknown"（sigmoid 拒絕）
正臉 margin 低於閾值 → "Unknown"（margin 拒絕）
餘弦驗證低於閾值 → "Unknown"（cosine 拒絕，預設關閉 COSINE_VERIFY_THRESH=-1.0）
KNN 距離超過閾值 → "Unknown"（KNN 拒絕，預設關閉）

自訓練負樣本類別：
  UNKNOWN_CLASS = "Unknown."（末尾句點，區別於上述四種拒絕）
  透過 Train Unknown. 按鈕選擇圖片目錄批次訓練
  建議訓練多人種/多年齡/多光線的多元圖片，樣本數盡量接近已知人總樣本數
  OvR 不平衡由 class_weight='balanced' 自動補償
```

---

## 六、時序穩定追蹤（StealEatStep）

```
正臉辨識成功（非 Unknown）→ 儲存 _StableFace（bbox, name, conf）
非正臉 frame：
  IoU(當前bbox, _StableFace.bbox) ≥ 0.35 → 沿用 _StableFace.name
  IoU < 0.35 → 使用 SVM 當次結果
連續 10 tick 無偵測 → 清除 _StableFace
StealEatStep = True 啟用 / False 停用
```

---

## 七、檔案清單

| 檔案 | 說明 |
|------|------|
| `mp_face_landmarker.py` | MediaPipe FaceLandmarker 封裝，回傳 (468, 3) landmarks |
| `face_feature_3d.py` | 姿態正規化 + 325 維特徵萃取 |
| `face_pose_classifier.py` | Signed Yaw / Pitch → 五類姿態分類（0~4） |
| `svm_classifier_np.py` | sklearn LinearSVC + Z-Score/L2 + KNN 驗證 |
| `face_recognizer.py` | 整合以上四個模組，提供訓練/推論/存取模型 API |
| `main.py` | CustomTkinter UI，含 Learning / Detect / Train Unknown. / Remove |
| `requirements.txt` | 依賴套件清單 |

---

## 八、模型儲存格式（face_model.npz）

```python
persons : object array    # 人名列表 ['Alice', 'Bob', 'Unknown.']
X       : float (N, 325)  # 特徵矩陣
Y       : int   (N,)      # 人名索引
P       : int   (N,)      # 姿態類別 0~4
```

SVM 權重不存入 npz，每次程式啟動由 `LoadModel()` 讀取特徵向量後重新訓練。

---

## 九、UI 功能說明

| 元件 | 功能 |
|------|------|
| 姓名欄位 + Learning | Webcam 正臉學習（FrontalOnly，100 幀 / 120 秒） |
| Train Unknown. | 選擇圖片目錄，批次萃取特徵加入 Unknown. 類別 |
| Remove | 移除指定人物所有訓練資料並重訓 |
| Export | 將姓名欄指定人物的資料匯出為獨立 .npz（分散訓練用） |
| Import & Merge | 多選個人 .npz，合併進主模型並重訓儲存 |
| Detect / Stop | 啟動/停止人臉辨識，滑動窗口多數決（10 幀） |
| SVM 信心度閾值 | 0.10~0.99，即時調整 sigmoid 門檻 |
| 分差閾值(Margin) | 0.0~3.0，即時調整正臉 margin 門檻 |
| 餘弦驗證閾值(Cos) | −1.0（關閉）~0.8，即時調整 cosine 驗證門檻 |
| 姿態顯示 | 即時顯示 Yaw/Pitch 值與姿態類別 |

---

## 十、分散訓練工作流程（Export / Import & Merge）

每人可在各自電腦獨立訓練，再由主機合併：

```
同事A電腦：訓練 Joey → Export → face_model_Joey.npz
同事B電腦：訓練 Henry → Export → face_model_Henry.npz
主機：Import & Merge（多選所有 .npz）→ 自動合併重訓 → 儲存 face_model.npz
```

**個人 .npz 格式**（與主模型相同）：
```python
persons : ['Joey']          # 只含一人
X       : float (N, 325)
Y       : int   (N,)        # 全為 0（只有一人）
P       : int   (N,)        # 姿態類別 0~4
```

**注意事項**：
- Import 時若人名已存在，樣本會**追加**（不覆蓋）
- 若要重新匯入同一人，建議先 Remove 再 Import
- 整包備份直接複製 `face_model.npz` 即可（含所有人）

---

## 十一、商用授權

MediaPipe Apache 2.0、OpenCV Apache 2.0、NumPy BSD、
scikit-learn BSD、CustomTkinter CC0、Pillow MIT
均為寬鬆授權，可安全商用。
