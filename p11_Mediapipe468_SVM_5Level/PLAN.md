# p11 五類姿態 SVM 人臉辨識 — 設計規劃

## 一、與 p10 的核心差異

| 項目 | p10 | p11 |
|------|-----|-----|
| SVM 分類器數量 | 1 個（所有姿態混合） | 5 個（每種姿態各一） |
| 側臉辨識 | 動態閾值補償（治標） | 姿態對應 SVM（治本） |
| 訓練樣本儲存 | `{人名: [特徵向量]}` | `{人名: {姿態類別: [特徵向量]}}` |
| NPZ 格式 | X, Y, persons | X, Y, P, persons（新增 P=姿態標籤） |
| 動態閾值補償 | 有 | 不需要 |

---

## 二、五類姿態定義（從觀察者視角）

| 代號 | 名稱 | 條件（signed 值） |
|------|------|-------------------|
| 0 | 正臉 | \|Yaw\| < 0.15 且 \|Pitch\| < 0.10 |
| 1 | 臉朝左上方 | Yaw ≤ −0.15 且 Pitch ≤ −0.10 |
| 2 | 臉朝右上方 | Yaw ≥ +0.15 且 Pitch ≤ −0.10 |
| 3 | 臉朝左下方 | Yaw ≤ −0.15 且 Pitch ≥ +0.10 |
| 4 | 臉朝右下方 | Yaw ≥ +0.15 且 Pitch ≥ +0.10 |

> 不在正臉範圍的幀，依 Yaw/Pitch 的符號落入最近象限（以符號決定，非閾值限制）

**Signed Yaw（水平）**：
```
SignedYaw = (DistA − DistB) / FaceWidth
  DistA = NoseTipX − min(LeftCheekX, RightCheekX)
  DistB = max(LeftCheekX, RightCheekX) − NoseTipX
  負 = 鼻子偏左（臉朝左），正 = 鼻子偏右（臉朝右）
```

**Signed Pitch（垂直）**：
```
EyeRatio = (EyeY − ForeheadY) / (ChinY − ForeheadY)
SignedPitch = EyeRatio − REF_EYE_RATIO（≈ 0.38）
  正 = 眼睛相對偏低（臉朝下），負 = 眼睛相對偏高（臉朝上）
```

---

## 三、整體流程

```
Webcam 影像
    │
    ▼
MediaPipe FaceLandmarker（mp_face_landmarker.py）
    │  468 個 3D 歸一化座標
    ├──► face_pose_classifier.py → 判斷姿態類別（0~4）
    └──► face_feature_3d.py    → 351 維臉部比例特徵向量
    
【訓練】
    每幀 → (特徵向量, 姿態類別) → _Samples[人名][姿態類別].append(Vec)
    FinishLearning → 5 個 SvmClassifier 各自訓練（各取對應姿態的樣本）

【推論】
    當前幀 → 計算姿態類別 → 選對應 SvmClassifier → predict → 人名 + 信心度
    （若該姿態無訓練資料 → 備援使用正臉 SVM）
```

---

## 四、特徵萃取改版（主管三點建議）

| 主管建議 | 實作方式 |
|---------|---------|
| 重心向量化（鼻尖為原點） | `v_i = P_i − P_鼻尖`（同 p10，鼻尖是「重心參考點」） |
| 單位化（除以臉部寬度） | `‖v_i‖ / 臉部寬度`（臉部寬度 = 左右顴骨間距，p10 用 IOD） |
| 點對點向量夾角 | `θ_ij = arccos(unit_i · unit_j)`，幾何不變量 |

**新特徵向量（325 維）**：
- Part A：C(25,2) = 300 個向量夾角（0~π radians）← 幾何不變量
- Part B：25 個正規化距離 ‖v_i‖/臉部寬度 ← 臉型比例
- 鼻尖本身只當原點，不加入特徵（p10 是 26 點含鼻尖，p11 是 25 點不含鼻尖）

---

## 五、檔案清單

| 檔案 | 來源 | 說明 |
|------|------|------|
| `mp_face_landmarker.py` | 複製 p10 | MediaPipe 封裝，不修改 |
| `face_feature_3d.py` | **改寫** | 向量夾角特徵，325 維 |
| `svm_classifier_np.py` | 複製 p10 | 純 NumPy SVM，不修改 |
| `face_pose_classifier.py` | **新增** | 5 類姿態分類（Signed Yaw/Pitch） |
| `face_recognizer.py` | **改寫** | 姿態感知訓練 + 5 分類器推論 |
| `main.py` | 依 Refmain.py | UI 主程式，訓練+偵測均顯示即時姿態 |
| `requirements.txt` | 複製 p10 | 依賴不變 |

---

## 五、訓練資料結構

```python
# face_recognizer.py 內部
_Samples = {
    'Alice': {
        0: [vec, vec, ...],  # 正臉
        1: [vec, ...],       # 左上
        2: [vec, ...],       # 右上
        3: [vec, ...],       # 左下
        4: [vec, ...],       # 右下
    },
    'Bob': { ... }
}

# face_model.npz 格式
persons: object array    # ['Alice', 'Bob']
X:       float (N, 351)  # 特徵矩陣
Y:       int   (N,)      # 人名索引
P:       int   (N,)      # 姿態類別 0~4  ← 新增
```

---

## 六、SVM 訓練策略（5 個分類器）

```python
for PoseCat in range(5):
    PoseSamples = {}
    for Name, PoseDict in _Samples.items():
        Vecs = PoseDict.get(PoseCat, [])
        if Vecs:
            PoseSamples[Name] = Vecs
    if PoseSamples:
        _Classifiers[PoseCat] = SvmClassifier().fit(PoseSamples)
    else:
        _Classifiers[PoseCat] = None  # 該姿態尚無訓練資料
```

---

## 七、推論策略

```
當前幀 → classifyPose(Landmarks3D) → PoseCat
若 _Classifiers[PoseCat] 有效 → 使用該分類器
否則 → 備援使用 _Classifiers[POSE_FRONTAL]
```

---

## 八、UI 主要調整

1. **學習進度**：新增「各姿態收集數量」顯示，例：
   `正臉:12  左上:5  右上:3  左下:7  右下:3`
2. **推論結果**：在姓名旁顯示偵測到的姿態，例：
   `Hi, Alice 您好 (右上)`
3. **學習 Frame 數**：從 30 增加到 **100**，Timeout 從 60s 增加到 **120s**
   → 確保每個姿態類別至少能收集 15~20 幀

---

## 九、商用授權

所有元件同 p10，均為 Apache 2.0 / BSD / MIT / CC0 寬鬆授權，可安全商用。
