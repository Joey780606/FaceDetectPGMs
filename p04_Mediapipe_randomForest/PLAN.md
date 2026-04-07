# PLAN.md — p04 MediaPipe + RandomForest

## 與 p03 的核心差異

| 項目 | p03 (face_recognition) | p04 (MediaPipe) |
|------|------------------------|-----------------|
| 偵測函式 | `face_recognition.face_locations()` | `mediapipe.solutions.face_mesh` |
| Landmark 數 | dlib 68 個點（dict 格式） | MediaPipe 468 個 3D 點 → 映射為 68 點 dict |
| 授權 | face_recognition(MIT)、dlib(Boost) | MediaPipe (Apache 2.0) |
| 安裝難度 | dlib 需要 cmake 編譯，Windows 困難 | pip install mediapipe 即可 |
| 新增模組 | — | `mp_face_detector.py`（偵測+映射） |
| 複用模組 | — | `face_feature.py`、`random_forest_np.py` 幾乎不變 |

---

## 專案結構

```
p04_Mediapipe_randomForest/
├── main.py                 # 主程式（CustomTkinter UI + Webcam）
├── mp_face_detector.py     # MediaPipe 偵測器：468 點 → 68 點 dict（核心新增）
├── face_feature.py         # 68 landmark dict → 23 維特徵向量（複用 p03）
├── face_recognizer.py      # FaceRecognizer 類別（改用 mp_face_detector）
├── random_forest_np.py     # 純 NumPy 實作（直接複用 p03，不修改）
├── face_model.npz          # 訓練資料儲存（執行後自動產生）
├── PLAN.md                 # 本文件
├── CLAUDE.md               # Claude Code 專案指引
└── requirements.txt        # 套件依賴
```

---

## 辨識流程（Predict）

```
Webcam frame (BGR ndarray 640×480)
    │
    ▼
mp_face_detector.detect(Frame)
    │  MediaPipe FaceMesh 取得每張臉的 468 個 3D 歸一化座標
    │  landmarkToPixel() 轉為像素座標
    │  selectDlib68() 從 468 點中選出 68 個等效點
    │  buildLandmarkDict() 轉成與 p03 相同的 dict 格式
    │  回傳 [(BoundingBox, LandmarkDict), ...]
    ▼
face_feature.extractFeatures(LandmarkDict)
    │  同 p03：23 維特徵向量
    │  IOD < 30px（側臉）→ 回傳 None，跳過此幀
    ▼
分類器（同 p03）
    ├─ 1 人 → OnePerson（馬氏距離 ≤ 4.0）
    └─ 2+ 人 → RandomForest + 馬氏距離二次驗證
    ▼
回傳 [(Top, Right, Bottom, Left, Name, Confidence), ...]
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
    ├─ mp_face_detector.detect() → BoundingBox + LandmarkDict
    ├─ extractFeatures() → 特徵向量
    ├─ 加入 _Samples[name]
    └─ 自動重訓分類器（_trainClassifier）
    │
    ▼
30 frames 完成 → SaveModel() → 儲存 face_model.npz
```

---

## 核心新模組：`mp_face_detector.py`

### 設計目標
讓 `face_feature.py` 完全不需要修改，只需讓輸出格式與 p03 的
`face_recognition.face_landmarks()` 回傳的 dict 相同。

### MediaPipe 468 → dlib 68 映射表

MediaPipe FaceMesh 的 468 個 3D 歸一化座標，
依照下列索引對應 dlib 的 68 個 landmark 位置：

```python
# 每個 key 對應 dlib 的部位，value 是 MediaPipe 468 點的索引列表
DLIB68_MEDIAPIPE_MAP = {
    'chin': [
        234, 93, 132, 58, 172, 136, 150, 149,
        176, 148, 152, 377, 400, 378, 379, 365, 454
    ],                                              # dlib 0–16（下顎線 17 點）
    'left_eyebrow': [70, 63, 105, 66, 107],         # dlib 17–21
    'right_eyebrow': [336, 296, 334, 293, 300],     # dlib 22–26
    'nose_bridge': [168, 6, 197, 195],              # dlib 27–30
    'nose_tip': [209, 198, 1, 422, 429],            # dlib 31–35
    'left_eye': [33, 160, 158, 133, 153, 144],      # dlib 36–41
    'right_eye': [362, 385, 387, 263, 373, 380],    # dlib 42–47
    'top_lip': [61, 40, 37, 0, 267, 270, 291,
                321, 405, 17, 181, 78],             # dlib 48–59（外唇 12 點）
    'bottom_lip': [78, 81, 13, 311, 308,
                   402, 14, 178],                   # dlib 60–67（內唇 8 點）
}
```

> **注意：** top_lip / bottom_lip 的部位名稱與 p03 face_recognition 保持一致，
> 但點的含義略有不同（外唇上方 / 口腔內圈）。
> face_feature.py 使用 `top_lip` 抓嘴角，`bottom_lip` 計算嘴高，
> 只要左右端點位置正確，辨識效果即可維持。

### 公開 API

```python
class MpFaceDetector:
    def __init__(self, MaxFaces: int = 5, MinDetectConf: float = 0.5):
        ...

    def detect(self, Frame: np.ndarray) -> list[tuple[tuple, dict]]:
        """
        輸入：BGR ndarray
        輸出：[(BoundingBox, LandmarkDict), ...]
            BoundingBox = (Top, Right, Bottom, Left)  # 像素座標，同 face_recognition
            LandmarkDict = {
                'chin': [(x,y), ...],          # 17 點
                'left_eyebrow': [...],          #  5 點
                'right_eyebrow': [...],         #  5 點
                'nose_bridge': [...],           #  4 點
                'nose_tip': [...],              #  5 點
                'left_eye': [...],              #  6 點
                'right_eye': [...],             #  6 點
                'top_lip': [...],               # 12 點
                'bottom_lip': [...],            #  8 點
            }
        """
```

### 內部實作重點

```python
def _landmarkToPixel(self, Landmark, W: int, H: int) -> tuple[int, int]:
    """MediaPipe 歸一化座標（0~1）→ 像素座標"""
    return (int(Landmark.x * W), int(Landmark.y * H))

def _buildBoundingBox(self, Landmarks, W: int, H: int) -> tuple:
    """從所有 landmark 的 min/max 計算 BoundingBox (Top, Right, Bottom, Left)"""

def _selectDlib68(self, Landmarks, W: int, H: int) -> dict:
    """依 DLIB68_MEDIAPIPE_MAP 選出 468 點中的 68 個，轉為像素 (x,y) tuple"""
```

---

## `face_recognizer.py` 修改摘要

只需替換 face_recognition 呼叫部分，其餘邏輯不變：

| p03 | p04 |
|-----|-----|
| `import face_recognition` | `from mp_face_detector import MpFaceDetector` |
| `face_recognition.face_locations(RgbFrame)` | `self._Detector.detect(Frame)` → 直接回傳 BoundingBox |
| `face_recognition.face_landmarks(RgbFrame, Locations)` | 同上（detect 一次回傳 Box + Dict） |
| BGR → RGB 轉換 | 不需要（MediaPipe 在 detect 內處理） |

---

## `face_feature.py` 修改摘要

**幾乎不需要修改**，因為 `MpFaceDetector.detect()` 的輸出 dict 格式
與 `face_recognition.face_landmarks()` 相同。

唯一可能的差異：p03 的 `top_lip` 含上唇 + 下唇外圍 12 點，
若點的語義有位移，需微調 F07（嘴角寬）、F17（嘴角夾角）的取點索引。

---

## `main.py` 修改摘要

與 p03 的 UI 佈局完全相同（4 列版面）：

| Row | 元件 | 功能 |
|-----|------|------|
| 0 | Detect 按鈕、辨識結果標籤 | 啟動/停止持續辨識 |
| 1 | 姓名輸入、Learning / Remove 按鈕、進度條 | 學習與管理 |
| 2 | Webcam 畫面（640×480） | 即時影像 + 人臉框覆蓋 |
| 3 | Log 文字框 | 已登錄人物與樣本數統計 |

**偵測機制：** 滑動窗口多數決（5 frames），同 p03。

---

## 資料儲存格式（face_model.npz）

與 p03 完全相同，無需變更：

| Key | 型別 | 說明 |
|-----|------|------|
| `X` | `float64 (N, 23)` | 所有人的特徵向量 |
| `Y` | `int (N,)` | 對應的人名索引 |
| `persons` | `object array` | 人名字串陣列 |

---

## requirements.txt

```
mediapipe>=0.10
opencv-python>=4.8
numpy>=1.26
customtkinter>=5.2
Pillow>=10.0
```

> **說明：** 移除 `face_recognition`、`dlib`，改用 `mediapipe`。
> dlib 在 Windows Python 3.13 安裝困難（需要 cmake + Visual Studio），
> MediaPipe 直接 `pip install mediapipe` 即可。

---

## 授權（商用安全）

| 套件 | 授權 | 商用 |
|------|------|------|
| MediaPipe | Apache 2.0 | ✅ 免費商用 |
| OpenCV | Apache 2.0 | ✅ 免費商用 |
| NumPy | BSD | ✅ 免費商用 |
| customtkinter | MIT | ✅ 免費商用 |
| Pillow | HPND (MIT-like) | ✅ 免費商用 |

> **注意：** MediaPipe 使用的模型（BlazeFace、FaceMesh）均以 Apache 2.0
> 授權發布，包含模型權重，商用使用合法。
> 但臉部辨識屬生物特徵資料，商用前需確認符合當地隱私法規
>（台灣個資法第 6 條、GDPR 等）。建議在使用前明確告知使用者並取得同意。

---

## 實作順序

| 步驟 | 檔案 | 工作 |
|------|------|------|
| 1 | `requirements.txt` | 確認 mediapipe 版本，移除 face_recognition/dlib |
| 2 | `mp_face_detector.py` | 實作 MpFaceDetector，驗證 468→68 映射 |
| 3 | `random_forest_np.py` | 直接複製 p03（不修改） |
| 4 | `face_feature.py` | 複製 p03，視需要微調 top_lip/bottom_lip 取點 |
| 5 | `face_recognizer.py` | 複製 p03，替換 face_recognition → MpFaceDetector |
| 6 | `main.py` | 複製 p03，視需要調整 import |

---

## 驗證步驟

1. `pip install -r requirements.txt` 無錯誤
2. `python main.py` 啟動，Webcam 畫面正常顯示，可看到人臉框
3. 輸入姓名 → 點 "Learning" → 進度條更新，收集 30 frames 後自動結束
4. Row 3 Log 顯示樣本數
5. 點 "Detect" → 已訓練人臉出現綠框 + 姓名
6. 未訓練的人臉出現紅框 + "您好,我不認識你,我可以認識你嗎?"
7. 訓練第二個人 → 兩人均能正確辨識
8. "Remove" 後該人消失，`GetSampleCounts` 不再包含此人
