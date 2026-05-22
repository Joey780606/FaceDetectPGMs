# PLAN.md — p15_VideoGetFace

## 目標

輸入 YouTube URL，分析影片中人臉，依三層過濾條件篩選，將符合條件的片段存成 mp4 至 `face_video/` 目錄。

---

## 三層過濾邏輯

| Filter | 條件 | 判斷方式 |
|--------|------|----------|
| F1 | 連續出現 ≥ 10 秒 | `(LastFrame - FirstFrame + 1) / Fps >= 10` |
| F2 | 正臉累計 ≥ 8 秒 | `IsFrontal=True 的幀數 / Fps >= 8` |
| F3 | 出現時畫面只有他自己 | `all(Track.IsAlone.values()) == True` |

---

## 資料結構

```python
class FaceTrack:
    TrackId   : int
    Boxes     : dict[int, tuple]   # FrameIdx → (x, y, w, h)
    IsFrontal : dict[int, bool]    # FrameIdx → bool
    IsAlone   : dict[int, bool]    # FrameIdx → bool
    FirstFrame: int
    LastFrame : int
```

---

## 處理流程

```
YouTube URL
    ↓ yt-dlp
face_video/_tmp_download.mp4
    ↓ cv2.VideoCapture + Mediapipe FaceLandmarker (VIDEO mode)
list[FaceTrack]          ← IoU 追蹤，gap ≤ 0.3s 可延續
    ↓ _filterByDuration  (F1)
    ↓ _filterByFrontal   (F2)
    ↓ _filterByAlone     (F3)
list[FaceTrack]（符合者）
    ↓ cv2.VideoWriter (mp4v / XVID 備援)
face_video/face_001.mp4, face_002.mp4, ...
```

---

## 檔案職責

| 檔案 | 職責 |
|------|------|
| `main.py` | UI（customtkinter）、Submit 觸發、daemon thread、`self.after()` 回呼 |
| `face_processor.py` | `FaceTrack`、`FaceProcessor`（下載、偵測、過濾、存檔） |
| `requirements.txt` | 套件版本清單 |
| `face_landmarker.task` | Mediapipe 478點模型（從 p05 複製） |
| `face_video/` | 輸出目錄（程式自動建立） |

---

## 關鍵參數

```python
MIN_CONTINUOUS_SEC    = 10.0   # Filter 1 門檻（秒）
MIN_FRONTAL_SEC       = 8.0    # Filter 2 門檻（秒）
IOU_THRESHOLD         = 0.3    # IoU 追蹤最低匹配值
MIN_FACE_HEIGHT_RATIO = 0.08   # 忽略太小的臉（影片高度 8%）
GapFrames             = max(5, int(Fps * 0.3))  # track 斷開容忍幀數
```

---

## 正臉判斷算法

Landmark 索引（478 點集）：
- 鼻尖：4
- 左眼中心：(33 + 133) / 2
- 右眼中心：(362 + 263) / 2

```
SymRatio = |NoseTip.x - LeftEye.x| / |RightEye.x - NoseTip.x|
正臉條件：0.6 ≤ SymRatio ≤ 1.67
```

---

## UI 佈局

```
┌──────────────────────────────────────────────────┐
│ [URL Entry  (width=440)          ] [Submit(w=90)]│
│ [CTkProgressBar ─────────────────────────────── ]│
│ [StatusLabel]                                    │
│ ┌────────────────────────────────────────────┐   │
│ │ CTkTextbox  height=300  （最新訊息在上方） │   │
│ └────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
視窗大小：580 × 440，不可調整大小
```

---

## Thread 設計

```
UI Thread
  └─ onSubmit()
       └─ threading.Thread(daemon=True)
            └─ _processingWorker()
                 ├─ FaceProcessor.process()
                 │    ├─ ProgressCallback → self.after(0, lambda P=Pct: ...)
                 │    └─ LogCallback     → self.after(0, lambda M=Msg: ...)
                 ├─ _onComplete(SavedFiles)   # via self.after()
                 └─ _onError(ErrMsg)          # via self.after()
```

---

## 驗證步驟

1. `pip install -r requirements.txt`
2. 確認 `face_landmarker.task` 存在
3. `python main.py`
4. 輸入含人臉的 YouTube 短片（建議 1～3 分鐘）
5. 按 Submit，進度條應持續更新，UI 不卡頓
6. 完成後 `face_video/` 應出現 `face_001.mp4`
7. 用播放器確認片段內容正確

---

## 已知限制

- 僅做位置式 IoU 追蹤，無跨 track 身份識別
- 若同一人離鏡頭再回來超過 GapFrames，會被視為兩個不同人
- Filter 3 要求「每幀」獨自，與他人短暫同框即失格
