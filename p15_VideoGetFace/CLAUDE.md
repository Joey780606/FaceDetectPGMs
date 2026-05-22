# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

給予一個Youtube的影片連結,按下Submit按鍵後,把影片中有臉部人物,且連續超過10秒的部分,另存成影片檔,供其他專案測試使用

## Code Specification
1. 開發語言：Python 3.13
2. UI library：customtkinter
3. 影像擷取：OpenCV (cv2)，使用 cv2.CAP_DSHOW 後端
4. 非阻塞串流：threading（WebCam 迴圈在 daemon 執行緒中執行）
5. 註解請用中文
6. Variable Naming：CamelCase 一致使用
7. Error Handling：所有 API 呼叫必須包含 try-except
8. Function 名稱使用英文，不要用中文
9. 臉部偵測可用Mediapipe來處理

## Design Decisions
1. 舉例,如果一個影片發現有5個人出現(每次畫面都只有一個人,臉不要太小),若有4人連續出現10秒,該4人有3個人都曾以正臉出現8秒以上,而該3人有二個人,在有他的畫面時,只有他自已,沒有別人的臉,就可將這二個人的影片擷取下來,另存到目前目錄裡下的face_video目錄.檔名不要重覆即可 

## UI
1. 要有一個影片網址輸入區,Submit按鍵,以及處理結果(有沒有找到相符資料?有幾人?存的檔名是叫什麼?),資料顯示在一個可放置多行的顯示區裡.

## File Structure
```
p15_VideoGetFace/
├── main.py              # UI + threading 調度
├── face_processor.py    # FaceTrack / FaceProcessor / 三層過濾 / 存檔
├── requirements.txt     # 相依套件
├── face_landmarker.task # Mediapipe 模型檔（從 p05 複製）
└── face_video/          # 輸出目錄
```

## Key Implementation Notes

### 三層過濾條件（face_processor.py）
- Filter 1：`MIN_CONTINUOUS_SEC = 10.0`　→ `(LastFrame - FirstFrame + 1) / Fps >= 10`
- Filter 2：`MIN_FRONTAL_SEC = 8.0`　→ 正臉幀數 / Fps >= 8
- Filter 3：`Track.IsAlone` 全部為 True（每幀出現時畫面只有一張臉）

### 正臉判斷（`_detectFrontality`）
使用 Mediapipe 478 點 landmark，計算鼻尖到左右眼中心的水平距離比：
- 左眼中心：landmark 33, 133 的 x 平均
- 右眼中心：landmark 362, 263 的 x 平均
- 鼻尖：landmark 4
- 比值 `0.6 ≤ LeftDist/RightDist ≤ 1.67` 視為正臉

### 臉部追蹤（IoU Tracking）
- IoU 閾值：`IOU_THRESHOLD = 0.3`
- Gap 容忍：`GapFrames = max(5, int(Fps * 0.3))` 秒，超過則關閉 track
- 最小臉高：`MIN_FACE_HEIGHT_RATIO = 0.08`（影片高度 8%）

### Mediapipe 模式
`FaceLandmarker` + `RunningMode.VIDEO`，時戳計算：
`TimestampMs = int(FrameIdx * 1000 / Fps)`

### Thread 安全
worker thread → `self.after(0, lambda M=Msg: ...)` 更新 UI，
必須用 captured variable 避免 closure 陷阱。

### 存檔規則
- 編碼優先用 `mp4v`（.mp4），失敗備援 `XVID`（.avi）
- 檔名：掃描 `face_video/` 找最大 `face_NNN` 編號 +1

## Dependencies
```
customtkinter>=5.2
opencv-python>=4.8
mediapipe>=0.10
yt-dlp>=2024.1
numpy>=1.26
Pillow>=10.0
```

## Run
```
pip install -r requirements.txt
python main.py
```

