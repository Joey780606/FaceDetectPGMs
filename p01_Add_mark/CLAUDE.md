# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

使用 face_recognition 庫的內建68個關鍵點,來協助自動在圖片中標出各器官的點或區域。
透過 CustomTkinter GUI進行開發。

## Architecture

**人臉辨識核心**：[ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)（MIT 授權，可免費商用）
**髮際線偵測**：[MediaPipe Face Mesh](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)（Google，Apache 2.0，可商用）— landmark 10 為額頭頂部中心點
**自動標點** 讓使用者輸入處理型態和目錄後,UI可讓使用者選單張測試,整個目錄處理。目前會把處理後的結果,以Yolo格式檔案放在使用者指定的目錄.

## Code Specification

1. 開發語言: Python 3.13
2. UI library: customtkinter
3. 註解請用中文
4. Variable Naming: CamelCase is used consistently
5. Error Handling: All API calls must include a try-catch block

## Design Decisions / Constraints

1. 顯示照片時,要把相關的器官的範圍標示出來

## UI Design（4 rows）

- **Row 0**：1是drop-down menu,目前就一項,"Yolo+眼眉鼻口髮際下巴8點",2是"單次測試"的button,3是"全目錄處理"的button.
- **Row 1**：1是Label,顯示"圖片目錄",2是供人輸入處理路徑的Entry,3是按下時可選取路徑的button,名叫"選擇",選好路徑後會顯示在2的Entry裡.
- **Row 2**：1是Label,顯示"輸出目錄",2是供人輸出資料路徑的Entry,3是按下時可選取路徑的button,名叫"選擇",選好路徑後會顯示在2的Entry裡.
- **Row 3**：顯示照片的地方,名叫PicArea.
- **Row 4**：再分二Row,1是Label,顯示 "Information:" 顯示多行的UI text,2是能放多行的Text,名叫"textInfo",有新的資料,都放在最上面一行.

1. 使用者選擇 "Yolo+眼眉鼻口髮際下巴8點" 的drop-down後,需再選擇"圖片目錄","輸出目錄"後,若按下"單次測試"button,會做以下事:
 a. 顯示68個關鍵點.
 b. 顯示左眼區(類別0),右眼區(類別1),左眉區(類別2),右眉區(類別3),鼻子區(類別4),嘴巴區(類別5),髮際線最低的中心點(MediaPipe landmark 10，失敗時fallback幾何估算)(類別6)和下巴點(類別7).
 c. 若發現圖裡的臉有側臉,會影響到算各類之間的距離,就在"textInfo"裡顯示: 檔案名,可能是側臉(角度:XXX度). 若能算出側臉的角度就標示出來.然後不要做下面d的部分.
 d. 將上述b點這八類以Yolo格式檔 `<class_id> <x_center> <y_center> <width> <height>` 的方式寫到 相同主檔名.txt 裡,並存在輸出目錄裡.處理完後, "textInfo"顯示處理的檔案名稱與處理時間(XX秒).
 
2. 使用者選擇 "Yolo+眼眉鼻口髮際下巴8點" 的drop-down後,需再選擇圖片目錄,輸出目錄後,若按下"全目錄處理"button,就依上述1將目錄中所有檔案都進行處理,
 也把每個檔的處理狀況,都寫到"textInfo"裡.

## 套件相容性注意事項

### setuptools 版本限制
`requirements.txt` 中鎖定 `setuptools==69.5.1`，原因如下：

- `face_recognition_models` 使用 `pkg_resources`（屬於 setuptools）來定位模型檔路徑
- setuptools 82 起已將 `pkg_resources` 完全移除，導致 `face_recognition_models` 無法 import
- Python 3.14 預設不附帶 setuptools，安裝時會直接裝到最新版（82+）
- 解決方式：鎖定 setuptools==69.5.1，此版本仍包含完整的 `pkg_resources`

### 若出現 "Please install face_recognition_models" 錯誤
依序執行以下指令：
```
pip install setuptools==69.5.1
pip install --force-reinstall git+https://github.com/ageitgey/face_recognition_models
python main.py
```

## Commands

### 安裝依賴
```
pip install -r requirements.txt
```

### 執行主程式
```
python main.py
```

### 首次使用流程
```
pip install -r requirements.txt
python main.py
```

## File Structure
```
p01_Add_mark/
├── main.py             # 程式進入點（5行）
├── face_annotator.py   # 所有 UI 類別 + 處理函式
├── requirements.txt    # 套件版本
└── PLAN.md             # 實作規劃文件
```

## Key Classes and Functions
- `FaceAnnotatorApp` (face_annotator.py) — CustomTkinter 主視窗
- `取得人臉關鍵點(NpImage)` — 呼叫 face_recognition API
- `偵測側臉(Landmarks, ImgW, ImgH)` — 判斷側臉及估算角度
- `計算八類邊框(Landmarks, ImgW, ImgH)` — 計算8個 YOLO 類別邊框
- `_載入MediaPipe模型()` — Lazy 初始化 MediaPipe Face Mesh（Apache 2.0，可商用）
- `精確髮際邊框(PilImage, ImgW, ImgH)` — MediaPipe landmark 10 定位髮際中心點（class 6）
- `轉換為Yolo格式(x1,y1,x2,y2,ImgW,ImgH)` — 像素座標轉 YOLO 正規化
- `寫入Yolo檔案(BBoxList, OutputPath, ImgW, ImgH)` — 輸出 .txt 標記檔
- `繪製關鍵點與框(PilImage, Landmarks, BBoxList, IsSideface)` — PIL 標注圖

## MediaPipe 髮際線偵測說明
- 授權：Apache 2.0，完全可商用
- 使用 `mediapipe.tasks.vision.FaceLandmarker`（Tasks API，mediapipe >= 0.10 適用）
- 初始化時設定 `num_faces=10`，支援多人照片偵測
- Landmark 10：額頭頂部中心，468點中最接近髮際線的中心點
- 首次使用時自動下載模型檔 `face_landmarker.task`（約1MB）至程式同目錄
- class 6/7 的 width/height 動態取 class 0–5 中最小的 w/h 值（非固定 0.025）

### MediaPipe vs 幾何估算的切換邏輯（detect_hairline_box）
多人照片時 MediaPipe 可能對應到錯誤的臉，依偵測到的臉數決定策略：

| MediaPipe 偵測臉數 | 條件 | 髮際線來源 |
|--------------------|------|-----------|
| 1 張 | — | MediaPipe（可靠） |
| 2 張 | 兩臉重心距離 > 圖寬 25%，且最近臉距參考點 ≤ 圖寬 15% | MediaPipe 比對最近臉 |
| 2 張 | 兩臉重心距離 ≤ 圖寬 25% | 幾何估算（臉太近，比對不可靠） |
| 2 張 | 最近 MediaPipe 臉距參考點 > 圖寬 15% | 幾何估算（MediaPipe 漏偵測目標臉） |
| 3 張以上 | — | 幾何估算（臉太多太密） |

幾何估算（`_estimate_hairline_box`）完全基於 face_recognition 已選定的 `Landmarks`，
不涉及跨模型臉序比對，永遠對應正確人臉。

**注意**：兩個模型偵測到的臉數可能不同（如某張臉光線不足被其中一個模型漏掉），
距離驗證（15% 圖寬）能防止 MediaPipe 未偵測到目標臉時強行比對到錯誤臉。

### cv2 DLL 載入失敗的處理（Windows 相容性問題）
某些 Windows 電腦上，opencv 的 `cp37-abi3` wheel 會因缺少系統 DLL 而無法載入，
導致 mediapipe import 失敗（mediapipe 的繪圖工具 `drawing_utils.py` 會 import cv2）。

**問題鏈**：
cv2 DLL 載入失敗 → mediapipe import 失敗 → `_load_mediapipe_model()` 回傳 False → 顯示「MediaPipe 載入失敗，改用幾何估算髮際線」

**解決方式**（已實作於 `_load_mediapipe_model`）：
載入 mediapipe 前先嘗試 import cv2；若失敗，塞一個空模組佔位：
```python
if 'cv2' not in sys.modules:
    try:
        import cv2
    except Exception:
        sys.modules['cv2'] = types.ModuleType('cv2')
```
本程式只使用 `FaceLandmarker.detect()`，不呼叫任何 cv2 功能，空模組即可。