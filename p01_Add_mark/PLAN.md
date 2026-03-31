# 計畫：人臉自動標點工具 (p01_Add_mark)

## Context

使用 face_recognition 庫的68個關鍵點，自動在圖片中標出眼眉鼻口髮際下巴8個區域，以 YOLO 格式輸出標記檔。GUI 使用 CustomTkinter。目前專案只有 CLAUDE.md 和 .gitignore，需要從頭建立程式碼。

---

## 檔案結構

```
p01_Add_mark/
├── main.py             # 5行進入點，建立 App 並執行 mainloop
├── face_annotator.py   # 所有 UI 類別 + 處理函式
├── requirements.txt    # 套件版本鎖定
└── CLAUDE.md           # (已存在，最後補填 Commands/FileStructure)
```

---

## 類別與函式架構

### `face_annotator.py`

**常數區**
- `SUPPORTED_EXTS` — 支援的圖片副檔名
- `CLASS_COLORS` — 8類別的繪圖顏色
- `PROFILE_OFFSET_THRESHOLD = 0.20` — 側臉判斷閾值
- `PIC_AREA_MAX_W/H = 800/600` — 顯示區域最大尺寸

**類別 `FaceAnnotatorApp(ctk.CTk)`**

| 方法 | 說明 |
|------|------|
| `__init__` | 初始化視窗，呼叫 `_建立介面` |
| `_建立介面` | 建立所有 widgets（Grid 5行3列） |
| `_選擇輸入目錄` | 開啟資料夾選擇對話框，填入 EntryInputDir |
| `_選擇輸出目錄` | 開啟資料夾選擇對話框，填入 EntryOutputDir |
| `_單次測試` | 驗證設定 → 開啟單一圖片選取 → 背景執行緒處理 |
| `_全目錄處理` | 驗證設定 → 掃描目錄 → 背景執行緒批次處理 |
| `_批次處理執行緒` | 逐一呼叫 `_處理單張`，完成後重啟按鈕 |
| `_處理單張` | 核心：載圖→偵測→繪製→判側臉→輸出YOLO |
| `_更新圖片顯示` | PIL→CTkImage→PicArea (須在主執行緒呼叫) |
| `_寫入訊息` | 新訊息 prepend 到 textInfo 最上方 |
| `_驗證設定` | 檢查模式/目錄是否有效 |

**模組層級純函式（較容易測試）**

| 函式 | 說明 |
|------|------|
| `取得人臉關鍵點(NpImage)` | 呼叫 `face_recognition.face_landmarks`，有 try/except |
| `偵測側臉(Landmarks, ImgW, ImgH)` | 回傳 `(bool, float)` |
| `計算邊框(PointList)` | 回傳 `(x1, y1, x2, y2)` |
| `計算八類邊框(Landmarks, ImgW, ImgH)` | 組合8類邊框，含髮際/下巴估算 |
| `轉換為Yolo格式(x1,y1,x2,y2,ImgW,ImgH)` | 回傳正規化 `(xc,yc,w,h)` |
| `寫入Yolo檔案(BBoxList, OutputPath, ImgW, ImgH)` | 寫入 .txt，有 try/except |
| `繪製關鍵點與框(PilImage, Landmarks, BBoxList, IsSideface)` | 回傳標注後的 PIL Image |

---

## UI 版面（Grid）

```
Row 0:  DropMode(col=0)    BtnSingle(col=1)         BtnBatch(col=2)
Row 1:  LblInput(col=0)    EntryInputDir(col=1 EW)  BtnSelInput(col=2)
Row 2:  LblOutput(col=0)   EntryOutputDir(col=1 EW) BtnSelOutput(col=2)
Row 3:  PicArea(col=0..2   sticky=NSEW)
Row 4:  LblInfo(col=0)     TextInfo(col=1..2  EW)

grid_columnconfigure(1, weight=1)  → Entry/TextInfo 水平延伸
grid_rowconfigure(3, weight=1)     → PicArea 垂直延伸
```

---

## 核心演算法

### 側臉偵測
1. 計算左眼/右眼中心 X → 兩眼中點 EyeMidX
2. 計算鼻尖平均 X → NoseTipX
3. 偏移比例 = `abs(NoseTipX - EyeMidX) / 兩眼間距`
4. 若 > 0.20 → 側臉；估算角度 = `min(Offset × 180, 90)` 度

### 髮際線估算（Class 6）
1. 眉毛最高點 BrowTopY = min(所有眉毛 Y)
2. 鼻樑最上點 NoseBridgeTopY = `nose_bridge[0].y`
3. 髮際 Y = `BrowTopY - (NoseBridgeTopY - BrowTopY) × 1.2`
4. 髮際 X = 兩眼中心 X 平均
5. 邊框：以該點為中心，半徑 = ImgW × 3%

### 下巴點（Class 7）
- `chin[8]` 為68點模型的下巴最低中心點
- 邊框：以該點為中心，半徑 = ImgW × 3%

### YOLO 正規化
```
XCenter = (x1 + x2) / 2 / ImgW
YCenter = (y1 + y2) / 2 / ImgH
Width   = (x2 - x1) / ImgW
Height  = (y2 - y1) / ImgH
```
輸出前先 clamp 座標至圖片邊界。

---

## 執行緒安全規則
- face_recognition + PIL 運算在背景 `threading.Thread` 執行
- 所有 UI 更新（圖片、textInfo）透過 `self.after(0, func, *args)` 回主執行緒

---

## 圖片顯示流程
```
路徑 → face_recognition.load_image_file (RGB numpy)
     → PIL.Image.fromarray
     → COPY → ImageDraw 畫68點（金黃色）+ 8類邊框（各類別顏色）
     → resize 縮放至最大 800×600（保持比例，不放大）
     → CTkImage → self.PicArea.configure(image=...)
     → self._CurrentImage = ... (防止 GC)
```

---

## dependencies (requirements.txt)
```
customtkinter>=5.2.1
face-recognition>=1.3.0
Pillow>=10.0.0
numpy>=1.24.0
```

---

## 驗證方式
1. `python main.py` 啟動 GUI，確認5個 row 版面正確
2. 設定圖片目錄（含正面人臉圖）和輸出目錄，按"單次測試"
   - PicArea 顯示標注後圖片（68點 + 8個彩色框）
   - 輸出目錄出現同名 .txt 含8行 YOLO 格式
3. 用一張側臉圖測試：textInfo 出現 "可能是側臉(角度:XX度)"，無 .txt 輸出
4. 按"全目錄處理"，輸出目錄出現對應張數的 .txt 檔
