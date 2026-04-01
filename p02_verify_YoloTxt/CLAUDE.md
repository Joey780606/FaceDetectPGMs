# CLAUDE.md — p02_verify_YoloTxt

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Project Goal

讀取 YOLO 格式的 `.txt` 標記檔，在對應圖片上畫出各特徵點的位置與邊框，
供使用者確認 `p01_Add_mark` 產生的標記是否正確。

## Architecture

- UI：CustomTkinter
- 圖片處理：Pillow（PIL）
- 不依賴 face_recognition 或 MediaPipe，僅做純粹的標記驗證顯示

## Code Specification

1. 開發語言：Python 3.13+
2. UI library：customtkinter
3. 註解請用中文
4. Variable Naming：CamelCase
5. Error Handling：所有檔案 I/O 需有 try/except

## File Structure

```
p02_verify_YoloTxt/
├── main.py          # 程式進入點（5行）
├── verify_app.py    # 主程式（UI + 邏輯）
├── requirements.txt # 套件版本
├── .gitignore       # 排除 __pycache__
└── CLAUDE.md        # 本文件
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

## UI Design（5 rows）

- **Row 0**：Label「圖片目錄」、Entry（圖片路徑）、Button「選擇」
- **Row 1**：Label「標記目錄」、Entry（YOLO txt 路徑）、Button「選擇」
- **Row 2**：Button「◀ 前一個」、Label（目前檔名與進度 N/總數）、Button「後一個 ▶」
- **Row 3**：PicArea — 顯示圖片及標記（邊框 + 中心點 + 類別標籤）
- **Row 4**：Label「Information:」、TextInfo（多行，新訊息插最上方）

## Key Classes and Functions

- `VerifyApp` (verify_app.py) — CustomTkinter 主視窗
- `parse_yolo_file(TxtPath)` — 讀取 YOLO .txt，回傳 list of dict
- `find_image_for_txt(TxtPath, ImageDir)` — 找同主檔名的圖片
- `draw_yolo_annotations(PilImage, Records)` — 畫邊框、中心點、類別標籤
- `resize_to_fit(PilImage, MaxW, MaxH)` — 等比例縮小至顯示區域

## 使用流程

1. 選擇「圖片目錄」（.png / .jpg 等圖片所在處）
2. 選擇「標記目錄」（YOLO .txt 所在處）→ 自動載入第一筆
3. 用「◀ 前一個」/ 「後一個 ▶」逐筆瀏覽
4. PicArea 顯示對應圖片，並標出各類別的邊框（含中心點與標籤）
5. TextInfo 顯示目前檔名、圖片尺寸、標記筆數與類別清單

## 類別定義（與 p01_Add_mark 一致）

| Class ID | 名稱 | 顏色 |
|----------|------|------|
| 0 | 左眼 | 綠 #00FF00 |
| 1 | 右眼 | 青 #00FFFF |
| 2 | 左眉 | 黃 #FFFF00 |
| 3 | 右眉 | 橙 #FF8000 |
| 4 | 鼻   | 洋紅 #FF00FF |
| 5 | 嘴   | 紅 #FF0000 |
| 6 | 髮際 | 白 #FFFFFF |
| 7 | 下巴 | 藍 #4080FF |

## YOLO 座標還原公式

```
X1 = (xc - w/2) * ImgW
Y1 = (yc - h/2) * ImgH
X2 = (xc + w/2) * ImgW
Y2 = (yc + h/2) * ImgH
```

## 與 p01_Add_mark 的關係

- p01_Add_mark：產生 YOLO 標記檔（.txt）
- p02_verify_YoloTxt：驗證標記檔是否正確標記在正確位置
- 兩者的 CLASS_COLORS / CLASS_NAMES 定義需保持一致
