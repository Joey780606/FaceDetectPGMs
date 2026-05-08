# p13_Animation 規劃文件

## 專案目標

設計一個透明浮動 Widget，顯示在螢幕右下角，能順暢播放並切換 GIF 動畫，並依 GIF 切換輪流顯示不同的互動 UI 元件。

---

## 技術選型

| 項目 | 選擇 | 理由 |
|------|------|------|
| 語言 | Python 3.13 | 專案規範 |
| UI 框架 | PySide6 | 支援透明視窗、QMovie 播放 GIF 效能佳、商用免費（LGPL） |
| 動畫格式 | GIF | 現有素材，QMovie 支援多幀播放與 CacheAll 預載 |

---

## UI 佈局（4 Rows）

```
┌─────────────────────────────────────────┐
│  Row 1：[上一個] [下一個] [隨機撥放] [✕] │
│  ┌───────────────────────────────────┐  │
│  │ Row 2：文字 Label（白底黑框）       │  │
│  │ Row 3：互動區（白底黑框，共用背景） │  │
│  └───────────────────────────────────┘  │
│  Row 4：GIF 顯示區（200×200，置中）      │
└─────────────────────────────────────────┘
視窗：220×420px，透明背景，固定右下角
面板：全透明、圓角 14px
```

### Row 1 — 控制列（`row_controls.py`）
- 按鈕：上一個、下一個、隨機撥放（可切換 toggle）、✕ 關閉
- ✕ 靠右對齊，紅色調樣式，點擊發出 `CloseRequested` 訊號 → `QCoreApplication.quit()`
- 任何按鍵按下 → 同時觸發 Row 2 文字追加

### Row 2 — 文字 Label（`row_label.py`）
- 預設文字：`"This is test texts. Check UI is OK or not."`
- Row 1 任何按鍵按下時，追加一次預設文字
- 累積超過 2000 字元 → 恢復預設值
- 使用唯讀 QTextEdit，內容超出自動出現捲軸

### Row 3 — 互動區（`row_interaction.py`）
- 每次 GIF 切換後，依序輪流顯示三種模式：
  - **Mode 0**：OK / Cancel 雙按鈕
  - **Mode 1**：文字輸入框（Enter 送出）
  - **Mode 2**：隱藏（高度收合為 0）

### Row 2 + Row 3 共用容器
- 白底（`background: white`）、黑色邊框（`border: 2px solid #1a1a1a`）、圓角 6px
- Row 3 隱藏時容器高度自動縮減

### Row 4 — GIF 顯示區（`row_display.py`）
- 固定 400×400px、置中對齊
- GIF 切換時以淡入淡出過場（各 80ms），在完全透明瞬間才實際換檔，確保視覺順暢

---

## 模組架構

```
gif_widget.py          ← 程式進入點，建立 QApplication + WidgetWindow
widget_window.py       ← 主視窗，組裝四排 Layout，連接所有訊號
├── gif_player.py      ← GIF 播放引擎（QMovie 管理、切換、隨機、播完偵測）
├── row_controls.py    ← Row 1 控制按鈕列
├── row_label.py       ← Row 2 文字 Label
├── row_interaction.py ← Row 3 互動區（三種模式輪流）
├── row_display.py     ← Row 4 GIF 顯示區（淡入淡出過場）
└── style_constants.py ← 所有視覺常數與 Stylesheet 字串

gif_bg_remover.py      ← 獨立工具：GIF 去背景色（customtkinter UI）
```

---

## 訊號流程

```
RowControls.PrevRequested  ──→ GifPlayer.switchPrev
                           ──→ RowLabel.appendText

RowControls.NextRequested  ──→ GifPlayer.switchNext
                           ──→ RowLabel.appendText

RowControls.RandomToggled  ──→ GifPlayer.toggleRandom
                           ──→ RowLabel.appendText

RowControls.CloseRequested ──→ QCoreApplication.quit

GifPlayer.GifSwitched      ──→ RowDisplay.triggerCrossfade（淡入淡出換GIF）
                           ──→ WidgetWindow._onGifSwitched → RowInteraction.advanceMode

GifPlayer.PlaybackCompleted──→ WidgetWindow._onRandomCompleted
                               （隨機模式時自動呼叫 GifPlayer.switchTo）
```

---

## 視覺常數（`style_constants.py`）

| 常數 | 值 | 說明 |
|------|----|------|
| WIDGET_WIDTH | 220 | 視窗寬度 |
| WIDGET_HEIGHT | 420 | 視窗高度 |
| GIF_DISPLAY_SIZE | 200 | GIF 顯示區邊長 |
| CROSSFADE_MS | 80 | 淡入/淡出各 80ms |
| ROW2_HEIGHT | 80 | Row 2 文字區高度 |
| ROW3_HEIGHT | 60 | Row 3 互動區高度 |
| MARGIN_RIGHT | 10 | 視窗距螢幕右邊緣 |
| MARGIN_BOTTOM | 10 | 視窗距螢幕下邊緣 |

---

## 輔助工具

### gif_bg_remover.py — GIF 去背工具
- UI：customtkinter
- 功能：選擇目錄 → 點「Transparent background」→ 批次去除所有 GIF 的背景色
- 演算法：讀取第一幀四角顏色判斷背景色，以 numpy 計算歐氏距離，距離 ≤ 容差的像素設 alpha=0
- 輸出：預設存至 `transparent/` 子目錄（可勾選覆蓋原始檔案）

---

## GIF 素材目錄

`gif/` 目錄內依數字前綴排序播放，例如 `(4)morning.gif`、`(5)coffee.gif`。
GIF 切換順序依檔名內 `(N)` 數字由小到大排列。
