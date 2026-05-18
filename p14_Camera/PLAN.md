# Plan: p14_Camera — WebCam 權限與選擇工具

## Context

根據 CLAUDE.md，此專案要開發兩個功能：
1. **WebCam 權限檢查**：Windows 上偵測/取得 WebCam 權限並開啟
2. **多台 WebCam 選擇**：偵測 WebCam 數量，若超過一台則顯示選擇對話框

---

## 檔案結構

```
p14_Camera/
├── CLAUDE.md
├── PLAN.md          ← 本檔案
├── main.py          ← 主程式（UI + 邏輯）
└── requirements.txt ← 依賴清單
```

---

## 依賴套件（requirements.txt）

```
customtkinter
opencv-python
```

---

## main.py 架構

### UI 主視窗（customtkinter）

- `CTkFrame` 橫排：`CTkComboBox`（下拉選單）+ `CTkButton`（執行）
- ComboBox 選項：`["WebCam權限", "WebCam台數"]`，placeholder `"請選擇"`
- Button 標籤：`"執行"`
- 結果顯示：`CTkLabel` 顯示執行狀態訊息

### 功能一：`checkWebcamPermission()`

**流程：**
1. 讀取 Windows Registry：
   `HKCU\Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam`
   - `Value == "Allow"` → 有權限 → 開啟 WebCam
   - 其他 → 無權限 → 彈出對話框引導至設定
2. 有權限：用 `cv2.VideoCapture(0)` 開啟第一台 WebCam
3. 無權限：彈出 `CTkToplevel`，按鈕執行 `start ms-settings:privacy-webcam`

### 功能二：`checkWebcamCount()`

**流程：**
1. 用 `cv2.VideoCapture(i, cv2.CAP_DSHOW)` 迴圈（i=0..9）偵測可用 WebCam
2. 0 台：顯示錯誤訊息
3. 1 台：直接開啟
4. 超過 1 台：彈出 `CTkToplevel` 選擇對話框，使用者確認後開啟

### `openWebcam(Index)` 共用函式

- 在獨立 daemon Thread 執行 `cv2.imshow` 迴圈，避免 UI 凍結
- 按 `q` 或關閉視窗時結束
- 開啟前若已有 WebCam 執行中，先停止舊的

---

## 關鍵設計細節

| 項目 | 決策 |
|------|------|
| WebCam 影像顯示 | 在獨立 Thread 使用 `cv2.imshow`，避免阻塞 UI |
| 權限設定引導 | `subprocess.run("start ms-settings:privacy-webcam", shell=True)` |
| WebCam 枚舉 | `cv2.VideoCapture(i, cv2.CAP_DSHOW)` 逐一嘗試，成功則加入清單 |
| 視窗關閉偵測 | `cv2.getWindowProperty(WindowName, cv2.WND_PROP_VISIBLE) < 1` |

---

## 驗證方式

1. 執行 `python main.py`
2. 未選擇直接按「執行」→ 顯示提示訊息
3. 選「WebCam權限」→ 確認 Registry 讀取正確，有權限時開啟 WebCam 視窗
4. 選「WebCam台數」→ 確認偵測台數，多台時顯示選擇對話框
5. 按 `q` 或關閉 WebCam 視窗 → 狀態訊息更新為「WebCam 已關閉」
