# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

可能會有多個小程式,但目前想做二個功能:

1. 一個在Windows上判斷有無權限開啟WebCam,若有就開啟WebCam,否則的話看可否取得權限,並開啟WebCam.
2. 看目前機器有幾台WebCam,若超過一台,就跳出dialog讓使用者選要開啟那個Webcam?

## Code Specification

1. 開發語言：Python 3.13
2. UI library：customtkinter
3. 影像擷取：OpenCV (cv2)，使用 cv2.CAP\_DSHOW 後端
4. 系統存取：winreg（讀取 Windows Registry）、subprocess（開啟 Windows 設定頁）
5. 非阻塞串流：threading（WebCam 迴圈在 daemon 執行緒中執行）
6. 註解請用中文
7. Variable Naming：CamelCase 一致使用
8. Error Handling：所有 API 呼叫必須包含 try-except
9. Function 名稱使用英文，不要用中文

## Design Decisions

1. 主視窗固定大小 420×280，resizable=False
2. 訊息記錄框（CTkTextbox）唯讀，最新訊息插入頂部（insert "1.0"），帶時間戳 \[HH:MM:SS]
3. 功能一「WebCam權限」：

   * 依序讀取 HKLM 與 HKCU 的 CapabilityAccessManager\\ConsentStore\\webcam Value
   * HKLM 無機碼時視為 Allow（部分 Windows 版本不存在）
   * 任一層級為非 Allow 時，彈出對話框並提供「開啟 Windows 設定」按鈕（ms-settings:privacy-webcam）
   * 兩層均為 Allow 時直接開啟 WebCam index 0
4. 功能二「WebCam台數」：

   * 掃描 index 0–9，使用 cv2.VideoCapture(I, cv2.CAP\_DSHOW).isOpened() 判斷是否存在
   * 0 台：記錄提示；1 台：直接開啟；≥2 台：彈出選擇對話框
5. WebCam 影像迴圈在獨立 daemon 執行緒（\_webcamLoop）中執行，避免 UI 凍結
6. 使用者可按 q 鍵或直接關閉 cv2 視窗來停止 WebCam
7. 開啟新 WebCam 前，若已有執行中的執行緒，先設 WebcamRunning=False 並 join（timeout=3s）

## UI

1. 頂部 TopFrame（CTkFrame）：

   * 左側 CTkComboBox（width=160，state=readonly），選項為「WebCam權限」、「WebCam台數」，佔位符為「請選擇」
   * 右側 CTkButton「執行」（width=80），點擊後呼叫 onExecute()
2. 下方 CTkTextbox（height=150，唯讀，wrap=word）作為訊息記錄框
3. 功能一權限不足時彈出 CTkToplevel（360×160），含說明文字與「開啟 Windows 設定」按鈕
4. 功能二多台時彈出 CTkToplevel（300×180），含 CTkComboBox 列出所有 WebCam 與「確定」按鈕



= = = = =

程式記錄:



1\.            HklmKey = winreg.OpenKey(

&#x20;               winreg.HKEY\_LOCAL\_MACHINE,

&#x20;               r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\webcam",

&#x20;           )

2\.            HklmValue, \_ = winreg.QueryValueEx(HklmKey, "Value")

3\.            winreg.CloseKey(HklmKey)

4\. subprocess.run("start ms-settings:privacy-webcam", shell=True, check=False)



5\.                 Cap = cv2.VideoCapture(I, cv2.CAP\_DSHOW)

6\.                if Cap.isOpened():

&#x20;                   AvailableCams.append(I)

7\.                Cap.release()

8\. CamOptions = \[f"WebCam {I}" for I in CamList]

&#x20;.. CamIndex = int(SelectedText.split()\[-1])

&#x20;.. if self.WebcamThread and self.WebcamThread.is\_alive():

&#x20;.. self.WebcamThread.join(timeout=3)

&#x20;..         self.WebcamThread = threading.Thread(

&#x20;           target=self.\_webcamLoop, args=(Index,), daemon=True

&#x20;       )

&#x20;.. Ret, Frame = Cap.read()

&#x20;.. cv2.imshow(WindowName, Frame)

&#x20;.. Key = cv2.waitKey(1) \& 0xFF

&#x20;.. if Key == ord("q"):

&#x20;.. if cv2.getWindowProperty(WindowName, cv2.WND\_PROP\_VISIBLE) < 1:

&#x20;           Cap.release()

&#x20;           cv2.destroyAllWindows()

&#x20;.. ctk.set\_appearance\_mode("System")

