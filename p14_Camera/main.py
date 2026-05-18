import customtkinter as ctk
import cv2
import winreg
import subprocess
import threading
from datetime import datetime


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("WebCam 工具")
        self.geometry("420x280")
        self.resizable(False, False)

        # WebCam 執行緒狀態
        self.WebcamThread = None
        self.WebcamRunning = False

        self._buildUI()

    def _buildUI(self):
        # 頂部操作列框架
        TopFrame = ctk.CTkFrame(self)
        TopFrame.pack(padx=20, pady=(20, 10), fill="x")

        # 下拉選單
        self.ComboSelection = ctk.CTkComboBox(
            TopFrame,
            values=["WebCam權限", "WebCam台數"],
            state="readonly",
            width=160,
        )
        self.ComboSelection.set("請選擇")
        self.ComboSelection.pack(side="left", padx=(10, 10), pady=10)

        # 執行按鈕
        self.BtnExecute = ctk.CTkButton(TopFrame, text="執行", width=80, command=self.onExecute)
        self.BtnExecute.pack(side="left", pady=10)

        # 多行訊息記錄框（最新在上，唯讀）
        self.TextLog = ctk.CTkTextbox(self, height=150, state="disabled", wrap="word")
        self.TextLog.pack(padx=20, pady=(0, 15), fill="x")

    def _logMessage(self, Text):
        # 將訊息加上時間戳後插入記錄框頂部
        TimeStr = datetime.now().strftime("%H:%M:%S")
        Entry = f"[{TimeStr}] {Text}\n"
        self.TextLog.configure(state="normal")
        self.TextLog.insert("1.0", Entry)
        self.TextLog.configure(state="disabled")

    def onExecute(self):
        Selected = self.ComboSelection.get()
        if Selected == "請選擇":
            self._logMessage("請先選擇功能")
            return
        if Selected == "WebCam權限":
            self.checkWebcamPermission()
        elif Selected == "WebCam台數":
            self.checkWebcamCount()

    # ── 功能一：WebCam 權限 ──────────────────────────────────────────────────

    def checkWebcamPermission(self):
        # 檢查系統層級攝影機存取開關（HKLM）
        try:
            HklmKey = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam",
            )
            HklmValue, _ = winreg.QueryValueEx(HklmKey, "Value")
            winreg.CloseKey(HklmKey)
        except Exception:
            # 部分 Windows 版本無此機碼，視為允許
            HklmValue = "Allow"

        # 檢查使用者層級攝影機存取開關（HKCU）
        try:
            HkcuKey = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam",
            )
            HkcuValue, _ = winreg.QueryValueEx(HkcuKey, "Value")
            winreg.CloseKey(HkcuKey)
        except Exception as E:
            self._logMessage(f"Registry 讀取失敗：{E}")
            return

        if HklmValue != "Allow":
            self._logMessage("系統攝影機存取已關閉，需至設定開啟")
            self._showPermissionDialog("系統攝影機存取已關閉。\n請至「設定 → 隱私權 → 相機」\n開啟「攝影機存取權」。")
        elif HkcuValue != "Allow":
            self._logMessage("使用者攝影機存取已關閉，需至設定開啟")
            self._showPermissionDialog("使用者攝影機存取已關閉。\n請至「設定 → 隱私權 → 相機」\n開啟「讓應用程式存取您的相機」。")
        else:
            self._logMessage("WebCam 已有權限，正在開啟...")
            self.openWebcam(0)

    def _showPermissionDialog(self, Msg="目前無 WebCam 存取權限。\n請至 Windows 隱私設定開啟相機權限。"):
        # 顯示引導使用者開啟設定的對話框
        Dialog = ctk.CTkToplevel(self)
        Dialog.title("WebCam 權限不足")
        Dialog.geometry("360x160")
        Dialog.resizable(False, False)
        Dialog.grab_set()

        LabelMsg = ctk.CTkLabel(Dialog, text=Msg, justify="left")
        LabelMsg.pack(pady=20, padx=20)

        def openPrivacySettings():
            try:
                subprocess.run("start ms-settings:privacy-webcam", shell=True, check=False)
            except Exception:
                pass
            Dialog.destroy()

        BtnOpenSettings = ctk.CTkButton(Dialog, text="開啟 Windows 設定", command=openPrivacySettings)
        BtnOpenSettings.pack(pady=5)

    # ── 功能二：WebCam 台數 ──────────────────────────────────────────────────

    def checkWebcamCount(self):
        self._logMessage("正在偵測 WebCam 數量...")
        self.update()

        AvailableCams = []
        for I in range(10):
            try:
                Cap = cv2.VideoCapture(I, cv2.CAP_DSHOW)
                if Cap.isOpened():
                    AvailableCams.append(I)
                Cap.release()
            except Exception:
                break

        if len(AvailableCams) == 0:
            self._logMessage("未偵測到任何 WebCam")
        elif len(AvailableCams) == 1:
            self._logMessage("偵測到 1 台 WebCam，正在開啟...")
            self.openWebcam(AvailableCams[0])
        else:
            self._logMessage(f"偵測到 {len(AvailableCams)} 台 WebCam，請選擇")
            self._showCamSelectDialog(AvailableCams)

    def _showCamSelectDialog(self, CamList):
        # 顯示多台 WebCam 選擇對話框
        Dialog = ctk.CTkToplevel(self)
        Dialog.title("選擇 WebCam")
        Dialog.geometry("300x180")
        Dialog.resizable(False, False)
        Dialog.grab_set()

        LabelMsg = ctk.CTkLabel(Dialog, text=f"偵測到 {len(CamList)} 台 WebCam，請選擇：")
        LabelMsg.pack(pady=15)

        CamOptions = [f"WebCam {I}" for I in CamList]
        CamCombo = ctk.CTkComboBox(Dialog, values=CamOptions, state="readonly", width=160)
        CamCombo.set(CamOptions[0])
        CamCombo.pack(pady=5)

        def onConfirm():
            SelectedText = CamCombo.get()
            # 取出選項末尾的數字作為 index
            CamIndex = int(SelectedText.split()[-1])
            Dialog.destroy()
            self.openWebcam(CamIndex)

        BtnConfirm = ctk.CTkButton(Dialog, text="確定", width=80, command=onConfirm)
        BtnConfirm.pack(pady=15)

    # ── 共用：開啟 WebCam ────────────────────────────────────────────────────

    def openWebcam(self, Index):
        # 若已有 WebCam 執行中，先停止
        if self.WebcamRunning:
            self.WebcamRunning = False
            if self.WebcamThread and self.WebcamThread.is_alive():
                self.WebcamThread.join(timeout=3)

        self.WebcamRunning = True
        self.WebcamThread = threading.Thread(
            target=self._webcamLoop, args=(Index,), daemon=True
        )
        self.WebcamThread.start()

    def _webcamLoop(self, Index):
        # 在獨立執行緒中執行 WebCam 影像迴圈，避免 UI 凍結
        try:
            Cap = cv2.VideoCapture(Index, cv2.CAP_DSHOW)
            if not Cap.isOpened():
                self.after(0, lambda: self._logMessage(f"無法開啟 WebCam {Index}"))
                self.WebcamRunning = False
                return

            self.after(0, lambda: self._logMessage(f"WebCam {Index} 運行中（按 q 關閉）"))

            WindowName = f"WebCam {Index}"
            while self.WebcamRunning:
                try:
                    Ret, Frame = Cap.read()
                    if not Ret:
                        break
                    cv2.imshow(WindowName, Frame)
                    # waitKey 需 > 0 才能讓視窗正常回應
                    Key = cv2.waitKey(1) & 0xFF
                    if Key == ord("q"):
                        break
                    # 若使用者直接關閉 cv2 視窗
                    if cv2.getWindowProperty(WindowName, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except Exception:
                    break

            Cap.release()
            cv2.destroyAllWindows()
            self.WebcamRunning = False
            self.after(0, lambda: self._logMessage("WebCam 已關閉"))
        except Exception as E:
            self.WebcamRunning = False
            ErrMsg = str(E)
            self.after(0, lambda: self._logMessage(f"WebCam 錯誤：{ErrMsg}"))


if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    MainApp = App()
    MainApp.mainloop()
