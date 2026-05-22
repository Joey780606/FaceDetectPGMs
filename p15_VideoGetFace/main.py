import customtkinter as ctk
import threading
import os
from datetime import datetime
from face_processor import FaceProcessor

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("p15 - 影片人臉擷取")
        self.geometry("580x440")
        self.resizable(False, False)
        self._buildUI()

    def _buildUI(self):
        # 頂部：URL 輸入 + Submit 按鍵
        TopFrame = ctk.CTkFrame(self)
        TopFrame.pack(fill="x", padx=10, pady=(10, 5))

        self.UrlEntry = ctk.CTkEntry(
            TopFrame,
            placeholder_text="輸入 YouTube 網址...",
            width=440,
        )
        self.UrlEntry.pack(side="left", padx=(5, 5), pady=8)

        self.SubmitBtn = ctk.CTkButton(
            TopFrame, text="Submit", width=90, command=self.onSubmit
        )
        self.SubmitBtn.pack(side="left", padx=(0, 5), pady=8)

        # 進度區
        ProgressFrame = ctk.CTkFrame(self)
        ProgressFrame.pack(fill="x", padx=10, pady=(0, 5))

        self.ProgressBar = ctk.CTkProgressBar(ProgressFrame, mode="determinate")
        self.ProgressBar.pack(fill="x", padx=8, pady=(8, 2))
        self.ProgressBar.set(0)

        self.StatusLabel = ctk.CTkLabel(ProgressFrame, text="就緒", anchor="w")
        self.StatusLabel.pack(fill="x", padx=8, pady=(0, 6))

        # 多行結果顯示區
        self.LogBox = ctk.CTkTextbox(self, height=300, wrap="word", state="disabled")
        self.LogBox.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _logMessage(self, Text):
        """在文字區最上方插入一行（含時戳）"""
        Timestamp = datetime.now().strftime("%H:%M:%S")
        Line = f"[{Timestamp}] {Text}\n"
        self.LogBox.configure(state="normal")
        self.LogBox.insert("1.0", Line)
        self.LogBox.configure(state="disabled")

    def onSubmit(self):
        """驗證輸入後啟動 worker thread"""
        Url = self.UrlEntry.get().strip()
        if not Url:
            self._logMessage("請輸入 YouTube 網址")
            return

        self.SubmitBtn.configure(state="disabled")
        self.ProgressBar.set(0)
        self.StatusLabel.configure(text="處理中...")
        self._logMessage(f"開始處理：{Url}")

        Thread = threading.Thread(
            target=self._processingWorker, args=(Url,), daemon=True
        )
        Thread.start()

    def _processingWorker(self, Url):
        """在 daemon thread 中執行，透過 self.after() 更新 UI"""
        try:
            OutputDir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "face_video"
            )
            Processor = FaceProcessor(
                ProgressCallback=lambda Pct, Msg: self.after(
                    0, lambda P=Pct, M=Msg: self._onProgress(P, M)  # 由 face_processor 呼叫，更新進度條和狀態訊息
                ),
                LogCallback=lambda Msg: self.after(
                    0, lambda M=Msg: self._logMessage(M)
                ),
            )
            SavedFiles = Processor.process(Url, OutputDir)  #真的有 process function,參 face_processor.py
            self.after(0, lambda F=SavedFiles: self._onComplete(F))
        except Exception as E:
            ErrMsg = str(E)
            self.after(0, lambda M=ErrMsg: self._onError(M))

    def _onProgress(self, Pct, Msg):
        self.ProgressBar.set(Pct)
        self.StatusLabel.configure(text=Msg)

    def _onComplete(self, SavedFiles):
        self.ProgressBar.set(1.0)
        self.SubmitBtn.configure(state="normal")
        if SavedFiles:
            self.StatusLabel.configure(text=f"完成！找到 {len(SavedFiles)} 人")
            self._logMessage(f"符合條件共 {len(SavedFiles)} 人，已存以下檔案：")
            for F in SavedFiles:
                self._logMessage(f"  → {F}")
        else:
            self.StatusLabel.configure(text="完成，未找到符合條件的人")
            self._logMessage("未找到同時滿足三項條件的人臉片段")

    def _onError(self, ErrMsg):
        self.ProgressBar.set(0)
        self.StatusLabel.configure(text="發生錯誤")
        self.SubmitBtn.configure(state="normal")
        self._logMessage(f"錯誤：{ErrMsg}")


if __name__ == "__main__":
    MainApp = App()
    MainApp.mainloop()
