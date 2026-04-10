"""
main.py

使用 Haar Cascade 人臉偵測 + PyTorch CNN 人臉辨識。
UI 以 CustomTkinter 建立，支援學習、辨識、移除人員。

使用方式：
    python main.py

依賴套件：
    pip install -r requirements.txt
"""

import os
import threading
import time

import cv2
import numpy as np
import customtkinter
from PIL import Image
import tkinter.messagebox as MsgBox

from face_detector   import HaarFaceDetector
from face_recognizer import FaceRecognizer
from model_store     import ModelStore

# --- 常數 ---
LEARN_TARGET_FRAMES   = 100    # 學習目標收集 frame 數
LEARN_TIMEOUT_SECONDS = 120    # 學習逾時（秒）
LEARN_TICK_MS         = 500    # 學習抓圖間隔（ms）
DETECT_TICK_MS        = 300    # 辨識推論間隔（ms）
UI_REFRESH_MS         = 30     # Webcam 畫面更新間隔（ms）
WEBCAM_DISPLAY_W      = 640    # 顯示寬度（px）
WEBCAM_DISPLAY_H      = 480    # 顯示高度（px）


# ==============================================================================
# Class: WebcamManager
# ==============================================================================
class WebcamManager:
    """管理 webcam 資源，以 daemon 背景執行緒持續讀取 frame。"""

    def __init__(self, CameraIndex: int = 0):
        self._CameraIndex  = CameraIndex
        self._Cap          = None
        self._Lock         = threading.Lock()
        self._LatestFrame  = None
        self._Running      = False
        self._Thread       = None

    def open(self) -> bool:
        """開啟 webcam 並啟動背景擷取執行緒。"""
        try:
            self._Cap = cv2.VideoCapture(self._CameraIndex)
            if not self._Cap.isOpened():
                raise RuntimeError(f"無法開啟攝影機 {self._CameraIndex}")
            self._Running = True
            self._Thread  = threading.Thread(target=self._captureLoop, daemon=True)
            self._Thread.start()
            return True
        except Exception as Error:
            print(f"[WebcamManager] 開啟攝影機失敗：{Error}")
            return False

    def close(self) -> None:
        """停止背景執行緒並釋放 VideoCapture 資源。"""
        self._Running = False
        if self._Thread is not None:
            self._Thread.join(timeout=2.0)
            self._Thread = None
        if self._Cap is not None:
            self._Cap.release()
            self._Cap = None

    def getLatestFrame(self) -> tuple:
        """以執行緒安全方式取得最新 frame 的副本。"""
        with self._Lock:
            if self._LatestFrame is None:
                return False, None
            return True, self._LatestFrame.copy()

    def _captureLoop(self) -> None:
        """背景執行緒：持續從攝影機讀取 frame 並更新共用緩衝區。"""
        while self._Running:
            try:
                Ret, Frame = self._Cap.read()
                if Ret:
                    with self._Lock:
                        self._LatestFrame = Frame
                else:
                    time.sleep(0.01)
            except Exception as Error:
                print(f"[WebcamManager] 擷取 frame 失敗：{Error}")
                time.sleep(0.1)


# ==============================================================================
# Class: MainApp
# ==============================================================================
class MainApp(customtkinter.CTk):
    """主應用程式視窗，整合 Haar 人臉偵測與 PyTorch CNN 辨識。"""

    def __init__(self):
        super().__init__()

        # --- 學習狀態 ---
        self._LearnActive     = False
        self._LearnName       = ""
        self._LearnStartTime  = 0.0
        self._LearnFrameCount = 0

        # --- 辨識狀態 ---
        self._DetectActive    = False
        self._InferenceActive = False   # 防止推論執行緒重入

        # --- 訓練狀態 ---
        self._TrainingActive  = False

        # --- 顯示快取 ---
        # list of (X, Y, W, H, Name, Confidence)
        self._LastDetections    = []
        self._CurrentPhotoImage = None   # 防止 GC 回收

        # --- 核心元件 ---
        self._Webcam     = WebcamManager(CameraIndex=0)
        self._Detector   = HaarFaceDetector()
        self._Recognizer = FaceRecognizer()
        self._Store      = ModelStore()

        # 建立 UI 並初始化
        self._buildUI()
        self._initComponents()

    # --------------------------------------------------------------------------
    # UI 建立
    # --------------------------------------------------------------------------
    def _buildUI(self) -> None:
        """建立 4-row CustomTkinter UI。"""
        self.title("人臉辨識系統（Haar Cascade + CNN）")
        self.protocol("WM_DELETE_WINDOW", self._onClose)
        self.resizable(True, True)

        # grid 佈局：Row2（Webcam）垂直延伸，其餘固定高度
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # --- Row 0：辨識功能列 ---
        Row0 = customtkinter.CTkFrame(self)
        Row0.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        Row0.grid_columnconfigure(0, weight=1)

        Row0Top = customtkinter.CTkFrame(Row0)
        Row0Top.grid(row=0, column=0, sticky="ew")

        self._BtnDetect = customtkinter.CTkButton(
            Row0Top, text="Detect", width=120,
            command=self._onBtnDetect
        )
        self._BtnDetect.pack(side="left", padx=(5, 10), pady=5)

        self._LblDetectResult = customtkinter.CTkLabel(
            Row0Top, text="",
            font=customtkinter.CTkFont(size=16, weight="bold")
        )
        self._LblDetectResult.pack(side="left", padx=5, pady=5)

        # --- Row 1：學習功能列 ---
        Row1 = customtkinter.CTkFrame(self)
        Row1.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        Row1.grid_columnconfigure(0, weight=1)

        Row1Top = customtkinter.CTkFrame(Row1)
        Row1Top.grid(row=0, column=0, sticky="ew")

        self._EntryName = customtkinter.CTkEntry(
            Row1Top, placeholder_text="輸入姓名", width=200
        )
        self._EntryName.pack(side="left", padx=(5, 10), pady=5)

        self._BtnLearn = customtkinter.CTkButton(
            Row1Top, text="Learning", width=120,
            command=self._onBtnLearn
        )
        self._BtnLearn.pack(side="left", padx=5, pady=5)

        self._BtnRemove = customtkinter.CTkButton(
            Row1Top, text="Remove", width=100,
            fg_color="#8B0000", hover_color="#B22222",
            command=self._onBtnRemove
        )
        self._BtnRemove.pack(side="left", padx=(5, 5), pady=5)

        # 學習提示標籤（學習時才顯示）
        self._LblLearnHint = customtkinter.CTkLabel(
            Row1Top,
            text="  請保持正臉，勿大幅度轉動頭部",
            text_color="#FFA500",
            font=customtkinter.CTkFont(size=13)
        )
        self._LblLearnHint.pack(side="left", padx=10, pady=5)
        self._LblLearnHint.pack_forget()   # 預設隱藏

        # Row1 下半：學習進度條（學習時才顯示）
        self._Row1Bot = customtkinter.CTkFrame(Row1)
        self._Row1Bot.grid(row=1, column=0, sticky="ew")
        self._Row1Bot.grid_remove()   # 預設隱藏
        self._Row1Bot.grid_columnconfigure(0, weight=1)
        self._Row1Bot.grid_columnconfigure(1, weight=1)

        # Column 0：剩餘秒數 + 時間進度條
        Col0 = customtkinter.CTkFrame(self._Row1Bot)
        Col0.grid(row=0, column=0, sticky="ew", padx=(5, 3), pady=5)
        Col0.grid_columnconfigure(0, weight=1)

        self._LblRemain = customtkinter.CTkLabel(Col0, text="剩餘時間：--")
        self._LblRemain.grid(row=0, column=0, sticky="w", padx=5, pady=(5, 2))
        self._PbarTime = customtkinter.CTkProgressBar(Col0)
        self._PbarTime.grid(row=1, column=0, sticky="ew", padx=5, pady=(2, 5))
        self._PbarTime.set(0)

        # Column 1：已收集張數 + frame 進度條
        Col1 = customtkinter.CTkFrame(self._Row1Bot)
        Col1.grid(row=0, column=1, sticky="ew", padx=(3, 5), pady=5)
        Col1.grid_columnconfigure(0, weight=1)

        self._LblFrames = customtkinter.CTkLabel(Col1, text="已收集張數：0")
        self._LblFrames.grid(row=0, column=0, sticky="w", padx=5, pady=(5, 2))
        self._PbarFrames = customtkinter.CTkProgressBar(Col1)
        self._PbarFrames.grid(row=1, column=0, sticky="ew", padx=5, pady=(2, 5))
        self._PbarFrames.set(0)

        # --- Row 2：Webcam 畫面（垂直延伸） ---
        Row2 = customtkinter.CTkFrame(self)
        Row2.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        Row2.grid_columnconfigure(0, weight=1)
        Row2.grid_rowconfigure(0, weight=1)

        self._WebcamLabel = customtkinter.CTkLabel(
            Row2, text="攝影機畫面載入中...",
            width=WEBCAM_DISPLAY_W, height=WEBCAM_DISPLAY_H
        )
        self._WebcamLabel.grid(row=0, column=0, sticky="nsew")

        # --- Row 3：Log 區域 ---
        Row3 = customtkinter.CTkFrame(self)
        Row3.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        Row3.grid_columnconfigure(0, weight=1)

        self._TxtLog = customtkinter.CTkTextbox(Row3, height=120, state="disabled")
        self._TxtLog.grid(row=0, column=0, sticky="ew", padx=10, pady=(4, 8))

    # --------------------------------------------------------------------------
    # 初始化
    # --------------------------------------------------------------------------
    def _initComponents(self) -> None:
        """開啟攝影機，載入已有模型，啟動 Webcam 更新迴圈。"""
        # 開啟攝影機
        if not self._Webcam.open():
            MsgBox.showerror("錯誤", "無法開啟攝影機，請確認裝置連接。")

        # 載入已有模型
        if self._Recognizer.loadModel(self._Store.getModelPath()):
            Persons = self._Recognizer.getPersonList()
            self._appendLog(f"已載入模型，已知人員：{', '.join(Persons)}")
        else:
            self._appendLog("尚無模型，請先進行學習。")

        # 啟動 Webcam 畫面更新迴圈
        self._updateWebcamView()

    # --------------------------------------------------------------------------
    # 按鈕事件
    # --------------------------------------------------------------------------
    def _onBtnDetect(self) -> None:
        """切換辨識模式（開啟 / 停止）。"""
        if self._TrainingActive:
            MsgBox.showwarning("警告", "訓練中，請等待完成後再操作。")
            return

        if self._LearnActive:
            self._stopLearning()

        if self._DetectActive:
            # 停止辨識
            self._DetectActive = False
            self._LastDetections = []
            self._BtnDetect.configure(text="Detect")
            self._LblDetectResult.configure(text="")
        else:
            # 啟動辨識前確認模型已載入
            if not self._Recognizer.isModelLoaded():
                MsgBox.showwarning(
                    "警告",
                    "尚無模型。\n請先進行學習後再辨識。"
                )
                return
            self._DetectActive = True
            self._BtnDetect.configure(text="Stop")
            self._detectTick()

    def _onBtnLearn(self) -> None:
        """開始或停止學習模式。"""
        if self._TrainingActive:
            MsgBox.showwarning("警告", "訓練中，請等待完成後再操作。")
            return

        # 若正在學習則停止
        if self._LearnActive:
            self._stopLearning()
            return

        # 取得並驗證姓名
        Name = self._EntryName.get().strip()
        if not Name:
            MsgBox.showwarning("警告", "請輸入姓名後再按 Learning。")
            return

        # 停止辨識（學習中不偵測）
        if self._DetectActive:
            self._DetectActive = False
            self._BtnDetect.configure(text="Detect")
            self._LblDetectResult.configure(text="")
            self._LastDetections = []

        # 啟動學習
        self._LearnActive     = True
        self._LearnName       = Name
        self._LearnStartTime  = time.time()
        self._LearnFrameCount = 0
        self._LastDetections  = []

        self._BtnLearn.configure(text="Stop")
        self._LblLearnHint.pack(side="left", padx=10, pady=5)
        self._Row1Bot.grid()
        self._PbarTime.set(0)
        self._PbarFrames.set(0)
        self._LblRemain.configure(text="剩餘時間：--")
        self._LblFrames.configure(text="已收集張數：0")
        self._appendLog(f"開始學習：{Name}")

        self._learnTick()

    def _onBtnRemove(self) -> None:
        """移除指定人員的訓練資料。"""
        if self._TrainingActive or self._LearnActive:
            MsgBox.showwarning("警告", "學習或訓練進行中，請稍後再操作。")
            return

        Name = self._EntryName.get().strip()
        if not Name:
            MsgBox.showwarning("警告", "請輸入要移除的姓名。")
            return

        if Name not in self._Store.listPersons():
            MsgBox.showwarning("警告", f"找不到人員：{Name}")
            return

        if not MsgBox.askyesno("確認", f"確定要移除 {Name} 的所有訓練資料？"):
            return

        # 停止辨識
        if self._DetectActive:
            self._DetectActive = False
            self._BtnDetect.configure(text="Detect")
            self._LblDetectResult.configure(text="")
            self._LastDetections = []

        self._Store.removePerson(Name)
        self._appendLog(f"已刪除 {Name} 的資料。")

        # 判斷剩餘人員是否足夠重新訓練
        Remaining = self._Store.listPersons()
        if len(Remaining) >= 1:
            self._appendLog("開始重新訓練...")
            self._startTraining()
        else:
            # 無任何人員，清除模型
            ModelPath = self._Store.getModelPath()
            if os.path.exists(ModelPath):
                os.remove(ModelPath)
            self._Recognizer.clearModel()
            self._appendLog("已無人員資料，模型已清除。請重新學習。")

    # --------------------------------------------------------------------------
    # 學習邏輯
    # --------------------------------------------------------------------------
    def _learnTick(self) -> None:
        """每 LEARN_TICK_MS 執行一次：取 frame → Haar 偵測 → 儲存 ROI。"""
        if not self._LearnActive:
            return

        Elapsed = time.time() - self._LearnStartTime
        Remain  = max(0.0, LEARN_TIMEOUT_SECONDS - Elapsed)

        # 更新進度 UI
        self._LblRemain.configure(text=f"剩餘時間：{Remain:.0f} 秒")
        self._PbarTime.set(1.0 - (Remain / LEARN_TIMEOUT_SECONDS))
        self._LblFrames.configure(text=f"已收集張數：{self._LearnFrameCount}")
        self._PbarFrames.set(self._LearnFrameCount / LEARN_TARGET_FRAMES)

        # 逾時停止
        if Elapsed >= LEARN_TIMEOUT_SECONDS:
            self._appendLog("學習逾時，自動停止。")
            self._stopLearning()
            self._tryStartTraining()
            return

        # 取最新 frame 並偵測人臉
        Ret, Frame = self._Webcam.getLatestFrame()
        if Ret and Frame is not None:
            Faces = self._Detector.detect(Frame)
            if Faces:
                # 取面積最大的人臉
                Roi, X, Y, W, H = max(Faces, key=lambda F: F[3] * F[4])
                self._Store.saveTrainingImage(self._LearnName, Roi)
                self._LearnFrameCount += 1
                # 在畫面上顯示偵測框
                self._LastDetections = [(X, Y, W, H, self._LearnName, 1.0)]
            else:
                # 未偵測到臉時清除舊框
                self._LastDetections = []

        # 達到目標張數
        if self._LearnFrameCount >= LEARN_TARGET_FRAMES:
            self._appendLog(f"收集完成：{self._LearnFrameCount} 張。")
            self._stopLearning()
            self._tryStartTraining()
            return

        self.after(LEARN_TICK_MS, self._learnTick)

    def _stopLearning(self) -> None:
        """停止學習模式，還原 UI 狀態。"""
        self._LearnActive    = False
        self._LastDetections = []
        self._BtnLearn.configure(text="Learning")
        self._LblLearnHint.pack_forget()
        self._Row1Bot.grid_remove()
        self._LblDetectResult.configure(text="")

    def _tryStartTraining(self) -> None:
        """收集完成後啟動訓練（1 位人員即可，__unknown__ 類別由程式自動產生）。"""
        Persons = self._Store.listPersons()
        if len(Persons) < 1:
            self._appendLog("找不到任何人員資料，無法訓練。")
            return
        self._appendLog("開始訓練...")
        self._startTraining()

    # --------------------------------------------------------------------------
    # 訓練邏輯
    # --------------------------------------------------------------------------
    def _startTraining(self) -> None:
        """在背景執行緒啟動訓練，並鎖定相關按鈕。"""
        self._TrainingActive = True
        self._BtnLearn.configure(state="disabled")
        self._BtnRemove.configure(state="disabled")
        self._BtnDetect.configure(state="disabled")

        def TrainWorker():
            self._LastReportedEpoch = 0

            def OnProgress(Epoch, TotalEpochs, Loss):
                self._LastReportedEpoch = Epoch
                # 每 5 個 epoch 回報一次，或最後一個 epoch
                if Epoch % 5 == 0 or Epoch == TotalEpochs:
                    self.after(0, lambda E=Epoch, T=TotalEpochs, L=Loss:
                        self._appendLog(f"  Epoch {E:3d}/{T}  Loss: {L:.4f}"))

            Success = self._Recognizer.train(
                DataDir=self._Store.getDataDir(),
                ProgressCallback=OnProgress
            )
            if Success:
                self._Recognizer.saveModel(self._Store.getModelPath())
            self.after(0, lambda: self._onTrainingComplete(Success))

        threading.Thread(target=TrainWorker, daemon=True).start()

    def _onTrainingComplete(self, Success: bool) -> None:
        """訓練完成後恢復 UI，顯示結果。"""
        self._TrainingActive = False
        self._BtnLearn.configure(state="normal")
        self._BtnRemove.configure(state="normal")
        self._BtnDetect.configure(state="normal")

        if Success:
            Persons     = self._Recognizer.getPersonList()
            StoppedEpoch = getattr(self, "_LastReportedEpoch", "?")
            self._appendLog(f"訓練完成（共 {StoppedEpoch} epochs）！已知人員：{', '.join(Persons)}")
            MsgBox.showinfo("完成", f"訓練完成！\n已知人員：{', '.join(Persons)}")
        else:
            self._appendLog("訓練失敗，請確認至少有 2 位人員的資料。")
            MsgBox.showerror("錯誤", "訓練失敗，請確認至少有 2 位人員的資料。")

    # --------------------------------------------------------------------------
    # 辨識邏輯
    # --------------------------------------------------------------------------
    def _detectTick(self) -> None:
        """每 DETECT_TICK_MS 執行一次推論（背景執行緒，防重入）。"""
        if not self._DetectActive:
            return
        if self._InferenceActive:
            self.after(DETECT_TICK_MS, self._detectTick)
            return

        self._InferenceActive = True
        Ret, Frame = self._Webcam.getLatestFrame()

        def InferWorker():
            NewDetections = []
            try:
                if Ret and Frame is not None:
                    Faces = self._Detector.detect(Frame)
                    for Roi, X, Y, W, H in Faces:
                        Name, Confidence = self._Recognizer.predict(Roi)
                        NewDetections.append((X, Y, W, H, Name, Confidence))
            except Exception as Error:
                print(f"[MainApp] 推論失敗：{Error}")
            self.after(0, lambda: self._onDetectResult(NewDetections))

        threading.Thread(target=InferWorker, daemon=True).start()

    def _onDetectResult(self, Detections: list) -> None:
        """推論完成後更新顯示（在 UI 執行緒執行）。"""
        self._InferenceActive = False
        self._LastDetections  = Detections

        if Detections:
            # 顯示信心最高者的結果
            Best = max(Detections, key=lambda D: D[5])
            Name, Confidence = Best[4], Best[5]
            if Name == "Unknown":
                self._LblDetectResult.configure(text="Unknown")
            else:
                self._LblDetectResult.configure(
                    text=f"{Name}  ({Confidence * 100:.0f}%)"
                )
        else:
            self._LblDetectResult.configure(text="偵測中...")

        if self._DetectActive:
            self.after(DETECT_TICK_MS, self._detectTick)

    # --------------------------------------------------------------------------
    # Webcam 畫面更新
    # --------------------------------------------------------------------------
    def _updateWebcamView(self) -> None:
        """每 UI_REFRESH_MS 更新一次 Webcam 畫面（疊加人臉框與標籤）。"""
        try:
            Ret, Frame = self._Webcam.getLatestFrame()
            if Ret and Frame is not None:
                DrawFrame = Frame.copy()

                # 疊加人臉框與名稱標籤
                for X, Y, W, H, Name, Confidence in self._LastDetections:
                    if self._LearnActive:
                        # 學習模式：橘色框 + "Learning" 標籤
                        Color = (0, 165, 255)   # BGR: orange
                        Label = "Learning"
                    elif Name == "Unknown":
                        Color = (128, 128, 128)
                        Label = "Unknown"
                    else:
                        Color = (0, 255, 0)
                        Label = f"{Name} {Confidence * 100:.0f}%"

                    cv2.rectangle(DrawFrame, (X, Y), (X + W, Y + H), Color, 2)
                    # 標籤背景（提升可讀性）
                    (TextW, TextH), _ = cv2.getTextSize(
                        Label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    LabelY = max(Y - 10, TextH + 5)
                    cv2.rectangle(
                        DrawFrame,
                        (X, LabelY - TextH - 4),
                        (X + TextW + 4, LabelY + 2),
                        Color, -1
                    )
                    cv2.putText(
                        DrawFrame, Label,
                        (X + 2, LabelY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 2
                    )

                # 縮放至顯示尺寸並轉換為 CTkImage
                DrawFrame = cv2.resize(DrawFrame, (WEBCAM_DISPLAY_W, WEBCAM_DISPLAY_H))
                RgbFrame  = cv2.cvtColor(DrawFrame, cv2.COLOR_BGR2RGB)
                PilImg    = Image.fromarray(RgbFrame)
                CtkImg    = customtkinter.CTkImage(
                    light_image=PilImg,
                    dark_image=PilImg,
                    size=(WEBCAM_DISPLAY_W, WEBCAM_DISPLAY_H)
                )
                self._CurrentPhotoImage = CtkImg   # 防止 GC 回收
                self._WebcamLabel.configure(image=CtkImg, text="")

        except Exception as Error:
            print(f"[MainApp] 更新畫面失敗：{Error}")

        self.after(UI_REFRESH_MS, self._updateWebcamView)

    # --------------------------------------------------------------------------
    # 工具
    # --------------------------------------------------------------------------
    def _appendLog(self, Text: str) -> None:
        """在 Log 區域新增一行文字。"""
        try:
            self._TxtLog.configure(state="normal")
            self._TxtLog.insert("end", Text + "\n")
            self._TxtLog.see("end")
            self._TxtLog.configure(state="disabled")
        except Exception as Error:
            print(f"[MainApp] Log 寫入失敗：{Error}")

    def _onClose(self) -> None:
        """關閉視窗時清理資源。"""
        self._LearnActive  = False
        self._DetectActive = False
        self._Webcam.close()
        self.destroy()


# ==============================================================================
# 程式進入點
# ==============================================================================
if __name__ == "__main__":
    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("blue")
    App = MainApp()
    App.mainloop()
