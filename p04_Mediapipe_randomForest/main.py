"""
main.py

使用 MediaPipe FaceMesh + webcam 進行人臉辨識。
透過 CustomTkinter 建立 GUI，可學習並辨識不同人的身份。

使用方式：
    python main.py

依賴套件：
    pip install -r requirements.txt
"""

import os
import threading
import time
from collections import Counter

import cv2
import numpy as np
import customtkinter
from PIL import Image
import tkinter.messagebox as MsgBox

from face_recognizer import FaceRecognizer

# --- 應用程式常數 ---
LEARN_TARGET_FRAMES   = 30     # 學習模式目標收集 frame 數
LEARN_TIMEOUT_SECONDS = 60     # 學習模式最長等待時間（秒，保底避免卡住）
UI_REFRESH_MS         = 30     # webcam 畫面更新間隔（毫秒）
LEARN_TICK_MS         = 500    # 學習時每次抓 frame 的間隔（每秒 2 個樣本）
DETECT_TICK_MS        = 300    # 辨識時每次推論的間隔（MediaPipe 較快，縮短至 300ms）
DETECT_NONE_DETECT_TARGET = 5  # DetectNone 偵測階段目標 frame 數


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

    def Open(self) -> bool:
        """開啟 webcam 並啟動背景擷取執行緒。"""
        try:
            self._Cap = cv2.VideoCapture(self._CameraIndex)
            if not self._Cap.isOpened():
                raise RuntimeError(f"無法開啟攝影機 {self._CameraIndex}")
            self._Running = True
            self._Thread  = threading.Thread(target=self._CaptureLoop, daemon=True)
            self._Thread.start()
            return True
        except Exception as Error:
            print(f"[WebcamManager] 開啟攝影機失敗：{Error}")
            return False

    def Close(self) -> None:
        """停止背景執行緒並釋放 VideoCapture 資源。"""
        self._Running = False
        if self._Thread is not None:
            self._Thread.join(timeout=2.0)
            self._Thread = None
        if self._Cap is not None:
            self._Cap.release()
            self._Cap = None

    def GetLatestFrame(self) -> tuple:
        """以執行緒安全方式取得最新 frame 的副本。"""
        with self._Lock:
            if self._LatestFrame is None:
                return False, None
            return True, self._LatestFrame.copy()

    def _CaptureLoop(self) -> None:
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
    """主應用程式視窗，整合 webcam 與 MediaPipe 人臉辨識功能。"""

    def __init__(self, mode="normal", pet=None):
        super().__init__()

        # DetectNone 模式狀態
        self._DetectNoneActive   = False
        self._DetectNoneDtNames  = []   # 偵測階段累積的預測名稱（滑動窗口）

        # 學習模式狀態
        self._LearnActive       = False
        self._LearnStartTime    = 0.0
        self._LearnName         = ""
        self._LearnFrameCount   = 0

        # 推論執行緒防重入旗標
        self._InferenceActive   = False  #Joey: 寫法重要

        # 人臉偵測結果快取（供 _UpdateWebcamView 繪製框用）
        self._LastDetections    = []   # list of (top, right, bottom, left, name, confidence)

        # UI 圖像參照（防止被 GC 回收）
        self._CurrentPhotoImage = None

        # 核心元件
        self._Webcam     = WebcamManager(CameraIndex=0)
        self._Recognizer = FaceRecognizer()

        # 建立 UI
        self.mode = mode
        self.pet  = pet
        self._BuildUI(self.mode)

        # 初始化元件（開啟攝影機、載入人臉模型）
        self._InitComponents()

    # --------------------------------------------------------------------------
    # UI 建立
    # --------------------------------------------------------------------------
    def _BuildUI(self, mode) -> None:
        """建立 4-row CustomTkinter UI 介面。"""
        self.title("人臉辨識系統（MediaPipe）")
        self.protocol("WM_DELETE_WINDOW", self._OnClose)
        self.resizable(True, True)

        # 使用 grid 佈局，讓 Row2 可垂直延伸，Row0/1/3 固定高度
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)   # 只有 Row2 垂直延伸

        # Row 0：辨識功能列
        Row0 = customtkinter.CTkFrame(self)
        Row0.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        Row0.grid_columnconfigure(0, weight=1)

        # Row0 上半：按鈕 + 辨識結果標籤
        Row0Top = customtkinter.CTkFrame(Row0)
        Row0Top.grid(row=0, column=0, sticky="ew")

        self._BtnDetectName = customtkinter.CTkButton(
            Row0Top,
            text="Detect face",
            width=140,
            command=self._OnBtnDetectName
        )
        self._BtnDetectName.pack(side="left", padx=(5, 10), pady=5)
        self._BtnDetectName.pack_forget()  # 隱藏，改用 DetectNone

        self._BtnDetect2 = customtkinter.CTkButton(
            Row0Top,
            text="Detect2",
            width=120,
            command=self._OnBtnDetect2
        )
        self._BtnDetect2.pack(side="left", padx=(0, 10), pady=5)
        self._BtnDetect2.pack_forget()  # 隱藏，改用 DetectNone

        self._BtnDetectNone = customtkinter.CTkButton(
            Row0Top,
            text="Detect",
            width=120,
            command=self._OnBtnDetectNone
        )
        self._BtnDetectNone.pack(side="left", padx=(0, 10), pady=5)

        self._LblDetectName = customtkinter.CTkLabel(
            Row0Top,
            text="",
            font=customtkinter.CTkFont(size=16, weight="bold")
        )
        self._LblDetectName.pack(side="left", padx=5, pady=5)

        # Row0 下半：buffer 資訊（保留 widget，暫不使用）
        self._LblBufferInfo = customtkinter.CTkLabel(
            Row0,
            text="",
            anchor="w"
        )
        self._LblBufferInfo.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))

        # Row 1：學習功能列
        Row1 = customtkinter.CTkFrame(self)
        Row1.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        Row1.grid_columnconfigure(0, weight=1)

        # Row1 上半：姓名輸入 + 學習按鈕
        Row1Top = customtkinter.CTkFrame(Row1)
        Row1Top.grid(row=0, column=0, sticky="ew")

        self._TblMyName = customtkinter.CTkEntry(
            Row1Top,
            placeholder_text="輸入姓名",
            width=200
        )
        self._TblMyName.pack(side="left", padx=(5, 10), pady=5)

        self._BtnLearn = customtkinter.CTkButton(
            Row1Top,
            text="Learning",
            width=120,
            command=self._OnBtnLearn
        )
        self._BtnLearn.pack(side="left", padx=5, pady=5)

        self._BtnRemove = customtkinter.CTkButton(
            Row1Top,
            text="Remove",
            width=100,
            fg_color="#8B0000",
            hover_color="#B22222",
            command=self._OnBtnRemove
        )
        self._BtnRemove.pack(side="left", padx=(5, 5), pady=5)
        #self._BtnRemove.pack_forget()  # 隱藏

        # Row1 下半：學習進度（兩個 Column，各占 1/2），預設隱藏
        self._Row1Bot = customtkinter.CTkFrame(Row1)
        self._Row1Bot.grid(row=1, column=0, sticky="ew")
        self._Row1Bot.grid_remove()  # 預設隱藏
        Row1Bot = self._Row1Bot
        Row1Bot.grid_columnconfigure(0, weight=1)
        Row1Bot.grid_columnconfigure(1, weight=1)

        # Column 0：剩餘秒數 + 時間進度條
        Col0 = customtkinter.CTkFrame(Row1Bot)
        Col0.grid(row=0, column=0, sticky="ew", padx=(5, 3), pady=5)
        Col0.grid_columnconfigure(0, weight=1)

        self._LblRemain = customtkinter.CTkLabel(Col0, text="Remaining study seconds: --")
        self._LblRemain.grid(row=0, column=0, sticky="w", padx=5, pady=(5, 2))

        self._PbarTime = customtkinter.CTkProgressBar(Col0)
        self._PbarTime.grid(row=1, column=0, sticky="ew", padx=5, pady=(2, 5))
        self._PbarTime.set(0)

        # Column 1：已學習 frame 數 + frame 進度條
        Col1 = customtkinter.CTkFrame(Row1Bot)
        Col1.grid(row=0, column=1, sticky="ew", padx=(3, 5), pady=5)
        Col1.grid_columnconfigure(0, weight=1)

        self._LblLearnFrames = customtkinter.CTkLabel(Col1, text="Number of frames learned: 0")
        self._LblLearnFrames.grid(row=0, column=0, sticky="w", padx=5, pady=(5, 2))

        self._PbarFrames = customtkinter.CTkProgressBar(Col1)
        self._PbarFrames.grid(row=1, column=0, sticky="ew", padx=5, pady=(2, 5))
        self._PbarFrames.set(0)

        # Row 2：Webcam 畫面（垂直延伸）
        Row2 = customtkinter.CTkFrame(self)
        Row2.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        Row2.grid_columnconfigure(0, weight=1)
        Row2.grid_rowconfigure(0, weight=1)

        self._WebcamCanvas = customtkinter.CTkLabel(
            Row2,
            text="攝影機畫面載入中...",
            width=640,
            height=480
        )
        self._WebcamCanvas.grid(row=0, column=0, sticky="nsew")

        # Row 3：學習資料摘要 Log
        Row3 = customtkinter.CTkFrame(self)
        Row3.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        Row3.grid_columnconfigure(0, weight=1)

        self._TxtLog = customtkinter.CTkTextbox(Row3, height=120, state="disabled")
        self._TxtLog.grid(row=0, column=0, sticky="ew", padx=10, pady=(4, 8))

        if mode == "demo":
            # Demo 模式：只保留 Detect
            self._BtnLearn.pack_forget()
            self._TblMyName.pack_forget()
            Row1.grid_remove()

        elif mode == "learning":
            # Learning 模式：只保留 Learning
            self._BtnDetectNone.pack_forget()
            Row0.grid_remove()

    # --------------------------------------------------------------------------
    # 初始化
    # --------------------------------------------------------------------------
    def _InitComponents(self) -> None:
        """初始化攝影機與人臉辨識模型。"""
        # 1. 開啟攝影機
        CamOk = self._Webcam.Open()
        if not CamOk:
            MsgBox.showwarning(
                "攝影機錯誤",
                "無法開啟攝影機，請確認連線後重新啟動。\n應用程式將以無攝影機模式執行。"
            )

        # 2. 載入人臉編碼模型
        try:
            self._Recognizer.LoadModel()
            self._UpdateSummary()
        except Exception as Error:
            print(f"[MainApp] 模型載入：{Error}")

        # 3. 啟動 webcam 畫面更新迴圈
        self.after(UI_REFRESH_MS, self._UpdateWebcamView)

    # --------------------------------------------------------------------------
    # 工具方法
    # --------------------------------------------------------------------------
    def _UpdateSummary(self) -> None:
        """將目前學習資料摘要寫入 Log。"""
        Counts = self._Recognizer.GetSampleCounts()
        if not Counts:
            self._AppendLog("學習資料：（尚無資料）")
            return
        Parts = [f"{Name}: {N} 張" for Name, N in Counts.items()] # Joey: 寫法重要
        self._AppendLog("學習資料：" + "　|　".join(Parts))

    def _AppendLog(self, Msg: str) -> None:
        """將訊息插入 Log textbox 最上方（最新在最前）。"""
        import datetime
        Timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        Line = f"[{Timestamp}] {Msg}\n"
        self._TxtLog.configure(state="normal")
        self._TxtLog.insert("0.0", Line)
        self._TxtLog.configure(state="disabled")

    # --------------------------------------------------------------------------
    # Webcam 畫面更新迴圈（每 30ms 執行一次）
    # --------------------------------------------------------------------------
    def _UpdateWebcamView(self) -> None:
        """
        從 WebcamManager 取得最新 frame，繪製快取的人臉偵測框，
        轉換為 tkinter PhotoImage 並顯示在 UI 中。
        """
        try:
            Ok, Frame = self._Webcam.GetLatestFrame()
            if Ok and Frame is not None:
                # 繪製上次推論結果的人臉框（使用快取，不在此執行推論）
                if self._LastDetections:
                    Frame = self._DrawDetections(Frame, self._LastDetections) # Joey: 在UI畫偵測的方框

                # 轉換 BGR → RGB → PIL Image → CTkImage
                FrameRgb = cv2.cvtColor(Frame, cv2.COLOR_BGR2RGB)
                Img = Image.fromarray(FrameRgb)

                # 縮放至 canvas 尺寸，最大不超過 640×480
                W = min(self._WebcamCanvas.winfo_width(),  640)
                H = min(self._WebcamCanvas.winfo_height(), 480)
                #self.print_detailed_data(Id=1, W=self._WebcamCanvas.winfo_width(), H=self._WebcamCanvas.winfo_height())
                DisplaySize = (W, H) if W > 1 and H > 1 else (min(Img.width, 640), min(Img.height, 480))

                Photo = customtkinter.CTkImage(light_image=Img, size=DisplaySize)
                self._WebcamCanvas.configure(image=Photo, text="")
                self._CurrentPhotoImage = Photo   # 保持參照，防止 GC 回收

        except Exception as Error:
            print(f"[MainApp] 更新畫面失敗：{Error}")
        finally:
            self.after(UI_REFRESH_MS, self._UpdateWebcamView)

    def _DrawDetections(self, Frame: np.ndarray, Detections: list) -> np.ndarray:
        """
        在 Frame 上繪製人臉偵測框與姓名標籤。
        已知人物用綠框，Unknown 用紅框。
        """
        DrawFrame = Frame.copy()
        for Top, Right, Bottom, Left, Name, Confidence in Detections:
            Color = (0, 255, 0) if Name != "Unknown" else (0, 0, 255)
            cv2.rectangle(DrawFrame, (Left, Top), (Right, Bottom), Color, 2)
            # 姓名標籤背景
            Label = f"{Name} ({Confidence:.2f})" if Name != "Unknown" else "Unknown"
            cv2.rectangle(DrawFrame, (Left, Bottom - 25), (Right, Bottom), Color, cv2.FILLED)
            cv2.putText(DrawFrame, Label, (Left + 6, Bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return DrawFrame

    # --------------------------------------------------------------------------
    # DetectNone 功能（持續偵測，已知人物顯示姓名，陌生人顯示提示）
    # --------------------------------------------------------------------------
    def _OnBtnDetectNone(self) -> None:
        """Detect 按鈕點擊事件：切換啟動/停止。"""
        if self._DetectNoneActive:
            # 停止偵測
            self._DetectNoneActive  = False
            self._LastDetections    = []
            self._BtnDetectNone.configure(text="Detect", state="normal")
            self._LblDetectName.configure(text="")
            return

        # 確認有已登錄的人臉資料
        if not self._Recognizer.CanDetect():
            self._LblDetectName.configure(text="沒有資料,請先訓練")
            return

        # 開始偵測
        self._DetectNoneActive  = True
        self._DetectNoneDtNames = []
        self._BtnDetectNone.configure(text="Stop", state="normal")
        self._LblDetectName.configure(
            text=f"偵測中.... (0/{DETECT_NONE_DETECT_TARGET})"
        )
        self.after(0, self._DetectNoneTick)

    def _DetectNoneTick(self) -> None:
        """
        DetectNone 辨識 tick。
        在背景執行緒執行 MediaPipe 推論，避免阻塞 UI。
        累積 DETECT_NONE_DETECT_TARGET 次結果後取多數決顯示。
        """
        if not self._DetectNoneActive:
            return

        # 若上次推論尚未完成，跳過本次 tick
        if not self._InferenceActive:
            try:
                Ok, Frame = self._Webcam.GetLatestFrame()
                if Ok and Frame is not None:
                    self._InferenceActive = True
                    FrameCopy = Frame.copy()

                    def Worker():
                        try:
                            Results = self._Recognizer.Predict(FrameCopy)
                            self.after(0, lambda R=Results: self._OnDetectNoneResult(R))
                        except Exception as Error:
                            print(f"[MainApp] DetectNone 推論失敗：{Error}")
                        finally:
                            self._InferenceActive = False

                    threading.Thread(target=Worker, daemon=True).start()
            except Exception as Error:
                print(f"[MainApp] DetectNone tick 失敗：{Error}")

        self.after(DETECT_TICK_MS, self._DetectNoneTick)

    def _OnDetectNoneResult(self, Results: list) -> None:
        """DetectNone 推論結果回調（在主執行緒執行）。"""
        if not self._DetectNoneActive:
            return

        if Results:
            self._LastDetections = Results
            # 取置信度最高的人臉作為本次結果
            BestResult = max(Results, key=lambda R: R[5])
            Name = BestResult[4]

            # 加入滑動窗口
            self._DetectNoneDtNames.append(Name)
            if len(self._DetectNoneDtNames) > DETECT_NONE_DETECT_TARGET:
                self._DetectNoneDtNames.pop(0)

            # 累積夠了才顯示多數決結果
            if len(self._DetectNoneDtNames) >= DETECT_NONE_DETECT_TARGET:
                BestName = Counter(self._DetectNoneDtNames).most_common(1)[0][0]
                if BestName == "Unknown":
                    self._LblDetectName.configure(text="您好,我不認識你,我可以認識你嗎?")
                    if self.pet:
                        self.pet.face_detected.emit("Sorry, I don't know you...")   #這個是PySide,也是Michael寫的寵物程式對接用.
                else:
                    self._LblDetectName.configure(text=f"Hi, {BestName} 您好,又見到您了")
                    if self.pet:
                        self.pet.face_detected.emit(f"Hi, {BestName} !")
            else:
                self._LblDetectName.configure(
                    text=f"偵測中.... ({len(self._DetectNoneDtNames)}/{DETECT_NONE_DETECT_TARGET})"
                )
        else:
            # 未偵測到人臉，清除框
            self._LastDetections = []

    # --------------------------------------------------------------------------
    # 學習功能
    # --------------------------------------------------------------------------
    def _OnBtnLearn(self) -> None:
        """學習按鈕點擊事件處理。"""
        if self._LearnActive:
            return   # 學習進行中，忽略重複點擊

        PersonName = self._TblMyName.get().strip()
        if not PersonName:
            MsgBox.showwarning("缺少姓名", "請先在姓名欄位輸入要學習的姓名。")
            return

        self._StartLearning(PersonName)

    def _OnBtnRemove(self) -> None:
        """Remove 按鈕點擊事件：移除姓名欄位中指定人物的所有訓練資料。"""
        PersonName = self._TblMyName.get().strip()
        if not PersonName:
            MsgBox.showwarning("缺少姓名", "請先在姓名欄位輸入要移除的姓名。")
            return

        KnownPersons = self._Recognizer.GetAccumulatedPersons()
        if PersonName not in KnownPersons:
            MsgBox.showwarning(
                "找不到人物",
                f"資料中沒有 [{PersonName}] 的學習資料。\n"
                f"目前已有人物：{', '.join(KnownPersons) if KnownPersons else '（無）'}"
            )
            return

        Confirm = MsgBox.askyesno(
            "確認移除",
            f"確定要移除 [{PersonName}] 的所有訓練資料嗎？\n此動作無法還原。"
        )
        if not Confirm:
            return

        Ok = self._Recognizer.RemovePerson(PersonName)
        if Ok:
            MsgBox.showinfo("移除完成", f"已成功移除 [{PersonName}] 的所有訓練資料。")
            self._UpdateSummary()
        else:
            MsgBox.showerror("移除失敗", f"移除 [{PersonName}] 時發生錯誤，請查看 console 輸出。")

    def _StartLearning(self, PersonName: str) -> None:
        """開始學習模式，收集指定人物的人臉編碼樣本。"""
        self._LearnActive     = True
        self._LearnName       = PersonName
        self._LearnStartTime  = time.time()
        self._LearnFrameCount = 0

        # 停用相關按鈕（學習期間不允許其他操作）
        self._BtnLearn.configure(state="disabled")
        self._LblRemain.configure(text=f"Remaining study seconds: {LEARN_TIMEOUT_SECONDS}  ← 請慢慢左右、上下轉動頭部，以提升辨識多樣性")
        self._LblLearnFrames.configure(text="Number of frames learned: 0")
        self._PbarTime.set(0)
        self._PbarFrames.set(0)
        self._Row1Bot.grid()  # 顯示進度區域

        # 啟動學習 tick
        self.after(0, self._LearningTick)
        if self.pet:
            self.pet.face_detected.emit("Start learning!")

    def _StopLearning(self) -> None:
        """結束學習模式，儲存人臉編碼。"""
        self._LearnActive    = False
        self._LastDetections = []   # 清除人臉框覆蓋層
        self._LblRemain.configure(text="Remaining study seconds: --")
        self._LblLearnFrames.configure(text="Number of frames learned: 0")
        self._PbarTime.set(0)
        self._PbarFrames.set(0)
        self._Row1Bot.grid_remove()  # 隱藏進度區域
        self._BtnLearn.configure(state="normal")

        # 儲存人臉編碼
        SaveOk = self._Recognizer.SaveModel()
        KnownPersons = self._Recognizer.GetKnownPersons()
        PersonCount  = len(KnownPersons)
        PersonList   = ', '.join(KnownPersons) if KnownPersons else "（無）"

        if SaveOk:
            Msg = (
                f"已完成 [{self._LearnName}] 的學習！\n"
                f"本次學習 {self._LearnFrameCount} 次\n\n"
                f"目前已登錄人物（共 {PersonCount} 人）：\n{PersonList}"
            )
        else:
            Msg = (
                f"學習完成，但模型儲存失敗。\n"
                f"本次學習 {self._LearnFrameCount} 次\n"
                f"目前已登錄人物（共 {PersonCount} 人）：{PersonList}"
            )
        MsgBox.showinfo("學習完成", Msg)

        self._UpdateSummary()
        if self.pet:
            self.pet.face_detected.emit("Finished learning!")

    def _LearningTick(self) -> None:
        """學習模式每次 tick（每 500ms 執行一次）：收集人臉樣本並更新進度。"""
        if not self._LearnActive:
            return

        # 更新時間與進度（在主執行緒）
        Elapsed = time.time() - self._LearnStartTime
        Remain  = max(0, LEARN_TIMEOUT_SECONDS - int(Elapsed))
        self._LblRemain.configure(text=f"Remaining study seconds: {Remain}")
        self._PbarTime.set(min(1.0, Elapsed / LEARN_TIMEOUT_SECONDS))
        self._PbarFrames.set(min(1.0, self._LearnFrameCount / LEARN_TARGET_FRAMES))

        # 達到目標 frame 數或超時，停止學習
        if self._LearnFrameCount >= LEARN_TARGET_FRAMES or Elapsed >= LEARN_TIMEOUT_SECONDS:
            self._StopLearning()
            return

        # 在背景執行緒進行人臉偵測與編碼抽取（避免阻塞 UI）
        if not self._InferenceActive:
            try:
                Ok, Frame = self._Webcam.GetLatestFrame()
                if Ok and Frame is not None:
                    self._InferenceActive = True
                    FrameCopy  = Frame.copy()
                    PersonName = self._LearnName

                    def Worker():
                        try:
                            Added = self._Recognizer.AddSample(FrameCopy, PersonName)
                            self.after(0, lambda A=Added: self._OnLearnSampleAdded(A))
                        except Exception as Error:
                            print(f"[MainApp] 學習推論失敗：{Error}")
                        finally:
                            self._InferenceActive = False

                    threading.Thread(target=Worker, daemon=True).start()
            except Exception as Error:
                print(f"[MainApp] 學習 tick 失敗：{Error}")

        # 排程下一次 tick
        self.after(LEARN_TICK_MS, self._LearningTick)

    def _OnLearnSampleAdded(self, Added: bool) -> None:
        """學習樣本加入回調（在主執行緒執行）。"""
        if not self._LearnActive:
            return
        if Added:
            self._LearnFrameCount += 1
            self._LblLearnFrames.configure(
                text=f"Number of frames learned: {self._LearnFrameCount}"
            )

    # --------------------------------------------------------------------------
    # 隱藏按鈕的 stub（保留 widget 定義相容性）
    # --------------------------------------------------------------------------
    def _OnBtnDetectName(self) -> None:
        """Detect face 按鈕（目前隱藏，保留 stub）。"""
        pass

    def _OnBtnDetect2(self) -> None:
        """Detect2 按鈕（目前隱藏，保留 stub）。"""
        pass

    # --------------------------------------------------------------------------
    # 關閉處理
    # --------------------------------------------------------------------------
    def _OnClose(self) -> None:
        """程式關閉前，先停止所有活動並釋放攝影機與 MediaPipe 資源。"""
        print("[MainApp] 程式關閉中...")
        self._DetectNoneActive = False
        self._LearnActive      = False
        self._Webcam.Close()
        # 釋放 MediaPipe FaceMesh 資源
        try:
            self._Recognizer._Detector.close()
        except Exception:
            pass
        self.destroy()

    def print_detailed_data(self, **kwargs): # 用法: self.print_detailed_data(Name="Bob", Age=30, City="Taoyuan")
        for key, value in kwargs.items():
            print(f"{key}: {value}")


# ==============================================================================
# 程式進入點
# ==============================================================================
if __name__ == "__main__":
    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("blue")
    App = MainApp()
    App.mainloop()
