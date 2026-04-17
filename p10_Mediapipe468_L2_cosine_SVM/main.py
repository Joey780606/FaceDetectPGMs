"""
main.py

使用 MediaPipe FaceLandmarker（468 個 3D 特徵點）+ Cosine 相似度比對進行人臉辨識。
透過 CustomTkinter 建立 GUI，可學習並辨識不同人的身份。

使用方式：
    python main.py

依賴套件：
    pip install -r requirements.txt
"""

import datetime
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
from svm_classifier_np import SVM_UNKNOWN_THRESH

# --- 應用程式常數 ---
LEARN_TARGET_FRAMES       = 100     # 學習模式目標收集 frame 數
LEARN_TIMEOUT_SECONDS     = 60     # 學習模式最長等待時間（秒，保底避免卡住）
UI_REFRESH_MS             = 30     # webcam 畫面更新間隔（毫秒）
LEARN_TICK_MS             = 300    # 學習時每次抓 frame 的間隔（每秒 2 個樣本）
DETECT_TICK_MS            = 300    # 辨識時每次推論的間隔
DETECT_NONE_DETECT_TARGET = 5      # 多數決累積 frame 數


# ==============================================================================
# Class: WebcamManager
# ==============================================================================
class WebcamManager:
    """管理 webcam 資源，以 daemon 背景執行緒持續讀取 frame。"""

    def __init__(self, CameraIndex: int = 0):
        self._CameraIndex = CameraIndex
        self._Cap         = None
        self._Lock        = threading.Lock()
        self._LatestFrame = None
        self._Running     = False
        self._Thread      = None

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
    """主應用程式視窗，整合 webcam 與 MediaPipe 468 點人臉辨識功能。"""

    def __init__(self):
        super().__init__()

        # DetectNone 模式狀態
        self._DetectNoneActive  = False
        self._DetectNoneDtNames = []   # 偵測階段累積的預測名稱（滑動窗口）

        # 學習模式狀態
        self._LearnActive     = False
        self._LearnStartTime  = 0.0
        self._LearnName       = ""
        self._LearnFrameCount = 0

        # 推論執行緒防重入旗標
        self._InferenceActive = False

        # 人臉偵測結果快取（供 _UpdateWebcamView 繪製框用）
        self._LastDetections     = []   # list of (top, right, bottom, left, name, confidence)

        # 學習時的關鍵點快取（供 _UpdateWebcamView 在學習中疊加顯示）
        self._LastLearnKeyPoints = []   # list of {"left_eye":(cx,cy), "right_eye":..., "nose":..., "mouth":...}

        # UI 圖像參照（防止被 GC 回收）
        self._CurrentPhotoImage = None

        # 核心元件（Recognizer 在背景執行緒初始化，先設為 None）
        self._Webcam     = WebcamManager(CameraIndex=0)
        self._Recognizer = None

        # 建立 UI
        self._BuildUI()

        # 初始化元件（開啟攝影機、載入人臉模型）
        self._InitComponents()

    # --------------------------------------------------------------------------
    # UI 建立
    # --------------------------------------------------------------------------
    def _BuildUI(self) -> None:
        """建立 5-row CustomTkinter UI 介面。"""
        self.title("人臉辨識系統（MediaPipe 468 + SVM 辨識）")
        self.protocol("WM_DELETE_WINDOW", self._OnClose)
        self.resizable(True, True)

        # 使用 grid 佈局，讓 Row3 可垂直延伸，Row0/1/2/4 固定高度
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)   # 只有 Row3（Webcam）垂直延伸

        # Row 0：辨識功能列
        Row0 = customtkinter.CTkFrame(self)
        Row0.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        Row0.grid_columnconfigure(0, weight=1)

        # Row0 上半：Detect 按鈕 + 辨識結果標籤
        Row0Top = customtkinter.CTkFrame(Row0)
        Row0Top.grid(row=0, column=0, sticky="ew")

        self._BtnDetectNone = customtkinter.CTkButton(
            Row0Top,
            text="Detect",
            width=120,
            command=self._OnBtnDetectNone
        )
        self._BtnDetectNone.pack(side="left", padx=(5, 10), pady=5)

        self._LblDetectName = customtkinter.CTkLabel(
            Row0Top,
            text="",
            font=customtkinter.CTkFont(size=16, weight="bold")
        )
        self._LblDetectName.pack(side="left", padx=5, pady=5)

        # Row0 下半：buffer 資訊（保留 widget 供日後使用）
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

        # Row1 上半：姓名輸入 + 學習按鈕 + 移除按鈕
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

        # Row1 下半：學習進度（兩欄，各占 1/2），預設隱藏
        self._Row1Bot = customtkinter.CTkFrame(Row1)
        self._Row1Bot.grid(row=1, column=0, sticky="ew")
        self._Row1Bot.grid_remove()   # 預設隱藏
        Row1Bot = self._Row1Bot
        Row1Bot.grid_columnconfigure(0, weight=1)
        Row1Bot.grid_columnconfigure(1, weight=1)

        # Column 0：剩餘秒數 + 時間進度條
        Col0 = customtkinter.CTkFrame(Row1Bot)
        Col0.grid(row=0, column=0, sticky="ew", padx=(5, 3), pady=5)
        Col0.grid_columnconfigure(0, weight=1)

        self._LblRemain = customtkinter.CTkLabel(
            Col0, text="Remaining study seconds: --"
        )
        self._LblRemain.grid(row=0, column=0, sticky="w", padx=5, pady=(5, 2))

        self._PbarTime = customtkinter.CTkProgressBar(Col0)
        self._PbarTime.grid(row=1, column=0, sticky="ew", padx=5, pady=(2, 5))
        self._PbarTime.set(0)

        # Column 1：已學習 frame 數 + frame 進度條
        Col1 = customtkinter.CTkFrame(Row1Bot)
        Col1.grid(row=0, column=1, sticky="ew", padx=(3, 5), pady=5)
        Col1.grid_columnconfigure(0, weight=1)

        self._LblLearnFrames = customtkinter.CTkLabel(
            Col1, text="Number of frames learned: 0"
        )
        self._LblLearnFrames.grid(row=0, column=0, sticky="w", padx=5, pady=(5, 2))

        self._PbarFrames = customtkinter.CTkProgressBar(Col1)
        self._PbarFrames.grid(row=1, column=0, sticky="ew", padx=5, pady=(2, 5))
        self._PbarFrames.set(0)

        # Row 2：閾值調整（Cosine 相似度閾值）
        Row2Thresh = customtkinter.CTkFrame(self)
        Row2Thresh.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        Row2Thresh.grid_columnconfigure(1, weight=1)

        customtkinter.CTkLabel(Row2Thresh, text="SVM 信心度閾值", anchor="w").grid(
            row=0, column=0, sticky="w", padx=(8, 4), pady=8
        )
        self._SldCosine = customtkinter.CTkSlider(
            Row2Thresh, from_=0.10, to=0.99,
            number_of_steps=890,
            command=self._OnCosineThreshChanged
        )
        self._SldCosine.set(SVM_UNKNOWN_THRESH)
        self._SldCosine.grid(row=0, column=1, sticky="ew", padx=4, pady=8)
        self._LblCosineVal = customtkinter.CTkLabel(
            Row2Thresh, text=f"{SVM_UNKNOWN_THRESH:.2f}", width=44, anchor="e"
        )
        self._LblCosineVal.grid(row=0, column=2, padx=(4, 8), pady=8)

        # Row 3：Webcam 畫面（垂直延伸）
        Row2 = customtkinter.CTkFrame(self)
        Row2.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
        Row2.grid_columnconfigure(0, weight=1)
        Row2.grid_rowconfigure(0, weight=1)

        self._WebcamCanvas = customtkinter.CTkLabel(
            Row2,
            text="攝影機畫面載入中...",
            width=640,
            height=480
        )
        self._WebcamCanvas.grid(row=0, column=0, sticky="nsew")

        # Row 4：學習資料摘要 Log
        Row3 = customtkinter.CTkFrame(self)
        Row3.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 10))
        Row3.grid_columnconfigure(0, weight=1)

        self._TxtLog = customtkinter.CTkTextbox(Row3, height=120, state="disabled")
        self._TxtLog.grid(row=0, column=0, sticky="ew", padx=10, pady=(4, 8))

    # --------------------------------------------------------------------------
    # 初始化
    # --------------------------------------------------------------------------
    def _InitComponents(self) -> None:
        """初始化攝影機，並以背景執行緒載入 MediaPipe 與 RF 分類器。"""
        # 1. 開啟攝影機（快速）
        CamOk = self._Webcam.Open()
        if not CamOk:
            MsgBox.showwarning(
                "攝影機錯誤",
                "無法開啟攝影機，請確認連線後重新啟動。\n應用程式將以無攝影機模式執行。"
            )

        # 2. 立刻啟動 webcam 畫面更新（攝影機已開，不需等模型）
        self.after(UI_REFRESH_MS, self._UpdateWebcamView)

        # 3. 停用功能按鈕，Log 提示初始化中
        self._SetButtonsEnabled(False)
        self._AppendLog("系統初始化中，請稍候...")

        # 4. 背景執行緒：建立 FaceRecognizer（含 MediaPipe 載入）+ LoadModel（重建 Cosine 比對器）
        def InitWorker():
            try:
                Recognizer = FaceRecognizer()
                Recognizer.LoadModel()
                self._Recognizer = Recognizer
            except Exception as Error:
                print(f"[MainApp] 背景初始化失敗：{Error}")
            finally:
                self.after(0, self._OnInitDone)

        threading.Thread(target=InitWorker, daemon=True).start()

    def _OnInitDone(self) -> None:
        """背景初始化完成後回到主執行緒：啟用按鈕並更新資料摘要。"""
        self._SetButtonsEnabled(True)
        if self._Recognizer is not None:
            self._UpdateSummary()
            self._AppendLog("初始化完成，系統就緒。")
        else:
            self._AppendLog("初始化失敗，請查看 console 輸出。")

    def _SetButtonsEnabled(self, Enabled: bool) -> None:
        """統一啟用或停用功能按鈕（初始化未完成時停用）。"""
        State = "normal" if Enabled else "disabled"
        self._BtnDetectNone.configure(state=State)
        self._BtnLearn.configure(state=State)
        self._BtnRemove.configure(state=State)

    # --------------------------------------------------------------------------
    # 工具方法
    # --------------------------------------------------------------------------
    def _UpdateSummary(self) -> None:
        """將目前學習資料摘要寫入 Log。"""
        Counts = self._Recognizer.GetSampleCounts()
        if not Counts:
            self._AppendLog("學習資料：（尚無資料）")
            return
        Parts = [f"{Name}: {N} 張" for Name, N in Counts.items()]
        self._AppendLog("學習資料：" + "　|　".join(Parts))

    def _AppendLog(self, Msg: str) -> None:
        """將訊息插入 Log textbox 最上方（最新在最前）。"""
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
                    Frame = self._DrawDetections(Frame, self._LastDetections)

                # 學習模式：疊加雙眼/鼻子/嘴巴中心點
                if self._LearnActive and self._LastLearnKeyPoints:
                    Frame = self._DrawKeyPoints(Frame, self._LastLearnKeyPoints)

                # 轉換 BGR → RGB → PIL Image → CTkImage
                FrameRgb = cv2.cvtColor(Frame, cv2.COLOR_BGR2RGB)
                Img = Image.fromarray(FrameRgb)

                # 縮放至 canvas 尺寸，最大不超過 640×480
                W = min(self._WebcamCanvas.winfo_width(),  640)
                H = min(self._WebcamCanvas.winfo_height(), 480)
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
        for Top, Right, Bottom, Left, Name, Confidence, *_ in Detections:
            Color = (0, 255, 0) if Name != "Unknown" else (0, 0, 255)
            cv2.rectangle(DrawFrame, (Left, Top), (Right, Bottom), Color, 2)
            # 姓名標籤背景
            Label = f"{Name} ({Confidence:.2f})" if Name != "Unknown" else "Unknown"
            cv2.rectangle(DrawFrame, (Left, Bottom - 25), (Right, Bottom), Color, cv2.FILLED)
            cv2.putText(DrawFrame, Label, (Left + 6, Bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return DrawFrame

    def _DrawKeyPoints(self, Frame: np.ndarray, KeyPointsList: list) -> np.ndarray:
        """
        在 Frame 上繪製學習時的關鍵點中心（雙眼、鼻子、嘴巴）。

        Parameters
        ----------
        KeyPointsList : list of dict，每個 dict 對應一張臉：
                        {"left_eye": (cx,cy), "right_eye": (cx,cy),
                         "nose": (cx,cy), "mouth": (cx,cy)}
        """
        DrawFrame = Frame.copy()
        # 各部位的顏色（BGR）與標籤
        StyleMap = {
            'left_eye':  ((255, 180,   0), "L"),   # 藍色
            'right_eye': ((255, 180,   0), "R"),   # 藍色
            'nose':      ((  0, 220,   0), "N"),   # 綠色
            'mouth':     ((  0,   0, 220), "M"),   # 紅色
        }
        for FaceKP in KeyPointsList:
            for PartName, (Color, Label) in StyleMap.items():
                Pt = FaceKP.get(PartName)
                if Pt is None:
                    continue
                Cx, Cy = Pt
                cv2.circle(DrawFrame, (Cx, Cy), 6, Color, -1)
                cv2.putText(DrawFrame, Label, (Cx + 8, Cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, Color, 1)
        return DrawFrame

    # --------------------------------------------------------------------------
    # DetectNone 功能（持續偵測，已知人物顯示姓名，陌生人顯示 Unknown）
    # --------------------------------------------------------------------------
    def _OnBtnDetectNone(self) -> None:
        """Detect 按鈕點擊事件：切換啟動/停止。"""
        if self._DetectNoneActive:
            # 停止偵測
            self._DetectNoneActive  = False
            self._LastDetections    = []
            self._BtnDetectNone.configure(text="Detect", state="normal")
            self._LblDetectName.configure(text="")
            self._LblBufferInfo.configure(text="")
            return

        # 確認有已登錄的人臉資料
        if not self._Recognizer.CanDetect():
            self._LblDetectName.configure(text="沒有資料，請先訓練")
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
        在背景執行緒執行推論，避免阻塞 UI。
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
            Name       = BestResult[4]
            YawRatio   = BestResult[6]
            PitchRatio = BestResult[7]

            # 更新頭部轉角顯示
            self._LblBufferInfo.configure(text=self._headPoseDesc(YawRatio, PitchRatio))

            # 加入滑動窗口
            self._DetectNoneDtNames.append(Name)
            if len(self._DetectNoneDtNames) > DETECT_NONE_DETECT_TARGET:
                self._DetectNoneDtNames.pop(0)

            # 累積夠了才顯示多數決結果
            if len(self._DetectNoneDtNames) >= DETECT_NONE_DETECT_TARGET:
                BestName = Counter(self._DetectNoneDtNames).most_common(1)[0][0]
                if BestName == "Unknown":
                    self._LblDetectName.configure(text="您好，我不認識你，我可以認識你嗎？")
                else:
                    self._LblDetectName.configure(text=f"Hi, {BestName} 您好，又見到您了")
            else:
                self._LblDetectName.configure(
                    text=f"偵測中.... ({len(self._DetectNoneDtNames)}/{DETECT_NONE_DETECT_TARGET})"
                )
        else:
            # 未偵測到人臉，清除框與轉角顯示
            self._LastDetections = []
            self._LblBufferInfo.configure(text="")

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
            self._Recognizer.SaveModel()
            MsgBox.showinfo("移除完成", f"已成功移除 [{PersonName}] 的所有訓練資料。")
            self._UpdateSummary()
        else:
            MsgBox.showerror("移除失敗", f"移除 [{PersonName}] 時發生錯誤，請查看 console 輸出。")

    def _StartLearning(self, PersonName: str) -> None:
        """開始學習模式，收集指定人物的人臉樣本。"""
        # 若 Detect 正在執行中，先停止
        if self._DetectNoneActive:
            self._DetectNoneActive = False
            self._LastDetections   = []
            self._BtnDetectNone.configure(text="Detect", state="normal")
            self._LblDetectName.configure(text="")

        self._LearnActive     = True
        self._LearnName       = PersonName
        self._LearnStartTime  = time.time()
        self._LearnFrameCount = 0

        # 停用按鈕（學習期間不允許其他操作）
        self._BtnLearn.configure(state="disabled")
        self._LblRemain.configure(
            text=f"Remaining study seconds: {LEARN_TIMEOUT_SECONDS}  ← 請慢慢左右、上下轉動頭部"
        )
        self._LblLearnFrames.configure(text="Number of frames learned: 0")
        self._PbarTime.set(0)
        self._PbarFrames.set(0)
        self._Row1Bot.grid()   # 顯示進度區域

        # 啟動學習 tick
        self.after(0, self._LearningTick)

    def _StopLearning(self) -> None:
        """結束學習模式：清除 UI 狀態，並以背景執行緒執行重訓與儲存。"""
        self._LearnActive        = False
        self._LastDetections     = []   # 清除人臉框覆蓋層
        self._LastLearnKeyPoints = []   # 清除學習關鍵點覆蓋層
        self._LblRemain.configure(text="Remaining study seconds: --")
        self._LblLearnFrames.configure(text="Number of frames learned: 0")
        self._PbarTime.set(0)
        self._PbarFrames.set(0)
        self._Row1Bot.grid_remove()    # 隱藏進度區域

        # 訓練期間停用所有功能按鈕，Log 提示
        self._SetButtonsEnabled(False)
        self._AppendLog(f"[{self._LearnName}] 樣本收集完畢，正在訓練 SVM 分類器...")

        # 快照學習結果（避免背景執行緒讀取時被覆蓋）
        LearnName  = self._LearnName
        FrameCount = self._LearnFrameCount

        def TrainWorker():
            """背景執行緒：重訓 + 儲存，完成後通知主執行緒。"""
            try:
                self._Recognizer.FinishLearning()
                SaveOk = self._Recognizer.SaveModel()
            except Exception as Error:
                print(f"[MainApp] 訓練/儲存失敗：{Error}")
                SaveOk = False
            self.after(0, lambda: self._OnTrainDone(LearnName, FrameCount, SaveOk))

        threading.Thread(target=TrainWorker, daemon=True).start()

    def _OnTrainDone(self, LearnName: str, FrameCount: int, SaveOk: bool) -> None:
        """訓練完成後回到主執行緒：啟用按鈕、顯示結果對話框。"""
        self._SetButtonsEnabled(True)
        KnownPersons = self._Recognizer.GetKnownPersons()
        PersonCount  = len(KnownPersons)
        PersonList   = ', '.join(KnownPersons) if KnownPersons else "（無）"

        if SaveOk:
            Msg = (
                f"已完成 [{LearnName}] 的學習！\n"
                f"本次學習 {FrameCount} 次\n\n"
                f"目前已登錄人物（共 {PersonCount} 人）：\n{PersonList}"
            )
        else:
            Msg = (
                f"學習完成，但模型儲存失敗。\n"
                f"本次學習 {FrameCount} 次\n"
                f"目前已登錄人物（共 {PersonCount} 人）：{PersonList}"
            )
        MsgBox.showinfo("學習完成", Msg)
        self._UpdateSummary()

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
                            # Retrain=False：學習期間只收集樣本，不重訓（避免每幀訓練 100 棵樹）
                            Added, KeyPoints = self._Recognizer.AddSample(
                                FrameCopy, PersonName, Retrain=False
                            )
                            self.after(0, lambda A=Added, KP=KeyPoints: self._OnLearnSampleAdded(A, KP))
                        except Exception as Error:
                            print(f"[MainApp] 學習推論失敗：{Error}")
                        finally:
                            self._InferenceActive = False

                    threading.Thread(target=Worker, daemon=True).start()
            except Exception as Error:
                print(f"[MainApp] 學習 tick 失敗：{Error}")

        # 排程下一次 tick
        self.after(LEARN_TICK_MS, self._LearningTick)

    def _OnLearnSampleAdded(self, Added: bool, KeyPoints: list) -> None:
        """學習樣本加入回調（在主執行緒執行）。"""
        if not self._LearnActive:
            return
        if Added:
            self._LearnFrameCount += 1
            self._LblLearnFrames.configure(
                text=f"Number of frames learned: {self._LearnFrameCount}"
            )
            # 更新關鍵點快取，供 _UpdateWebcamView 疊加顯示
            self._LastLearnKeyPoints = KeyPoints

    # --------------------------------------------------------------------------
    # 工具方法：頭部轉角描述
    # --------------------------------------------------------------------------
    def _headPoseDesc(self, YawRatio: float, PitchRatio: float) -> str:
        """將 YawRatio 與 PitchRatio（各 0~1）轉換為可讀的頭部姿態描述字串。"""
        YawDeg   = int(YawRatio   * 60)
        PitchDeg = int(PitchRatio * 30)

        if YawRatio < 0.10:
            YawDesc = "正臉"
        elif YawRatio < 0.30:
            YawDesc = "微轉"
        elif YawRatio < 0.55:
            YawDesc = "中等轉"
        elif YawRatio < 0.80:
            YawDesc = "明顯轉"
        else:
            YawDesc = "側臉"

        if PitchRatio < 0.15:
            PitchDesc = "水平"
        elif PitchRatio < 0.45:
            PitchDesc = "微仰/低"
        elif PitchRatio < 0.75:
            PitchDesc = "中等仰/低"
        else:
            PitchDesc = "大幅仰/低"

        return f"左右：約 {YawDeg}°（{YawDesc}）  上下：約 {PitchDeg}°（{PitchDesc}）"

    # --------------------------------------------------------------------------
    # 閾值調整 Slider 回調
    # --------------------------------------------------------------------------
    def _OnCosineThreshChanged(self, Value: float) -> None:
        """Cosine 相似度閾值 Slider 拖動時，即時更新顯示值與辨識器閾值。"""
        self._LblCosineVal.configure(text=f"{Value:.2f}")
        if self._Recognizer is not None:
            self._Recognizer.SetThresholds(CosineThresh=Value)

    # --------------------------------------------------------------------------
    # 關閉處理
    # --------------------------------------------------------------------------
    def _OnClose(self) -> None:
        """程式關閉前，先停止所有活動並釋放攝影機與 MediaPipe 資源。"""
        print("[MainApp] 程式關閉中...")
        self._DetectNoneActive = False
        self._LearnActive      = False
        self._Webcam.Close()
        if self._Recognizer is not None:
            try:
                self._Recognizer._Detector.close()
            except Exception:
                pass
        self.destroy()


# ==============================================================================
# 程式進入點
# ==============================================================================
if __name__ == "__main__":
    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("blue")
    App = MainApp()
    App.mainloop()
