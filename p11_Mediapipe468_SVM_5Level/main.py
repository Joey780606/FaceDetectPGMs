"""
main.py  (p11)

使用 MediaPipe FaceMesh + 五類姿態 SVM 進行人臉辨識。
透過 CustomTkinter 建立 GUI，可學習並辨識不同人的身份。
訓練與偵測過程均即時顯示當前臉部姿態（置中/左上/右上/左下/右下）。

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
from face_pose_classifier import POSE_NAMES, POSE_NAMES_EN
from svm_classifier_np import SVM_UNKNOWN_THRESH

# ── 應用程式常數 ──────────────────────────────────────────────────────────────
#LEARN_TARGET_FRAMES   = 100    # 學習模式目標收集 frame 數（分5類需較多樣本）
#LEARN_TIMEOUT_SECONDS = 120    # 學習模式最長等待時間（秒）
LEARN_TARGET_FRAMES   = 30    # 學習模式目標收集 frame 數（分5類需較多樣本）
LEARN_TIMEOUT_SECONDS = 30    # 學習模式最長等待時間（秒）
UI_REFRESH_MS         = 30     # webcam 畫面更新間隔（毫秒）
LEARN_TICK_MS         = 500    # 學習時每次抓 frame 的間隔（每秒 2 個樣本）
DETECT_TICK_MS        = 300    # 偵測時每次推論的間隔
DETECT_NONE_DETECT_TARGET = 5  # 滑動窗口多數決所需幀數


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
        """背景執行緒：持續讀取 frame 並更新共用緩衝區。"""
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
    """主應用程式視窗，整合 webcam 與五類姿態 SVM 人臉辨識。"""

    def __init__(self, mode="normal", pet=None):
        super().__init__()

        # DetectNone 模式狀態
        self._DetectNoneActive  = False
        self._DetectNoneDtNames = []

        # 學習模式狀態
        self._LearnActive    = False
        self._LearnStartTime = 0.0
        self._LearnName      = ""
        self._LearnFrameCount = 0

        # 推論執行緒防重入旗標
        self._InferenceActive = False

        # 人臉偵測結果快取（供 _UpdateWebcamView 繪製框）
        self._LastDetections = []

        # 學習關鍵點快取（供學習中疊加顯示）
        self._LastLearnKeyPoints = []

        # UI 圖像參照（防止 GC 回收）
        self._CurrentPhotoImage = None

        # 核心元件
        self._Webcam     = WebcamManager(CameraIndex=0)
        self._Recognizer = FaceRecognizer()

        self.mode = mode
        self.pet  = pet
        self._BuildUI(self.mode)
        self._InitComponents()

    # ──────────────────────────────────────────────────────────────────────────
    # UI 建立
    # ──────────────────────────────────────────────────────────────────────────
    def _BuildUI(self, mode) -> None:
        """建立 CustomTkinter UI 介面。"""
        self.title("人臉辨識系統（MediaPipe 五類姿態 SVM）")
        self.protocol("WM_DELETE_WINDOW", self._OnClose)
        self.resizable(True, True)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # ── Row 0：偵測功能列 ────────────────────────────────────────────────
        Row0 = customtkinter.CTkFrame(self)
        Row0.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        Row0.grid_columnconfigure(0, weight=1)

        Row0Top = customtkinter.CTkFrame(Row0)
        Row0Top.grid(row=0, column=0, sticky="ew")

        self._BtnDetectNone = customtkinter.CTkButton(
            Row0Top, text="Detect", width=120,
            command=self._OnBtnDetectNone
        )
        self._BtnDetectNone.pack(side="left", padx=(5, 10), pady=5)

        # 姿態即時顯示（訓練與偵測共用）
        self._LblPoseStatus = customtkinter.CTkLabel(
            Row0Top, text="姿態：---",
            font=customtkinter.CTkFont(size=14, weight="bold"),
            width=110,
        )
        self._LblPoseStatus.pack(side="left", padx=(0, 10), pady=5)

        self._LblDetectName = customtkinter.CTkLabel(
            Row0Top, text="",
            font=customtkinter.CTkFont(size=16, weight="bold")
        )
        self._LblDetectName.pack(side="left", padx=5, pady=5)

        self._LblBufferInfo = customtkinter.CTkLabel(Row0, text="", anchor="w")
        self._LblBufferInfo.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))

        # ── Row 1：學習功能列 ────────────────────────────────────────────────
        Row1 = customtkinter.CTkFrame(self)
        Row1.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        Row1.grid_columnconfigure(0, weight=1)

        Row1Top = customtkinter.CTkFrame(Row1)
        Row1Top.grid(row=0, column=0, sticky="ew")

        self._TblMyName = customtkinter.CTkEntry(
            Row1Top, placeholder_text="輸入姓名", width=200
        )
        self._TblMyName.pack(side="left", padx=(5, 10), pady=5)

        self._BtnLearn = customtkinter.CTkButton(
            Row1Top, text="Learning", width=120,
            command=self._OnBtnLearn
        )
        self._BtnLearn.pack(side="left", padx=5, pady=5)

        self._BtnRemove = customtkinter.CTkButton(
            Row1Top, text="Remove", width=100,
            fg_color="#8B0000", hover_color="#B22222",
            command=self._OnBtnRemove
        )
        self._BtnRemove.pack(side="left", padx=(5, 5), pady=5)

        # 學習進度區（預設隱藏，學習中顯示）
        self._Row1Bot = customtkinter.CTkFrame(Row1)
        self._Row1Bot.grid(row=1, column=0, sticky="ew")
        self._Row1Bot.grid_remove()
        Row1Bot = self._Row1Bot
        Row1Bot.grid_columnconfigure(0, weight=1)
        Row1Bot.grid_columnconfigure(1, weight=1)

        # Col 0：剩餘秒數 + 時間進度條
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

        # Col 1：各姿態收集數量 + frame 進度條
        Col1 = customtkinter.CTkFrame(Row1Bot)
        Col1.grid(row=0, column=1, sticky="ew", padx=(3, 5), pady=5)
        Col1.grid_columnconfigure(0, weight=1)

        self._LblLearnFrames = customtkinter.CTkLabel(
            Col1, text="已收集 0 張：置中:0 左上:0 右上:0 左下:0 右下:0"
        )
        self._LblLearnFrames.grid(row=0, column=0, sticky="w", padx=5, pady=(5, 2))

        self._PbarFrames = customtkinter.CTkProgressBar(Col1)
        self._PbarFrames.grid(row=1, column=0, sticky="ew", padx=5, pady=(2, 5))
        self._PbarFrames.set(0)

        # ── Row 2：信心度閾值 Slider ─────────────────────────────────────────
        RowThresh = customtkinter.CTkFrame(self)
        RowThresh.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        RowThresh.grid_columnconfigure(1, weight=1)

        customtkinter.CTkLabel(RowThresh, text="SVM 信心度閾值", anchor="w").grid(
            row=0, column=0, sticky="w", padx=(8, 4), pady=8
        )
        self._SldCosine = customtkinter.CTkSlider(
            RowThresh, from_=0.10, to=0.99,
            number_of_steps=890,
            command=self._OnCosineThreshChanged
        )
        self._SldCosine.set(SVM_UNKNOWN_THRESH)
        self._SldCosine.grid(row=0, column=1, sticky="ew", padx=4, pady=8)
        self._LblCosineVal = customtkinter.CTkLabel(
            RowThresh, text=f"{SVM_UNKNOWN_THRESH:.2f}", width=44, anchor="e"
        )
        self._LblCosineVal.grid(row=0, column=2, padx=(4, 8), pady=8)

        # ── Row 3：Webcam 畫面 ───────────────────────────────────────────────
        Row2 = customtkinter.CTkFrame(self)
        Row2.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
        Row2.grid_columnconfigure(0, weight=1)
        Row2.grid_rowconfigure(0, weight=1)

        self._WebcamCanvas = customtkinter.CTkLabel(
            Row2, text="攝影機畫面載入中...", width=640, height=480
        )
        self._WebcamCanvas.grid(row=0, column=0, sticky="nsew")

        # ── Row 4：Log ───────────────────────────────────────────────────────
        Row3 = customtkinter.CTkFrame(self)
        Row3.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 10))
        Row3.grid_columnconfigure(0, weight=1)

        self._TxtLog = customtkinter.CTkTextbox(Row3, height=120, state="disabled")
        self._TxtLog.grid(row=0, column=0, sticky="ew", padx=10, pady=(4, 8))

        # ── Mode 調整 ────────────────────────────────────────────────────────
        if mode == "demo":
            self._BtnLearn.pack_forget()
            self._TblMyName.pack_forget()
            Row1.grid_remove()
        elif mode == "learning":
            self._BtnDetectNone.pack_forget()
            Row0.grid_remove()

    # ──────────────────────────────────────────────────────────────────────────
    # 初始化
    # ──────────────────────────────────────────────────────────────────────────
    def _InitComponents(self) -> None:
        """初始化攝影機與人臉辨識模型。"""
        CamOk = self._Webcam.Open()
        if not CamOk:
            MsgBox.showwarning(
                "攝影機錯誤",
                "無法開啟攝影機，請確認連線後重新啟動。\n應用程式將以無攝影機模式執行。"
            )
        try:
            self._Recognizer.LoadModel()
            self._UpdateSummary()
        except Exception as Error:
            print(f"[MainApp] 模型載入：{Error}")

        self.after(UI_REFRESH_MS, self._UpdateWebcamView)

    # ──────────────────────────────────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────────────────────────────────
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
        import datetime
        Timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        Line = f"[{Timestamp}] {Msg}\n"
        self._TxtLog.configure(state="normal")
        self._TxtLog.insert("0.0", Line)
        self._TxtLog.configure(state="disabled")

    def _UpdatePoseLabel(self, PoseCat: int,
                         Yaw: float = None, Pitch: float = None) -> None:
        """更新姿態即時顯示標籤（訓練與偵測共用）。附帶 Yaw/Pitch 原始值供調閾值參考。"""
        try:
            PoseName = POSE_NAMES[PoseCat] if 0 <= PoseCat < len(POSE_NAMES) else "---"
            if Yaw is not None and Pitch is not None:
                Text = f"姿態：{PoseName} (Y:{Yaw:+.2f} P:{Pitch:+.2f})"
            else:
                Text = f"姿態：{PoseName}"
            self._LblPoseStatus.configure(text=Text)
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    # Webcam 畫面更新迴圈（每 30ms）
    # ──────────────────────────────────────────────────────────────────────────
    def _UpdateWebcamView(self) -> None:
        """取得最新 frame，疊加偵測框後顯示於 UI。"""
        try:
            Ok, Frame = self._Webcam.GetLatestFrame()
            if Ok and Frame is not None:
                if self._LastDetections:
                    Frame = self._DrawDetections(Frame, self._LastDetections)

                if self._LearnActive and self._LastLearnKeyPoints:
                    Frame = self._DrawKeyPoints(Frame, self._LastLearnKeyPoints)

                FrameRgb = cv2.cvtColor(Frame, cv2.COLOR_BGR2RGB)
                Img = Image.fromarray(FrameRgb)

                W = min(self._WebcamCanvas.winfo_width(),  640)
                H = min(self._WebcamCanvas.winfo_height(), 480)
                DisplaySize = (W, H) if W > 1 and H > 1 else (
                    min(Img.width, 640), min(Img.height, 480)
                )

                Photo = customtkinter.CTkImage(light_image=Img, size=DisplaySize)
                self._WebcamCanvas.configure(image=Photo, text="")
                self._CurrentPhotoImage = Photo

        except Exception as Error:
            print(f"[MainApp] 更新畫面失敗：{Error}")
        finally:
            self.after(UI_REFRESH_MS, self._UpdateWebcamView)

    def _DrawDetections(self, Frame: np.ndarray, Detections: list) -> np.ndarray:
        """
        在 Frame 上繪製人臉偵測框與姓名標籤。
        已知人物用綠框，Unknown 用紅框，框內顯示姓名、信心度與姿態。
        """
        DrawFrame = Frame.copy()
        for Top, Right, Bottom, Left, Name, Confidence, PoseCat, _Yaw, _Pitch in Detections:
            Color     = (0, 255, 0) if Name != "Unknown" else (0, 0, 255)
            PoseShort = POSE_NAMES_EN[PoseCat] if 0 <= PoseCat < len(POSE_NAMES_EN) else "?"
            cv2.rectangle(DrawFrame, (Left, Top), (Right, Bottom), Color, 2)
            if Name != "Unknown":
                Label = f"{Name} ({Confidence:.2f}) [{PoseShort}]"
            else:
                Label = f"Unknown [{PoseShort}]"
            cv2.rectangle(DrawFrame, (Left, Bottom - 25), (Right, Bottom), Color, cv2.FILLED)
            cv2.putText(DrawFrame, Label, (Left + 6, Bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return DrawFrame

    def _DrawKeyPoints(self, Frame: np.ndarray, KeyPointsList: list) -> np.ndarray:
        """在 Frame 上繪製學習時偵測到的雙眼、鼻子、嘴巴中心點。"""
        DrawFrame = Frame.copy()
        StyleMap = {
            'left_eye':  ((255, 180,   0), "L"),
            'right_eye': ((255, 180,   0), "R"),
            'nose':      ((  0, 220,   0), "N"),
            'mouth':     ((  0,   0, 220), "M"),
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

    # ──────────────────────────────────────────────────────────────────────────
    # DetectNone 功能
    # ──────────────────────────────────────────────────────────────────────────
    def _OnBtnDetectNone(self) -> None:
        """Detect 按鈕點擊：切換啟動 / 停止。"""
        if self._DetectNoneActive:
            self._DetectNoneActive  = False
            self._LastDetections    = []
            self._BtnDetectNone.configure(text="Detect", state="normal")
            self._LblDetectName.configure(text="")
            self._LblPoseStatus.configure(text="姿態：---")
            return

        if not self._Recognizer.CanDetect():
            self._LblDetectName.configure(text="沒有資料，請先訓練")
            return

        self._DetectNoneActive  = True
        self._DetectNoneDtNames = []
        self._BtnDetectNone.configure(text="Stop", state="normal")
        self._LblDetectName.configure(
            text=f"偵測中.... (0/{DETECT_NONE_DETECT_TARGET})"
        )
        self.after(0, self._DetectNoneTick)

    def _DetectNoneTick(self) -> None:
        """偵測 tick：在背景執行緒執行 MediaPipe 推論，避免阻塞 UI。"""
        if not self._DetectNoneActive:
            return

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
        """偵測結果回調（主執行緒）。"""
        if not self._DetectNoneActive:
            return

        if Results:
            self._LastDetections = Results
            BestResult = max(Results, key=lambda R: R[5])
            Name    = BestResult[4]
            PoseCat = BestResult[6]
            Yaw     = BestResult[7]
            Pitch   = BestResult[8]

            # 即時更新姿態顯示（含原始數值）
            self._UpdatePoseLabel(PoseCat, Yaw, Pitch)

            # 滑動窗口多數決
            self._DetectNoneDtNames.append(Name)
            if len(self._DetectNoneDtNames) > DETECT_NONE_DETECT_TARGET:
                self._DetectNoneDtNames.pop(0)

            if len(self._DetectNoneDtNames) >= DETECT_NONE_DETECT_TARGET:
                BestName = Counter(self._DetectNoneDtNames).most_common(1)[0][0]
                if BestName == "Unknown":
                    self._LblDetectName.configure(text="您好，我不認識你，我可以認識你嗎？")
                    if self.pet:
                        self.pet.face_detected.emit("Sorry, I don't know you...")
                else:
                    self._LblDetectName.configure(text=f"Hi, {BestName} 您好，又見到您了")
                    if self.pet:
                        self.pet.face_detected.emit(f"Hi, {BestName} !")
            else:
                self._LblDetectName.configure(
                    text=f"偵測中.... ({len(self._DetectNoneDtNames)}/{DETECT_NONE_DETECT_TARGET})"
                )
        else:
            self._LastDetections = []

    # ──────────────────────────────────────────────────────────────────────────
    # 學習功能
    # ──────────────────────────────────────────────────────────────────────────
    def _OnBtnLearn(self) -> None:
        """Learning 按鈕點擊。"""
        if self._LearnActive:
            return
        PersonName = self._TblMyName.get().strip()
        if not PersonName:
            MsgBox.showwarning("缺少姓名", "請先在姓名欄位輸入要學習的姓名。")
            return
        self._StartLearning(PersonName)

    def _OnBtnRemove(self) -> None:
        """Remove 按鈕點擊：移除指定人物的訓練資料。"""
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
        """開始學習模式。"""
        if self._DetectNoneActive:
            self._DetectNoneActive = False
            self._LastDetections   = []
            self._BtnDetectNone.configure(text="Detect", state="normal")
            self._LblDetectName.configure(text="")

        self._LearnActive     = True
        self._LearnName       = PersonName
        self._LearnStartTime  = time.time()
        self._LearnFrameCount = 0

        self._BtnLearn.configure(state="disabled")
        self._LblRemain.configure(
            text=f"Remaining study seconds: {LEARN_TIMEOUT_SECONDS}"
            "  ← 請慢慢左右、上下轉動頭部，讓五種姿態都能收集到樣本"
        )
        self._LblLearnFrames.configure(
            text="已收集 0 張：置中:0 左上:0 右上:0 左下:0 右下:0"
        )
        self._PbarTime.set(0)
        self._PbarFrames.set(0)
        self._Row1Bot.grid()

        self.after(0, self._LearningTick)
        if self.pet:
            self.pet.face_detected.emit("Start learning!")

    def _StopLearning(self) -> None:
        """結束學習模式，儲存模型。"""
        self._LearnActive        = False
        self._LastDetections     = []
        self._LastLearnKeyPoints = []
        self._LblRemain.configure(text="Remaining study seconds: --")
        self._LblLearnFrames.configure(
            text="已收集 0 張：置中:0 左上:0 右上:0 左下:0 右下:0"
        )
        self._PbarTime.set(0)
        self._PbarFrames.set(0)
        self._Row1Bot.grid_remove()
        self._BtnLearn.configure(state="normal")
        self._LblPoseStatus.configure(text="姿態：---")

        self._Recognizer.FinishLearning()

        SaveOk       = self._Recognizer.SaveModel()
        KnownPersons = self._Recognizer.GetKnownPersons()
        PersonCount  = len(KnownPersons)
        PersonList   = ', '.join(KnownPersons) if KnownPersons else "（無）"

        # 顯示各姿態收集統計
        PoseCounts = self._Recognizer.GetPersonPoseCounts(self._LearnName)
        PoseDetail = "  ".join(
            f"{POSE_NAMES[c]}:{PoseCounts.get(c, 0)}"
            for c in range(5)
        )

        if SaveOk:
            Msg = (
                f"已完成 [{self._LearnName}] 的學習！\n"
                f"本次學習 {self._LearnFrameCount} 幀\n"
                f"各姿態：{PoseDetail}\n\n"
                f"目前已登錄人物（共 {PersonCount} 人）：\n{PersonList}"
            )
        else:
            Msg = (
                f"學習完成，但模型儲存失敗。\n"
                f"本次學習 {self._LearnFrameCount} 幀\n"
                f"各姿態：{PoseDetail}\n"
                f"目前已登錄人物（共 {PersonCount} 人）：{PersonList}"
            )
        MsgBox.showinfo("學習完成", Msg)

        self._UpdateSummary()
        if self.pet:
            self.pet.face_detected.emit("Finished learning!")

    def _LearningTick(self) -> None:
        """學習模式 tick（每 500ms）：收集人臉樣本，更新進度。"""
        if not self._LearnActive:
            return

        Elapsed = time.time() - self._LearnStartTime
        Remain  = max(0, LEARN_TIMEOUT_SECONDS - int(Elapsed))
        self._LblRemain.configure(text=f"Remaining study seconds: {Remain}")
        self._PbarTime.set(min(1.0, Elapsed / LEARN_TIMEOUT_SECONDS))
        self._PbarFrames.set(min(1.0, self._LearnFrameCount / LEARN_TARGET_FRAMES))

        if self._LearnFrameCount >= LEARN_TARGET_FRAMES or Elapsed >= LEARN_TIMEOUT_SECONDS:
            self._StopLearning()
            return

        if not self._InferenceActive:
            try:
                Ok, Frame = self._Webcam.GetLatestFrame()
                if Ok and Frame is not None:
                    self._InferenceActive = True
                    FrameCopy  = Frame.copy()
                    PersonName = self._LearnName

                    def Worker():
                        try:
                            Added, KP, PoseCat, Yaw, Pitch = self._Recognizer.AddSample(
                                FrameCopy, PersonName, Retrain=False
                            )
                            self.after(0, lambda A=Added, K=KP, P=PoseCat, Y=Yaw, Pi=Pitch:
                                       self._OnLearnSampleAdded(A, K, P, Y, Pi))
                        except Exception as Error:
                            print(f"[MainApp] 學習推論失敗：{Error}")
                        finally:
                            self._InferenceActive = False

                    threading.Thread(target=Worker, daemon=True).start()
            except Exception as Error:
                print(f"[MainApp] 學習 tick 失敗：{Error}")

        self.after(LEARN_TICK_MS, self._LearningTick)

    def _OnLearnSampleAdded(self, Added: bool, KeyPoints: list,
                            PoseCat: int, Yaw: float = 0.0,
                            Pitch: float = 0.0) -> None:
        """學習樣本加入回調（主執行緒）。"""
        if not self._LearnActive:
            return
        if Added:
            self._LearnFrameCount += 1
            self._LastLearnKeyPoints = KeyPoints

            # 更新姿態顯示（含原始數值，方便確認閾值是否適當）
            self._UpdatePoseLabel(PoseCat, Yaw, Pitch)

            # 更新各姿態收集數量
            PoseCounts = self._Recognizer.GetPersonPoseCounts(self._LearnName)
            PoseStr    = "  ".join(
                f"{POSE_NAMES[c]}:{PoseCounts.get(c, 0)}" for c in range(5)
            )
            self._LblLearnFrames.configure(
                text=f"已收集 {self._LearnFrameCount} 張：{PoseStr}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # 閾值調整 Slider 回調
    # ──────────────────────────────────────────────────────────────────────────
    def _OnCosineThreshChanged(self, Value: float) -> None:
        """SVM 信心度閾值 Slider 拖動時，即時更新顯示值與辨識器閾值。"""
        self._LblCosineVal.configure(text=f"{Value:.2f}")
        if self._Recognizer is not None:
            self._Recognizer.SetThresholds(CosineThresh=Value)

    # ──────────────────────────────────────────────────────────────────────────
    # 關閉處理
    # ──────────────────────────────────────────────────────────────────────────
    def _OnClose(self) -> None:
        """程式關閉前，釋放攝影機與 MediaPipe 資源。"""
        print("[MainApp] 程式關閉中...")
        self._DetectNoneActive = False
        self._LearnActive      = False
        self._Webcam.Close()
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
