# face_annotator.py — 人臉自動標點工具主程式
# 使用 face_recognition 的68個關鍵點，以 YOLO 格式輸出標記檔
# UI: CustomTkinter

import os
import glob
import time
import threading
from tkinter import filedialog

import customtkinter as ctk
import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── 常數 ──────────────────────────────────────────────────────────────────────

# 支援的圖片副檔名
SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# 側臉判斷閾值（鼻尖相對兩眼中點的偏移比例）
PROFILE_OFFSET_THRESHOLD = 0.20
# 歪頭判斷閾值（兩眼Y軸差異 / 兩眼間距）：歪頭時此值偏大
EYE_TILT_THRESHOLD = 0.15

# 圖片顯示區域最大尺寸
PIC_AREA_MAX_W = 800
PIC_AREA_MAX_H = 600

# 各類別顏色（BGR→RGB 16進位）
CLASS_COLORS = {
    0: '#00FF00',   # 左眼 — 綠
    1: '#00FFFF',   # 右眼 — 青
    2: '#FFFF00',   # 左眉 — 黃
    3: '#FF8000',   # 右眉 — 橙
    4: '#FF00FF',   # 鼻子 — 洋紅
    5: '#FF0000',   # 嘴巴 — 紅
    6: '#FFFFFF',   # 髮際 — 白
    7: '#4080FF',   # 下巴 — 藍
}

# 類別名稱（用於圖上標籤）
CLASS_NAMES = {
    0: '左眼', 1: '右眼',
    2: '左眉', 3: '右眉',
    4: '鼻',   5: '嘴',
    6: '髮際', 7: '下巴',
}

# 68點的顏色
LANDMARK_DOT_COLOR = '#FFD700'  # 金黃色
LANDMARK_DOT_RADIUS = 3

# ── MediaPipe Face Mesh（全域 Lazy Load）─────────────────────────────────────
# MediaPipe Face Mesh landmark 10：額頭頂部中心，最接近髮際線的中心點
MP_HAIRLINE_LANDMARK_INDEX = 10

_MpFaceMesh = None   # mediapipe FaceMesh 實例
_MpReady    = False  # 是否已初始化


# ── 模組層級純函式 ─────────────────────────────────────────────────────────────

def get_face_landmarks(NpImage: np.ndarray) -> list:
    """呼叫 face_recognition 取得68個關鍵點，失敗時回傳空列表"""
    try:
        return face_recognition.face_landmarks(NpImage)
    except Exception as E:
        return []


def detect_face_pose(Landmarks: dict, ImgW: int, ImgH: int) -> tuple:
    """
    判斷臉部姿態。
    回傳 (臉型: str, 估算角度: float)
    臉型: 'normal'（正常）| 'side'（側臉）| 'tilt'（歪頭）

    判斷邏輯：
    1. 兩眼Y軸差異 > EYE_TILT_THRESHOLD → 歪頭
    2. 鼻尖水平偏移 > PROFILE_OFFSET_THRESHOLD → 側臉
    3. 兩者皆未超過 → 正常
    """
    try:
        LeftEyePts  = Landmarks['left_eye']
        RightEyePts = Landmarks['right_eye']
        # 左眼中心
        LeftEyeX = sum(P[0] for P in LeftEyePts) / len(LeftEyePts)
        LeftEyeY = sum(P[1] for P in LeftEyePts) / len(LeftEyePts)
        # 右眼中心
        RightEyeX = sum(P[0] for P in RightEyePts) / len(RightEyePts)
        RightEyeY = sum(P[1] for P in RightEyePts) / len(RightEyePts)
        # 兩眼間距
        EyeSpan = abs(RightEyeX - LeftEyeX)
        if EyeSpan < 1:
            return 'normal', 0.0
        # 兩眼Y軸差異比例
        EyeYDiff = abs(LeftEyeY - RightEyeY) / EyeSpan
        # 鼻尖水平偏移比例（使用 nose_tip[2] 中心單點）
        NoseTipX = Landmarks['nose_tip'][2][0]
        EyeMidX  = (LeftEyeX + RightEyeX) / 2.0
        NoseOffset = abs(NoseTipX - EyeMidX) / EyeSpan
        #print(f'1-1 Eye: EyeYDiff={EyeYDiff:.4f}, EYE_TILT_THRESHOLD={EYE_TILT_THRESHOLD:.4f}')
        #print(f'1-2 Nose: NoseOffset={NoseOffset:.4f}, PROFILE_OFFSET_THRESHOLD={PROFILE_OFFSET_THRESHOLD:.4f}')
        # 先判歪頭，再判側臉
        if EyeYDiff > EYE_TILT_THRESHOLD:
            AngleDeg = min(EyeYDiff * 180.0, 90.0)
            #print(f'Info: tilt AngleDeg={AngleDeg:.2f}')
            return 'tilt', AngleDeg   # 歪頭
        if NoseOffset > PROFILE_OFFSET_THRESHOLD:
            AngleDeg = min(NoseOffset * 180.0, 90.0)
            #print(f'Info: side AngleDeg={AngleDeg:.2f}')
            return 'side', AngleDeg   # 側臉
        return 'normal', 0.0
    except Exception:
        return 'normal', 0.0


def calc_bounding_box(PointList: list) -> tuple:
    """從點列表計算最小外接矩形，回傳 (x1, y1, x2, y2)"""
    Xs = [P[0] for P in PointList]
    Ys = [P[1] for P in PointList]
    return int(min(Xs)), int(min(Ys)), int(max(Xs)), int(max(Ys))


def calc_eight_class_boxes(Landmarks: dict, ImgW: int, ImgH: int) -> list:
    """
    計算8個 YOLO 類別的邊框。
    回傳 list of dict: [{'class': N, 'x1':..., 'y1':..., 'x2':..., 'y2':...}, ...]
    """
    結果 = []
    # 邊框 padding (px)
    PAD = 4

    def _add_box(ClassId, PointList):
        if not PointList:
            return
        X1, Y1, X2, Y2 = calc_bounding_box(PointList)
        # 加 padding，並 clamp 至圖片邊界
        X1 = max(0, X1 - PAD)
        Y1 = max(0, Y1 - PAD)
        X2 = min(ImgW - 1, X2 + PAD)
        Y2 = min(ImgH - 1, Y2 + PAD)
        結果.append({'class': ClassId, 'x1': X1, 'y1': Y1, 'x2': X2, 'y2': Y2})

    # 類別 0：左眼
    _add_box(0, Landmarks.get('left_eye', []))
    # 類別 1：右眼
    _add_box(1, Landmarks.get('right_eye', []))
    # 類別 2：左眉
    _add_box(2, Landmarks.get('left_eyebrow', []))
    # 類別 3：右眉
    _add_box(3, Landmarks.get('right_eyebrow', []))
    # 類別 4：鼻子（鼻樑 + 鼻尖）
    NosePoints = Landmarks.get('nose_bridge', []) + Landmarks.get('nose_tip', [])
    _add_box(4, NosePoints)
    # 類別 5：嘴巴（上唇 + 下唇）
    MouthPoints = Landmarks.get('top_lip', []) + Landmarks.get('bottom_lip', [])
    _add_box(5, MouthPoints)

    # 類別 6：髮際線最低中心點（幾何估算）
    HairlineBox = _estimate_hairline_box(Landmarks, ImgW, ImgH)
    if HairlineBox:
        結果.append(HairlineBox)

    # 類別 7：下巴最低中心點
    ChinBox = _estimate_chin_box(Landmarks, ImgW, ImgH)
    if ChinBox:
        結果.append(ChinBox)

    return 結果


def _estimate_hairline_box(Landmarks: dict, ImgW: int, ImgH: int) -> dict | None:
    """估算髮際線最低中心點的邊框（類別6）"""
    try:
        LeftBrow = Landmarks.get('left_eyebrow', [])
        RightBrow = Landmarks.get('right_eyebrow', [])
        NoseBridge = Landmarks.get('nose_bridge', [])
        LeftEye = Landmarks.get('left_eye', [])
        RightEye = Landmarks.get('right_eye', [])

        if not LeftBrow or not RightBrow or not NoseBridge:
            return None

        # 眉毛最高點 Y（Y 越小 = 越靠上）
        AllBrowY = [P[1] for P in LeftBrow + RightBrow]
        BrowTopY = min(AllBrowY)

        # 鼻樑最上點 Y
        NoseBridgeTopY = NoseBridge[0][1]

        # 髮際 Y 估算：眉毛上方 1.2 倍的眉鼻距
        EyeToNoseDist = abs(NoseBridgeTopY - BrowTopY)
        HairlineY = int(BrowTopY - EyeToNoseDist * 1.2)

        # 髮際 X：兩眼中心的平均
        LeftEyeCx = sum(P[0] for P in LeftEye) / len(LeftEye) if LeftEye else ImgW / 2
        RightEyeCx = sum(P[0] for P in RightEye) / len(RightEye) if RightEye else ImgW / 2
        HairlineCx = int((LeftEyeCx + RightEyeCx) / 2)

        # 先把中心點 clamp 至圖片範圍內，避免邊框座標反轉
        HairlineCx = max(0, min(HairlineCx, ImgW - 1))
        HairlineY  = max(0, min(HairlineY,  ImgH - 1))
        # 固定邊框大小：正規化後 width=height=0.025（僅作為參考點）
        HalfW = max(1, int(ImgW * 0.025 / 2))
        HalfH = max(1, int(ImgH * 0.025 / 2))
        X1 = max(0, HairlineCx - HalfW)
        Y1 = max(0, HairlineY  - HalfH)
        X2 = min(ImgW - 1, HairlineCx + HalfW)
        Y2 = min(ImgH - 1, HairlineY  + HalfH)
        return {'class': 6, 'x1': X1, 'y1': Y1, 'x2': X2, 'y2': Y2}
    except Exception:
        return None


def _estimate_chin_box(Landmarks: dict, ImgW: int, ImgH: int) -> dict | None:
    """取 chin[8] 作為下巴最低中心點（類別7）"""
    try:
        ChinPts = Landmarks.get('chin', [])
        if len(ChinPts) < 9:
            return None
        # chin[8] 為68點模型的下巴最低中心點
        ChinPoint = ChinPts[8]
        # 固定邊框大小：正規化後 width=height=0.025（僅作為參考點）
        HalfW = int(ImgW * 0.025 / 2)
        HalfH = int(ImgH * 0.025 / 2)
        X1 = max(0, ChinPoint[0] - HalfW)
        Y1 = max(0, ChinPoint[1] - HalfH)
        X2 = min(ImgW - 1, ChinPoint[0] + HalfW)
        Y2 = min(ImgH - 1, ChinPoint[1] + HalfH)
        return {'class': 7, 'x1': X1, 'y1': Y1, 'x2': X2, 'y2': Y2}
    except Exception:
        return None


def _load_mediapipe_model() -> bool:
    """
    初始化 MediaPipe FaceLandmarker（Tasks API，適用 mediapipe >= 0.10）。
    首次呼叫時若模型檔不存在，自動下載（約1MB）。
    回傳 True 表示已就緒。
    """
    global _MpFaceMesh, _MpReady
    if _MpReady:
        return True
    try:
        import urllib.request
        import mediapipe as mp

        # 模型檔存放於程式同目錄
        ModelPath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'face_landmarker.task',
        )

        # 首次使用時自動下載模型檔（約1MB）
        if not os.path.exists(ModelPath):
            ModelUrl = (
                'https://storage.googleapis.com/mediapipe-models/'
                'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            )
            urllib.request.urlretrieve(ModelUrl, ModelPath)

        Options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=ModelPath),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        _MpFaceMesh = mp.tasks.vision.FaceLandmarker.create_from_options(Options)
        _MpReady = True
        return True
    except Exception:
        return False


def detect_hairline_box(PilImage: Image.Image,
                  ImgW: int, ImgH: int) -> dict | None:
    """
    使用 MediaPipe FaceLandmarker landmark 10 精確定位髮際線最低中心點（類別6）。
    Apache 2.0 授權，可商用。失敗時回傳 None，由外層 fallback 至幾何估算。
    """
    try:
        import mediapipe as mp

        # MediaPipe Tasks API 需要 mp.Image 格式
        NpRgb = np.array(PilImage)
        MpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=NpRgb)
        Result = _MpFaceMesh.detect(MpImage)

        if not Result.face_landmarks:
            return None

        # 取 landmark 10（額頭頂部中心，最靠近髮際的中心點）
        Lm = Result.face_landmarks[0][MP_HAIRLINE_LANDMARK_INDEX]
        HairlineX = int(Lm.x * ImgW)
        HairlineY = int(Lm.y * ImgH)

        # clamp 至圖片範圍
        HairlineX = max(0, min(HairlineX, ImgW - 1))
        HairlineY = max(0, min(HairlineY, ImgH - 1))

        # 固定邊框大小：正規化後 width=height=0.025（僅作為參考點）
        HalfW = max(1, int(ImgW * 0.025 / 2))
        HalfH = max(1, int(ImgH * 0.025 / 2))
        X1 = max(0, HairlineX - HalfW)
        Y1 = max(0, HairlineY - HalfH)
        X2 = min(ImgW - 1, HairlineX + HalfW)
        Y2 = min(ImgH - 1, HairlineY + HalfH)
        return {'class': 6, 'x1': X1, 'y1': Y1, 'x2': X2, 'y2': Y2}
    except Exception:
        return None


def convert_to_yolo(X1: int, Y1: int, X2: int, Y2: int,
                   ImgW: int, ImgH: int) -> tuple:
    """像素座標轉換為 YOLO 正規化格式，回傳 (xc, yc, w, h)"""
    # clamp
    X1 = max(0, min(X1, ImgW - 1))
    Y1 = max(0, min(Y1, ImgH - 1))
    X2 = max(0, min(X2, ImgW - 1))
    Y2 = max(0, min(Y2, ImgH - 1))

    Xc = (X1 + X2) / 2.0 / ImgW
    Yc = (Y1 + Y2) / 2.0 / ImgH
    W  = (X2 - X1) / ImgW
    H  = (Y2 - Y1) / ImgH
    return Xc, Yc, W, H


def write_yolo_file(BBoxList: list, OutputPath: str,
                  ImgW: int, ImgH: int) -> bool:
    """將邊框列表以 YOLO 格式寫入檔案，成功回傳 True"""
    try:
        with open(OutputPath, 'w', encoding='utf-8') as F:
            for Item in sorted(BBoxList, key=lambda X: X['class']):
                Xc, Yc, W, H = convert_to_yolo(
                    Item['x1'], Item['y1'], Item['x2'], Item['y2'], ImgW, ImgH)
                F.write(f"{Item['class']} {Xc:.6f} {Yc:.6f} {W:.6f} {H:.6f}\n")
        return True
    except Exception as E:
        return False


def draw_landmarks_and_boxes(PilImage: Image.Image,
                   Landmarks: dict,
                   BBoxList: list,
                   FaceType: str) -> Image.Image:
    """
    在圖片副本上繪製68個關鍵點（金黃色點）及8類邊框（各類別顏色）。
    回傳標注後的 PIL Image。
    """
    AnnotatedImg = PilImage.copy()
    Draw = ImageDraw.Draw(AnnotatedImg)

    # 繪製68個關鍵點
    AllLandmarkKeys = [
        'chin', 'left_eyebrow', 'right_eyebrow',
        'nose_bridge', 'nose_tip',
        'left_eye', 'right_eye',
        'top_lip', 'bottom_lip',
    ]
    R = LANDMARK_DOT_RADIUS
    for Key in AllLandmarkKeys:
        for Pt in Landmarks.get(Key, []):
            X, Y = Pt
            Draw.ellipse(
                [X - R, Y - R, X + R, Y + R],
                fill=LANDMARK_DOT_COLOR,
                outline=LANDMARK_DOT_COLOR,
            )

    # 繪製8類邊框
    for Item in BBoxList:
        Cid = Item['class']
        Color = CLASS_COLORS.get(Cid, '#FFFFFF')
        # 防護：跳過無效邊框（座標反轉時 PIL 會 raise）
        if Item['x2'] <= Item['x1'] or Item['y2'] <= Item['y1']:
            continue
        Draw.rectangle(
            [Item['x1'], Item['y1'], Item['x2'], Item['y2']],
            outline=Color,
            width=2,
        )
        # 類別標籤
        LabelText = CLASS_NAMES.get(Cid, str(Cid))
        Draw.text(
            (Item['x1'] + 2, Item['y1'] + 1),
            LabelText,
            fill=Color,
        )

    # 若姿態異常，在圖上加提示文字
    if FaceType == 'side':
        Draw.text((10, 10), '⚠ 側臉', fill='#FF4444')
    elif FaceType == 'tilt':
        Draw.text((10, 10), '⚠ 歪頭', fill='#FF8800')

    return AnnotatedImg


# ── UI 主類別 ──────────────────────────────────────────────────────────────────

class FaceAnnotatorApp(ctk.CTk):
    """人臉自動標點工具主視窗"""

    def __init__(self):
        super().__init__()
        self.title('人臉自動標點工具')
        self.minsize(900, 750)

        # 保留 CTkImage 參考，防止被垃圾回收
        self._CurrentImage = None

        self._build_ui()
        self._setup_layout()

    # ── 建立介面 ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        """建立所有 widgets"""

        # Row 0：模式選單 + 功能按鈕
        self.DropMode = ctk.CTkOptionMenu(
            self,
            values=['Yolo+眼眉鼻口髮際下巴8點'],
            width=220,
        )
        self.DropMode.grid(row=0, column=0, padx=8, pady=8, sticky='w')

        self.BtnSingle = ctk.CTkButton(
            self, text='單次測試', width=120,
            command=self._single_test,
        )
        self.BtnSingle.grid(row=0, column=1, padx=8, pady=8)

        self.BtnBatch = ctk.CTkButton(
            self, text='全目錄處理', width=120,
            command=self._batch_process,
        )
        self.BtnBatch.grid(row=0, column=2, padx=8, pady=8)

        # Row 1：圖片目錄
        ctk.CTkLabel(self, text='圖片目錄').grid(
            row=1, column=0, padx=8, pady=4, sticky='e')
        self.EntryInputDir = ctk.CTkEntry(self, placeholder_text='請選擇圖片目錄...')
        self.EntryInputDir.grid(row=1, column=1, padx=4, pady=4, sticky='ew')
        ctk.CTkButton(
            self, text='選擇', width=60,
            command=self._select_input_dir,
        ).grid(row=1, column=2, padx=8, pady=4)

        # Row 2：輸出目錄
        ctk.CTkLabel(self, text='輸出目錄').grid(
            row=2, column=0, padx=8, pady=4, sticky='e')
        self.EntryOutputDir = ctk.CTkEntry(self, placeholder_text='請選擇輸出目錄...')
        self.EntryOutputDir.grid(row=2, column=1, padx=4, pady=4, sticky='ew')
        ctk.CTkButton(
            self, text='選擇', width=60,
            command=self._select_output_dir,
        ).grid(row=2, column=2, padx=8, pady=4)

        # Row 3：圖片顯示區 PicArea
        self.PicArea = ctk.CTkLabel(
            self,
            text='尚無圖片',
            fg_color=('#2B2B2B', '#1C1C1C'),
            width=PIC_AREA_MAX_W,
            height=PIC_AREA_MAX_H,
            compound='center',
        )
        self.PicArea.grid(row=3, column=0, columnspan=3,
                          padx=8, pady=8, sticky='nsew')

        # Row 4：訊息區
        ctk.CTkLabel(self, text='Information:').grid(
            row=4, column=0, padx=8, pady=(4, 2), sticky='ne')
        self.TextInfo = ctk.CTkTextbox(
            self, height=120, wrap='word', state='disabled')
        self.TextInfo.grid(row=4, column=1, columnspan=2,
                           padx=8, pady=(4, 8), sticky='ew')

    def _setup_layout(self):
        """設定 Grid 權重，讓 Entry/TextInfo 水平延伸，PicArea 垂直延伸"""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=1)

    # ── 目錄選擇 ──────────────────────────────────────────────────────────────

    def _select_input_dir(self):
        """開啟資料夾選擇對話框，填入 EntryInputDir"""
        DirPath = filedialog.askdirectory(title='選擇圖片目錄')
        if DirPath:
            self.EntryInputDir.delete(0, 'end')
            self.EntryInputDir.insert(0, DirPath)

    def _select_output_dir(self):
        """開啟資料夾選擇對話框，填入 EntryOutputDir"""
        DirPath = filedialog.askdirectory(title='選擇輸出目錄')
        if DirPath:
            self.EntryOutputDir.delete(0, 'end')
            self.EntryOutputDir.insert(0, DirPath)

    # ── 驗證設定 ──────────────────────────────────────────────────────────────

    def _validate_settings(self) -> bool:
        """驗證目錄是否已設定，回傳 True 表示可繼續"""
        InputDir = self.EntryInputDir.get().strip()
        OutputDir = self.EntryOutputDir.get().strip()

        if not InputDir:
            self._log_message('⚠ 請先選擇圖片目錄')
            return False
        if not os.path.isdir(InputDir):
            self._log_message(f'⚠ 圖片目錄不存在: {InputDir}')
            return False
        if not OutputDir:
            self._log_message('⚠ 請先選擇輸出目錄')
            return False

        # 輸出目錄不存在時自動建立
        try:
            os.makedirs(OutputDir, exist_ok=True)
        except Exception as E:
            self._log_message(f'⚠ 無法建立輸出目錄: {E}')
            return False

        return True

    # ── 單次測試 ──────────────────────────────────────────────────────────────

    def _single_test(self):
        """選取單張圖片後，在背景執行緒處理"""
        if not self._validate_settings():
            return

        # 開啟圖片選取對話框
        ImagePath = filedialog.askopenfilename(
            title='選擇要處理的圖片',
            filetypes=[
                ('圖片檔案', '*.jpg *.jpeg *.png *.bmp *.webp'),
                ('所有檔案', '*.*'),
            ],
        )
        if not ImagePath:
            return

        self.BtnSingle.configure(state='disabled')
        self.BtnBatch.configure(state='disabled')

        T = threading.Thread(
            target=self._process_single_thread,
            args=(ImagePath,),
            daemon=True,
        )
        T.start()

    # ── 全目錄處理 ────────────────────────────────────────────────────────────

    def _batch_process(self):
        """掃描目錄中所有圖片，在背景執行緒批次處理"""
        if not self._validate_settings():
            return

        InputDir = self.EntryInputDir.get().strip()

        # 收集所有支援的圖片檔案
        ImageList = []
        for Ext in SUPPORTED_EXTS:
            ImageList.extend(glob.glob(os.path.join(InputDir, f'*{Ext}')))
            ImageList.extend(glob.glob(os.path.join(InputDir, f'*{Ext.upper()}')))
        # 去重（大小寫可能重複）
        ImageList = list(dict.fromkeys(ImageList))

        if not ImageList:
            self._log_message('⚠ 目錄中無支援的圖片檔案')
            return

        self._log_message(f'開始批次處理，共 {len(ImageList)} 張圖片...')
        self.BtnSingle.configure(state='disabled')
        self.BtnBatch.configure(state='disabled')

        T = threading.Thread(
            target=self._batch_process_thread,
            args=(ImageList,),
            daemon=True,
        )
        T.start()

    def _batch_process_thread(self, ImageList: list):
        """批次處理執行緒，逐一處理每張圖片"""
        for ImagePath in ImageList:
            self._process_single(ImagePath)
        # 完成後在主執行緒重啟按鈕
        self.after(0, self._enable_buttons)
        self.after(0, self._log_message,
                   f'✓ 全目錄處理完成，共 {len(ImageList)} 張')

    # ── 核心處理 ──────────────────────────────────────────────────────────────

    def _process_single_thread(self, ImagePath: str):
        """單次測試用執行緒包裝，完成後重啟按鈕"""
        self._process_single(ImagePath)
        self.after(0, self._enable_buttons)

    def _process_single(self, ImagePath: str):
        """
        核心處理流程：
        1. 載入圖片
        2. 偵測關鍵點
        3. 判斷側臉
        4. 繪製標注並更新顯示
        5. 輸出 YOLO 格式檔（非側臉才執行）
        """
        StartTime = time.time()
        BaseName = os.path.basename(ImagePath)
        OutputDir = self.EntryOutputDir.get().strip()

        # 載入圖片
        try:
            NpImage = face_recognition.load_image_file(ImagePath)
        except Exception as E:
            self.after(0, self._log_message, f'⚠ {BaseName}: 無法載入圖片 ({E})')
            return

        ImgH, ImgW = NpImage.shape[:2]
        PilImage = Image.fromarray(NpImage)

        # get_face_landmarks
        LandmarksList = get_face_landmarks(NpImage)
        if not LandmarksList:
            self.after(0, self._log_message, f'⚠ {BaseName}: 無法偵測人臉')
            return

        # 只取第一張臉
        Landmarks = LandmarksList[0]

        # detect_face_pose / 歪頭
        FaceType, Angle = detect_face_pose(Landmarks, ImgW, ImgH)

        # 計算8類邊框（側臉也計算，用於繪製顯示）
        BBoxList = calc_eight_class_boxes(Landmarks, ImgW, ImgH)

        # 用 MediaPipe landmark 10 精確定位髮際線，取代幾何估算的 class 6
        if not _MpReady:
            ModelFile = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'face_landmarker.task')
            if not os.path.exists(ModelFile):
                self.after(0, self._log_message,
                           '首次使用 MediaPipe：正在下載模型（約1MB），請稍候...')
        if _load_mediapipe_model():
            HairlineBox = detect_hairline_box(PilImage, ImgW, ImgH)
            if HairlineBox:
                BBoxList = [Item for Item in BBoxList if Item['class'] != 6]
                BBoxList.append(HairlineBox)
        else:
            self.after(0, self._log_message,
                       '⚠ MediaPipe 載入失敗，改用幾何估算髮際線')

        # 繪製並顯示圖片
        AnnotatedPil = draw_landmarks_and_boxes(PilImage, Landmarks, BBoxList, FaceType)
        self.after(0, self._update_image_display, AnnotatedPil)

        # 側臉：記錄訊息，不輸出 YOLO 檔
        if FaceType == 'side':
            self.after(0, self._log_message,
                       f'{BaseName}, 可能是側臉(角度:{Angle:.1f}度)')
            return
        # 歪頭：記錄訊息，但繼續輸出 YOLO 檔
        if FaceType == 'tilt':
            self.after(0, self._log_message,
                       f'{BaseName}, 可能是歪頭(角度:{Angle:.1f}度)')

        # 輸出 YOLO 格式檔
        StemName = os.path.splitext(BaseName)[0]
        OutputPath = os.path.join(OutputDir, StemName + '.txt')
        Success = write_yolo_file(BBoxList, OutputPath, ImgW, ImgH)

        Elapsed = time.time() - StartTime
        if Success:
            self.after(0, self._log_message,
                       f'✓ {BaseName}  處理完成 ({Elapsed:.2f}秒)')
        else:
            self.after(0, self._log_message,
                       f'⚠ {BaseName}: 寫入 YOLO 檔案失敗')

    # ── UI 更新（須在主執行緒呼叫）──────────────────────────────────────────────

    def _update_image_display(self, PilImage: Image.Image):
        """將 PIL Image 縮放後顯示在 PicArea"""
        # 計算縮放比例（不放大）
        ScaleW = PIC_AREA_MAX_W / PilImage.width
        ScaleH = PIC_AREA_MAX_H / PilImage.height
        Scale = min(ScaleW, ScaleH, 1.0)
        DisplayW = max(1, int(PilImage.width * Scale))
        DisplayH = max(1, int(PilImage.height * Scale))

        DisplayPil = PilImage.resize((DisplayW, DisplayH), Image.LANCZOS)
        CtkImg = ctk.CTkImage(
            light_image=DisplayPil,
            dark_image=DisplayPil,
            size=(DisplayW, DisplayH),
        )
        self.PicArea.configure(image=CtkImg, text='')
        # 保留參考防止 GC
        self._CurrentImage = CtkImg

    def _log_message(self, Msg: str):
        """將訊息 prepend 到 TextInfo 最上方"""
        self.TextInfo.configure(state='normal')
        self.TextInfo.insert('0.0', Msg + '\n')
        self.TextInfo.configure(state='disabled')

    def _enable_buttons(self):
        """重新啟用功能按鈕"""
        self.BtnSingle.configure(state='normal')
        self.BtnBatch.configure(state='normal')
