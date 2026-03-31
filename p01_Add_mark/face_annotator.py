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


# ── 模組層級純函式 ─────────────────────────────────────────────────────────────

def 取得人臉關鍵點(NpImage: np.ndarray) -> list:
    """呼叫 face_recognition 取得68個關鍵點，失敗時回傳空列表"""
    try:
        return face_recognition.face_landmarks(NpImage)
    except Exception as E:
        return []


def 偵測側臉(Landmarks: dict, ImgW: int, ImgH: int) -> tuple:
    """
    判斷是否為側臉。
    回傳 (是否側臉: bool, 估算角度: float)
    """
    try:
        # 左眼中心 X
        LeftEyeX = sum(P[0] for P in Landmarks['left_eye']) / len(Landmarks['left_eye'])
        # 右眼中心 X
        RightEyeX = sum(P[0] for P in Landmarks['right_eye']) / len(Landmarks['right_eye'])
        # 兩眼中點 X
        EyeMidX = (LeftEyeX + RightEyeX) / 2.0
        # 鼻尖平均 X
        NoseTipX = sum(P[0] for P in Landmarks['nose_tip']) / len(Landmarks['nose_tip'])
        # 兩眼間距
        EyeSpan = abs(RightEyeX - LeftEyeX)
        if EyeSpan < 1:
            return False, 0.0
        # 偏移比例
        Offset = abs(NoseTipX - EyeMidX) / EyeSpan
        if Offset > PROFILE_OFFSET_THRESHOLD:
            # 線性估算角度：Offset=0 → 0度，Offset=0.5 → 90度
            AngleDeg = min(Offset * 180.0, 90.0)
            return True, AngleDeg
        return False, 0.0
    except Exception:
        return False, 0.0


def 計算邊框(PointList: list) -> tuple:
    """從點列表計算最小外接矩形，回傳 (x1, y1, x2, y2)"""
    Xs = [P[0] for P in PointList]
    Ys = [P[1] for P in PointList]
    return int(min(Xs)), int(min(Ys)), int(max(Xs)), int(max(Ys))


def 計算八類邊框(Landmarks: dict, ImgW: int, ImgH: int) -> list:
    """
    計算8個 YOLO 類別的邊框。
    回傳 list of dict: [{'class': N, 'x1':..., 'y1':..., 'x2':..., 'y2':...}, ...]
    """
    結果 = []
    # 邊框 padding (px)
    PAD = 4

    def _加邊框(ClassId, PointList):
        if not PointList:
            return
        X1, Y1, X2, Y2 = 計算邊框(PointList)
        # 加 padding，並 clamp 至圖片邊界
        X1 = max(0, X1 - PAD)
        Y1 = max(0, Y1 - PAD)
        X2 = min(ImgW - 1, X2 + PAD)
        Y2 = min(ImgH - 1, Y2 + PAD)
        結果.append({'class': ClassId, 'x1': X1, 'y1': Y1, 'x2': X2, 'y2': Y2})

    # 類別 0：左眼
    _加邊框(0, Landmarks.get('left_eye', []))
    # 類別 1：右眼
    _加邊框(1, Landmarks.get('right_eye', []))
    # 類別 2：左眉
    _加邊框(2, Landmarks.get('left_eyebrow', []))
    # 類別 3：右眉
    _加邊框(3, Landmarks.get('right_eyebrow', []))
    # 類別 4：鼻子（鼻樑 + 鼻尖）
    NosePoints = Landmarks.get('nose_bridge', []) + Landmarks.get('nose_tip', [])
    _加邊框(4, NosePoints)
    # 類別 5：嘴巴（上唇 + 下唇）
    MouthPoints = Landmarks.get('top_lip', []) + Landmarks.get('bottom_lip', [])
    _加邊框(5, MouthPoints)

    # 類別 6：髮際線最低中心點（幾何估算）
    HairlineBox = _估算髮際邊框(Landmarks, ImgW, ImgH)
    if HairlineBox:
        結果.append(HairlineBox)

    # 類別 7：下巴最低中心點
    ChinBox = _估算下巴邊框(Landmarks, ImgW, ImgH)
    if ChinBox:
        結果.append(ChinBox)

    return 結果


def _估算髮際邊框(Landmarks: dict, ImgW: int, ImgH: int) -> dict | None:
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

        # 固定邊框大小：正規化後 width=height=0.025（僅作為參考點）
        HalfW = int(ImgW * 0.025 / 2)
        HalfH = int(ImgH * 0.025 / 2)
        X1 = max(0, HairlineCx - HalfW)
        Y1 = max(0, HairlineY - HalfH)
        X2 = min(ImgW - 1, HairlineCx + HalfW)
        Y2 = min(ImgH - 1, HairlineY + HalfH)
        return {'class': 6, 'x1': X1, 'y1': Y1, 'x2': X2, 'y2': Y2}
    except Exception:
        return None


def _估算下巴邊框(Landmarks: dict, ImgW: int, ImgH: int) -> dict | None:
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


def 轉換為Yolo格式(X1: int, Y1: int, X2: int, Y2: int,
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


def 寫入Yolo檔案(BBoxList: list, OutputPath: str,
                  ImgW: int, ImgH: int) -> bool:
    """將邊框列表以 YOLO 格式寫入檔案，成功回傳 True"""
    try:
        with open(OutputPath, 'w', encoding='utf-8') as F:
            for Item in BBoxList:
                Xc, Yc, W, H = 轉換為Yolo格式(
                    Item['x1'], Item['y1'], Item['x2'], Item['y2'], ImgW, ImgH)
                F.write(f"{Item['class']} {Xc:.6f} {Yc:.6f} {W:.6f} {H:.6f}\n")
        return True
    except Exception as E:
        return False


def 繪製關鍵點與框(PilImage: Image.Image,
                   Landmarks: dict,
                   BBoxList: list,
                   IsSideface: bool) -> Image.Image:
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

    # 若是側臉，在圖上加提示文字
    if IsSideface:
        Draw.text((10, 10), '⚠ 側臉', fill='#FF4444')

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

        self._建立介面()
        self._設定版面()

    # ── 建立介面 ──────────────────────────────────────────────────────────────

    def _建立介面(self):
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
            command=self._單次測試,
        )
        self.BtnSingle.grid(row=0, column=1, padx=8, pady=8)

        self.BtnBatch = ctk.CTkButton(
            self, text='全目錄處理', width=120,
            command=self._全目錄處理,
        )
        self.BtnBatch.grid(row=0, column=2, padx=8, pady=8)

        # Row 1：圖片目錄
        ctk.CTkLabel(self, text='圖片目錄').grid(
            row=1, column=0, padx=8, pady=4, sticky='e')
        self.EntryInputDir = ctk.CTkEntry(self, placeholder_text='請選擇圖片目錄...')
        self.EntryInputDir.grid(row=1, column=1, padx=4, pady=4, sticky='ew')
        ctk.CTkButton(
            self, text='選擇', width=60,
            command=self._選擇輸入目錄,
        ).grid(row=1, column=2, padx=8, pady=4)

        # Row 2：輸出目錄
        ctk.CTkLabel(self, text='輸出目錄').grid(
            row=2, column=0, padx=8, pady=4, sticky='e')
        self.EntryOutputDir = ctk.CTkEntry(self, placeholder_text='請選擇輸出目錄...')
        self.EntryOutputDir.grid(row=2, column=1, padx=4, pady=4, sticky='ew')
        ctk.CTkButton(
            self, text='選擇', width=60,
            command=self._選擇輸出目錄,
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

    def _設定版面(self):
        """設定 Grid 權重，讓 Entry/TextInfo 水平延伸，PicArea 垂直延伸"""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=1)

    # ── 目錄選擇 ──────────────────────────────────────────────────────────────

    def _選擇輸入目錄(self):
        """開啟資料夾選擇對話框，填入 EntryInputDir"""
        DirPath = filedialog.askdirectory(title='選擇圖片目錄')
        if DirPath:
            self.EntryInputDir.delete(0, 'end')
            self.EntryInputDir.insert(0, DirPath)

    def _選擇輸出目錄(self):
        """開啟資料夾選擇對話框，填入 EntryOutputDir"""
        DirPath = filedialog.askdirectory(title='選擇輸出目錄')
        if DirPath:
            self.EntryOutputDir.delete(0, 'end')
            self.EntryOutputDir.insert(0, DirPath)

    # ── 驗證設定 ──────────────────────────────────────────────────────────────

    def _驗證設定(self) -> bool:
        """驗證目錄是否已設定，回傳 True 表示可繼續"""
        InputDir = self.EntryInputDir.get().strip()
        OutputDir = self.EntryOutputDir.get().strip()

        if not InputDir:
            self._寫入訊息('⚠ 請先選擇圖片目錄')
            return False
        if not os.path.isdir(InputDir):
            self._寫入訊息(f'⚠ 圖片目錄不存在: {InputDir}')
            return False
        if not OutputDir:
            self._寫入訊息('⚠ 請先選擇輸出目錄')
            return False

        # 輸出目錄不存在時自動建立
        try:
            os.makedirs(OutputDir, exist_ok=True)
        except Exception as E:
            self._寫入訊息(f'⚠ 無法建立輸出目錄: {E}')
            return False

        return True

    # ── 單次測試 ──────────────────────────────────────────────────────────────

    def _單次測試(self):
        """選取單張圖片後，在背景執行緒處理"""
        if not self._驗證設定():
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
            target=self._處理單張執行緒,
            args=(ImagePath,),
            daemon=True,
        )
        T.start()

    # ── 全目錄處理 ────────────────────────────────────────────────────────────

    def _全目錄處理(self):
        """掃描目錄中所有圖片，在背景執行緒批次處理"""
        if not self._驗證設定():
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
            self._寫入訊息('⚠ 目錄中無支援的圖片檔案')
            return

        self._寫入訊息(f'開始批次處理，共 {len(ImageList)} 張圖片...')
        self.BtnSingle.configure(state='disabled')
        self.BtnBatch.configure(state='disabled')

        T = threading.Thread(
            target=self._批次處理執行緒,
            args=(ImageList,),
            daemon=True,
        )
        T.start()

    def _批次處理執行緒(self, ImageList: list):
        """批次處理執行緒，逐一處理每張圖片"""
        for ImagePath in ImageList:
            self._處理單張(ImagePath)
        # 完成後在主執行緒重啟按鈕
        self.after(0, self._重啟按鈕)
        self.after(0, self._寫入訊息,
                   f'✓ 全目錄處理完成，共 {len(ImageList)} 張')

    # ── 核心處理 ──────────────────────────────────────────────────────────────

    def _處理單張執行緒(self, ImagePath: str):
        """單次測試用執行緒包裝，完成後重啟按鈕"""
        self._處理單張(ImagePath)
        self.after(0, self._重啟按鈕)

    def _處理單張(self, ImagePath: str):
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
            self.after(0, self._寫入訊息, f'⚠ {BaseName}: 無法載入圖片 ({E})')
            return

        ImgH, ImgW = NpImage.shape[:2]
        PilImage = Image.fromarray(NpImage)

        # 取得人臉關鍵點
        LandmarksList = 取得人臉關鍵點(NpImage)
        if not LandmarksList:
            self.after(0, self._寫入訊息, f'⚠ {BaseName}: 無法偵測人臉')
            return

        # 只取第一張臉
        Landmarks = LandmarksList[0]

        # 偵測側臉
        IsSideface, Angle = 偵測側臉(Landmarks, ImgW, ImgH)

        # 計算8類邊框（側臉也計算，用於繪製顯示）
        BBoxList = 計算八類邊框(Landmarks, ImgW, ImgH)

        # 繪製並顯示圖片
        AnnotatedPil = 繪製關鍵點與框(PilImage, Landmarks, BBoxList, IsSideface)
        self.after(0, self._更新圖片顯示, AnnotatedPil)

        # 側臉：記錄訊息，不輸出 YOLO 檔
        if IsSideface:
            self.after(0, self._寫入訊息,
                       f'{BaseName}, 可能是側臉(角度:{Angle:.1f}度)')
            return

        # 輸出 YOLO 格式檔
        StemName = os.path.splitext(BaseName)[0]
        OutputPath = os.path.join(OutputDir, StemName + '.txt')
        Success = 寫入Yolo檔案(BBoxList, OutputPath, ImgW, ImgH)

        Elapsed = time.time() - StartTime
        if Success:
            self.after(0, self._寫入訊息,
                       f'✓ {BaseName}  處理完成 ({Elapsed:.2f}秒)')
        else:
            self.after(0, self._寫入訊息,
                       f'⚠ {BaseName}: 寫入 YOLO 檔案失敗')

    # ── UI 更新（須在主執行緒呼叫）──────────────────────────────────────────────

    def _更新圖片顯示(self, PilImage: Image.Image):
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

    def _寫入訊息(self, Msg: str):
        """將訊息 prepend 到 TextInfo 最上方"""
        self.TextInfo.configure(state='normal')
        self.TextInfo.insert('0.0', Msg + '\n')
        self.TextInfo.configure(state='disabled')

    def _重啟按鈕(self):
        """重新啟用功能按鈕"""
        self.BtnSingle.configure(state='normal')
        self.BtnBatch.configure(state='normal')
