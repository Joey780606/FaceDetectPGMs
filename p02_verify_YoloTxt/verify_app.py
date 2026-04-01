# verify_app.py — YOLO 標記驗證工具
# 讀取 YOLO .txt 標記檔，在對應圖片上畫出各特徵點，供使用者確認標記正確性

import os
import glob
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFont

# ── 常數 ──────────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

PIC_AREA_MAX_W = 800
PIC_AREA_MAX_H = 600

# 各類別顏色（與 p01_Add_mark 一致）
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

CLASS_NAMES = {
    0: '左眼', 1: '右眼',
    2: '左眉', 3: '右眉',
    4: '鼻',   5: '嘴',
    6: '髮際', 7: '下巴',
}


# ── 純函式 ────────────────────────────────────────────────────────────────────

def parse_yolo_file(TxtPath: str) -> list:
    """
    讀取 YOLO 格式標記檔，回傳 list of dict。
    每筆格式：{'class': int, 'xc': float, 'yc': float, 'w': float, 'h': float}
    """
    Records = []
    try:
        with open(TxtPath, 'r', encoding='utf-8') as F:
            for Line in F:
                Parts = Line.strip().split()
                if len(Parts) != 5:
                    continue
                try:
                    Records.append({
                        'class': int(Parts[0]),
                        'xc':    float(Parts[1]),
                        'yc':    float(Parts[2]),
                        'w':     float(Parts[3]),
                        'h':     float(Parts[4]),
                    })
                except ValueError:
                    continue
    except Exception:
        pass
    return Records


def find_image_for_txt(TxtPath: str, ImageDir: str) -> str | None:
    """
    在 ImageDir 中尋找與 TxtPath 同主檔名的圖片，
    回傳完整圖片路徑；找不到回傳 None。
    """
    BaseName = os.path.splitext(os.path.basename(TxtPath))[0]
    for Ext in SUPPORTED_EXTS:
        Candidate = os.path.join(ImageDir, BaseName + Ext)
        if os.path.exists(Candidate):
            return Candidate
    return None


def draw_yolo_annotations(PilImage: Image.Image, Records: list) -> Image.Image:
    """
    在圖片副本上依 YOLO 記錄畫出邊框與類別標籤，回傳標注後的 PIL Image。
    """
    ImgW, ImgH = PilImage.size
    Annotated = PilImage.copy()
    Draw = ImageDraw.Draw(Annotated)

    # 嘗試載入字型，失敗時用預設字型
    try:
        Font = ImageFont.truetype('arial.ttf', max(12, ImgW // 60))
    except Exception:
        Font = ImageFont.load_default()

    for Rec in Records:
        Cls = Rec['class']
        Color = CLASS_COLORS.get(Cls, '#FFFFFF')

        # YOLO 正規化座標還原為像素
        Xc = Rec['xc'] * ImgW
        Yc = Rec['yc'] * ImgH
        W  = Rec['w']  * ImgW
        H  = Rec['h']  * ImgH
        X1 = int(Xc - W / 2)
        Y1 = int(Yc - H / 2)
        X2 = int(Xc + W / 2)
        Y2 = int(Yc + H / 2)

        # 畫邊框
        Draw.rectangle([(X1, Y1), (X2, Y2)], outline=Color, width=2)

        # 畫中心點
        R = 4
        Draw.ellipse([(Xc - R, Yc - R), (Xc + R, Yc + R)], fill=Color)

        # 類別標籤
        Label = f"{Cls}:{CLASS_NAMES.get(Cls, str(Cls))}"
        Draw.text((X1 + 2, Y1 + 2), Label, fill=Color, font=Font)

    return Annotated


def resize_to_fit(PilImage: Image.Image,
                  MaxW: int = PIC_AREA_MAX_W,
                  MaxH: int = PIC_AREA_MAX_H) -> Image.Image:
    """等比例縮小至最大顯示尺寸，不放大。"""
    W, H = PilImage.size
    Scale = min(MaxW / W, MaxH / H, 1.0)
    if Scale < 1.0:
        return PilImage.resize((int(W * Scale), int(H * Scale)), Image.LANCZOS)
    return PilImage


# ── 主視窗 ────────────────────────────────────────────────────────────────────

class VerifyApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title('YOLO 標記驗證工具')
        self.resizable(True, True)

        # 狀態
        self._TxtFileList = []   # 目前目錄下所有 .txt 檔的完整路徑清單
        self._CurrentIdx  = -1  # 目前顯示的索引
        self._CurrentImage = None  # 防止 GC

        self._build_ui()
        self._setup_layout()

    def _build_ui(self):
        """建立所有 widgets"""

        # Row 0：圖片目錄
        ctk.CTkLabel(self, text='圖片目錄').grid(
            row=0, column=0, padx=8, pady=6, sticky='e')
        self.EntryImageDir = ctk.CTkEntry(
            self, placeholder_text='請選擇圖片目錄...')
        self.EntryImageDir.grid(row=0, column=1, padx=4, pady=6, sticky='ew')
        ctk.CTkButton(
            self, text='選擇', width=60,
            command=self._select_image_dir,
        ).grid(row=0, column=2, padx=8, pady=6)

        # Row 1：標記目錄
        ctk.CTkLabel(self, text='標記目錄').grid(
            row=1, column=0, padx=8, pady=6, sticky='e')
        self.EntryMarkDir = ctk.CTkEntry(
            self, placeholder_text='請選擇 YOLO txt 目錄...')
        self.EntryMarkDir.grid(row=1, column=1, padx=4, pady=6, sticky='ew')
        ctk.CTkButton(
            self, text='選擇', width=60,
            command=self._select_mark_dir,
        ).grid(row=1, column=2, padx=8, pady=6)

        # Row 2：導覽列
        self.BtnPrev = ctk.CTkButton(
            self, text='◀ 前一個', width=120,
            command=self._go_prev, state='disabled',
        )
        self.BtnPrev.grid(row=2, column=0, padx=8, pady=6)

        self.LblFileInfo = ctk.CTkLabel(
            self, text='尚未載入', anchor='center')
        self.LblFileInfo.grid(row=2, column=1, padx=4, pady=6, sticky='ew')

        self.BtnNext = ctk.CTkButton(
            self, text='後一個 ▶', width=120,
            command=self._go_next, state='disabled',
        )
        self.BtnNext.grid(row=2, column=2, padx=8, pady=6)

        # Row 3：圖片顯示區
        self.PicArea = ctk.CTkLabel(
            self,
            text='請選擇圖片目錄與標記目錄',
            fg_color=('#2B2B2B', '#1C1C1C'),
            width=PIC_AREA_MAX_W,
            height=PIC_AREA_MAX_H,
            compound='center',
        )
        self.PicArea.grid(row=3, column=0, columnspan=3,
                          padx=8, pady=8, sticky='nsew')

        # Row 4：訊息區
        ctk.CTkLabel(self, text='Information:').grid(
            row=4, column=0, padx=8, pady=(4, 8), sticky='ne')
        self.TextInfo = ctk.CTkTextbox(
            self, height=100, wrap='word', state='disabled')
        self.TextInfo.grid(row=4, column=1, columnspan=2,
                           padx=8, pady=(4, 8), sticky='ew')

    def _setup_layout(self):
        """Grid 權重設定"""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(3, weight=1)

    # ── 目錄選擇 ──────────────────────────────────────────────────────────────

    def _select_image_dir(self):
        """選擇圖片目錄"""
        DirPath = filedialog.askdirectory(title='選擇圖片目錄')
        if DirPath:
            self.EntryImageDir.delete(0, 'end')
            self.EntryImageDir.insert(0, DirPath)

    def _select_mark_dir(self):
        """選擇標記目錄，選好後自動掃描 .txt 檔並載入第一筆"""
        DirPath = filedialog.askdirectory(title='選擇 YOLO 標記目錄')
        if not DirPath:
            return
        self.EntryMarkDir.delete(0, 'end')
        self.EntryMarkDir.insert(0, DirPath)
        self._load_txt_list(DirPath)

    # ── 檔案清單 ──────────────────────────────────────────────────────────────

    def _load_txt_list(self, MarkDir: str):
        """掃描目錄內所有 .txt 檔，排序後顯示第一筆"""
        TxtFiles = sorted(glob.glob(os.path.join(MarkDir, '*.txt')))
        if not TxtFiles:
            self._log_message(f'⚠ {MarkDir} 裡找不到任何 .txt 檔')
            self._TxtFileList = []
            self._CurrentIdx  = -1
            self._update_nav_buttons()
            return

        self._TxtFileList = TxtFiles
        self._CurrentIdx  = 0
        self._update_nav_buttons()
        self._show_current()

    # ── 導覽 ──────────────────────────────────────────────────────────────────

    def _go_prev(self):
        if self._CurrentIdx > 0:
            self._CurrentIdx -= 1
            self._show_current()

    def _go_next(self):
        if self._CurrentIdx < len(self._TxtFileList) - 1:
            self._CurrentIdx += 1
            self._show_current()

    def _update_nav_buttons(self):
        """依目前索引更新前/後按鈕的可用狀態"""
        Total = len(self._TxtFileList)
        self.BtnPrev.configure(
            state='normal' if self._CurrentIdx > 0 else 'disabled')
        self.BtnNext.configure(
            state='normal' if self._CurrentIdx < Total - 1 else 'disabled')

    # ── 顯示 ──────────────────────────────────────────────────────────────────

    def _show_current(self):
        """顯示目前索引對應的圖片與標記"""
        if not self._TxtFileList or self._CurrentIdx < 0:
            return

        TxtPath   = self._TxtFileList[self._CurrentIdx]
        Total     = len(self._TxtFileList)
        BaseName  = os.path.basename(TxtPath)
        ImageDir  = self.EntryImageDir.get().strip()

        # 更新檔名標籤
        self.LblFileInfo.configure(
            text=f'{BaseName}  ({self._CurrentIdx + 1} / {Total})')
        self._update_nav_buttons()

        # 尋找對應圖片
        if not ImageDir:
            self._log_message('⚠ 請先選擇圖片目錄')
            return

        ImagePath = find_image_for_txt(TxtPath, ImageDir)
        if not ImagePath:
            self._log_message(
                f'⚠ 找不到 {os.path.splitext(BaseName)[0]} 的對應圖片')
            return

        # 載入圖片
        try:
            PilImage = Image.open(ImagePath).convert('RGB')
        except Exception as E:
            self._log_message(f'⚠ 無法載入圖片 {ImagePath}：{E}')
            return

        ImgW, ImgH = PilImage.size

        # 讀取 YOLO 標記
        Records = parse_yolo_file(TxtPath)

        # 畫標記
        Annotated = draw_yolo_annotations(PilImage, Records)

        # 縮放至顯示區
        Displayed = resize_to_fit(Annotated)

        # 更新 PicArea
        CtkImg = ctk.CTkImage(light_image=Displayed,
                               dark_image=Displayed,
                               size=Displayed.size)
        self.PicArea.configure(image=CtkImg, text='')
        self._CurrentImage = CtkImg  # 防止 GC

        # 訊息
        ClassList = ', '.join(
            f"{R['class']}:{CLASS_NAMES.get(R['class'], '?')}"
            for R in Records
        )
        self._log_message(
            f'{BaseName}  圖片:{ImgW}×{ImgH}  '
            f'標記:{len(Records)}筆 [{ClassList}]'
        )

    # ── 訊息 ──────────────────────────────────────────────────────────────────

    def _log_message(self, Msg: str):
        """新訊息插入 TextInfo 最上方"""
        self.TextInfo.configure(state='normal')
        self.TextInfo.insert('0.0', Msg + '\n')
        self.TextInfo.configure(state='disabled')
