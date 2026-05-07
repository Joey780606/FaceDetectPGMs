#!/usr/bin/env python3
"""
GIF 背景去除工具
將指定目錄內所有 GIF 檔案的背景色轉換為透明
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image
import numpy as np
import os
import threading


def GetCornerBackgroundColor(RgbaArray):
    """從四個角落像素判斷背景色，取出現最多次的顏色"""
    Height, Width = RgbaArray.shape[:2]
    Corners = [
        tuple(RgbaArray[0, 0, :3]),
        tuple(RgbaArray[0, Width - 1, :3]),
        tuple(RgbaArray[Height - 1, 0, :3]),
        tuple(RgbaArray[Height - 1, Width - 1, :3]),
    ]
    ColorCount = {}
    for Color in Corners:
        ColorCount[Color] = ColorCount.get(Color, 0) + 1
    return max(ColorCount, key=ColorCount.get)


def ApplyTransparencyToFrame(Frame, BgColor, Tolerance):
    """將單一幀的背景色像素設為透明"""
    RgbaArray = np.array(Frame.convert('RGBA'), dtype=np.float32)

    BgR, BgG, BgB = BgColor
    # 計算每個像素與背景色的歐氏距離
    Distance = np.sqrt(
        (RgbaArray[:, :, 0] - BgR) ** 2 +
        (RgbaArray[:, :, 1] - BgG) ** 2 +
        (RgbaArray[:, :, 2] - BgB) ** 2
    )
    # 距離在容差範圍內的像素設為完全透明
    Mask = Distance <= Tolerance
    RgbaArray[Mask, 3] = 0

    return Image.fromarray(RgbaArray.astype(np.uint8), 'RGBA')


def MakeGifTransparent(InputPath, OutputPath, Tolerance=20):
    """將 GIF 檔案所有幀的背景色轉換為透明並儲存"""
    try:
        GifImg = Image.open(InputPath)

        # 以第一幀判斷背景色
        FirstArray = np.array(GifImg.convert('RGBA'))
        BgColor = GetCornerBackgroundColor(FirstArray)

        Frames = []
        Durations = []

        try:
            while True:
                Duration = GifImg.info.get('duration', 100)
                Durations.append(Duration)

                TransparentFrame = ApplyTransparencyToFrame(GifImg, BgColor, Tolerance)
                Frames.append(TransparentFrame)

                GifImg.seek(GifImg.tell() + 1)
        except EOFError:
            pass

        if not Frames:
            raise ValueError("無法讀取任何 GIF 幀")

        # 儲存含透明通道的 GIF
        Frames[0].save(
            OutputPath,
            save_all=True,
            append_images=Frames[1:],
            loop=GifImg.info.get('loop', 0),
            duration=Durations,
            disposal=2,
        )
    except Exception as E:
        raise RuntimeError(f"處理 GIF 失敗：{E}")


class GifBgRemoverApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("GIF 背景去除工具")
        self.geometry("640x560")
        self.resizable(False, False)

        self._BuildDirRow()
        self._BuildOptionsRow()
        self._BuildActionRow()
        self._BuildProgressRow()
        self._BuildLogArea()

    def _BuildDirRow(self):
        """建立目錄選擇列"""
        DirFrame = ctk.CTkFrame(self)
        DirFrame.pack(fill="x", padx=15, pady=(15, 5))

        ctk.CTkLabel(DirFrame, text="目錄：", width=50).pack(side="left", padx=(10, 5), pady=8)

        self.DirVar = tk.StringVar()
        ctk.CTkEntry(DirFrame, textvariable=self.DirVar, width=430).pack(side="left", padx=5, pady=8)

        ctk.CTkButton(DirFrame, text="瀏覽", width=70, command=self._BrowseDir).pack(side="left", padx=(5, 10), pady=8)

    def _BuildOptionsRow(self):
        """建立選項列（容差、覆蓋設定）"""
        OptFrame = ctk.CTkFrame(self)
        OptFrame.pack(fill="x", padx=15, pady=5)

        # 顏色容差滑桿
        ctk.CTkLabel(OptFrame, text="顏色容差：", width=80).pack(side="left", padx=(10, 5), pady=10)

        self.ToleranceVar = tk.IntVar(value=20)
        self.ToleranceLabel = ctk.CTkLabel(OptFrame, text="20", width=30)

        Slider = ctk.CTkSlider(
            OptFrame, from_=0, to=80,
            variable=self.ToleranceVar,
            width=280,
            command=lambda v: self.ToleranceLabel.configure(text=str(int(v)))
        )
        Slider.pack(side="left", padx=5, pady=10)
        self.ToleranceLabel.pack(side="left", padx=(0, 20), pady=10)

        # 覆蓋原始檔案選項
        self.OverwriteVar = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            OptFrame,
            text="覆蓋原始檔案",
            variable=self.OverwriteVar
        ).pack(side="left", padx=10, pady=10)

    def _BuildActionRow(self):
        """建立執行按鈕列"""
        self.RunBtn = ctk.CTkButton(
            self,
            text="Transparent background",
            height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._StartProcess
        )
        self.RunBtn.pack(fill="x", padx=15, pady=8)

    def _BuildProgressRow(self):
        """建立進度條列"""
        ProgFrame = ctk.CTkFrame(self)
        ProgFrame.pack(fill="x", padx=15, pady=5)

        self.ProgressBar = ctk.CTkProgressBar(ProgFrame)
        self.ProgressBar.pack(fill="x", padx=10, pady=8)
        self.ProgressBar.set(0)

        self.StatusVar = tk.StringVar(value="就緒")
        ctk.CTkLabel(ProgFrame, textvariable=self.StatusVar).pack(pady=(0, 6))

    def _BuildLogArea(self):
        """建立日誌文字區域"""
        self.LogBox = ctk.CTkTextbox(self, height=220)
        self.LogBox.pack(fill="both", expand=True, padx=15, pady=(5, 15))

    # ── 事件處理 ──────────────────────────────────────────────

    def _BrowseDir(self):
        """瀏覽選擇目錄"""
        DirPath = filedialog.askdirectory(title="選擇含有 GIF 檔案的目錄")
        if DirPath:
            self.DirVar.set(DirPath)

    def _Log(self, Message):
        """在 UI 執行緒更新日誌文字框"""
        self.LogBox.insert("end", f"{Message}\n")
        self.LogBox.see("end")

    def _SetStatus(self, Text):
        self.StatusVar.set(Text)

    def _StartProcess(self):
        """驗證輸入後以背景執行緒啟動處理"""
        DirPath = self.DirVar.get().strip()
        if not DirPath:
            messagebox.showwarning("請選擇目錄", "請先選擇包含 GIF 檔案的目錄。")
            return
        if not os.path.isdir(DirPath):
            messagebox.showerror("目錄不存在", f"找不到目錄：{DirPath}")
            return

        self.RunBtn.configure(state="disabled")
        self.ProgressBar.set(0)
        self.LogBox.delete("1.0", "end")

        Thread = threading.Thread(
            target=self._ProcessAllGifs,
            args=(DirPath,),
            daemon=True
        )
        Thread.start()

    def _ProcessAllGifs(self, DirPath):
        """掃描並處理目錄內所有 GIF 檔案（背景執行緒）"""
        try:
            GifFiles = sorted(
                f for f in os.listdir(DirPath) if f.lower().endswith('.gif')
            )

            if not GifFiles:
                self.after(0, lambda: self._Log("⚠  找不到任何 GIF 檔案。"))
                self.after(0, lambda: self._SetStatus("完成（無 GIF）"))
                return

            TotalCount = len(GifFiles)
            self.after(0, lambda: self._Log(f"找到 {TotalCount} 個 GIF 檔案，開始處理…\n"))

            Tolerance = self.ToleranceVar.get()
            Overwrite = self.OverwriteVar.get()

            # 決定輸出目錄
            if not Overwrite:
                OutputDir = os.path.join(DirPath, "transparent")
                try:
                    os.makedirs(OutputDir, exist_ok=True)
                except Exception as E:
                    self.after(0, lambda: self._Log(f"❌ 無法建立輸出目錄：{E}"))
                    return
                self.after(0, lambda: self._Log(f"輸出目錄：{OutputDir}\n"))

            SuccessCount = 0
            FailCount = 0

            for Idx, GifFile in enumerate(GifFiles):
                InputPath = os.path.join(DirPath, GifFile)
                OutputPath = InputPath if Overwrite else os.path.join(OutputDir, GifFile)

                self.after(0, lambda F=GifFile: self._Log(f"  處理：{F}"))
                self.after(0, lambda F=GifFile: self._SetStatus(f"處理中：{F}"))

                try:
                    MakeGifTransparent(InputPath, OutputPath, Tolerance)
                    self.after(0, lambda F=GifFile: self._Log(f"  ✓ 完成：{F}"))
                    SuccessCount += 1
                except Exception as E:
                    self.after(0, lambda F=GifFile, Err=E: self._Log(f"  ✗ 失敗：{F} → {Err}"))
                    FailCount += 1

                Progress = (Idx + 1) / TotalCount
                self.after(0, lambda P=Progress: self.ProgressBar.set(P))

            Summary = f"\n全部完成：成功 {SuccessCount} 個，失敗 {FailCount} 個。"
            self.after(0, lambda: self._Log(Summary))
            self.after(0, lambda: self._SetStatus("完成"))

        except Exception as E:
            self.after(0, lambda: self._Log(f"❌ 發生未預期錯誤：{E}"))
            self.after(0, lambda: self._SetStatus("錯誤"))
        finally:
            self.after(0, lambda: self.RunBtn.configure(state="normal"))


if __name__ == "__main__":
    App = GifBgRemoverApp()
    App.mainloop()
