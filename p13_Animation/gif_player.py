# GIF 播放引擎：管理所有 QMovie 物件、切換邏輯、隨機撥放狀態機

import os
import re
import random
from PySide6.QtCore import QObject, Signal, Qt, QSize
from PySide6.QtGui import QMovie
from style_constants import GIF_DISPLAY_SIZE


class GifPlayer(QObject):
    """管理所有 GIF 的載入、播放與切換。"""

    # 切換完成後發出（傳入新索引）,widget_window.py 會用，在過場完成後啟動新 GIF
    GifSwitched = Signal(int)
    # 當前 GIF 播放完一輪時發出, widget_window.py  會用，觸發下一輪播放（依隨機模式決定下一個索引）
    PlaybackCompleted = Signal()

    def __init__(self, Parent=None):
        super().__init__(Parent)
        self._Movies = []           # 所有 QMovie 物件
        self._GifNames = []         # 對應的檔案名稱
        self._CurrentIndex = 0      # 當前播放索引
        self._RandomMode = False    # 是否為隨機撥放模式
        self._JustStarted = False   # 剛啟動標記，避免誤判 frame==0 為播完

    def loadAll(self, GifDir: str) -> None:
        """掃描指定目錄，依數字前綴排序後載入所有 GIF。"""
        try:
            AllFiles = os.listdir(GifDir)
            GifFiles = [F for F in AllFiles if F.lower().endswith('.gif')]

            # 依檔名開頭的數字排序（如 "(4)xxx.gif" → 4）
            def extractNumber(FileName: str) -> int:
                Match = re.search(r'\((\d+)\)', FileName)
                return int(Match.group(1)) if Match else 9999

            GifFiles.sort(key=extractNumber)
            self._GifNames = GifFiles

            # 建立每個 GIF 的 QMovie，並預載所有 frame
            for FileName in GifFiles:
                try:
                    FullPath = os.path.join(GifDir, FileName)
                    Movie = QMovie(FullPath)
                    Movie.setCacheMode(QMovie.CacheAll)
                    # 縮放至顯示區尺寸，避免原始尺寸過大被截圖
                    Movie.setScaledSize(QSize(GIF_DISPLAY_SIZE, GIF_DISPLAY_SIZE))
                    # 觸發 Qt 內部預載
                    Movie.jumpToFrame(0)
                    self._Movies.append(Movie)
                except Exception as E:
                    print(f'載入 GIF 失敗 [{FileName}]: {E}')

            if self._Movies:
                self._startMovie(0)

        except Exception as E:
            print(f'掃描 GIF 目錄失敗: {E}')

    def _startMovie(self, Index: int) -> None:
        """啟動指定索引的 QMovie，連接 frameChanged 訊號。"""
        try:
            Movie = self._Movies[Index]
            # 先斷開舊連接，避免多次連接
            try:
                Movie.frameChanged.disconnect()
            except Exception:
                pass
            self._JustStarted = True
            Movie.frameChanged.connect(self._onFrameChanged)
            Movie.start()
        except Exception as E:
            print(f'啟動 QMovie 失敗 [index={Index}]: {E}')

    def _onFrameChanged(self, FrameNum: int) -> None:
        """偵測 GIF 播放完一輪（frame 回到 0）。"""
        try:
            if FrameNum == 0:
                if self._JustStarted:
                    # 剛啟動時的第一個 frame==0，不算播完
                    self._JustStarted = False
                else:
                    # 真正播完一輪
                    self.PlaybackCompleted.emit()
            else:
                self._JustStarted = False
        except Exception as E:
            print(f'frameChanged 處理失敗: {E}')

    def switchTo(self, Index: int) -> None:
        """切換至指定索引的 GIF。"""
        try:
            if not self._Movies:
                return
            Index = max(0, min(Index, len(self._Movies) - 1))

            # 停止舊的 movie 並斷開訊號
            if self._Movies:
                try:
                    OldMovie = self._Movies[self._CurrentIndex]
                    OldMovie.stop()
                    OldMovie.frameChanged.disconnect()
                except Exception:
                    pass

            self._CurrentIndex = Index
            self.GifSwitched.emit(Index)
            # 注意：_startMovie 由 RowDisplay 在 opacity=0 時呼叫
            # 此處只發出訊號，讓 RowDisplay 觸發過場，過場完成後再 start
        except Exception as E:
            print(f'GIF 切換失敗: {E}')

    def startCurrentMovie(self) -> None:
        """由 RowDisplay 過場完成後呼叫，啟動新 GIF 的播放。"""
        try:
            self._startMovie(self._CurrentIndex)
        except Exception as E:
            print(f'啟動當前 GIF 失敗: {E}')

    def switchNext(self) -> None:
        """切換至下一個 GIF（循環）。"""
        try:
            if not self._Movies:
                return
            NextIndex = (self._CurrentIndex + 1) % len(self._Movies)
            self.switchTo(NextIndex)
        except Exception as E:
            print(f'切換下一個失敗: {E}')

    def switchPrev(self) -> None:
        """切換至上一個 GIF（循環）。"""
        try:
            if not self._Movies:
                return
            PrevIndex = (self._CurrentIndex - 1) % len(self._Movies)
            self.switchTo(PrevIndex)
        except Exception as E:
            print(f'切換上一個失敗: {E}')

    def toggleRandom(self, Enabled: bool) -> None:
        """開啟或關閉隨機撥放模式。"""
        self._RandomMode = Enabled

    def pickNextRandom(self) -> int:
        """隨機選擇一個不同於當前的索引。"""
        try:
            if len(self._Movies) <= 1:
                return self._CurrentIndex
            NextIndex = self._CurrentIndex
            while NextIndex == self._CurrentIndex:
                NextIndex = random.randint(0, len(self._Movies) - 1)
            return NextIndex
        except Exception as E:
            print(f'隨機選擇失敗: {E}')
            return 0

    def getMovie(self, Index: int) -> QMovie:
        """取得指定索引的 QMovie 物件。"""
        try:
            return self._Movies[Index]
        except Exception as E:
            print(f'取得 QMovie 失敗 [index={Index}]: {E}')
            return None

    def getCurrentIndex(self) -> int:
        return self._CurrentIndex

    def isRandomMode(self) -> bool:
        return self._RandomMode

    def getCount(self) -> int:
        return len(self._Movies)

    def getGifNames(self) -> list:
        return self._GifNames
