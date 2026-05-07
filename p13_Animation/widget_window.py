# 主視窗：透明浮動視窗，組裝三排 Layout，連接所有訊號

import os
from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication
from PySide6.QtCore import Qt, QCoreApplication
from gif_player import GifPlayer
from row_controls import RowControls
from row_display import RowDisplay
from row_interaction import RowInteraction
from style_constants import WIDGET_WIDTH, WIDGET_HEIGHT, MARGIN_RIGHT, MARGIN_BOTTOM, STYLE_PANEL


class WidgetWindow(QWidget):
    """主視窗：透明背景、固定在螢幕右下角的浮動 Widget。"""

    def __init__(self, Parent=None):
        super().__init__(Parent)
        self._setupWindow()
        self._setupComponents()
        self._setupLayout()
        self._connectSignals()
        self._loadGifs()
        self._positionBottomRight()

    def _setupWindow(self) -> None:
        """設定視窗屬性：無框、置頂、透明背景、不在工作列顯示。"""
        try:
            self.setWindowFlags(
                Qt.FramelessWindowHint |
                Qt.WindowStaysOnTopHint |
                Qt.Tool
            )
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.setFixedSize(WIDGET_WIDTH, WIDGET_HEIGHT)
        except Exception as E:
            print(f'視窗設定失敗: {E}')

    def _setupComponents(self) -> None:
        """建立各子元件。"""
        try:
            self._Player = GifPlayer(self)
            self._RowControls = RowControls(self)
            self._RowDisplay = RowDisplay(self)
            self._RowInteraction = RowInteraction(self)
        except Exception as E:
            print(f'子元件建立失敗: {E}')

    def _setupLayout(self) -> None:
        """組裝深色玻璃面板 + 三排 Layout。"""
        try:
            # 外層：透明的頂層視窗
            OuterLayout = QVBoxLayout(self)
            OuterLayout.setContentsMargins(0, 0, 0, 0)
            OuterLayout.setSpacing(0)

            # 深色玻璃面板（含圓角背景）
            self._Panel = QWidget(self)
            self._Panel.setObjectName('MainPanel')
            self._Panel.setStyleSheet(STYLE_PANEL)

            PanelLayout = QVBoxLayout(self._Panel)
            PanelLayout.setContentsMargins(0, 0, 0, 0)
            PanelLayout.setSpacing(0)

            # Row 1：控制按鈕
            PanelLayout.addWidget(self._RowControls)

            # Row 2：互動區
            PanelLayout.addWidget(self._RowInteraction)

            # Row 3：GIF 顯示區（置中）
            PanelLayout.addWidget(self._RowDisplay, 0, Qt.AlignHCenter)
            OuterLayout.addWidget(self._Panel)

        except Exception as E:
            print(f'Layout 建立失敗: {E}')

    def _connectSignals(self) -> None:
        """連接所有元件之間的訊號。"""
        try:
            # Row 1 按鈕 → GifPlayer
            self._RowControls.PrevRequested.connect(self._Player.switchPrev)
            self._RowControls.NextRequested.connect(self._Player.switchNext)
            self._RowControls.RandomToggled.connect(self._Player.toggleRandom)
            self._RowControls.CloseRequested.connect(QCoreApplication.quit)

            # GifPlayer 切換 → RowDisplay 過場 + Row3 模式推進
            self._Player.GifSwitched.connect(self._RowDisplay.triggerCrossfade)
            self._Player.GifSwitched.connect(self._onGifSwitched)

            # GifPlayer 播完一輪 → 隨機模式自動切換
            self._Player.PlaybackCompleted.connect(self._onRandomCompleted)

            # Row 3 事件（供後續功能擴充用）
            self._RowInteraction.OkPressed.connect(self._onOkPressed)
            self._RowInteraction.CancelPressed.connect(self._onCancelPressed)
            self._RowInteraction.TextSubmitted.connect(self._onTextSubmitted)

        except Exception as E:
            print(f'訊號連接失敗: {E}')

    def _loadGifs(self) -> None:
        """載入 gif 目錄下的所有 GIF，並連接至顯示元件。"""
        try:
            GifDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gif')
            self._Player.loadAll(GifDir)
            self._RowDisplay.attachPlayer(self._Player)
        except Exception as E:
            print(f'載入 GIF 失敗: {E}')

    def _positionBottomRight(self) -> None:
        """將視窗定位在螢幕右下角（排除任務列）。"""
        try:
            Screen = QApplication.primaryScreen()
            Available = Screen.availableGeometry()
            X = Available.right() - WIDGET_WIDTH - MARGIN_RIGHT
            Y = Available.bottom() - WIDGET_HEIGHT - MARGIN_BOTTOM
            self.move(X, Y)
        except Exception as E:
            print(f'視窗定位失敗: {E}')

    def _onGifSwitched(self, NewIndex: int) -> None:
        """GIF 切換時，推進 Row3 模式。"""
        try:
            self._RowInteraction.advanceMode()
        except Exception as E:
            print(f'Row3 模式推進失敗: {E}')

    def _onRandomCompleted(self) -> None:
        """當前 GIF 播完一輪，若隨機模式啟用則自動切換。"""
        try:
            if self._Player.isRandomMode():
                NextIndex = self._Player.pickNextRandom()
                self._Player.switchTo(NextIndex)
        except Exception as E:
            print(f'隨機切換失敗: {E}')

    def _onOkPressed(self) -> None:
        """OK 按鈕按下（預留擴充）。"""
        print('OK 按下')

    def _onCancelPressed(self) -> None:
        """Cancel 按鈕按下（預留擴充）。"""
        print('Cancel 按下')

    def _onTextSubmitted(self, Text: str) -> None:
        """文字輸入框送出（預留擴充）。"""
        print(f'輸入文字: {Text}')
