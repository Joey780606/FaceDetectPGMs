# Row 1：控制按鈕列（上一個 / 下一個 / 隨機撥放）

from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton
from PySide6.QtCore import Signal, Qt
from style_constants import STYLE_BUTTON, STYLE_BUTTON_CLOSE


class RowControls(QWidget):
    """Row 1 控制列，含三個操作按鈕與關閉按鈕。"""

    PrevRequested  = Signal()
    NextRequested  = Signal()
    RandomToggled  = Signal(bool)
    CloseRequested = Signal()

    def __init__(self, Parent=None):
        super().__init__(Parent)
        self._setupUi()

    def _setupUi(self) -> None:
        """建立三個按鈕並排版。"""
        try:
            Layout = QHBoxLayout(self)
            Layout.setContentsMargins(12, 8, 12, 8)
            Layout.setSpacing(10)

            # 上一個按鈕
            self._BtnPrev = QPushButton('上一個')
            self._BtnPrev.setStyleSheet(STYLE_BUTTON)
            self._BtnPrev.setCursor(Qt.PointingHandCursor)
            self._BtnPrev.clicked.connect(self.PrevRequested)

            # 下一個按鈕
            self._BtnNext = QPushButton('下一個')
            self._BtnNext.setStyleSheet(STYLE_BUTTON)
            self._BtnNext.setCursor(Qt.PointingHandCursor)
            self._BtnNext.clicked.connect(self.NextRequested)

            # 隨機撥放按鈕（可切換狀態）
            self._BtnRandom = QPushButton('隨機撥放')
            self._BtnRandom.setStyleSheet(STYLE_BUTTON)
            self._BtnRandom.setCursor(Qt.PointingHandCursor)
            self._BtnRandom.setCheckable(True)
            self._BtnRandom.toggled.connect(self.RandomToggled)

            # 關閉按鈕（靠右）
            self._BtnClose = QPushButton('✕')
            self._BtnClose.setStyleSheet(STYLE_BUTTON_CLOSE)
            self._BtnClose.setCursor(Qt.PointingHandCursor)
            self._BtnClose.setToolTip('關閉程式')
            self._BtnClose.clicked.connect(self.CloseRequested)

            Layout.addWidget(self._BtnPrev)
            Layout.addWidget(self._BtnNext)
            Layout.addWidget(self._BtnRandom)
            Layout.addStretch()
            Layout.addWidget(self._BtnClose)

        except Exception as E:
            print(f'RowControls UI 建立失敗: {E}')

    def isRandomActive(self) -> bool:
        """回傳隨機撥放是否啟用中。"""
        return self._BtnRandom.isChecked()
