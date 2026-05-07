# Row 3：互動區，三種模式輪流顯示（OK/Cancel、文字輸入、隱藏）

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QStackedWidget, QPushButton, QLineEdit
)
from PySide6.QtCore import Signal, Qt
from style_constants import (
    ROW3_HEIGHT, STYLE_BUTTON_OK, STYLE_BUTTON_CANCEL,
    STYLE_LINE_EDIT, STYLE_ROW3
)

# Row3 模式常數
MODE_OK_CANCEL = 0
MODE_TEXT_INPUT = 1
MODE_HIDDEN = 2


class RowInteraction(QWidget):
    """Row 3 互動區，每次 GIF 切換後輪流顯示三種模式。"""

    OkPressed = Signal()
    CancelPressed = Signal()
    TextSubmitted = Signal(str)

    def __init__(self, Parent=None):
        super().__init__(Parent)
        self.setObjectName('Row3')
        self._ModeIndex = MODE_OK_CANCEL    # 從第一種模式開始
        self._setupUi()

    def _setupUi(self) -> None:
        """建立 QStackedWidget 含三個頁面。"""
        try:
            self.setStyleSheet(STYLE_ROW3)
            self.setFixedHeight(ROW3_HEIGHT)

            Layout = QHBoxLayout(self)
            Layout.setContentsMargins(12, 6, 12, 6)
            Layout.setSpacing(0)

            self._Stack = QStackedWidget()
            self._Stack.setStyleSheet('background: transparent;')

            # 頁面 0：OK + Cancel 按鈕
            self._Stack.addWidget(self._buildPage0())
            # 頁面 1：文字輸入框
            self._Stack.addWidget(self._buildPage1())
            # 頁面 2：空白（隱藏用）
            self._Stack.addWidget(self._buildPage2())

            self._Stack.setCurrentIndex(MODE_OK_CANCEL)
            Layout.addWidget(self._Stack)

        except Exception as E:
            print(f'RowInteraction UI 建立失敗: {E}')

    def _buildPage0(self) -> QWidget:
        """頁面 0：OK 和 Cancel 兩個按鈕。"""
        try:
            Page = QWidget()
            Page.setStyleSheet('background: transparent;')
            Layout = QHBoxLayout(Page)
            Layout.setContentsMargins(0, 0, 0, 0)
            Layout.setSpacing(16)
            Layout.addStretch()

            BtnOk = QPushButton('OK')
            BtnOk.setStyleSheet(STYLE_BUTTON_OK)
            BtnOk.setCursor(Qt.PointingHandCursor)
            BtnOk.clicked.connect(self.OkPressed)

            BtnCancel = QPushButton('Cancel')
            BtnCancel.setStyleSheet(STYLE_BUTTON_CANCEL)
            BtnCancel.setCursor(Qt.PointingHandCursor)
            BtnCancel.clicked.connect(self.CancelPressed)

            Layout.addWidget(BtnOk)
            Layout.addWidget(BtnCancel)
            Layout.addStretch()
            return Page
        except Exception as E:
            print(f'Page0 建立失敗: {E}')
            return QWidget()

    def _buildPage1(self) -> QWidget:
        """頁面 1：文字輸入框（按 Enter 送出）。"""
        try:
            Page = QWidget()
            Page.setStyleSheet('background: transparent;')
            Layout = QHBoxLayout(Page)
            Layout.setContentsMargins(0, 0, 0, 0)

            self._LineEdit = QLineEdit()
            self._LineEdit.setStyleSheet(STYLE_LINE_EDIT)
            self._LineEdit.setPlaceholderText('請輸入訊息…')
            self._LineEdit.returnPressed.connect(self._onTextSubmit)

            Layout.addWidget(self._LineEdit)
            return Page
        except Exception as E:
            print(f'Page1 建立失敗: {E}')
            return QWidget()

    def _buildPage2(self) -> QWidget:
        """頁面 2：空白，Row3 收合時使用。"""
        Page = QWidget()
        Page.setStyleSheet('background: transparent;')
        return Page

    def _onTextSubmit(self) -> None:
        """使用者在輸入框按 Enter。"""
        try:
            Text = self._LineEdit.text().strip()
            if Text:
                self.TextSubmitted.emit(Text)
                self._LineEdit.clear()
        except Exception as E:
            print(f'文字送出失敗: {E}')

    def advanceMode(self) -> None:
        """每次 GIF 切換後呼叫，輪流切換至下一種模式。"""
        try:
            self._ModeIndex = (self._ModeIndex + 1) % 3
            self._stack_setMode(self._ModeIndex)
        except Exception as E:
            print(f'切換 Row3 模式失敗: {E}')

    def _stack_setMode(self, Mode: int) -> None:
        """切換到指定模式，並調整高度。"""
        try:
            self._Stack.setCurrentIndex(Mode)
            if Mode == MODE_HIDDEN:
                # 收合 Row3，不佔空間且不跳動視窗
                self.setMaximumHeight(0)
                self.setMinimumHeight(0)
            else:
                self.setMaximumHeight(ROW3_HEIGHT)
                self.setMinimumHeight(ROW3_HEIGHT)
        except Exception as E:
            print(f'Row3 高度調整失敗: {E}')

    def getCurrentMode(self) -> int:
        return self._ModeIndex

    def getText(self) -> str:
        """取得輸入框目前內容（MODE_TEXT_INPUT 時使用）。"""
        try:
            return self._LineEdit.text()
        except Exception as E:
            print(f'取得輸入內容失敗: {E}')
            return ''
