# Row 2：文字顯示 Label，Row 1 任何按鍵按下時追加預設文字

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PySide6.QtCore import Qt
from style_constants import ROW2_HEIGHT, STYLE_ROW2_TEXT

# 預設文字與字數上限
DEFAULT_TEXT = 'This is test texts. Check UI is OK or not.'
MAX_LENGTH    = 2000


class RowLabel(QWidget):
    """Row 2 文字顯示區，按 Row 1 按鍵追加文字，超過上限後重設。"""

    def __init__(self, Parent=None):
        super().__init__(Parent)
        self._Content = DEFAULT_TEXT
        self._setupUi()

    def _setupUi(self) -> None:
        """建立唯讀文字框。"""
        try:
            self.setFixedHeight(ROW2_HEIGHT)

            Layout = QVBoxLayout(self)
            Layout.setContentsMargins(6, 6, 6, 6)
            Layout.setSpacing(0)

            self._TextEdit = QTextEdit()
            self._TextEdit.setReadOnly(True)
            self._TextEdit.setStyleSheet(STYLE_ROW2_TEXT)
            self._TextEdit.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self._TextEdit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self._TextEdit.setPlainText(DEFAULT_TEXT)

            Layout.addWidget(self._TextEdit)
        except Exception as E:
            print(f'RowLabel UI 建立失敗: {E}')

    def appendText(self, *_) -> None:
        """追加一次預設文字；超過 MAX_LENGTH 則恢復預設值。"""
        try:
            self._Content += DEFAULT_TEXT
            if len(self._Content) > MAX_LENGTH:
                self._Content = DEFAULT_TEXT
            self._TextEdit.setPlainText(self._Content)
            # 捲動至最新內容
            Scrollbar = self._TextEdit.verticalScrollBar()
            Scrollbar.setValue(Scrollbar.maximum())
        except Exception as E:
            print(f'追加文字失敗: {E}')
