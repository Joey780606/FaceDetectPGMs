# 視覺設計常數：所有顏色、尺寸、stylesheet 字串集中於此

# 視窗尺寸
WIDGET_WIDTH = 420
WIDGET_HEIGHT = 570

# GIF 顯示區尺寸
GIF_DISPLAY_SIZE = 400

# 過場動畫時間（毫秒）
CROSSFADE_MS = 80

# 視窗與螢幕邊緣間距
MARGIN_RIGHT = 10
MARGIN_BOTTOM = 10

# Row 3 高度
ROW3_HEIGHT = 60

# 主面板 stylesheet（深色玻璃感）
STYLE_PANEL = """
    QWidget#MainPanel {
        background: rgba(25, 25, 35, 200);
        border-radius: 14px;
    }
"""

# Row 1 容器 stylesheet
STYLE_ROW1 = """
    QWidget#Row1 {
        background: transparent;
        border-radius: 10px 10px 0px 0px;
    }
"""

# 按鈕 stylesheet（深色半透明）
STYLE_BUTTON = """
    QPushButton {
        background: rgba(60, 60, 80, 180);
        color: #E0E0F0;
        border: 1px solid rgba(120, 120, 160, 100);
        border-radius: 8px;
        font-size: 13px;
        font-family: "Microsoft JhengHei", "Microsoft YaHei", sans-serif;
        padding: 6px 12px;
        min-height: 28px;
    }
    QPushButton:hover {
        background: rgba(80, 80, 110, 220);
        border: 1px solid rgba(150, 150, 200, 160);
    }
    QPushButton:pressed {
        background: rgba(40, 40, 60, 220);
    }
    QPushButton:checked {
        background: rgba(70, 100, 160, 220);
        border: 1px solid rgba(100, 140, 220, 200);
        color: #FFFFFF;
    }
    QPushButton:checked:hover {
        background: rgba(85, 115, 175, 240);
    }
"""

# OK/Cancel 按鈕 stylesheet（略有差異色調）
STYLE_BUTTON_OK = """
    QPushButton {
        background: rgba(50, 80, 130, 200);
        color: #D0E0FF;
        border: 1px solid rgba(100, 140, 200, 120);
        border-radius: 8px;
        font-size: 13px;
        font-family: "Microsoft JhengHei", "Microsoft YaHei", sans-serif;
        padding: 6px 20px;
        min-height: 30px;
        min-width: 80px;
    }
    QPushButton:hover {
        background: rgba(65, 100, 160, 230);
    }
    QPushButton:pressed {
        background: rgba(35, 60, 110, 230);
    }
"""

STYLE_BUTTON_CANCEL = """
    QPushButton {
        background: rgba(80, 50, 50, 180);
        color: #FFD0D0;
        border: 1px solid rgba(180, 100, 100, 100);
        border-radius: 8px;
        font-size: 13px;
        font-family: "Microsoft JhengHei", "Microsoft YaHei", sans-serif;
        padding: 6px 20px;
        min-height: 30px;
        min-width: 80px;
    }
    QPushButton:hover {
        background: rgba(110, 60, 60, 210);
    }
    QPushButton:pressed {
        background: rgba(55, 35, 35, 220);
    }
"""

# 關閉按鈕 stylesheet（紅色調，視覺上明顯區隔）
STYLE_BUTTON_CLOSE = """
    QPushButton {
        background: rgba(120, 40, 40, 180);
        color: #FFB0B0;
        border: 1px solid rgba(200, 80, 80, 100);
        border-radius: 8px;
        font-size: 14px;
        font-weight: bold;
        font-family: "Microsoft JhengHei", "Microsoft YaHei", sans-serif;
        padding: 4px 10px;
        min-height: 28px;
        min-width: 32px;
    }
    QPushButton:hover {
        background: rgba(180, 50, 50, 220);
        border: 1px solid rgba(230, 100, 100, 180);
        color: #FFFFFF;
    }
    QPushButton:pressed {
        background: rgba(90, 25, 25, 230);
    }
"""

# 文字輸入框 stylesheet
STYLE_LINE_EDIT = """
    QLineEdit {
        background: rgba(40, 40, 55, 200);
        color: #E0E0F0;
        border: 1px solid rgba(120, 120, 160, 130);
        border-radius: 8px;
        font-size: 13px;
        font-family: "Microsoft JhengHei", "Microsoft YaHei", sans-serif;
        padding: 6px 12px;
        min-height: 30px;
    }
    QLineEdit:focus {
        border: 1px solid rgba(100, 140, 220, 200);
        background: rgba(50, 50, 70, 220);
    }
    QLineEdit::placeholder {
        color: rgba(150, 150, 170, 180);
    }
"""

# GIF 顯示 QLabel stylesheet
STYLE_GIF_LABEL = """
    QLabel {
        background: transparent;
        border: none;
    }
"""

# Row 3 容器 stylesheet
STYLE_ROW3 = """
    QWidget#Row3 {
        background: transparent;
    }
"""
