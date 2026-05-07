# 進入點：啟動 GIF Widget 小工具

import sys
from PySide6.QtWidgets import QApplication
from widget_window import WidgetWindow


if __name__ == '__main__':
    try:
        App = QApplication(sys.argv)
        App.setApplicationName('GifWidget')
        # 使程式在所有視窗關閉後仍維持執行（Widget 模式）
        App.setQuitOnLastWindowClosed(False)

        Window = WidgetWindow()
        Window.show()

        sys.exit(App.exec())
    except Exception as E:
        print(f'啟動失敗: {E}')
        sys.exit(1)
