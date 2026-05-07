# GIF 顯示區：QLabel + 淡入淡出過場（opacity=0 瞬間才換 GIF）

from PySide6.QtWidgets import QLabel, QGraphicsOpacityEffect
from PySide6.QtCore import Qt, QPropertyAnimation, QSequentialAnimationGroup, QAbstractAnimation
from PySide6.QtGui import QMovie
from style_constants import GIF_DISPLAY_SIZE, CROSSFADE_MS, STYLE_GIF_LABEL


class RowDisplay(QLabel):
    """GIF 顯示區，負責淡入淡出過場切換。"""

    def __init__(self, Parent=None):
        super().__init__(Parent)
        self._Player = None
        self._PendingIndex = 0      # 等待過場完成後要顯示的索引
        self._IsFading = False      # 是否正在過場中

        self._setupUi()
        self._setupAnimation()

    def _setupUi(self) -> None:
        """設定顯示區外觀。"""
        self.setFixedSize(GIF_DISPLAY_SIZE, GIF_DISPLAY_SIZE)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(STYLE_GIF_LABEL)

    def _setupAnimation(self) -> None:
        """建立淡入淡出動畫群組。"""
        # 透明度效果
        self._OpacityEffect = QGraphicsOpacityEffect(self)
        self._OpacityEffect.setOpacity(1.0)
        self.setGraphicsEffect(self._OpacityEffect)

        # 淡出動畫：1.0 → 0.0
        self._FadeOutAnim = QPropertyAnimation(self._OpacityEffect, b'opacity')
        self._FadeOutAnim.setDuration(CROSSFADE_MS)
        self._FadeOutAnim.setStartValue(1.0)
        self._FadeOutAnim.setEndValue(0.0)

        # 淡入動畫：0.0 → 1.0
        self._FadeInAnim = QPropertyAnimation(self._OpacityEffect, b'opacity')
        self._FadeInAnim.setDuration(CROSSFADE_MS)
        self._FadeInAnim.setStartValue(0.0)
        self._FadeInAnim.setEndValue(1.0)

        # 淡出結束時換 GIF，再開始淡入
        self._FadeOutAnim.finished.connect(self._onFadeOutFinished)

    def attachPlayer(self, Player) -> None:
        """連接 GifPlayer，並顯示第一個 GIF。"""
        try:
            self._Player = Player
            if Player and Player.getCount() > 0:
                FirstMovie = Player.getMovie(0)
                if FirstMovie:
                    self.setMovie(FirstMovie)
                    FirstMovie.start()
        except Exception as E:
            print(f'連接 GifPlayer 失敗: {E}')

    def triggerCrossfade(self, NewIndex: int) -> None:
        """由外部呼叫，觸發淡出→換GIF→淡入的過場流程。"""
        try:
            self._PendingIndex = NewIndex
            if self._IsFading:
                # 若正在過場中，直接在淡出完成時換至最新的 PendingIndex
                return
            self._IsFading = True
            self._FadeOutAnim.start()
        except Exception as E:
            print(f'觸發過場失敗: {E}')

    def _onFadeOutFinished(self) -> None:
        """淡出完成後（opacity=0）：換 GIF → 啟動 → 淡入。"""
        try:
            if self._Player is None:
                return

            # 在完全透明的瞬間換 GIF（使用者看不到）
            NewMovie = self._Player.getMovie(self._PendingIndex)
            if NewMovie:
                self.setMovie(NewMovie)
                # 通知 Player 啟動新 GIF（連接 frameChanged 訊號）
                self._Player.startCurrentMovie()

            # 開始淡入
            self._FadeInAnim.finished.connect(self._onFadeInFinished)
            self._FadeInAnim.start()

        except Exception as E:
            print(f'過場換GIF失敗: {E}')
            self._IsFading = False
            self._OpacityEffect.setOpacity(1.0)

    def _onFadeInFinished(self) -> None:
        """淡入完成，重置過場狀態。"""
        try:
            self._FadeInAnim.finished.disconnect(self._onFadeInFinished)
            self._IsFading = False
        except Exception as E:
            print(f'淡入完成處理失敗: {E}')
            self._IsFading = False
