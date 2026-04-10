"""
face_detector.py

使用 OpenCV 內建 Haar Cascade 偵測人臉。
回傳每張臉的灰階 ROI（96×96）及在原始 Frame 中的位置座標。
授權：OpenCV Haar Cascade XML 屬於 Apache 2.0，商用安全。
"""

import cv2
import numpy as np

# CNN 輸入尺寸（與 face_recognizer.py 保持一致）
CNN_INPUT_SIZE = 96


class HaarFaceDetector:
    """Haar Cascade 人臉偵測器。"""

    def __init__(self):
        # 使用 OpenCV 內建的 Haar Cascade 模型（Apache 2.0）
        CascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._Cascade = cv2.CascadeClassifier(CascadePath)
        if self._Cascade.empty():
            raise RuntimeError(f"無法載入 Haar Cascade：{CascadePath}")

    def detect(self, Frame: np.ndarray) -> list:
        """
        偵測 Frame 中所有正臉。

        參數:
            Frame: BGR 格式的 numpy array（原始 webcam frame）

        回傳:
            list of (RoiGray96, X, Y, W, H)
            - RoiGray96: 96×96 灰階 numpy array（uint8），供 CNN 使用
            - X, Y, W, H: 在原始 Frame 中的人臉框座標（int）
        """
        try:
            # 轉灰階供 Haar 偵測使用
            GrayFrame = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)

            # 偵測人臉
            Faces = self._Cascade.detectMultiScale(
                GrayFrame,
                scaleFactor=1.1,    # 縮放比例（越小越精細但越慢）
                minNeighbors=5,     # 最少鄰近矩形數（越大誤偵越少）
                minSize=(60, 60)    # 最小偵測尺寸（60px 以下忽略）
            )

            Result = []
            if len(Faces) == 0:
                return Result

            for (X, Y, W, H) in Faces:
                # 裁切臉部區域（灰階）
                FaceRoi = GrayFrame[Y:Y + H, X:X + W]
                # 縮放至 CNN 輸入尺寸
                FaceRoi = cv2.resize(FaceRoi, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                Result.append((FaceRoi, int(X), int(Y), int(W), int(H)))

            return Result

        except Exception as Error:
            print(f"[HaarFaceDetector] 偵測失敗：{Error}")
            return []
