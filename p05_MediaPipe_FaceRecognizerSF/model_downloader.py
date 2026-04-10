"""
model_downloader.py

首次執行時自動下載所需的模型檔案：
  1. face_landmarker.task       — MediaPipe FaceLandmarker（Apache 2.0）
  2. face_recognition_sface_2021dec.onnx — OpenCV FaceRecognizerSF（Apache 2.0）

使用方式：
    from model_downloader import ensureModels
    ensureModels()
"""

import os
import urllib.request

# ──────────────────────────────────────────────────────────────────────────────
# 模型下載來源（皆為 Apache 2.0 授權，商用安全）
# ──────────────────────────────────────────────────────────────────────────────

# MediaPipe FaceLandmarker — Google AI Edge 官方發佈
_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_LANDMARKER_FILE = "face_landmarker.task"

# OpenCV FaceRecognizerSF (SFace) — OpenCV Zoo 官方發佈
_SFNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
_SFNET_FILE = "face_recognition_sface_2021dec.onnx"


def _downloadFile(Url: str, FilePath: str) -> bool:
    """
    從指定 URL 下載檔案並儲存至本機路徑。
    顯示下載進度百分比。

    Returns
    -------
    True 表示下載成功，False 表示失敗。
    """
    try:
        print(f"[ModelDownloader] 下載中：{os.path.basename(FilePath)}")
        print(f"  來源：{Url}")

        def _Progress(BlockCount, BlockSize, TotalSize):
            if TotalSize > 0:
                Pct = min(100.0, BlockCount * BlockSize / TotalSize * 100)
                print(f"\r  進度：{Pct:5.1f}%", end="", flush=True)

        urllib.request.urlretrieve(Url, FilePath, reporthook=_Progress)
        print()  # 換行
        print(f"[ModelDownloader] 下載完成：{FilePath}")
        return True

    except Exception as Error:
        print(f"\n[ModelDownloader] 下載失敗：{Error}")
        # 清除可能產生的不完整檔案
        if os.path.exists(FilePath):
            try:
                os.remove(FilePath)
            except Exception:
                pass
        return False


def ensureModels() -> bool:
    """
    確認所有必要的模型檔案存在，不存在時自動下載。

    Returns
    -------
    True 表示所有模型就緒，False 表示至少一個模型下載失敗。
    """
    AllOk = True

    # 1. MediaPipe FaceLandmarker
    try:
        if not os.path.exists(_LANDMARKER_FILE):
            print(f"[ModelDownloader] 未找到 {_LANDMARKER_FILE}，開始下載...")
            Ok = _downloadFile(_LANDMARKER_URL, _LANDMARKER_FILE)
            if not Ok:
                AllOk = False
        else:
            print(f"[ModelDownloader] 已存在：{_LANDMARKER_FILE}")
    except Exception as Error:
        print(f"[ModelDownloader] FaceLandmarker 模型處理失敗：{Error}")
        AllOk = False

    # 2. OpenCV FaceRecognizerSF
    try:
        if not os.path.exists(_SFNET_FILE):
            print(f"[ModelDownloader] 未找到 {_SFNET_FILE}，開始下載...")
            Ok = _downloadFile(_SFNET_URL, _SFNET_FILE)
            if not Ok:
                AllOk = False
        else:
            print(f"[ModelDownloader] 已存在：{_SFNET_FILE}")
    except Exception as Error:
        print(f"[ModelDownloader] FaceRecognizerSF 模型處理失敗：{Error}")
        AllOk = False

    return AllOk
