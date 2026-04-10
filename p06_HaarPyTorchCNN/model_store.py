"""
model_store.py

管理訓練資料（臉部 ROI 圖片）與模型檔案的存取。

目錄結構：
    data/faces/{人名}/img_0001.jpg ... img_0060.jpg
    model/face_cnn.pth
"""

import os
import shutil
import cv2
import numpy as np


UNKNOWN_DIR_NAME = "__unknown__"   # 合成 unknown 類別的資料夾名稱


class ModelStore:
    """臉部訓練資料與模型檔案的存取管理。"""

    # 以專案目錄為基準的相對路徑
    _BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR    = os.path.join(_BASE_DIR, "data",  "faces")
    MODEL_DIR   = os.path.join(_BASE_DIR, "model")
    MODEL_PATH  = os.path.join(MODEL_DIR, "face_cnn.pth")

    def __init__(self):
        # 確保目錄存在
        os.makedirs(self.DATA_DIR,  exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # 訓練圖片管理
    # --------------------------------------------------------------------------
    def saveTrainingImage(self, Name: str, Roi: np.ndarray) -> bool:
        """
        儲存單張臉部 ROI 到 data/faces/{Name}/。

        Name: 人員姓名（作為資料夾名稱）
        Roi : 96×96 灰階 numpy array（uint8）
        """
        try:
            PersonDir = os.path.join(self.DATA_DIR, Name)
            os.makedirs(PersonDir, exist_ok=True)
            # 計算下一張圖片的序號
            Existing = [
                F for F in os.listdir(PersonDir)
                if F.lower().endswith(".jpg")
            ]
            NextIdx  = len(Existing) + 1
            FilePath = os.path.join(PersonDir, f"img_{NextIdx:04d}.jpg")
            cv2.imwrite(FilePath, Roi)
            return True
        except Exception as Error:
            print(f"[ModelStore] 儲存圖片失敗：{Error}")
            return False

    def getImageCount(self, Name: str) -> int:
        """回傳某人已儲存的圖片數量。"""
        try:
            PersonDir = os.path.join(self.DATA_DIR, Name)
            if not os.path.isdir(PersonDir):
                return 0
            return len([
                F for F in os.listdir(PersonDir)
                if F.lower().endswith(".jpg")
            ])
        except Exception as Error:
            print(f"[ModelStore] 取得圖片數量失敗：{Error}")
            return 0

    def listPersons(self) -> list:
        """回傳所有已有訓練資料的人員名稱（排序，不含 __unknown__ 內部類別）。"""
        try:
            return sorted([
                D for D in os.listdir(self.DATA_DIR)
                if os.path.isdir(os.path.join(self.DATA_DIR, D))
                and D != UNKNOWN_DIR_NAME
            ])
        except Exception as Error:
            print(f"[ModelStore] 列出人員失敗：{Error}")
            return []

    def removePerson(self, Name: str) -> bool:
        """刪除某人的所有訓練資料，並清除 __unknown__ 樣本（下次訓練重新產生）。"""
        try:
            PersonDir = os.path.join(self.DATA_DIR, Name)
            if os.path.isdir(PersonDir):
                shutil.rmtree(PersonDir)
            # 移除人員後，__unknown__ 的 Type A 樣本已過時，清除讓下次重新產生
            UnknownDir = os.path.join(self.DATA_DIR, UNKNOWN_DIR_NAME)
            if os.path.isdir(UnknownDir):
                shutil.rmtree(UnknownDir)
            return True
        except Exception as Error:
            print(f"[ModelStore] 刪除人員資料失敗：{Error}")
            return False

    # --------------------------------------------------------------------------
    # 路徑查詢
    # --------------------------------------------------------------------------
    def getDataDir(self) -> str:
        """回傳 data/faces/ 根目錄路徑。"""
        return self.DATA_DIR

    def getModelPath(self) -> str:
        """回傳 model/face_cnn.pth 路徑。"""
        return self.MODEL_PATH

    def modelExists(self) -> bool:
        """回傳模型檔案是否存在。"""
        return os.path.exists(self.MODEL_PATH)
