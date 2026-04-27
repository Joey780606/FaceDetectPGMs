# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

MediaPipe 參考網址: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

開啟電腦 Webcam，輸入人名後，找出 468 個 3D face landmarks（x, y, z 軸），
萃取 325 維臉部幾何特徵向量，交給sklearn SVM 的 OneClassSVM 分類器做訓練後，再偵測人臉辨識來者是誰。

## Code Specification

1. 開發語言：Python 3.13
2. UI library：customtkinter
3. 註解請用中文
4. Variable Naming：CamelCase 一致使用
5. Error Handling：所有 API 呼叫必須包含 try-except
6. Function 名稱使用英文，不要用中文

## Design Decisions

1. **臉部寬度歸一化**：所有距離除以左右顴骨間距（臉部寬度），消除臉距鏡頭遠近造成的縮放干擾。
2. **Z-Score + L2 標準化**：消除「人類臉部拓撲相似」造成的干擾，使 SVM 特徵向量真正具有個人區分能力。
3. **辨識引擎（sklearn OneClassSVM）**：
   - 一個人臉,一個OneClassSVM
4. **姿態正規化（pose normalization）**：
   - `face_feature_3d.py` 以左右顴骨（X 軸）與下巴/額頭（Y 軸）建立旋轉矩陣 R
   - `R.T` 將所有 landmark 旋轉至正臉座標系，消除 Yaw / Pitch / Roll 影響
   - 訓練與偵測走同一條路，特徵一致
5. **單一 SVM**：姿態正規化後只需一個 SVM，所有角度樣本混合訓練。
   訓練時 `FrontalOnly=True`，只收正臉樣本。
6. **Unknown 偵測（四層）**：
   - 以下面四層為偵測。但這是以LinearSVC的概念設計,如在OneClassSVM不適用,請再提出。
   - 第一層：sigmoid 信心度閾值（低於閾值 → "Unknown"）
   - 第二層：margin 分差閾值（正臉時 top-1/top-2 分差小 → "Unknown"；側臉停用）
   - 第三層：餘弦驗證（`COSINE_VERIFY_THRESH`，預設 −1.0 關閉；query 與該人平均向量 cosine < 閾值 → "Unknown"）
   - 第四層：KNN 驗證（`KNN_VERIFY_ENABLED`，預設關閉）
7. **側臉閾值調整**：非正臉時 sigmoid 閾值 −0.1，margin 閾值強制設為 0.0（停用）
8. **商用授權**：MediaPipe Apache 2.0、OpenCV Apache 2.0、NumPy BSD、scikit-learn BSD、CustomTkinter CC0、Pillow MIT，均可安全商用。

## UI
可參考Refmain.py. (只需參考,還是建立main.py. Refmain.py日後我會刪除)
輸入人名，按 Learning 進行正臉學習，按 Detect 推論是誰。
Remove 按鈕：移除指定人物的所有訓練資料。
Export 按鈕：將姓名欄指定人物的訓練資料匯出為獨立 .npz 檔（分散訓練用）。
Import & Merge 按鈕：多選個人 .npz 檔，合併進主模型並重訓儲存。
SVM 信心度閾值 slider（0.10~0.99）、分差閾值 slider（0.0~3.0）、
餘弦驗證閾值 slider（−1.0 關閉 ～ 0.8）可即時調整。
