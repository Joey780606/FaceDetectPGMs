# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

MediaPipe 參考網址: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

開啟電腦 Webcam，輸入人名後，找出 468 個 3D face landmarks（x, y, z 軸），
萃取 325 維臉部幾何特徵向量，交給 sklearn OneClassSVM 分類器做訓練後，再偵測人臉辨識來者是誰。
每人獨立一個 OneClassSVM，適合陌生人偵測場景。

## Code Specification

1. 開發語言：Python 3.13
2. UI library：customtkinter
3. 註解請用中文
4. Variable Naming：CamelCase 一致使用
5. Error Handling：所有 API 呼叫必須包含 try-except
6. Function 名稱使用英文，不要用中文

## Project Files

| 檔案 | 職責 |
|------|------|
| `main.py` | 主程式 UI（CustomTkinter）+ Webcam 管理 |
| `face_recognizer.py` | 整合偵測、特徵萃取、SVM 的高層 API |
| `svm_classifier_np.py` | OneClassSVM 分類器（每人一個 SVM） |
| `face_feature_3d.py` | 325 維特徵萃取（姿態正規化 + 向量夾角） |
| `face_pose_classifier.py` | 5 級姿態分類（置中/左上/右上/左下/右下） |
| `mp_face_landmarker.py` | MediaPipe 468 點 3D landmark 偵測 |
| `face_model.npz` | 模型檔（訓練資料，執行時自動生成） |
| `face_landmarker.task` | MediaPipe 模型（首次執行自動下載） |

## Design Decisions

1. **臉部寬度歸一化**：所有距離除以左右顴骨間距（臉部寬度），消除臉距鏡頭遠近造成的縮放干擾。
2. **Z-Score + L2 標準化**：消除「人類臉部拓撲相似」造成的干擾，使 SVM 特徵向量真正具有個人區分能力。
3. **辨識引擎（sklearn OneClassSVM）**：
   - 每人獨立一個 OneClassSVM（`kernel='rbf'`, `nu=0.1`, `gamma='scale'`）
   - 推論時取各人 `decision_function()` 最高分者為候選
   - `decision_function > 0`：inlier（認識）；`< 0`：outlier（不認識）
4. **姿態正規化（pose normalization）**：
   - `face_feature_3d.py` 以左右顴骨（X 軸）與下巴/額頭（Y 軸）建立旋轉矩陣 R
   - `R.T` 將所有 landmark 旋轉至正臉座標系，消除 Yaw / Pitch / Roll 影響
   - 訓練與偵測走同一條路，特徵一致
5. **訓練策略**：`FrontalOnly=True`，只收正臉樣本（100 幀目標）。
   姿態正規化後正臉樣本已能覆蓋各角度，單一 SVM per person 即可。
6. **Unknown 偵測（四層）**：
   - 第一層：raw decision score 閾值（`SVM_CONF_THRESH=0.0`；低於閾值 → "Unknown"；slider: -1.0~1.0）
   - 第二層：margin 分差閾值（`SVM_MARGIN_THRESH=0.3`；正臉時 top-1/top-2 分差小 → "Unknown"；slider: 0.0~3.0）
   - 第三層：餘弦驗證（`COSINE_VERIFY_THRESH=-1.0` 關閉；query 與該人平均向量 cosine < 閾值 → "Unknown"；slider: -1.0~0.8）
   - 第四層：KNN 驗證（`KNN_VERIFY_ENABLED=False` 預設關閉）
7. **側臉閾值調整**：非正臉時 score 閾值 −0.1，margin 閾值強制設為 0.0（停用分差檢查）
8. **商用授權**：MediaPipe Apache 2.0、OpenCV Apache 2.0、NumPy BSD、scikit-learn BSD、CustomTkinter CC0、Pillow MIT，均可安全商用。

## UI

`main.py` 已實作完成（Refmain.py 為舊版參考，可刪除）。

輸入人名，按 Learning 進行正臉學習，按 Detect 推論是誰。
- **Remove 按鈕**：移除指定人物的所有訓練資料。
- **Export 按鈕**：將姓名欄指定人物的訓練資料匯出為獨立 .npz 檔（分散訓練用）。
- **Import & Merge 按鈕**：多選個人 .npz 檔，合併進主模型並重訓儲存。
- **SVM 信心度閾值 slider**（-1.0~1.0，預設 0.0）：OneClassSVM raw decision score 閾值。
- **分差閾值 slider**（0.0~3.0，預設 0.3）：top-1 與 top-2 分差閾值。
- **餘弦驗證閾值 slider**（-1.0 關閉 ~ 0.8）：關閉時顯示「關閉」。
