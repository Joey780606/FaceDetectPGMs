# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

MediaPipe 參考網址: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

開啟電腦 Webcam，輸入人名後，找出 468 個 3D face landmarks（x, y, z 軸），
進行歸一化後，建立五個 SVM 象限（正臉、左上、右上、左下、右下），
依臉的角度進行訓練，並偵測人臉。

## Code Specification

1. 開發語言：Python 3.12.8
2. UI library：customtkinter
3. 註解請用中文
4. Variable Naming：CamelCase 一致使用
5. Error Handling：所有 API 呼叫必須包含 try-except
6. Function 名稱使用英文，不要用中文

## Design Decisions

1. **特徵歸一化（IOD）**：
   - 所有 468 個 landmarks 的 (x, y, z) 各減去兩眼中點，再除以瞳距（IOD）
   - IOD = 左右眼中心的 2D 距離
   - 目的：消除臉距鏡頭遠近造成的縮放干擾
   - 特徵向量維度：468 × 3 = 1404-dim（np.float32）

2. **辨識引擎：sklearn 的 OneClassSVM**
   - 每人、每象限各自訓練一個 OneClassSVM（kernel='rbf', nu=0.1）
   - 推論時取各人 decision_function 最高分者
   - 分數 > 0.0 → 認識；≦ 0.0 → Unknown
   - 優點：不需要多人才能訓練；單人時即可做 Unknown 偵測

3. **五象限分類（Yaw + Pitch）**：
   - Yaw（左右）= (鼻尖.x - 眼中點.x) / IOD，正臉時 ≈ 0
   - Pitch（上下）= (鼻尖.y - 額頭.y) / 臉高 - 0.50，正臉時 ≈ 0
   - **注意**：Pitch 使用臉高歸一化（非 IOD），因為鼻尖天生低於眼睛約 0.3~0.5 IOD，
     若用 IOD 歸一化則 Pitch 永遠為正值，導致所有幀落入「下」象限
   - YAW_THRESHOLD = 0.05（IOD 單位）
   - PITCH_THRESHOLD = 0.06（臉高比例單位）
   - 象限：|Yaw|<0.05 且 |Pitch|<0.06 → 正臉；其餘依 Yaw/Pitch 正負分四角

4. **訓練目標**：每象限每人 20 張（共 100 張），各象限滿 20 自動跳過，全滿自動停止

5. **Unknown 偵測**：每人各象限獨立 OneClassSVM；decision_function ≦ 0 判為 Unknown

6. **模型儲存**：使用 joblib 存成 face_model.npz（含 TrainData + SVMs dict）

## UI

以 Refmain.py 為藍圖設計 main.py（Refmain.py 僅供參考，專案完成後由使用者刪除）。

- Row 0：Detect 按鈕 + 辨識結果
- Row 1：姓名輸入 + Learning + Remove + 五象限進度（學習中才顯示）
- Row 2：Webcam 畫面
- Row 3：Log

## 詳細設計請參 PLAN.md
