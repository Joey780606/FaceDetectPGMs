# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

MediaPipe 參考網址: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

開啟電腦 Webcam，輸入人名後，找出 468 個 3D face landmarks（x, y, z 軸），
萃取 351 維臉部比例特徵向量，交給線性 SVM 做分類訓練後，再偵測人臉辨識來者是誰。

## Architecture

```
Webcam → MediaPipe FaceLandmarker（468 個 3D 點）
       → face_feature_3d.py（351 維比例特徵：325 成對距離 + 26 z 深度，IOD 歸一化）
       → svm_classifier_np.py（Z-Score + L2 正規化 → OvR 線性 SVM / 最大 Cosine）
       → face_recognizer.py（整合層：學習、推論、存檔、載入）
       → main.py（CustomTkinter UI）
```

詳細設計說明請參閱 `PLAN.md`。

## Code Specification

1. 開發語言：Python 3.13
2. UI library：customtkinter
3. 註解請用中文
4. Variable Naming：CamelCase 一致使用
5. Error Handling：所有 API 呼叫必須包含 try-except
6. Function 名稱使用英文，不要用中文

## Design Decisions

1. **IOD 歸一化**：所有距離除以瞳距（IOD），消除臉距鏡頭遠近造成的縮放干擾。
2. **Z-Score 標準化**：消除「人類臉部拓撲相似」造成的干擾，使 Cosine 相似度真正具有個人區分能力。
3. **辨識引擎（純 NumPy）**：
   - 多人（≥ 2 人）：OvR 線性 SVM，SGD hinge loss 訓練（`svm_classifier_np.py`）
   - 單人（1 人）：最大 Cosine 相似度比對
   - 不依賴 sklearn / scipy（避免 Windows Application Control 封鎖 DLL）
4. **Unknown 偵測**：不需預先收集陌生人樣本，SVM 分數低於信心度閾值即判為 Unknown。
5. **動態閾值補償（頭部姿態）**：推論時估算水平（Yaw）與垂直（Pitch）轉角比例，
   以兩軸 2-norm 合成後動態降低信心度閾值（最多補償 0.20），避免側臉或抬/低頭被誤判為 Unknown。
   - Yaw：左右顴骨（landmark 234, 454）與鼻尖（landmark 1）的 x 軸不對稱度
   - Pitch：眼睛中心 y 位置相對額頭（landmark 10）到下巴（landmark 152）的偏移量
6. **商用授權**：所有依賴（MediaPipe Apache 2.0、OpenCV Apache 2.0、NumPy BSD、CustomTkinter CC0、Pillow MIT）均為寬鬆授權，可安全商用。

## Key Files

| 檔案 | 用途 |
|------|------|
| `mp_face_landmarker.py` | MediaPipe 封裝，勿修改 |
| `face_feature_3d.py` | 特徵萃取邏輯，勿修改 |
| `svm_classifier_np.py` | 核心辨識引擎（p10 新增） |
| `cosine_matcher_np.py` | 舊版 k-NN 引擎（保留參考，不在主流程） |
| `face_recognizer.py` | 整合層 |
| `main.py` | UI 主程式 |
| `face_model.npz` | 訓練資料（刪除可重新訓練） |
| `PLAN.md` | 完整技術設計文件 |

## UI

輸入人名，按 Learning 進行學習（收集 100 幀），按 Detect 推論來者是誰。
信心度閾值 Slider 可即時調整嚴格程度（預設 0.60）。
Detect 期間 UI 會即時顯示估算的頭部姿態（左右轉角 + 上下傾角），格式如：
`左右：約 18°（微轉）  上下：約 8°（水平）`
