# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

開啟電腦 Webcam，輸入人名後，使用 MediaPipe FaceLandmarker 取得 468 個 3D face landmarks（含 x, y, z），
萃取 351 維臉部比例特徵向量，再以 Z-Score 標準化 + k-NN Cosine 相似度進行人臉辨識，判斷此人是誰或為 Unknown。

MediaPipe 參考網址: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

## Architecture

```
Webcam 影像
    │
    ▼
MediaPipe FaceLandmarker（mp_face_landmarker.py）
    │  偵測人臉並輸出 468 個 3D 座標點
    ▼
特徵萃取（face_feature_3d.py）
    │  從 468 點中選 26 個關鍵點，計算 351 維臉部比例特徵向量
    │  - Part A：C(26,2) = 325 維成對 3D 距離 / IOD
    │  - Part B：26 維 z 深度 / IOD
    ▼
Z-Score 標準化 + L2 正規化（cosine_matcher_np.py）
    │  消除「人類臉部拓撲相似」的干擾，突顯個人差異
    ▼
k-NN Cosine 相似度比對（K=7，閾值可調）
    │  與所有訓練樣本比較，多數決投票
    ▼
判斷結果：此人 / Unknown
```

## Code Specification

1. 開發語言：Python 3.13
2. UI library：customtkinter
3. 註解請用中文
4. Variable Naming：CamelCase 一致使用
5. Error Handling：所有 API 呼叫必須包含 try-except
6. Function 名稱使用英文，不要用中文

## Design Decisions

1. **IOD 歸一化（縮放不變性）**：所有距離與 z 深度均除以 IOD（瞳距），確保臉離鏡頭遠近不影響特徵值。
   - IOD = 左眼中心（Landmark 33/159/145/133 平均）與右眼中心（362/386/374/263 平均）的 3D 距離。
   - 若 IOD < 1e-5（MediaPipe 偵測退化），丟棄此幀。

2. **351 維臉部比例特徵**：選取 26 個關鍵 Landmark（鼻子、眼睛、眉毛、嘴巴、下巴、顴骨、額頭），
   計算 325 維成對 3D 距離 + 26 維 z 深度，共 351 維。不使用全部 468 點（Cosine 相似度無法區分）。

3. **Z-Score 標準化**：使用訓練集各維度的全局均值 μ 與標準差 σ 做標準化，
   使向量描述「與平均臉的偏差方向」，而非「這是一張人臉」，Cosine 才具區分能力。

4. **Unknown 偵測**：無需預先收集陌生人樣本。K 個鄰居的加權 Cosine 平均值低於閾值即判為 Unknown。
   UI 提供閾值 Slider（範圍 0.10 ~ 0.99，預設 0.75）供調整嚴格度。

## Main Files

| 檔案 | 功能 |
|------|------|
| `mp_face_landmarker.py` | MediaPipe FaceLandmarker 封裝，輸出 468 個 3D 點 |
| `face_feature_3d.py` | 特徵萃取：468 點 → 351 維臉部比例向量 |
| `cosine_matcher_np.py` | 辨識引擎：Z-Score 標準化 + k-NN Cosine 比對 |
| `face_recognizer.py` | 整合層：學習、辨識、存檔、載入 |
| `main.py` | CustomTkinter UI |
| `face_model.npz` | 儲存所有訓練樣本（原始特徵向量） |
| `face_landmarker.task` | MediaPipe 模型檔 |

## UI

請參 Refmain.py 設計 UI，輸入人名，進行學習，並按 Detect 來推論是誰。
有什麼需要跟我討論的地方都可以說。
Refmain.py 只是供 UI 設計藍圖，請做出自己的 main.py。
Refmain.py 這個檔，在專案完成後，我會刪除掉。（我會刪該檔，您不用幫我刪。）
