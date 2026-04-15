# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

MediaPipe參考網址: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker,
開啟電腦Web cam,輸入人名後,找出468個3-dimensional face landmarks,有x,y,z軸的資料,
把這些資料交給LBPH (OpenCV)做分類訓練後,再偵測做人臉辨識,查看該人是誰。

## Architecture

### 模組架構

| 檔案 | 職責 |
|------|------|
| `mp_face_landmarker.py` | MediaPipe FaceLandmarker，回傳 468 個 3D landmark + BoundingBox + KeyPoints |
| `face_feature_3d.py` | IOD 歸一化特徵萃取（方案 A 研究用，方案 B 不使用） |
| `lbph_recognizer.py` | 5 點仿射對齊（`alignFace`）+ `LbphRecognizer`（fit/predict/write/read） |
| `face_recognizer.py` | 整合 MediaPipe + LBPH，對外 API 與 p07 保持一致 |
| `main.py` | CustomTkinter 主程式（以 Refmain.py 為藍圖） |

### 辨識流程（方案 B：5 點對齊）

```
webcam BGR frame
  → MpFaceLandmarker.detect()
      → BoundingBox, Landmarks3D(468×3), KeyPoints
  → _extractFivePts(Landmarks3D)
      → 5 個像素座標：[左眼中心, 右眼中心, 鼻尖, 左嘴角, 右嘴角]
  → alignFace(Frame, FivePts)
      → cv2.estimateAffinePartial2D → warpAffine → cvtColor(GRAY)
      → 100×100 uint8 灰階人臉
  → LbphRecognizer.predict()
      → (label_index, confidence 0~1)
  → label_index → person_name
```

### 模型儲存
- `face_model_lbph.yml`：LBPH 訓練模型（OpenCV XML）
- `face_model_lbph_meta.npz`：人名對應表 + 各人對齊臉部影像（供 RemovePerson 重訓）

## Code Specification

1. 開發語言：Python 3.13
2. UI library：customtkinter
3. 註解請用中文
4. Variable Naming：CamelCase 一致使用
5. Error Handling：所有 API 呼叫必須包含 try-except
6. Function 名稱使用英文，不要用中文

## Design Decisions
1. 要能找出距離比例 (Ratios),不要因為臉離鏡頭近一點（臉變大），座標差值就會變大。模型必須處理這種縮放（Scaling）帶來的干擾。

## UI
請參 Refmain.py 設計UI,
輸入人名,進行學習,並按Detect來推論是誰.
有什麼需要跟我討論的地方都可以說.
最後 Refmain.py 只是供UI設計,請還是以他為藍圖,做出自己的main.py.
Refmain.py這個檔,在專案完成後,我會刪除掉. (我會刪該檔,您不用幫我刪.)
