# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

MediaPipe 參考網址: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

開啟電腦 Webcam，輸入人名後，找出 468 個 3D face landmarks（x, y, z 軸），
萃取 351 維臉部比例特徵向量，依臉部角度,交給相關的類別對 SVM 做分類訓練後，再偵測人臉辨識來者是誰。

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
5. **五類SVM比較**：之前版本會有側臉成效不佳的狀況,此版希望在訓練時,能依照臉部角度變化,分成以下五類:
   正臉,臉朝左上方,臉朝右上方,臉朝左下方,臉朝右下方.
   依MediaPipe取點後,判斷臉的角度進行分類.
   在detect時,依五類評比出最像是誰,並顯示出來.
6. **商用授權**：所有依賴（MediaPipe Apache 2.0、OpenCV Apache 2.0、NumPy BSD、CustomTkinter CC0、Pillow MIT）均為寬鬆授權，可安全商用。


## UI
請參 Refmain.py 設計UI,
輸入人名,進行學習,並按Detect來推論是誰.
有什麼需要跟我討論的地方都可以說.
最後 Refmain.py 只是供UI設計,請還是以他為藍圖,做出自己的main.py.
Refmain.py這個檔,在專案完成後,我會刪除掉. (我會刪該檔,您不用幫我刪.)