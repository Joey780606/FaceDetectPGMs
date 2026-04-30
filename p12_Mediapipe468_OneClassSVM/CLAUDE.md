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
| `face_pose_classifier.py` | 5 級姿態分類（置中/左上/右上/左下/右下）+ Roll 偵測 |
| `mp_face_landmarker.py` | MediaPipe 468 點 3D landmark 偵測 |
| `face_model.npz` | 模型檔（訓練資料，執行時自動生成） |
| `face_landmarker.task` | MediaPipe 模型（首次執行自動下載） |

## Design Decisions

1. **臉部寬度歸一化**：所有距離除以左右顴骨間距（臉部寬度），消除臉距鏡頭遠近造成的縮放干擾。

2. **Z-Score + L2 標準化**：消除「人類臉部拓撲相似」造成的干擾，使 SVM 特徵向量真正具有個人區分能力。
   - 注意：GlobalMean/GlobalStd 由所有人資料計算，若各人樣本數差異懸殊（如 A:1000、B:100），Mean 會偏向樣本多的人，輕微影響樣本少者的歸一化品質。建議各人樣本數盡量接近。

3. **辨識引擎（sklearn OneClassSVM）**：
   - 每人獨立一個 OneClassSVM（`kernel='rbf'`, `nu=0.1`, `gamma='scale'`）
   - 推論時取各人 `decision_function()` 最高分者為候選
   - `decision_function > 0`：inlier（認識）；`< 0`：outlier（不認識）
   - `nu` 語意：訓練樣本中允許落在邊界外的最大比例；**越高邊界越緊**（更嚴格但更容易誤拒自己），越低邊界越寬鬆（更穩定但對陌生人容忍度較高）

4. **姿態正規化（pose normalization）**：
   - `face_feature_3d.py` 以左右顴骨（X 軸）與下巴/額頭（Y 軸）建立旋轉矩陣 R
   - `R.T` 將所有 landmark 旋轉至正臉座標系，理論上能消除 Yaw / Pitch / Roll 三軸影響
   - 實際上 MediaPipe z 軸為估計深度（精度低於 x, y），Roll 補正有殘差；歪頭時分數仍會輕微下降
   - 訓練與偵測走同一條路，特徵一致

5. **訓練策略**：`FrontalOnly=True`，只收正臉樣本（100 幀目標）。
   姿態正規化後正臉樣本已能覆蓋各角度，單一 SVM per person 即可。

6. **Unknown 偵測（四層）**：
   - 第一層：raw decision score 閾值（`SVM_CONF_THRESH=0.0`；低於閾值 → "Unknown"；slider: -1.0~1.0）
   - 第二層：margin 分差閾值（`SVM_MARGIN_THRESH=0.3`；正臉時 top-1/top-2 分差小 → "Unknown"；slider: 0.0~3.0）
   - 第三層：餘弦驗證（`COSINE_VERIFY_THRESH=-1.0` 關閉；query 與該人平均向量 cosine < 閾值 → "Unknown"；slider: -1.0~0.8）
   - 第四層：KNN 驗證（`KNN_VERIFY_ENABLED=False` 預設關閉）

7. **姿態閾值調整（Yaw / Pitch / Roll）**：
   - `face_pose_classifier.py` 偵測三軸：Yaw（左右轉）、Pitch（上下俯仰）、Roll（歪頭）
   - `ROLL_THRESH=0.15 rad`（≈8.6°）：超過視為歪頭
   - 側臉（Yaw/Pitch 超標）或歪頭（Roll 超標）時：score 閾值 −0.1，margin 閾值強制 0.0（停用）
   - `classifyPoseWithValues()` 回傳 4-tuple：`(PoseCat, Yaw, Pitch, Roll)`

8. **時序穩定追蹤（StealEatStep）**：
   - 正臉且頭直立（Yaw/Pitch/Roll 均在閾值內）辨識成功 → 更新快取 `_StableFace`
   - 正臉 Unknown 且分數 < `STABLE_FACE_CLEAR_THRESH(-0.30)` → 清快取（真的不認識）
   - 正臉 Unknown 但分數介於 −0.30~0（邊界模糊）→ 沿用快取（避免微小 jitter 導致名稱閃爍）
   - 側臉或歪頭 → `_isSameFace()` 判斷是否同一張臉，符合則沿用快取
   - 連續 `STABLE_FACE_MAX_MISS=10` tick 偵測不到臉 → 清快取

9. **_isSameFace()：IoU + 中心點距離雙重判斷**：
   - 主判斷：IoU ≥ `STABLE_FACE_IOU_THRESH(0.35)` → 同一張臉
   - Fallback：IoU 不足時，中心點距離 / 臉寬 < `STABLE_FACE_CENTER_THRESH(0.50)` → 同一張臉
   - 歪頭時 bounding box 因 landmark 極值偏移而形狀改變，IoU 易低估；中心點幾乎不動，作為更穩定的判斷依據

10. **PoseLabels debug 輸出**：
    - `svm_classifier_np.py predict()` 接受 `PoseLabels: list` 參數
    - `face_recognizer.py` 組合姿態字串（正臉/歪頭/側臉+角度值）傳入
    - 輸出格式：`[OCSVM/全角度][正臉 Y:+0.02] Scores=[...] thresh=0.00 → Joey cos=0.895`

11. **Predict result tuple（10 值）**：
    ```python
    (Top, Right, Bottom, Left, Name, Conf, PoseCat, Yaw, Pitch, Roll)
    ```
    `main.py` 中所有解包必須對應 10 個值。

12. **商用授權**：MediaPipe Apache 2.0、OpenCV Apache 2.0、NumPy BSD、scikit-learn BSD、CustomTkinter CC0、Pillow MIT，均可安全商用。

## UI

`main.py` 已實作完成（Refmain.py 為舊版參考，可刪除）。

輸入人名，按 Learning 進行正臉學習，按 Detect 推論是誰。
- **Remove 按鈕**：移除指定人物的所有訓練資料。
- **Export 按鈕**：將姓名欄指定人物的訓練資料匯出為獨立 .npz 檔（分散訓練用）。
- **Import & Merge 按鈕**：多選個人 .npz 檔，合併進主模型並重訓儲存。
- **SVM 信心度閾值 slider**（-1.0~1.0，預設 0.0）：OneClassSVM raw decision score 閾值。
- **分差閾值 slider**（0.0~3.0，預設 0.3）：top-1 與 top-2 分差閾值。
- **餘弦驗證閾值 slider**（-1.0 關閉 ~ 0.8）：關閉時顯示「關閉」。
- **姿態顯示**：即時顯示 Yaw / Pitch / Roll 原始值，格式 `Y:+0.02 P:+0.01 R:+0.03`。

## 調優建議

| 問題 | 建議調整 |
|------|---------|
| 分數在 0 附近跳動 | 降低 `SVM_NU`（如 0.05）或降低 `SVM_CONF_THRESH` slider |
| 側臉/歪頭變 Unknown | `STABLE_FACE_CLEAR_THRESH` 調高（如 -0.20）或降低 SVM slider |
| 陌生人被誤認 | 提高 `SVM_CONF_THRESH` slider 或提高 `SVM_NU` |
| 歪頭快取失效 | 調高 `STABLE_FACE_CENTER_THRESH`（如 0.60） |
