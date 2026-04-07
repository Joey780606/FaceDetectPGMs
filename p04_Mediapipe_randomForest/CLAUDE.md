# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

MediaPipe 參考網址: https://github.com/google-ai-edge/mediapipe
使用 MediaPipe FaceMesh，從 468 個 3D 歸一化座標中選出 68 個等效關鍵點（對應 dlib 68 點位置），
當電腦的 Webcam 偵測到人臉時，標記出關鍵點並交給混合分類器（Random Forest + 馬氏距離驗證）做人臉學習與辨識。

## 檔案職責

| 檔案 | 說明 |
|------|------|
| `main.py` | 主程式，CustomTkinter UI + Webcam 管理 |
| `mp_face_detector.py` | MediaPipe FaceMesh 偵測器，468 點 → 68 點 dict（核心新模組） |
| `face_feature.py` | 從 68 個 landmark 萃取 23 維幾何特徵向量 |
| `face_recognizer.py` | `FaceRecognizer` 類別，辨識核心，由 main.py 引入 |
| `random_forest_np.py` | 純 NumPy 實作：`DecisionTree`、`RandomForest`、`OnePerson` |
| `face_model.npz` | 訓練資料儲存檔（執行後自動產生） |
| `face_landmarker.task` | 是 MediaPipe 的打包模型檔，副檔名 .task 是 MediaPipe 自訂的容器格式,只會讀取,不會修改 |

詳細流程與設計說明見 `PLAN.md`。

## Architecture

**人臉偵測與關鍵點**：`MpFaceDetector.detect(Frame)`
- 以 `mediapipe.solutions.face_mesh` 取得 468 個 3D 歸一化座標
- 依 `DLIB68_MEDIAPIPE_MAP` 選出 68 個等效點，轉為像素 (x, y) tuple
- 輸出格式與 `face_recognition.face_landmarks()` 完全相同（dict by 部位名稱）

**特徵萃取**：`extractFeatures(LandmarkDict)` → 23 維幾何特徵
- 距離特徵（÷ 瞳距 IOD 歸一化）、角度特徵、比例特徵
- IOD < 30px（側臉）→ 跳過此幀

**辨識策略（混合方案）**：
- 1 人已訓練 → `OnePerson`（馬氏距離閾值 4.0）
- 2+ 人已訓練 → `RandomForest` 初步分類 + 各人 `OnePerson` 二次驗證
  - RF max_prob < 0.45 → Unknown
  - RF 選出人名，但馬氏距離 > 4.0 → 改判 Unknown
  - 兩者均通過 → 顯示人名

**模型儲存**：`face_model.npz`（純 NumPy 壓縮，無 sklearn/scipy 依賴）

## MediaPipe 468 → 68 點映射（DLIB68_MEDIAPIPE_MAP）

定義於 `mp_face_detector.py`：

| 部位 | MediaPipe 索引 | 點數 |
|------|---------------|------|
| chin（下顎線） | 234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,454 | 17 |
| left_eyebrow | 70,63,105,66,107 | 5 |
| right_eyebrow | 336,296,334,293,300 | 5 |
| nose_bridge | 168,6,197,195 | 4 |
| nose_tip | 209,198,1,422,429 | 5 |
| left_eye | 33,160,158,133,153,144 | 6 |
| right_eye | 362,385,387,263,373,380 | 6 |
| top_lip | 61,40,37,0,267,270,291,321,405,17,181,78 | 12 |
| bottom_lip | 78,81,13,311,308,402,14,178 | 8 |

> top_lip[0]=61（左嘴角）、top_lip[6]=291（右嘴角），與 face_feature.py 的取點邏輯一致。

## Code Specification

1. 開發語言：Python 3.13
2. UI library：customtkinter
3. 註解請用中文
4. Variable Naming：CamelCase 一致使用
5. Error Handling：所有 API 呼叫必須包含 try-except
6. Function 名稱使用英文，不要用中文

## Design Decisions

| 項目 | 決策 |
|------|------|
| 特徵維度 | 23 維（距離 15、角度 3、比例 5），以瞳距 IOD 歸一化 |
| 缺失值 | -1.0（部位偵測失敗時填入） |
| 側臉處理 | IOD < 30px → extractFeatures 回傳 None，該幀略過 |
| Unknown 判定 | RF 信心度 < 0.45，或馬氏距離 > 4.0 |
| 單人模式 | 只用 OnePerson 馬氏距離，不啟動 Random Forest |
| 多人模式 | RF + 每人各自的馬氏距離驗證器（混合方案） |
| MediaPipe 模式 | static_image_mode=False（影片串流），refine_landmarks=True（啟用虹膜精細化） |
| GPU | 不需要，純 CPU 可執行 |
| 序列化 | 儲存原始樣本（X、Y、persons），載入後重新訓練，避免 pickle 相容問題 |
| 偵測間隔 | 學習 500ms/frame，辨識 300ms/frame（MediaPipe 比 dlib 快，縮短至 300ms） |

## 授權（商用安全）

| 套件 | 授權 | 商用 |
|------|------|------|
| MediaPipe | Apache 2.0 | ✅ 免費商用（含模型權重） |
| OpenCV | Apache 2.0 | ✅ 免費商用 |
| NumPy | BSD | ✅ 免費商用 |
| customtkinter | MIT | ✅ 免費商用 |
| Pillow | HPND (MIT-like) | ✅ 免費商用 |

> 臉部辨識屬生物特徵資料，商用前需確認符合當地隱私法規（台灣個資法第 6 條、GDPR 等）。
> 建議在使用前明確告知使用者並取得同意。
