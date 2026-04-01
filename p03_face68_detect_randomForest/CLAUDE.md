# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

使用 face_recognition 庫的內建 68 個關鍵點，當電腦的 Webcam 偵測到人臉時，
標記出 68 個關鍵點並交給混合分類器（Random Forest + 馬氏距離驗證）做人臉學習與辨識。

## 檔案職責

| 檔案 | 說明 |
|------|------|
| `main.py` | 主程式，CustomTkinter UI + Webcam 管理 |
| `face_recognizer.py` | `FaceRecognizer` 類別，辨識核心，由 main.py 引入 |
| `face_feature.py` | 從 68 個 landmark 萃取 23 維幾何特徵向量 |
| `random_forest_np.py` | 純 NumPy 實作：`DecisionTree`、`RandomForest`、`OnePerson` |
| `face_model.npz` | 訓練資料儲存檔（執行後自動產生） |

詳細流程與設計說明見 `PLAN.md`。

## Architecture

**人臉偵測**：`face_recognition.face_locations(model="hog")`（純 CPU，不需 GPU）

**特徵萃取**：`face_recognition.face_landmarks()` → 68 個 landmark → 23 維幾何特徵
- 距離特徵（÷ 瞳距 IOD 歸一化）、角度特徵、比例特徵
- IOD < 30px（側臉）→ 跳過此幀

**辨識策略（混合方案）**：
- 1 人已訓練 → `OnePerson`（馬氏距離閾值 4.0）
- 2+ 人已訓練 → `RandomForest` 初步分類 + 各人 `OnePerson` 二次驗證
  - RF max_prob < 0.45 → Unknown
  - RF 選出人名，但馬氏距離 > 4.0 → 改判 Unknown
  - 兩者均通過 → 顯示人名

**模型儲存**：`face_model.npz`（純 NumPy 壓縮，無 sklearn/scipy 依賴）

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
| 旋轉角度 | HOG 約支援 ±30°；學習時顯示提示引導使用者轉頭以收集多角度樣本 |
| GPU | 不需要，純 CPU 可執行（HOG 偵測模型） |
| 序列化 | 儲存原始樣本（X、Y、persons），載入後重新訓練，避免 pickle 相容問題 |

## 授權（商用安全）

所有套件均為商用友善授權：face_recognition (MIT)、dlib (Boost)、numpy (BSD)、OpenCV (Apache 2.0)、customtkinter (MIT)。

> 臉部辨識屬生物特徵資料，商用前需確認符合當地隱私法規（台灣個資法第6條、GDPR 等）。建議在使用前明確告知使用者並取得同意。
