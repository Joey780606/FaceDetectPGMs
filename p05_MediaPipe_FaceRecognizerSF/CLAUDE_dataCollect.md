# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Joey參考資料1
1. https://blog.csdn.net/ljygood2/article/details/144446082
2. https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html

## Joey參考資料2
1.
將 MediaPipe 與 OpenCV DNN (YuNet / FaceRecognizerSF) 結合，是目前在商用法律合規性、效能、以及精準度之間最平衡的方案之一。

1. 為什麼這個組合在「商用」上相對安全？
OpenCV 官方近年來為了應對商用需求，在其擴展庫與範例中推行了一套名為 YuNet (偵測) 與 FaceRecognizerSF (辨識) 的模型。

模型授權： 這些模型通常以 Apache 2.0 授權釋出，這是對商業極其友善的協議，允許修改、散佈且不強制開源你的代碼。

訓練資料： FaceRecognizerSF 主要基於 WiderFace 和 MS1M 等較為大型且在學術/商用界限定義較清晰的資料集進行訓練。

官方維護： 這些模型是由 OpenCV 官方團隊（如英特爾或相關貢獻者）審核過的，比起個人開發者的 GitHub 專案，法律風險低很多。

2. MediaPipe 如何與 OpenCV DNN 結合？
雖然 OpenCV 有自己的偵測器（YuNet），但 MediaPipe 在「穩定性」和「多角度追蹤」上表現優異。你可以採取以下「混血」架構：

運作流程圖：
MediaPipe (偵測與對齊)： 利用 MediaPipe 抓取 478 個特徵點，計算臉部的旋轉角度（Roll, Pitch, Yaw），並根據眼睛與嘴巴的位置將臉部裁切並「拉正」。

OpenCV DNN (特徵提取)： 將對齊後的臉部圖片（通常是 112x112 像素）丟進 OpenCV 的 FaceRecognizerSF。

向量比對： FaceRecognizerSF 會回傳一串 128 維的浮點數向量。你只需計算兩組向量的 L2 距離 或 餘弦相似度 即可完成辨識。

## Joey參考資料3
[FaceAligner] FaceLandmarker 載入失敗：Unable to open file at face_landmarker.task
[FaceRecognizer] 找不到模型檔：face_recognition_sface_2021dec.onnx
[ModelDownloader] 未找到 face_landmarker.task，開始下載...
[ModelDownloader] 下載中：face_landmarker.task
  來源：https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  進度：100.0%
[ModelDownloader] 下載完成：face_landmarker.task
[ModelDownloader] 未找到 face_recognition_sface_2021dec.onnx，開始下載...
[ModelDownloader] 下載中：face_recognition_sface_2021dec.onnx
  來源：https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
  
## Project Goal

MediaPipe參考網址: https://github.com/google-ai-edge/mediapipe
使用 MediaPipe, 找出像 face_recognition 庫的 68 個關鍵點，當電腦的 Webcam 偵測到人臉時，
標記出 68 個關鍵點並交給混合分類器（Random Forest + 馬氏距離驗證）做人臉學習與辨識。


## Architecture

## Code Specification

1. 開發語言：Python 3.13
2. UI library：customtkinter
3. 註解請用中文
4. Variable Naming：CamelCase 一致使用
5. Error Handling：所有 API 呼叫必須包含 try-except
6. Function 名稱使用英文，不要用中文

## Design Decisions


## 授權（商用安全）
一定要做到商用安全,像 MediaPipe 的訓練集是否可以免費商用? 若是我們有訓練人臉時所需要做的商用保護等,請幫我們考慮.
