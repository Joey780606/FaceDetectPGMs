# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

MediaPipe參考網址: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker,
開啟電腦Web cam,輸入人名後,找出468個3-dimensional face landmarks,有x,y,z軸的資料,
把這些資料交給Random Forest做分類訓練後,再偵測做人臉辨識,查看該人是誰。

## Architecture

## Code Specification

1. 開發語言：Python 3.13
2. UI library：customtkinter
3. 註解請用中文
4. Variable Naming：CamelCase 一致使用
5. Error Handling：所有 API 呼叫必須包含 try-except
6. Function 名稱使用英文，不要用中文

## Design Decisions
1. 使用相對座標化： 以「鼻尖（Index 1）」為原點 (0, 0, 0)，計算其他點與鼻尖的相對位移。
2. 要能找出距離比例 (Ratios),不要因為臉離鏡頭近一點（臉變大），座標差值就會變大。模型必須處理這種縮放（Scaling）帶來的干擾。

## UI
請參 Refmain.py 設計UI,
輸入人名,進行學習,並按Detect來推論是誰.
有什麼需要跟我討論的地方都可以說.
最後 Refmain.py 只是供UI設計,請還是以他為藍圖,做出自己的main.py.
Refmain.py這個檔,在專案完成後,我會刪除掉. (我會刪該檔,您不用幫我刪.)