# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
