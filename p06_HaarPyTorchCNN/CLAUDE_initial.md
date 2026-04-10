# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

使用Haar Cascade來做人臉偵測,然後用PyTorch 寫一個小型 CNN,
來做人臉辨識的程式.
做到結合 AI 又授權乾淨的唯一方法

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
一定要做到商用安全.

## UI
請參 Refmain.py 設計UI,
輸入人名,進行學習,但一次需要訓練多少張?
頭上下左右轉動到多少度,是可以做人臉辨識的極限值等等,請幫我擬定計劃.
有什麼需要跟我討論的地方都可以說.
最後 Refmain.py 只是供UI設計,請還是以他為藍圖,做出自己的main.py.
Refmain.py這個檔,在專案完成後,我會刪除掉. (我會刪該檔,您不用幫我刪.)