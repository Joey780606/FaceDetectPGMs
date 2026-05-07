# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

設計一個能讓動畫顯示和切換順暢的Widget小工具.
目前動畫我是以gif圖檔為主,但歡迎您規劃時提供其他動畫設計和顯示的建議.

## Code Specification

1. 開發語言：Python 3.13
2. UI library：若有在gif動畫表現效果較好的UI工具,且商用不需付費的,可以推薦,我之前都用customtkinter,但不確定合不合適. 
3. 註解請用中文.
4. Variable Naming：CamelCase 一致使用.
5. Error Handling：所有 API 呼叫必須包含 try-except.
6. Function 名稱使用英文，不要用中文.

## Design Decisions

1. **切換gif圖檔順暢**：會在gif圖檔播放到一半做gif檔案切換的動作,視覺效果要順暢。
2. **UI元件要顯示的漂亮**：切換不同gif時,可能要顯示相對的UI元件。UI元件也要顯示的漂亮。
3. **商用授權**：均需可安全商用。

## UI
1. 設計為Widget小工具,所以使用的元件,背景最好都是透明的.
2. 畫面出現在螢幕右下角.
3. 目前若以gif變化為設計方向,現在有"gif"目錄,裡面存放gif檔.UI預計有3 rows:
 Row 1: 有三個按鈕.分別是"上一個","下一個","隨機撥放".按"上一個",換上個gif檔. 按"下一個",換下個gif檔. 按"隨機撥放",若gif動畫撥放完,換下個動畫.
 Row 2: gif圖檔顯示區.
 Row 3: 先設計三種型態,第一種是二個button,"OK"和"Cancel". 第二種是文字輸入框. 第三種是Row3看不見. 每切換一個gif檔,就按上述型態輪流顯示.
