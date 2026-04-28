# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

MediaPipe 參考網址: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

開啟電腦 Webcam，輸入人名後，找出 468 個 3D face landmarks（x, y, z 軸），
萃取 325 維臉部幾何特徵向量，交給sklearn SVM 的 OneClassSVM 分類器做訓練後，再偵測人臉辨識來者是誰。

## Code Specification

1. 開發語言：Python 3.13
2. UI library：customtkinter
3. 註解請用中文
4. Variable Naming：CamelCase 一致使用
5. Error Handling：所有 API 呼叫必須包含 try-except
6. Function 名稱使用英文，不要用中文

## Design Decisions

1. **臉部寬度歸一化**：所有距離除以左右顴骨間距（臉部寬度），消除臉距鏡頭遠近造成的縮放干擾。
2. **Z-Score + L2 標準化**：消除「人類臉部拓撲相似」造成的干擾，使 SVM 特徵向量真正具有個人區分能力。
3. **辨識引擎（sklearn OneClassSVM）**：
   - 一個人臉,一個OneClassSVM
4. **姿態正規化（pose normalization）**：
   - `face_feature_3d.py` 以左右顴骨（X 軸）與下巴/額頭（Y 軸）建立旋轉矩陣 R
   - `R.T` 將所有 landmark 旋轉至正臉座標系，消除 Yaw / Pitch / Roll 影響
   - 訓練與偵測走同一條路，特徵一致
5. **單一 SVM**：姿態正規化後只需一個 SVM，所有角度樣本混合訓練。
   訓練時 `FrontalOnly=True`，只收正臉樣本。
6. **Unknown 偵測（四層）**：
   - 以下面四層為偵測。但這是以LinearSVC的概念設計,如在OneClassSVM不適用,請再提出。
   - 第一層：sigmoid 信心度閾值（低於閾值 → "Unknown"）
   - 第二層：margin 分差閾值（正臉時 top-1/top-2 分差小 → "Unknown"；側臉停用）
   - 第三層：餘弦驗證（`COSINE_VERIFY_THRESH`，預設 −1.0 關閉；query 與該人平均向量 cosine < 閾值 → "Unknown"）
   - 第四層：KNN 驗證（`KNN_VERIFY_ENABLED`，預設關閉）
7. **側臉閾值調整**：非正臉時 sigmoid 閾值 −0.1，margin 閾值強制設為 0.0（停用）
8. **商用授權**：MediaPipe Apache 2.0、OpenCV Apache 2.0、NumPy BSD、scikit-learn BSD、CustomTkinter CC0、Pillow MIT，均可安全商用。

## UI
可參考Refmain.py. (只需參考,還是建立main.py. Refmain.py日後我會刪除)
輸入人名，按 Learning 進行正臉學習，按 Detect 推論是誰。
Remove 按鈕：移除指定人物的所有訓練資料。
Export 按鈕：將姓名欄指定人物的訓練資料匯出為獨立 .npz 檔（分散訓練用）。
Import & Merge 按鈕：多選個人 .npz 檔，合併進主模型並重訓儲存。
SVM 信心度閾值 slider（0.10~0.99）、分差閾值 slider（0.0~3.0）、
餘弦驗證閾值 slider（−1.0 關閉 ～ 0.8）可即時調整。


== 001: > 目前detect時,在正臉時,遇到歪著頭時,分數還是會下降,這是什麼原因,也是468個點會因歪著頭,相關值會有影響嗎?

● 這個問題很值得深入看。讓我先讀 face_feature_3d.py 確認姿態正規化的實作細節。

  Read 1 file (ctrl+o to expand)

● 現在我可以完整解釋這個問題了。

  ---
  歪頭（Roll 旋轉）影響分數的原因

  三個旋轉軸的定義

  ┌───────────────┬──────────┬────────────────────────────────┐
  │      軸       │   動作   │              偵測              │
  ├───────────────┼──────────┼────────────────────────────────┤
  │ Yaw（偏航）   │ 左右轉頭 │ face_pose_classifier.py 有偵測 │
  ├───────────────┼──────────┼────────────────────────────────┤
  │ Pitch（俯仰） │ 上下點頭 │ face_pose_classifier.py 有偵測 │
  ├───────────────┼──────────┼────────────────────────────────┤
  │ Roll（橫滾）  │ 歪著頭   │ ❌ 沒有偵測                    │
  └───────────────┴──────────┴────────────────────────────────┘

  ---
  問題 1：face_pose_classifier.py 不認識 Roll

  # face_pose_classifier.py 只算 Yaw + Pitch
  SignedYaw   = _computeSignedYaw(Lm)
  SignedPitch = _computeSignedPitch(Lm)

  if abs(SignedYaw) < YAW_THRESH and abs(SignedPitch) < PITCH_THRESH:
      return POSE_FRONTAL, SignedYaw, SignedPitch   # ← 歪頭也會走這條路

  歪著頭時，若 Yaw 和 Pitch 都在閾值內，程式認定這是「正臉」，套用最嚴格的信心度閾值。

  ---
  問題 2：face_feature_3d.py 的 R^T 補正理論上能處理 Roll，但有殘差

  _buildFaceRotationMatrix 用顴骨連線做 X 軸：

  XAxis = (Landmarks3D[454] - Landmarks3D[234]) / norm   # 左顴骨 → 右顴骨
  YAxis = Gram-Schmidt(額頭-下巴, 垂直於 XAxis)
  ZAxis = XAxis × YAxis
  R = [XAxis | YAxis | ZAxis]
  Canonical = (R.T @ Centered.T).T   # 旋轉回正臉座標系

  歪頭時 XAxis 在螢幕上是斜的，R^T 理論上能旋轉回來。但問題出在 Z 軸（深度）：

  - MediaPipe 的 x, y 是精確的螢幕座標（0-1 歸一化）
  - MediaPipe 的 z 是神經網路估計的深度，精度比 x, y 差
  - 歪頭時，臉部各點的真實 3D 深度改變，但 MediaPipe 的 z 估計可能有偏差
  - R^T 套用在含有 z 誤差的座標上，補正就不完全

  ---
  問題 3：訓練資料沒有 Roll 樣本

  # face_recognizer.py
  Added, KP, PoseCat, Yaw, Pitch = self._Recognizer.AddSample(
      FrameCopy, PersonName, Retrain=False, FrontalOnly=True  ← 只收 POSE_FRONTAL
  )

  FrontalOnly=True 只過濾 Yaw 和 Pitch，完全不過濾 Roll。但訓練時使用者通常頭是直的，所以訓練樣本事實上都是 Roll≈0°。

  OneClassSVM 的邊界就緊繞這批 Roll≈0° 的特徵，歪頭時即使 R^T 補正，殘差也足以讓分數下降。

  ---
  小結

  歪頭（Roll）
    → face_pose_classifier：還是判 POSE_FRONTAL（只看 Yaw/Pitch）
    → face_feature_3d：R^T 理論補正但 z 估計誤差造成殘差
    → 特徵向量微偏離訓練分布
    → OneClassSVM 邊界緊（nu=0.1）→ 分數下降

  ---
  改善方向

  方案 A（輕量）：在 face_pose_classifier.py 加入 Roll 偵測
  → 歪頭時降低閾值（-0.1），與側臉邏輯一致

  方案 B（根本）：訓練時讓使用者也刻意歪頭收幾張
  → 目前 FrontalOnly 機制無法強制這點，需要另外設計

  方案 C（維持現狀）：如果使用場景是「人直立面對鏡頭」，歪頭本來就不是正常輸入，可接受分數下降後報 Unknown。

  你覺得哪個方向比較符合你的需求？

 ------- 後序處理:
 > 如果用方案A,發現歪頭時,也套入穩定臉追蹤的方式,之前若正臉,就用正臉的辨識結果.這樣可行嗎?

● 完全可行，而且改動範圍很小，是個乾淨的延伸。

  ---
  設計邏輯

  目前 StealEatStep 的觸發條件是「PoseCat != POSE_FRONTAL」（Yaw 或 Pitch 超標）。
  只要把 Roll 也納入觸發條件，架構不需要動：

  目前：
    正臉（Yaw/Pitch 在閾值內）→ 更新 / 清除快取
    側臉（Yaw 或 Pitch 超標） → 套用 IoU 快取

  改成：
    正臉且不歪頭（Yaw/Pitch/Roll 都在閾值內）→ 更新 / 清除快取
    側臉 OR 歪頭（任一超標）               → 套用 IoU 快取

  ---
  需要修改的地方（3 個檔）

  1. face_pose_classifier.py（加 Roll 計算）

  ROLL_THRESH = 0.15   # radians ≈ 8.6°

  def computeRoll(Landmarks3D):
      """歪頭角度：右顴骨 - 左顴骨 的連線相對水平的弧度。"""
      Dx = Landmarks3D[454, 0] - Landmarks3D[234, 0]
      Dy = Landmarks3D[454, 1] - Landmarks3D[234, 1]
      return np.arctan2(Dy, Dx)   # 接近 0 = 頭直立

  # classifyPoseWithValues 回傳多一個 Roll
  return PoseCat, SignedYaw, SignedPitch, Roll

  2. face_recognizer.py（拆解多一個回傳值）

  Predict() 的結果 tuple 從 9 個值改為 10 個：
  (Top, Right, Bottom, Left, Name, Conf, PoseCat, Yaw, Pitch, Roll)

  3. main.py（StealEatStep 觸發條件加 Roll 判斷）

  IsRolled = abs(Roll) > ROLL_THRESH

  if PoseCat == POSE_FRONTAL and not IsRolled:
      # 正臉且頭直立 → 正常更新 / 清除快取
  else:
      # 側臉 OR 歪頭 → IoU 快取邏輯

  ---
  行為比較

  ┌───────────────┬────────────────────┬───────────────────────────────┐
  │     場景      │        目前        │             改後              │
  ├───────────────┼────────────────────┼───────────────────────────────┤
  │ 正臉 + 頭直立 │ 更新快取 ✓         │ 更新快取 ✓                    │
  ├───────────────┼────────────────────┼───────────────────────────────┤
  │ 正臉 + 歪頭   │ 更新快取（分數低） │ 套用快取（沿用上次正臉結果）✓ │
  ├───────────────┼────────────────────┼───────────────────────────────┤
  │ 側臉（轉頭）  │ 套用快取 ✓         │ 套用快取 ✓                    │
  ├───────────────┼────────────────────┼───────────────────────────────┤
  │ 正臉 Unknown  │ 清除快取 ✓         │ 清除快取 ✓                    │
  └───────────────┴────────────────────┴───────────────────────────────┘

  ---
  唯一需要注意的點

  ROLL_THRESH 的值需要實測微調。太小（< 0.08）連正常人姿勢也會觸發；太大（> 0.25）歪頭幅度夠大才有用。預設 0.15
  radians（≈8.6°） 是合理起點。

  ---
  要我直接實作嗎？
  
== 002: OneClassSVM 介紹
OneClassSVM 主要用於異常檢測 (Outlier Detection)。它會學習資料的邊界，將大部分資料點包圍在內，超出邊界的則視為異常。

參數	說明		預設值
kernel	核函數類型。常用的有 'rbf' (徑向基函數)、'linear'、'poly'。(預設'rbf')
nu	最關鍵參數。	代表訓練集中「異常值」比例的上限，同時也是「支持向量」比例的下限。(預設0.5)
gamma	核函數係數（僅限 rbf, poly, sigmoid）。影響決策邊界的平滑度。	(預設'scale')
tol	停止訓練的誤差容忍度。(預設1e-3)
shrinking	是否使用啟發式收縮手段來加速訓練。(預設True)

2. 參數kernel的值和說明:
'linear': 線性核。在原始空間中畫一條直線（或平面）來區分資料。適用於資料特徵非常多，且類別之間本來就能被直線分開時。
'poly'多項式核。透過特徵的次方組合（如 x^2, x^3）建立非線性邊界。適用於資料具有明顯的多項式特徵關係。計算量通常較大，較少作為首選。
'rbf'	徑向基函數 (高斯核)。將資料映射到無限維空間。它會以資料點為中心，向外擴散形成圓形的邊界。最通用。能處理複雜的非線性分布，是大多數任務的預設值。

3. ● 是反過來的。nu 的語意是「訓練樣本中允許落在邊界外的最大比例」：

  ---
  nu 直覺圖解

  nu = 0.1（目前值）            nu = 0.9（你問的）
  ───────────────────           ───────────────────
  訓練樣本 100 個               訓練樣本 100 個

    ┌─────────────┐               ┌──┐
    │  ○○○ ○○○○  │               │○ │  ← 邊界只圍住
    │ ○○○○○○○○○ │               └──┘     ~10 個核心樣本
    │  ○ ○○○○○  │
    └─────────────┘             其他 90 個 ○ 在邊界外
     ~90 個在邊界內

  邊界寬鬆，自己的臉         邊界極緊，自己的臉
  比較容易進來 ✓             也常被擋在外面 ✗

  ---
  參數對照表

  ┌─────────────┬──────────┬───────────────┬────────────────┐
  │     nu      │ 邊界大小 │ 自己臉的分數  │ 陌生人誤判風險 │
  ├─────────────┼──────────┼───────────────┼────────────────┤
  │ 0.05        │ 非常寬鬆 │ 穩定在高正值  │ 較高           │
  ├─────────────┼──────────┼───────────────┼────────────────┤
  │ 0.1（目前） │ 適中偏緊 │ 約 0.2~0.8    │ 適中           │
  ├─────────────┼──────────┼───────────────┼────────────────┤
  │ 0.3         │ 緊       │ 常在 0 附近跳 │ 低             │
  ├─────────────┼──────────┼───────────────┼────────────────┤
  │ 0.9         │ 極緊     │ 幾乎全負值    │ 非常低         │
  └─────────────┴──────────┴───────────────┴────────────────┘

  ---
  針對你目前的問題

  分數在 0 附近跳動 → 邊界剛好切在訓練樣本邊緣，稍微動一下就跨越。

  解法是往下調 nu，讓邊界更寬鬆：

  # svm_classifier_np.py
  SVM_NU = 0.05   # 從 0.1 降到 0.05，邊界更寬，自己的臉更穩定落在正值

  代價是對陌生人的拒絕力稍微降低，但可以透過 SVM_CONF_THRESH slider 來補。
  
== 003: 歪頭無法處理的原因和解法:
  為什麼 IoU 會失效於歪頭

  MediaPipe 的 bounding box 是從所有 468 個 landmark 的 min/max(x, y) 算出來的。歪頭 42°（R:+0.74）時：
  - 額頭 landmark 往右偏、往下偏
  - 下巴 landmark 往左偏、往上偏
  - Bounding box 變高、形狀扭曲、上下偏移

  結果 IoU 從 0.6+ 掉到 0.25 左右，低於 0.35 threshold → 快取不生效。

  解法：中心點距離 fallback

  臉的中心點在螢幕上歪頭時幾乎不動（你人沒往旁邊走）。只要：

  中心距離 / 臉寬 < 0.50  → 視為同一張臉

  現在的判斷邏輯：
  1. IoU ≥ 0.35 → 同一張臉（原本的判斷）
  2. IoU 不足但中心距離 < 臉寬的 50% → 也視為同一張臉（歪頭 fallback）

  debug print 會顯示：
  [MainApp] StealEat IoU=0.21 centerDist=0.08 → 沿用快取

  確認正常運作後，可以把那行 print 移除。
  
== 004: 目前OneClassSVM比較大的問題,是明明訓練過了,但在detect時,即使是正臉的狀態,還是有可能稍微動一下,分數就會變負值,然後又回到正值.
  這是否就是因為純粹判斷像或不像的關係?有時偵測到影像,但其實是有點模糊的?
  
● 這個問題分幾個層面來解釋：

  ---
  為什麼分數會在 0 附近來回跳

  1. OneClassSVM 邊界本身的特性

  OneClassSVM 學的是「把訓練樣本圍起來的超球面邊界」：

  分數 > 0 → 在邊界內（認識）
  分數 < 0 → 在邊界外（不認識）
  分數 ≈ 0 → 剛好站在邊界上

  訓練時 nu=0.1 表示最多允許 10% 的訓練樣本落在邊界外，換句話說邊界刻意畫得緊。實際使用時的樣本因為各種微小變動，很容易在 0 附近震盪。

  2. 兩個主要的擾動來源

  MediaPipe landmark 的 frame-to-frame jitter：
  同一張靜止的臉，每一幀的 468 個點座標都有 ~0.5~2 px 的微小差異（這是 MediaPipe 內部神經網路的輸出雜訊，不是因為臉真的在動）。這微小差異乘以 325 維特徵計算，就足以讓 decision score 跨越 0。

  輕微移動 / 模糊：
  稍微動一下 → landmark 位置改變 → Z-Score + L2 正規化後的向量偏移 → 分數變動。模糊影像則因為 MediaPipe 對模糊較敏感，landmark 誤差更大。

  3. 這是 OneClassSVM 的設計取捨

  ┌────────────────────┬──────────────────┬────────────────────────┐
  │       分類器       │     決策依據     │       抗雜訊能力       │
  ├────────────────────┼──────────────────┼────────────────────────┤
  │ 多類別 SVM（p11）  │ 哪個類別離最遠   │ 相對比較，較穩定       │
  ├────────────────────┼──────────────────┼────────────────────────┤
  │ OneClassSVM（p12） │ 有沒有落在邊界內 │ 絕對判斷，邊界附近易跳 │
  └────────────────────┴──────────────────┴────────────────────────┘

  多類別 SVM 問的是「你比較像誰」，OneClassSVM 問的是「你夠不夠像我」，後者天生對邊界附近的樣本更敏感。

  ---
  目前的緩解機制

  程式已有滑動窗口多數決（DETECT_NONE_DETECT_TARGET = 10）：

  # 收集 10 幀，取最多人投票的名字
  BestName = Counter(self._DetectNoneDtNames).most_common(1)[0][0]

  只要 10 幀中多數是 Joey，UI 就顯示 Joey，偶爾幾幀 Unknown 不影響結果。

  ---
  可調整的方向

  方向 A（最直接）：降低 SVM_CONF_THRESH
  目前預設 0.0，改成 -0.1 或 -0.2，允許「稍微落在邊界外」也算認識：
  SVM_CONF_THRESH = -0.15  # 容忍小幅邊界外的樣本
  風險：對陌生人的辨識容忍度也跟著提高。

  方向 B：增加訓練樣本多樣性
  訓練時刻意輕微移動、改變距離，讓邊界覆蓋更大的範圍。目前固定 100 幀收集，可以改成收更多（200 幀）。

  方向 C：調大 nu
  nu=0.1 → nu=0.2，邊界更寬鬆，訓練樣本有更多落在邊界外的容忍空間，邊界本身就不會那麼緊。

  方向 D（不改 code）：拉大窗口
  把 DETECT_NONE_DETECT_TARGET 從 10 改到 15，讓多數決更穩健。

  ---
  結論： 你的判斷是正確的。這是「純粹判斷像或不像」的固有特性，加上 MediaPipe 每幀都有微小 jitter。建議先試試把 SVM slider 往左拉到 -0.1~-0.2 看效果，不夠再考慮增加訓練樣本或調大 nu。
  
== 004: 目前OneClassSVM的好處是判斷Unknown的效