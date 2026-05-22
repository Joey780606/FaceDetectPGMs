import cv2
import mediapipe as mp
import os
import re
import yt_dlp
from mediapipe.tasks import python as MpPython
from mediapipe.tasks.python import vision as MpVision


class FaceTrack:
    """記錄單一人臉在影片中的追蹤資料"""

    def __init__(self, TrackId, FirstFrameIdx, Box, IsFrontal, IsAlone):
        self.TrackId = TrackId
        self.Boxes = {FirstFrameIdx: Box}          # FrameIdx → (x, y, w, h), 建立 dict, key是 FrameIdx(int), value 是 Box (tuple,臉部位置框)
        self.IsFrontal = {FirstFrameIdx: IsFrontal}  # FrameIdx → bool
        self.IsAlone = {FirstFrameIdx: IsAlone}      # FrameIdx → bool
        self.FirstFrame = FirstFrameIdx
        self.LastFrame = FirstFrameIdx

    def update(self, FrameIdx, Box, IsFrontal, IsAlone):
        self.Boxes[FrameIdx] = Box
        self.IsFrontal[FrameIdx] = IsFrontal
        self.IsAlone[FrameIdx] = IsAlone
        self.LastFrame = FrameIdx


class FaceProcessor:
    """負責下載影片、偵測人臉、套用三層過濾、存出片段"""

    # 正臉對稱比值範圍（左右距離比）
    FRONTAL_RATIO_MIN = 0.6
    FRONTAL_RATIO_MAX = 1.67
    # 最小臉部高度（相對影片高度的比例）
    MIN_FACE_HEIGHT_RATIO = 0.08
    # 過濾門檻（秒）
    MIN_CONTINUOUS_SEC = 10.0
    MIN_FRONTAL_SEC = 8.0
    # IoU 最低匹配門檻
    IOU_THRESHOLD = 0.3

    def __init__(self, ProgressCallback=None, LogCallback=None):
        self.ProgressCallback = ProgressCallback
        self.LogCallback = LogCallback
        self._NextTrackId = 0

    def _log(self, Msg):
        if self.LogCallback:
            self.LogCallback(Msg)

    def _progress(self, Pct, Msg):
        if self.ProgressCallback:
            self.ProgressCallback(Pct, Msg)

    def process(self, VideoUrl, OutputDir):
        """主入口：下載 → 偵測 → 過濾 → 存檔，回傳已存檔名列表"""
        os.makedirs(OutputDir, exist_ok=True)

        # 步驟一：下載影片
        self._log("開始下載影片...")
        self._progress(0.0, "下載中...")
        TmpPath = os.path.join(OutputDir, "_tmp_download.mp4")
        try:
            VideoPath = self._downloadVideo(VideoUrl, TmpPath)
        except Exception as E:
            raise RuntimeError(f"影片下載失敗：{E}")
        self._log(f"下載完成：{os.path.basename(VideoPath)}")

        # 步驟二：逐幀偵測人臉
        self._log("開始分析人臉...")
        self._progress(0.1, "分析人臉中...")
        try:
            AllTracks = self._runDetectionPass(VideoPath)
        except Exception as E:
            raise RuntimeError(f"人臉偵測失敗：{E}")
        self._log(f"共追蹤到 {len(AllTracks)} 個人臉軌跡")

        # 取得影片 FPS
        Cap = cv2.VideoCapture(VideoPath)
        Fps = Cap.get(cv2.CAP_PROP_FPS) or 30.0
        Cap.release()

        # 步驟三：三層過濾
        TracksF1 = self._filterByDuration(AllTracks, Fps)
        self._log(f"Filter 1（連續≥{self.MIN_CONTINUOUS_SEC:.0f}秒）通過：{len(TracksF1)} 人")

        TracksF2 = self._filterByFrontal(TracksF1, Fps)
        self._log(f"Filter 2（正臉≥{self.MIN_FRONTAL_SEC:.0f}秒）通過：{len(TracksF2)} 人")

        # Filter 3 暫時停用（條件過嚴，日後視需要再開啟）
        # TracksF3 = self._filterByAlone(TracksF2)
        # self._log(f"Filter 3（全程獨自出現）通過：{len(TracksF3)} 人")
        TracksF3 = TracksF2

        # 步驟四：存出片段
        SavedFiles = []
        Total = max(len(TracksF3), 1)
        for I, Track in enumerate(TracksF3):
            self._progress(0.9 + 0.1 * I / Total, f"存檔 {I + 1}/{len(TracksF3)}...")   # 字串整合,重要
            try:
                FileName = self._saveSegment(VideoPath, Track, OutputDir, Fps)
                SavedFiles.append(FileName)
                self._log(f"已存：{FileName}")
            except Exception as E:
                self._log(f"存檔失敗（TrackId={Track.TrackId}）：{E}")

        # 清理暫存下載檔（暫存檔與輸出片段名稱不同，可直接刪）
        try:
            if os.path.exists(TmpPath):
                os.remove(TmpPath)
        except Exception:
            pass

        self._progress(1.0, "完成")
        return SavedFiles

    def _downloadVideo(self, VideoUrl, TmpPath):
        """使用 yt-dlp 下載影片，回傳實際檔案路徑"""
        # 先刪舊暫存檔，否則 yt-dlp 會跳過已存在的檔案
        if os.path.exists(TmpPath):
            os.remove(TmpPath)

        YdlOpts = {
            'format': 'best[ext=mp4]/best', # 優先 mp4 格式，沒有才接受其他格式質（可能是 webm）
            'outtmpl': TmpPath,
            'quiet': True,
            'no_warnings': True,
            'overwrites': True,  # 強制覆寫，雙重保險
        }
        with yt_dlp.YoutubeDL(YdlOpts) as Ydl:
            Ydl.download([VideoUrl])

        # yt-dlp 可能自動附加副檔名
        # yt-dlp 下載時，outtmpl 設定的是 _tmp_download.mp4，但它可能：
        # - 實際下載成 _tmp_download.webm（因為影片源沒有 mp4）, YouTube 內部儲存的影片很多是 WebM 格式（尤其是高畫質版本）。
        # - 下載中途產生 _tmp_download.mp4.part（未完成的暫存）

        if not os.path.exists(TmpPath):
            Dir = os.path.dirname(TmpPath)
            Base = os.path.splitext(os.path.basename(TmpPath))[0]   # 取檔案的主檔名（不含副檔名）
            for F in os.listdir(Dir):   # 在掃描目錄，找出「以 _tmp_download 開頭，且副檔名不是 .part」的檔案。
                if F.startswith(Base) and not F.endswith('.part'):
                    return os.path.join(Dir, F)

        return TmpPath

    def _runDetectionPass(self, VideoPath):
        """逐幀跑 Mediapipe FaceLandmarker，建立並回傳所有 FaceTrack"""
        ModelPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
        if not os.path.exists(ModelPath):
            raise RuntimeError(f"找不到 Mediapipe 模型檔：{ModelPath}")

        BaseOptions = MpPython.BaseOptions(model_asset_path=ModelPath)
        Options = MpVision.FaceLandmarkerOptions(
            base_options=BaseOptions,
            running_mode=MpVision.RunningMode.VIDEO,
            num_faces=10,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )

        Cap = cv2.VideoCapture(VideoPath)
        if not Cap.isOpened():
            raise RuntimeError(f"無法開啟影片：{VideoPath}")

        Fps = Cap.get(cv2.CAP_PROP_FPS) or 30.0
        TotalFrames = int(Cap.get(cv2.CAP_PROP_FRAME_COUNT))
        FrameW = int(Cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        FrameH = int(Cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 超過此幀數未出現則關閉 track
        GapFrames = max(5, int(Fps * 0.3))

        ActiveTracks = []
        ClosedTracks = []
        FrameIdx = 0

        try:
            Detector = MpVision.FaceLandmarker.create_from_options(Options)
            with Detector:
                while True:
                    Ret, Frame = Cap.read()
                    if not Ret:
                        break

                    FrameRgb = cv2.cvtColor(Frame, cv2.COLOR_BGR2RGB)   # 將cv2讀取的 BGR 格式轉成 Mediapipe 需要的 RGB 格式
                    MpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=FrameRgb)  # 建立 Mediapipe 圖像物件，指定Mediapipe格式為 SRGB（即一般 RGB）
                    TimestampMs = int(FrameIdx * 1000 / Fps)

                    try:
                        Result = Detector.detect_for_video(MpImage, TimestampMs)    # 呼叫 Mediapipe 的偵測函式，傳入圖像和對應的時間戳（毫秒），回傳偵測結果物件
                    except Exception:
                        FrameIdx += 1
                        continue

                    # 解析偵測結果
                    Faces = []  # [(Box, IsFrontal)]
                    for FaceLandmarks in Result.face_landmarks: # Result.face_landmarks 是 Mediapipe 回傳的偵測結果物件中的一個屬性，包含了所有偵測到的人臉的關鍵點資料
                        Xs = [Lm.x for Lm in FaceLandmarks] # 從每個關鍵點物件 Lm 中提取 x 座標，建立一個列表 Xs
                        Ys = [Lm.y for Lm in FaceLandmarks]
                        X1 = int(min(Xs) * FrameW)
                        Y1 = int(min(Ys) * FrameH)
                        X2 = int(max(Xs) * FrameW)
                        Y2 = int(max(Ys) * FrameH)
                        BoxH = Y2 - Y1
                        # 忽略太小的臉
                        if BoxH < FrameH * self.MIN_FACE_HEIGHT_RATIO:
                            continue
                        Box = (X1, Y1, X2 - X1, BoxH)
                        IsFrontal = self._detectFrontality(FaceLandmarks)   #bool,判斷是否為正臉
                        Faces.append((Box, IsFrontal))

                    # 此幀只有一張臉才算獨自出現
                    IsAlone = (len(Faces) == 1)

                    self._matchFacesToTracks(Faces, ActiveTracks, ClosedTracks, FrameIdx, IsAlone, GapFrames)

                    FrameIdx += 1

                    # 每 30 幀回報一次進度
                    if FrameIdx % 30 == 0 and TotalFrames > 0:
                        Pct = 0.1 + 0.8 * (FrameIdx / TotalFrames)
                        self._progress(Pct, f"分析中 {FrameIdx}/{TotalFrames} 幀")
        finally:
            Cap.release()

        # 仍活躍的 tracks 也納入結果
        ClosedTracks.extend(ActiveTracks)
        return ClosedTracks

    def _matchFacesToTracks(self, Faces, ActiveTracks, ClosedTracks, FrameIdx, IsAlone, GapFrames):
        """以 IoU 配對新偵測臉與現有 track；未匹配的建立新 track；過期的移至 ClosedTracks"""
        MatchedTrackIds = set()
        MatchedFaceIdxs = set()

        # 貪婪匹配：每張臉找 IoU 最高的活躍 track
        for FaceIdx, (Box, IsFrontal) in enumerate(Faces):
            BestIoU = self.IOU_THRESHOLD
            BestTrack = None
            for Track in ActiveTracks:
                if Track.TrackId in MatchedTrackIds:
                    continue
                LastBox = Track.Boxes[Track.LastFrame]
                IoU = self._calcIoU(Box, LastBox)
                if IoU > BestIoU:
                    BestIoU = IoU
                    BestTrack = Track

            if BestTrack is not None:
                BestTrack.update(FrameIdx, Box, IsFrontal, IsAlone)
                MatchedTrackIds.add(BestTrack.TrackId)
                MatchedFaceIdxs.add(FaceIdx)

        # 未匹配的偵測結果建立新 track
        for FaceIdx, (Box, IsFrontal) in enumerate(Faces):
            if FaceIdx not in MatchedFaceIdxs:
                NewTrack = FaceTrack(self._NextTrackId, FrameIdx, Box, IsFrontal, IsAlone)
                self._NextTrackId += 1
                ActiveTracks.append(NewTrack)

        # 超過 gap 容忍未更新的 track 關閉
        ToClose = [
            T for T in ActiveTracks
            if T.TrackId not in MatchedTrackIds and (FrameIdx - T.LastFrame) > GapFrames
        ]
        for Track in ToClose:
            ActiveTracks.remove(Track)
            ClosedTracks.append(Track)

    def _calcIoU(self, BoxA, BoxB):
        """計算兩個 (x, y, w, h) 矩形的 IoU"""
        Ax1, Ay1, Aw, Ah = BoxA
        Bx1, By1, Bw, Bh = BoxB
        Ax2, Ay2 = Ax1 + Aw, Ay1 + Ah
        Bx2, By2 = Bx1 + Bw, By1 + Bh

        InterX1 = max(Ax1, Bx1)
        InterY1 = max(Ay1, By1)
        InterX2 = min(Ax2, Bx2)
        InterY2 = min(Ay2, By2)

        InterArea = max(0, InterX2 - InterX1) * max(0, InterY2 - InterY1)
        UnionArea = Aw * Ah + Bw * Bh - InterArea
        return InterArea / UnionArea if UnionArea > 0 else 0.0

    def _detectFrontality(self, FaceLandmarks):
        """用鼻尖到左右眼的水平距離比判斷是否為正臉"""
        NoseTip = FaceLandmarks[4]
        # 左眼內外角
        LeftEyeX = (FaceLandmarks[33].x + FaceLandmarks[133].x) / 2
        # 右眼內外角
        RightEyeX = (FaceLandmarks[362].x + FaceLandmarks[263].x) / 2
        NoseX = NoseTip.x

        RightDist = abs(RightEyeX - NoseX)
        if RightDist < 1e-6:
            return False

        Ratio = abs(NoseX - LeftEyeX) / RightDist
        return self.FRONTAL_RATIO_MIN <= Ratio <= self.FRONTAL_RATIO_MAX

    def _filterByDuration(self, Tracks, Fps):
        """Filter 1：連續出現時間 ≥ MIN_CONTINUOUS_SEC"""
        Result = []
        for Track in Tracks:
            DurationSec = (Track.LastFrame - Track.FirstFrame + 1) / Fps
            if DurationSec >= self.MIN_CONTINUOUS_SEC:
                Result.append(Track)
        return Result

    def _filterByFrontal(self, Tracks, Fps):
        """Filter 2：正臉累計時間 ≥ MIN_FRONTAL_SEC"""
        Result = []
        for Track in Tracks:
            FrontalFrames = sum(1 for V in Track.IsFrontal.values() if V)
            if FrontalFrames / Fps >= self.MIN_FRONTAL_SEC:
                Result.append(Track)
        return Result

    def _filterByAlone(self, Tracks):
        """Filter 3：每一幀出現時畫面中只有他自己"""
        Result = []
        for Track in Tracks:
            if Track.IsAlone and all(Track.IsAlone.values()):
                Result.append(Track)
        return Result

    def _saveSegment(self, VideoPath, Track, OutputDir, Fps):
        """將 Track 對應的影片片段存成 mp4，回傳檔名"""
        Cap = cv2.VideoCapture(VideoPath)
        W = int(Cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(Cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        FileName = self._generateFilename(OutputDir)
        OutputPath = os.path.join(OutputDir, FileName)

        try:
            Fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            Out = cv2.VideoWriter(OutputPath, Fourcc, Fps, (W, H))
            if not Out.isOpened():
                raise RuntimeError("mp4v 編碼開啟失敗")
        except Exception:
            # 備援：改用 XVID + .avi
            FileName = FileName.replace('.mp4', '.avi')
            OutputPath = os.path.join(OutputDir, FileName)
            Fourcc = cv2.VideoWriter_fourcc(*'XVID')
            Out = cv2.VideoWriter(OutputPath, Fourcc, Fps, (W, H))

        try:
            Cap.set(cv2.CAP_PROP_POS_FRAMES, Track.FirstFrame)
            for _ in range(Track.LastFrame - Track.FirstFrame + 1):
                Ret, Frame = Cap.read()
                if Ret:
                    Out.write(Frame)
        finally:
            Out.release()
            Cap.release()

        return FileName

    def _generateFilename(self, OutputDir):
        """掃描 OutputDir 找最大 face_NNN 編號，回傳下一個不重複的檔名"""
        MaxNum = 0
        Pattern = re.compile(r'^face_(\d+)\.(mp4|avi)$')
        try:
            for F in os.listdir(OutputDir):
                M = Pattern.match(F)
                if M:
                    MaxNum = max(MaxNum, int(M.group(1)))
        except Exception:
            pass
        return f"face_{MaxNum + 1:03d}.mp4"
