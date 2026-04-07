#### 1.使用技術:
a. Face_recognition: https://github.com/ageitgey/face_recognition (用來抓取68個點)
b. 混合分類器（Random Forest + 馬氏距離驗證）, 馬氏距離是再協助判斷Random forest的結果,是不是現在WebCam的人,做到"找不到"的效果.

#### 2.程式功能和疑問:
{main.py} 主程式,669行
{face_recognizer.py} FaceRecognizer類別，辨識核心，由 main.py 引入,354行
{face_feature.py} 從 68 個 landmark 萃取 23 維幾何特徵向量,219行 (OK)
{random_forest_np.py} 純 NumPy 實作：`DecisionTree`、`RandomForest`、`OnePerson`,334行
{face_model.npz} 訓練資料儲存檔（執行後自動產生）

#### 3. 學習,辨識流程都在 PLAN.md裡記錄

#### {main.py}
a. OK區段:
001~038
136~297: def _BuildUI(self, mode) 建UI,可以晚點看

b. self._LatestFrame
是從camera抓下來的畫面,是 OpenCV 影像 = 3D numpy 陣列.
np.ndarray。可以把它想成一個三層的數字表格

```
 Frame.shape = (480, 640, 3)
                │    │    └─ 3 個顏色通道：Blue, Green, Red
                │    └─ 每行 640 個像素（寬）
                └─ 共 480 行（高）

 Frame[0, 0]   = [210, 180, 140]  ← 左上角像素：B=210, G=180, R=140
```

#### {face_recognizer.py}  Joey:看過一次,很多要解析的
a. OK區段:
```
 定義:
 import face_recognition  #這是 https://github.com/ageitgey/face_recognition 的libaray
 
 1. 公用變數 __init__ 
  _ModelPath : String - 通常是 "face_model.npz"
  _Samples : dict 的 key-value 
  _Classifier : None
  _Validators : dict 的 key-value 
  _IsTrained : Boolean, False
  
 2. def LoadModel(self) -> bool:  # Joey: 要試著印 Persons, X, Y出來.
 3. def SaveModel(self) -> bool:  # Joey: 要印出資料.
   np.savez_compressed(... # Joey: 要查
 4. def Predict(self, Frame: np.ndarray) -> list:
   for i, (Loc, Lm) in enumerate(zip(Locations, LandmarksList)): #Joey: 要查
   
 要研究的部分:
 1. del self._Samples[PersonName]
 2. ValidPersons = {Name: Vecs for Name, Vecs in self._Samples.items() if Vecs}
 3. for Vec, Name, Conf in zip(Vecs, Names, Confs):
```

b. 陣列處理
```
 1. def AddSample ...
  若 Frame[0, 0] = [210, 180, 140]  ← 左上角像素：B=210, G=180, R=140
  RgbFrame = Frame[:, :, ::-1].copy()

  意思就是把第3維反轉：BGR → RGB
  
  face_recognition 的函式:
  Locations  = face_recognition.face_locations(RgbFrame, model="hog")
  LandmarksList = face_recognition.face_landmarks(RgbFrame, Locations)
```

c. face_locations 和 face_landmarks 都是 face_recognition library 內的函式.不是我們寫的

#### {face_feature.py}
1\. extractFeatures(Landmarks: dict)
 a. 從68個特徵點萃取成23維特徵向量
```
 眼睛位置一定要用,眉毛,鼻子,嘴巴,下巴若沒有,CC是採用估算的方式,如下眉毛處理部分:
 LBrowCenter = _center(LeftBrow)  if LeftBrow  else LeftEyeCenter  - np.array([0.0, 15.0])
```

 b. 找出15個距離特徵 (距離除以 IOD)
F01, F02: 左眼寬度, 右眼寬度 
F03, F04: 左眉寬度, 左眉寬邊 (若沒有就放 -1)
F05: 鼻梁長度 (若點數未超過2點,就放 -1)
F06: 鼻翼寬度 (若沒有就放 -1)
F07: 嘴角寬度
F08: 左眼→左眉距離 (沒有眉毛就 -1)
F09: 右眼→右眉距離 (沒有眉毛就 -1)
F10: 鼻尖→嘴中心距離 (二個值都沒有就 -1)
F11: 下巴寬度 (沒有就-1)
F12: 下巴到鼻梁最高點 (二個值都沒有就 -1)
F13: 左眼中心→鼻尖 (沒鼻尖就 -1)
F14: 右眼中心→鼻尖 (沒鼻尖就 -1)
F15: 眼寬不對稱率 (左眼寬 - 右眼寬) /IOD

 c. 角度特徵
F16: 左眼外角—鼻尖—右眼外角 夾角
F17: 左嘴角—嘴中心—右嘴角 夾角
F18: 兩眉連線與水平的仰角（度），仰頭為正 (難)

 d. 比例特徵
F19: 鼻翼寬 / 嘴角寬
F20: 嘴角寬 / 瞳距
F21: 左眼高 / 左眼寬
F22: 右眼高 / 右眼寬
F23: 下巴寬 / 臉高

#### {random_forest_np.py}
```
```