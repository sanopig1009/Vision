# 需要安裝: pip install pypylon opencv-python
from pypylon import pylon
import cv2

# 建立相機物件
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# 開始擷取
camera.Open()
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        # 取得影像資料
        img = grabResult.Array
        # 顯示影像
        cv2.imshow('Camera', img)
        # 按下 q 鍵離開
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    grabResult.Release()
