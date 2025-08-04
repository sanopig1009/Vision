import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 無法打開相機")
    exit()

# 建立調整視窗與滑桿
cv2.namedWindow("Mask")
cv2.createTrackbar("H_low", "Mask", 0, 180, nothing)
cv2.createTrackbar("S_low", "Mask", 0, 255, nothing)
cv2.createTrackbar("V_low", "Mask", 180, 255, nothing)
cv2.createTrackbar("H_high", "Mask", 180, 180, nothing)
cv2.createTrackbar("S_high", "Mask", 60, 255, nothing)
cv2.createTrackbar("V_high", "Mask", 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 無法擷取畫面")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 讀取滑桿數值
    h_low = cv2.getTrackbarPos("H_low", "Mask")
    s_low = cv2.getTrackbarPos("S_low", "Mask")
    v_low = cv2.getTrackbarPos("V_low", "Mask")
    h_high = cv2.getTrackbarPos("H_high", "Mask")
    s_high = cv2.getTrackbarPos("S_high", "Mask")
    v_high = cv2.getTrackbarPos("V_high", "Mask")

    lower = np.array([h_low, s_low, v_low])
    upper = np.array([h_high, s_high, v_high])
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 1000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Width: {w} px", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("USB Camera", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()