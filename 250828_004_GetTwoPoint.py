import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 無法打開相機")
    exit()

cv2.namedWindow("Mask")
cv2.createTrackbar("H_low", "Mask", 0, 180, nothing)
cv2.createTrackbar("S_low", "Mask", 0, 255, nothing)
cv2.createTrackbar("V_low", "Mask", 0, 255, nothing)
cv2.createTrackbar("H_high", "Mask", 180, 180, nothing)
cv2.createTrackbar("S_high", "Mask", 255, 255, nothing)
cv2.createTrackbar("V_high", "Mask", 50, 255, nothing)  # 低V值抓黑色

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 無法擷取畫面")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
    centers = []
    if contours and len(contours) >= 2:
        # 取面積最大的兩個黑點
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
        if len(centers) == 2:
            cv2.line(frame, centers[0], centers[1], (0,255,0), 3)
            dist = np.linalg.norm(np.array(centers[0]) - np.array(centers[1]))
            mid = ((centers[0][0]+centers[1][0])//2, (centers[0][1]+centers[1][1])//2)
            cv2.putText(frame, f"Dist: {int(dist)} px", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.imshow("USB Camera", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()