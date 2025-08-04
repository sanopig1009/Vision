import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 無法打開相機")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 無法擷取畫面")
        break

    # 轉換為 HSV 色彩空間
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 設定筆尖顏色範圍（需根據實際顏色調整）
    # 例如：黑色筆尖
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 60])
    mask = cv2.inRange(hsv, lower, upper)

    # 形態學操作去雜訊
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 找輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 找最大輪廓
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 100:  # 過濾雜訊
            # 計算最小外接圓
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            # 標記筆尖
            cv2.circle(frame, center, int(radius), (0, 0, 255), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            cv2.putText(frame, "Pencil Tip", (center[0]+10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("USB Camera", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()