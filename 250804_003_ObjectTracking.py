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

    # 轉 HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 白色範圍（可依實際調整）
    lower = np.array([0, 0, 180])
    upper = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # 去雜訊
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 找輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 找最大白色區域
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 1000:
            # 多邊形近似
            epsilon = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            # 如果是四邊形
            if len(approx) == 4:
                for idx, point in enumerate(approx):
                    x, y = point[0]
                    cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
                    cv2.putText(frame, f"Corner {idx+1}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                cv2.drawContours(frame, [approx], -1, (0,255,0), 2)

    cv2.imshow("USB Camera", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()