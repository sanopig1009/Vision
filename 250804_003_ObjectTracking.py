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
            # 外接矩形
            x, y, w, h = cv2.boundingRect(c)
            # 畫出矩形
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 顯示寬度
            cv2.putText(frame, f"Width: {w} px", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("USB Camera", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()