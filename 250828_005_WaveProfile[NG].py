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
            # 取得影像中心 y 座標
            h, w = frame.shape[:2]
            center_y = h // 2

            # 找出輪廓上所有 y 接近 center_y 的點
            y_tolerance = 5  # 容許誤差
            points_on_center = [pt[0] for pt in c if abs(pt[0][1] - center_y) <= y_tolerance]

            if len(points_on_center) >= 2:
                # 依 x 排序，最左與最右即為寬度端點
                points_on_center = sorted(points_on_center, key=lambda p: p[0])
                left_pt = points_on_center[0]
                right_pt = points_on_center[-1]

                # 畫出端點與寬度線
                cv2.circle(frame, tuple(left_pt), 8, (255,0,255), -1)
                cv2.circle(frame, tuple(right_pt), 8, (0,0,255), -1)
                cv2.line(frame, tuple(left_pt), tuple(right_pt), (255,255,0), 3)
                width = np.linalg.norm(np.array(left_pt) - np.array(right_pt))
                cv2.putText(frame, f"Width: {int(width)} px", ((left_pt[0]+right_pt[0])//2, center_y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            else:
                cv2.putText(frame, "No width found at center", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 畫出輪廓
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

    cv2.imshow("USB Camera", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()