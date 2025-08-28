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
            h, w = frame.shape[:2]
            center_y = h // 2
            y_band = 15  # 中心帶寬度，可調整

            # 可視化中心帶
            cv2.rectangle(frame, (0, center_y - y_band), (w, center_y + y_band), (200, 200, 200), 1)

            # --- 計算平均中心線 ---
            contour_points = c.reshape(-1, 2)
            y_dict = {}
            for x, y in contour_points:
                if y not in y_dict:
                    y_dict[y] = []
                y_dict[y].append(x)
            center_line_points = []
            for y in sorted(y_dict.keys()):
                xs = y_dict[y]
                if len(xs) >= 2:
                    x_left = min(xs)
                    x_right = max(xs)
                    x_center = (x_left + x_right) // 2
                    center_line_points.append([x_center, y])
            if len(center_line_points) > 2:
                center_line_points = np.array(center_line_points, dtype=np.int32)
                cv2.polylines(frame, [center_line_points], False, (0,255,255), 2)

                # --- 以畫面正中央y為基準，找中心線點 ---
                # 找到y最接近center_y的中心線點
                center_pt = min(center_line_points, key=lambda pt: abs(pt[1] - center_y))
                idx = np.where((center_line_points == center_pt).all(axis=1))[0][0]

                # 計算切線方向（用前後點差分）
                window = 5
                if window < idx < len(center_line_points)-window:
                    pt_prev = center_line_points[idx-window]
                    pt_next = center_line_points[idx+window]
                else:
                    pt_prev = center_line_points[max(0, idx-1)]
                    pt_next = center_line_points[min(len(center_line_points)-1, idx+1)]
                tangent = pt_next - pt_prev
                if np.linalg.norm(tangent) == 0:
                    tangent = np.array([1,0])
                else:
                    tangent = tangent / np.linalg.norm(tangent)
                normal = np.array([-tangent[1], tangent[0]])  # 法線方向

                # 沿法線方向搜尋左右邊界
                max_search = 200  # 搜尋長度
                found_left = None
                found_right = None
                for d in range(1, max_search):
                    test_pt = (center_pt + normal * d).astype(int)
                    if 0 <= test_pt[0] < w and 0 <= test_pt[1] < h:
                        if mask[test_pt[1], test_pt[0]] > 0:
                            found_right = test_pt
                        else:
                            break
                for d in range(1, max_search):
                    test_pt = (center_pt - normal * d).astype(int)
                    if 0 <= test_pt[0] < w and 0 <= test_pt[1] < h:
                        if mask[test_pt[1], test_pt[0]] > 0:
                            found_left = test_pt
                        else:
                            break

                if found_left is not None and found_right is not None:
                    cv2.circle(frame, tuple(found_left), 8, (255,0,255), -1)
                    cv2.circle(frame, tuple(found_right), 8, (0,0,255), -1)
                    cv2.line(frame, tuple(found_left), tuple(found_right), (255,255,0), 3)
                    width = np.linalg.norm(found_left - found_right)
                    cv2.putText(frame, f"Width: {int(width)} px", ((found_left[0]+found_right[0])//2, (found_left[1]+found_right[1])//2-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                else:
                    cv2.putText(frame, "No width found (normal)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

    cv2.imshow("USB Camera", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()