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
            # 多邊形近似
            epsilon = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) == 4:
                # 取得四個角點，排序
                pts = approx.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]

                # 左右邊線
                pt_left_top = rect[0]
                pt_left_bottom = rect[3]
                pt_right_top = rect[1]
                pt_right_bottom = rect[2]

                # 畫出左右邊線
                cv2.line(frame, tuple(pt_left_top.astype(int)), tuple(pt_left_bottom.astype(int)), (255, 0, 255), 3)
                cv2.line(frame, tuple(pt_right_top.astype(int)), tuple(pt_right_bottom.astype(int)), (0, 0, 255), 3)

                # 畫出中心線
                center_top = ((pt_left_top + pt_right_top) / 2)
                center_bottom = ((pt_left_bottom + pt_right_bottom) / 2)
                cv2.line(frame, tuple(center_top.astype(int)), tuple(center_bottom.astype(int)), (0, 255, 255), 3)

                # 取中心線中點
                center_mid = ((center_top + center_bottom) / 2)

                # 求中心線方向向量
                dir_vec = center_bottom - center_top
                dir_vec = dir_vec / np.linalg.norm(dir_vec)

                # 求垂直於中心線的單位向量
                perp_vec = np.array([-dir_vec[1], dir_vec[0]])

                # 取一條足夠長的線段（與中心線垂直，通過中心點）
                line_len = 1000  # 足夠長即可
                pt1 = (center_mid + perp_vec * line_len).astype(int)
                pt2 = (center_mid - perp_vec * line_len).astype(int)
                cv2.line(frame, tuple(pt1), tuple(pt2), (0, 255, 0), 2)

                # 求這條線與左右邊線的交點
                def line_intersection(p1, p2, q1, q2):
                    # p1,p2: 第一條線的兩點
                    # q1,q2: 第二條線的兩點
                    s = np.vstack([p1, p2, q1, q2])        # s for stacked
                    h = np.hstack((s, np.ones((4, 1))))     # homogeneous
                    l1 = np.cross(h[0], h[1])               # line 1
                    l2 = np.cross(h[2], h[3])               # line 2
                    x, y, z = np.cross(l1, l2)
                    if z == 0:
                        return None
                    return np.array([x/z, y/z])

                # 左邊線交點
                cross_left = line_intersection(pt_left_top, pt_left_bottom, pt1, pt2)
                # 右邊線交點
                cross_right = line_intersection(pt_right_top, pt_right_bottom, pt1, pt2)

                if cross_left is not None and cross_right is not None:
                    cross_left = cross_left.astype(int)
                    cross_right = cross_right.astype(int)
                    cv2.circle(frame, tuple(cross_left), 8, (255,0,255), -1)
                    cv2.circle(frame, tuple(cross_right), 8, (0,0,255), -1)
                    cv2.line(frame, tuple(cross_left), tuple(cross_right), (255,255,0), 3)
                    width = np.linalg.norm(cross_left - cross_right)
                    cv2.putText(frame, f"Width: {int(width)} px", ((cross_left[0]+cross_right[0])//2, (cross_left[1]+cross_right[1])//2-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

                # 畫出四邊形
                cv2.polylines(frame, [approx], True, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Not detected as rectangle", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("USB Camera", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()