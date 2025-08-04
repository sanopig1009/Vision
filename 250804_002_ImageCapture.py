import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 無法打開相機")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 無法擷取畫面")
        break

    cv2.imshow("USB Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()