import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 無法打開相機")
    exit()

plt.ion()  # 啟用互動模式
fig, ax = plt.subplots()
line_plot, = ax.plot([], [], color='c')
ax.set_ylim(0, 255)
ax.set_xlabel('X')
ax.set_ylabel('Gray Intensity')
ax.set_title('Gray Intensity Along Center Line')

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 無法擷取畫面")
        break

    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2

    # 畫黑色中心點
    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 0), -1)
    # 畫青色水平線
    cv2.line(frame, (0, center_y), (width, center_y), (255, 255, 0), 2)

    # 取得中心線附近的灰階平均值 (取 5 條線平均)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    band = gray[center_y-2:center_y+3, :]  # 上下各取 2 條線
    intensity = np.mean(band, axis=0)

    # 除錯：印出前 10 個強度值
    print(intensity[:10])

    # 更新圖表
    line_plot.set_data(np.arange(width), intensity)
    ax.set_xlim(0, width)
    ax.relim()             # 重新計算數據範圍
    ax.autoscale_view()    # 自動縮放
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)        # 適度延遲，確保刷新

    cv2.imshow("USB Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
