import cv2
import os

video_path = './test_video/pitchlab.mp4'
output_dir = 'images'

# 如果資料夾不存在，就創建
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
interval = 3  # 每隔30幀儲存一次

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % interval == 0:
        # 使用序號存檔名，格式化成 00001.jpg, 00002.jpg ...
        filename = os.path.join(output_dir, f'frame_{frame_count:05d}.jpg')
        cv2.imwrite(filename, frame)

    frame_count += 1


cap.release()
cv2.destroyAllWindows()
print(f'總共儲存 {frame_count} 幀')