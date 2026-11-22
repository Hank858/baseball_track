import cv2
from ultralytics import YOLO


# 1. 載入模型
# model_path = './model/pitcher_hitter_catcher_detector_v3.pt'
# model_path = './model/ball_tracking_v3-YOLOv11.pt' # 可以偵測到本壘板
# model_path = './model/glove_tracking_v4_YOLOv11.pt' # 可以偵測到本壘板
model_path = './model/best.pt'
model = YOLO(model_path)

print("模型名稱/版本:", model.task)
model.info()

# 2. 設定測試的圖片路徑
# source_img = 'room.jpg'
source_video = './test_video/WBC_test.mp4'

print(model.names) # model class names

# 3. 進行預測 
print(f"\n 開始使用 {model_path} 進行推論...")
# results = model.predict(source=source_img, save=True, conf=0.2)
# results = model.predict(source=source_video, save=True, conf=0.2)
# results = model.track(source=source_video, show=True, tracker="bytetrack.yaml") # 物件追蹤(bytetrack)
# results = model.track(source=source_video, show=True, classes = [0])  # 物件追蹤(BoT-SORT)

# 開啟視訊鏡頭 (0 代表預設鏡頭)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# 3. 設定解碼格式與解析度
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_FPS, 30)

print("相機啟動成功！按下 'q' 鍵可關閉測試視窗")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("無法讀取攝影機畫面")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
        # break

        
#     # YOLO 推論
#     results = model(frame, conf=0.3)
    
#     # 畫上 Bounding Box
#     annotated_frame = results[0].plot()
    
#     # 顯示即時影像
#     cv2.imshow("Baseball Tracking Test", annotated_frame)
    
    if cv2.waitKey(50) & 0xFF == ord("q"):
        break

# cap.release()
# cv2.destroyAllWindows()

print("\n 推論完成")