import cv2
from ultralytics import YOLO


# 1. 載入你剛剛訓練出來的最強模型
# model_path = './model/pitcher_hitter_catcher_detector_v3.pt'
# model_path = './model/ball_tracking_v3-YOLOv11.pt' 
# model_path = './model/glove_tracking_v4_YOLOv11.pt' 
model_path = '../ultralytics/runs/detect/freeze_10_fulltracking/weights/best.pt' 
model = YOLO(model_path)

model.info()

# 2. 設定你要測試的圖片路徑
# 假設圖片放在上一層的 yolo_project 根目錄下
# source_img = 'room.jpg'
source_video = './test_video/mlb.mp4'

print(model.names) # model class names

# 3. 進行預測 (推論)
# save=True：讓 YOLO 自動把畫好框線的結果存成圖片
# conf=0.5：信心度門檻，代表只顯示模型有 50% 以上把握的目標（可自由調整 0.1 ~ 0.9）
print(f"\n🚀 開始使用 {model_path} 進行推論...")
# results = model.predict(source=source_img, save=True, conf=0.2)
# results = model.predict(source=source_video, save=True, conf=0.2)
# results = model.track(source=source_video, show=True, tracker="bytetrack.yaml") # 物件追蹤(bytetrack)
results = model.track(source=source_video, show=True, classes = [1], conf=0.4)  # 物件追蹤(BoT-SORT)

#開啟視訊鏡頭 (0 代表預設鏡頭)
# cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap = cv2.VideoCapture(source_video)

# 3. 設定解碼格式與解析度 (避免畫面卡死)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960) # 可依需求調整
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_FPS, 30)

print("相機啟動成功！按下 'q' 鍵可關閉測試視窗")

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         print("無法讀取攝影機畫面")
#         break
        
#     # YOLO 推論
#     results = model(frame, conf=0.3)
    
#     # 畫上 Bounding Box
#     annotated_frame = results[0].plot()
    
#     # 顯示即時影像
#     cv2.imshow("Baseball Tracking Test", annotated_frame)
    
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
        
# cap.release()
# cv2.destroyAllWindows()

print("\n✅ 推論完成！請去本機端的 yolo_project/runs/detect/predict/ 資料夾查看結果圖片。")