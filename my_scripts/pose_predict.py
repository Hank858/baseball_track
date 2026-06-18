import cv2
from ultralytics import YOLO

# 1. 載入你剛剛訓練出來的最強模型
# 注意：請根據你的實際路徑調整，這裡是假設你在 my_scripts 資料夾下執行
model_path_a = './model/pitcher_hitter_catcher_detector_v3.pt' # 偵測打者
model_path_b = './model/yolo11n-pose.pt' # 偵測打者pose
model_path_c = './model/glove_tracking_v4_YOLOv11.pt' # 替換成你的模型路徑
model_a = YOLO(model_path_a)
model_b = YOLO(model_path_b)
model_c = YOLO(model_path_c)

# 2. 設定你要測試的圖片路徑
# 假設圖片放在上一層的 yolo_project 根目錄下
source_img = 'room.jpg'
source_video = './test_video/WBC_test.mp4'
homeplate = [None, None]  # 初始化本壘板座標

print(model_a.names) # model class names

# 3. 進行預測 (推論)
# save=True：讓 YOLO 自動把畫好框線的結果存成圖片
# conf=0.5：信心度門檻，代表只顯示模型有 50% 以上把握的目標（可自由調整 0.1 ~ 0.9）

# 讀取影片
cap = cv2.VideoCapture('./test_video/WBC_test.mp4')

# # 3. 設定解碼格式與解析度 (避免畫面卡死)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960) # 可依需求調整
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
# cap.set(cv2.CAP_PROP_FPS, 30)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("無法讀取攝影機畫面")
        break
    
    # 測試是否能偵測到本壘板
    results_homeplate = model_c(frame, conf=0.1, classes=[1]) # 偵測本壘板
    results_homeplate = results_homeplate[0]
    if len(results_homeplate.boxes) > 0:
        homeplate_box = results_homeplate.boxes.xyxy[0]  # 取出第一個本壘板的 bounding box
        x1, y1, x2, y2 = homeplate_box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2) # 在畫面上畫出本壘板的 bounding box
    
    if homeplate is not None:
        results_homeplate = model_c(frame, conf=0.1, classes=[1]) # 偵測本壘板
        homeplate_result = results_homeplate[0]
        if len(homeplate_result.boxes) > 0:
            homeplate_box = homeplate_result.boxes.xyxy[0]  # 取出第一個本壘板的 bounding box
            x1, y1, x2, y2 = homeplate_box
            # 取本壘板兩側的x座標
            homeplate[0] = int(x1)  # 本壘板左側的x座標
            homeplate[1] = int(x2)  # 本壘板右側的x座標
            
            
    # YOLO 推論
    results_bat = model_a(frame, conf=0.2, classes = [0]) # 只偵測打者
    results_pose = model_b(frame, conf=0.2) # 偵測姿態
    
    result_bat = results_bat[0]  # 取出第一個結果 (因為我們只處理一個 frame)
    result_pose = results_pose[0]  # 取出第一個結果 (因為我們只處理一個 frame)
    
    # 確保畫面中有抓到人
    if len(result_pose.boxes) > 0:
        batter_index = 0
        
        # 找出打者的pose
        for i, box in enumerate(result_pose.boxes.xyxy):
            x1, y1, x2, y2 = box
            # 檢查這個pose的中心點是否在打者的bounding box內
            if (x1 + x2) / 2 > result_bat.boxes.xyxy[0][0] and (x1 + x2) / 2 < result_bat.boxes.xyxy[0][2] and (y1 + y2) / 2 > result_bat.boxes.xyxy[0][1] and (y1 + y2) / 2 < result_bat.boxes.xyxy[0][3]:
                batter_index = i
                break
            
        batter_only_result = result_pose[batter_index]  # 取出打者的pose結果
        kpts = batter_only_result.keypoints.xy[0] # 打者的關鍵點
        left_shoulder = kpts[5]  # 左肩
        right_shoulder = kpts[6] # 右肩
        left_knee = kpts[13] # 左膝
        right_knee = kpts[14] # 右膝
        left_hip = kpts[11] # 左臀
        right_hip = kpts[12] # 右臀
        
        left_x = int(left_shoulder[0])
        left_y = int(left_shoulder[1])
        right_x = int(right_shoulder[0])
        right_y = int(right_shoulder[1])
        left_knee_x = int(left_knee[0])
        left_knee_y = int(left_knee[1])
        right_knee_x = int(right_knee[0])
        right_knee_y = int(right_knee[1])
        left_hip_x = int(left_hip[0])
        left_hip_y = int(left_hip[1])
        right_hip_x = int(right_hip[0])
        right_hip_y = int(right_hip[1])
        
        # 好球帶上緣(右打者)
        strike_zone_top = [0, 0]
        strike_zone_top[0] = (left_shoulder[0] + left_hip[0]) / 2 # x
        strike_zone_top[1] = (left_shoulder[1] + left_hip[1]) / 2 # y

        # 畫一個實心的紅色圓點 (BGR格式，紅色是 0, 0, 255)
        cv2.circle(frame, (left_x, left_y), radius=6, color=(0, 0, 255), thickness=-1)
        cv2.circle(frame, (right_x, right_y), radius=6, color=(0, 0, 255), thickness=-1)
        cv2.circle(frame, (left_knee_x, left_knee_y), radius=6, color=(0, 0, 255), thickness=-1)
        cv2.circle(frame, (right_knee_x, right_knee_y), radius=6, color=(0, 0, 255), thickness=-1)
        cv2.circle(frame, (left_hip_x, left_hip_y), radius=6, color=(0, 0, 255), thickness=-1)
        cv2.circle(frame, (right_hip_x, right_hip_y), radius=6, color=(0, 0, 255), thickness=-1)

        # 畫出好球帶(42 * (中心點到膝蓋的距離))
        cv2.rectangle(frame, (homeplate[0], int(strike_zone_top[1])), (homeplate[1], int(left_knee_y)), (255, 0, 0), 2)
        # cv2.rectangle(frame, (int(left_shoulder[0]), int(strike_zone_top[1])), (int(left_knee_x), int(left_knee_y)), (255, 0, 0), 2)
        
        # 你甚至可以在點的旁邊寫上文字標籤
        cv2.putText(frame, "L-Shoulder", (left_x + 10, left_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "R-Shoulder", (right_x + 10, right_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "L-Knee", (left_knee_x + 10, left_knee_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "R-Knee", (right_knee_x + 10, right_knee_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "L-Hip", (left_hip_x + 10, left_hip_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "R-Hip", (right_hip_x + 10, right_hip_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 2. 將打者的骨架畫在原本的 frame 上
        # 參數解密：
        # boxes=False: 不要畫方形 Bounding Box 和標籤 (超級重要！)
        # kpt_radius=4, kpt_line=2: 控制骨架點的大小與線條粗細
        # frame = batter_only_result.plot(
        #     boxes=True,      # 只留姿態，不要方框
        #     kpt_radius=4,     # 關節點圓圈半徑
        #     kpt_line=4,       # 骨架連線粗細
        #     img=frame         # 指定畫在我們原本的影像上
        # )
        
        # (選用) 既然抓到了打者專屬的 batter_only_result，
        # 你也可以在這裡把他的肩膀 (點5, 6) 和膝蓋 (點13, 14) 座標抽出來算好球帶了！
        # keypoints = batter_only_result.keypoints.xy[0] 

        # 顯示即時影像
        cv2.imshow("Baseball Tracking Test", frame)
    
        if cv2.waitKey(1) & 0xFF == ord("q"):
             break
        
cap.release()
cv2.destroyAllWindows()
        
print("\n✅ 推論完成！請去本機端的 yolo_project/runs/detect/predict/ 資料夾查看結果圖片。")