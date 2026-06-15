from ultralytics import YOLO

# 1. 載入你剛剛訓練出來的最強模型
# 注意：請根據你的實際路徑調整，這裡是假設你在 my_scripts 資料夾下執行
model_path = '../ultralytics/runs/detect/train/weights/best.pt' 
model = YOLO(model_path)

# 2. 設定你要測試的圖片路徑
# 假設圖片放在上一層的 yolo_project 根目錄下
source_img = 'room.jpg'

# 3. 進行預測 (推論)
# save=True：讓 YOLO 自動把畫好框線的結果存成圖片
# conf=0.5：信心度門檻，代表只顯示模型有 50% 以上把握的目標（可自由調整 0.1 ~ 0.9）
print(f"\n🚀 開始使用 {model_path} 進行推論...")
results = model.predict(source=source_img, save=True, conf=0.2)

print("\n✅ 推論完成！請去本機端的 yolo_project/runs/detect/predict/ 資料夾查看結果圖片。")