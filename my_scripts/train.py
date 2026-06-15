from ultralytics import YOLO

# 1. 載入 YOLO11 預訓練模型
model = YOLO('./model/ball_tracking_v4-YOLOv11.pt')

# 2. 執行微調
results = model.train(
    data='./dataset/data.yaml',   # 資料集設定檔路徑
    epochs=50,                 # 訓練總輪數
    imgsz=640,                 # 輸入影像尺寸
    batch=8,                  # 批次大小（顯存不足可調小）
    device=0,                  # 運算裝置（0 = GPU, 'cpu' = CPU）
    freeze=10,                 # 凍結前 10 層（fine-tune 關鍵）
    lr0=0.001,                 # 初始學習率（fine-tune 建議調小）
    optimizer='AdamW',         # 優化器（fine-tune 推薦 AdamW）
    patience=20,               # Early Stopping 輪數
    name='yolo11_finetune',    # 實驗名稱
    save=True,                 # 儲存 best.pt / last.pt
    amp=True,                  # 混合精度訓練（節省顯存）
    plots=True,                # 產生訓練曲線圖
)

print(f"訓練完成！最佳模型：runs/train/yolo11_finetune/weights/best.pt")