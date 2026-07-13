from ultralytics import YOLO

# 1. 載入 YOLO11 預訓練模型
# model = YOLO('./model/ball_tracking_v4-YOLOv11.pt')
# model = YOLO('./model/glove_tracking_v4_YOLOv11.pt')
# model = YOLO('yolo11s.pt')  # 載入上次訓練的權重檔案
model = YOLO('../ultralytics/runs/detect/freeze_10_fulltracking/weights/last.pt')

# 2. 執行微調
# results = model.train(
#     data='./dataset/data.yaml',   # 資料集設定檔路徑
#     epochs=50,                 # 訓練總輪數
#     imgsz=640,                 # 輸入影像尺寸
#     batch=8,                  # 批次大小（顯存不足可調小）
#     device=0,                  # 運算裝置（0 = GPU, 'cpu' = CPU）
#     freeze=10,                 # 凍結前 10 層（fine-tune 關鍵）
#     lr0=0.001,                 # 初始學習率（fine-tune 建議調小）
#     optimizer='AdamW',         # 優化器（fine-tune 推薦 AdamW）
#     patience=20,               # Early Stopping 輪數
#     name='yolo11_finetune',    # 實驗名稱
#     save=True,                 # 儲存 best.pt / last.pt
#     amp=True,                  # 混合精度訓練（節省顯存）
#     plots=True,                # 產生訓練曲線圖
#     resume=True              # 從頭開始訓練（如果要繼續訓練，設為 True 並指定 weights）
# )

results = model.train(
    data='./dataset/data.yaml',
    epochs=100,
    imgsz=800,  #640,
    batch=8,
    device=0,
    optimizer='AdamW',
    lr0=0.001,
    freeze=10,      # 建議保留（穩定）
    patience=20,
    name='freeze_10_fulltracking',
    save=True,
    amp=True,
    plots=True,
    resume=True    # ❗一定要 False
)


print(f"訓練完成!")