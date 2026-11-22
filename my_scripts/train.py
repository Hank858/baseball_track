from ultralytics import YOLO

# 1. 載入 YOLO11 預訓練模型
# model = YOLO('./model/ball_tracking_v4-YOLOv11.pt')
# model = YOLO('./model/glove_tracking_v4_YOLOv11.pt')
# model = YOLO('yolo11s.pt')  # 載入上次訓練的權重檔案
model = YOLO('./model/best.pt')

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
    epochs=200,
    imgsz=800,  #640,
    batch=16,
    device=0,
    optimizer='AdamW',
    lr0=0.001,
    freeze=10,      # 建議保留（穩定）
    patience=20,
    name='200epoch_early_full_enhance',
    save=True,
    amp=True,
    plots=True,
    resume=False,    # ❗一定要 False
    # === 資料增強參數 ===
    degrees=10.0,        # 隨機旋轉 ±10 度
    translate=0.1,       # 隨機平移
    scale=0.5,           # 隨機縮放
    shear=2.0,           # 隨機剪切（斜切變形）
    perspective=0.0,     # 透視變換（視角度而定，通常 0.0 ~ 0.001）
    flipud=0.0,          # 上下翻轉（如果是車牌、棒球等有上下方向性的物件，設 0）
    fliplr=0.5,          # 左右翻轉（50% 機率）
    mosaic=1.0,          # 100% 啟用 Mosaic 增強
    mixup=0.15           # 啟用 15% 的 Mixup 增強
)


print(f"訓練完成!")