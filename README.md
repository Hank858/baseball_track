# 嘗試使用 YOLO 來偵測棒球比賽中的物件

## 學習項目

1. 使用標記軟體 Roboflow 進行圖片標記
2. 學習 YOLO 的訓練操作及推論測試
3. 理解 YOLO 的物件追蹤

## 實驗目標

1. 成功偵測比賽中各個物件，包括：
   - 本壘板
   - 投手板
   - 棒球手套
   - 打擊者
   - 投球者
   - 棒球

   **物件偵測：✅**

2. 連續追蹤棒球軌跡（從投手投出到本壘板）

   **物件追蹤：❌**

## 失敗原因檢討

### 1. 模型 / 演算法問題

- Detection 的 Bounding Box 不穩定
- Tracker 的 Matching 失敗

### 2. 輸入 / 硬體問題

- 測試影片的畫質與影格率不足：較低的影像解析度可能造成棒球等小型物體的特徵資訊不足，使 YOLO Detection 困難；較低的 FPS 則會增加相鄰影格間物體位置變化，使 Tracker 的 Matching 困難。
- 電腦運算速度不足，導致無法運行更大的 model 及推論速度跟不上影片 FPS，造成即時畫面延遲。

## 實驗成果

### 物件偵測

![YOLO 物件偵測結果](https://github.com/user-attachments/assets/a8126b09-8f37-43e3-aab6-5db5a7a40b33)

### 物件追蹤

![YOLO 物件追蹤結果](https://github.com/user-attachments/assets/d9abe326-ced6-4ddc-adcd-a36436a789c0)
