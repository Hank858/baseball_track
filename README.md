# 嘗試使用YOLO來偵測棒球比賽中的物件
## 學習項目：
1. 使用標記軟體roboflow進行圖片標記
2. 學習YOLO的訓練操作及推論測試
3. 理解YOLO的物件追蹤
## 實驗目標：
1. 成功偵測比賽中各個物件(包括：本壘板、投手板、棒球手套、打擊者、投球者、棒球)✅
2. 連續追蹤棒球軌跡（從投手投出到本壘板）❌
## 失敗原因檢討：
我對追蹤任務的失敗列出了以下幾點可能的原因   
### 1. 模型 / 演算法問題：  
- Detection 的 Bounding Box 不穩定
- Tracker 的 Matching 失敗
### 2. 輸入 / 硬體問題：  
- 測試影片的畫質與影格率不足：較低的影像解析度可能造成棒球等小型物體的特徵資訊不足，使 YOLO Detection 困難；較低的 FPS 則會增加相鄰影格間物體位置變化，使 Tracker 的 Matching 困難。
- 電腦運算速度不足，導致無法運行更大的model及推論速度跟不上影片FPS，造成即時畫面延遲
## 實驗成果：
<img width="935" height="832" alt="image" src="https://github.com/user-attachments/assets/a8126b09-8f37-43e3-aab6-5db5a7a40b33" />
