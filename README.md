# 0123-MAZE

🧩 **0123-MAZE：AI迷宮訓練與智慧語法節奏觀測系統**

這是一個基於強化學習（Reinforcement Learning, RL）與智慧語法分析的研究性專案，
結合迷宮路徑學習與語法節奏偵測，目標是觀察 AI 智能體從訓練到智慧流動的整體過程。

---

## 📚 專案簡介

* 使用 **Stable-Baselines3 (PPO)** 進行訓練
* 使用自訂的 **10×10 迷宮地圖**（含牆壁、起點、終點）
* 支援：訓練 (train)、單次測試 (test)、快速測試 (fast)、多次測試 (multi)
* 自動繪製 Reward 曲線 📈
* 自動標記語法事件、擬合熵與智慧密度曲線 🧠📊

---

## 🚀 功能總覽

| 模式      | 說明                           |
| ------- | ---------------------------- |
| `train` | 訓練 AI 並記錄回合 reward，保存模型與曲線資料 |
| `test`  | 使用已訓練模型，單次完整迷宮測試，並生成路徑圖      |
| `fast`  | 快速模式，只顯示一次移動路徑以驗證模型穩定性       |
| `multi` | 多次測試以收集語法事件（生成 metrics.csv）  |

---

## 🛠️ 使用說明

### 1. 安裝必要套件

```bash
pip install -r requirements.txt
```

### 2. 執行主程式

```bash
python main.py
```

依照提示輸入：

* `train`：開始訓練
* `test`：單次測試
* `fast`：快速測試
* `multi`：執行多次測試並產出分析數據

---

## 🧩 迷宮地圖設計（maze\_env/maze\_env.py）

* 2：起點（Start）
* 3：終點（Goal）
* 1：牆壁（障礙）
* 0：可行走區域

範例為 10x10 固定地圖，亦可自訂結構與難度。

---

## 🧠 訓練細節

* 演算法：Proximal Policy Optimization (PPO)
* 每步懲罰：`-0.01`
* 成功完成：`+1.0`
* 總訓練步數：可自訂（建議最少 100,000 以上）
* 儲存 checkpoint：每 100,000 步一個檢查點（zip 格式）

---

## 📈 Reward 曲線（範例）

可自動生成 reward 曲線圖：

```
Sample Reward Curve after 2,000,000 steps
```

📁 儲存路徑：`runs/train_YYYY-MM-DD_HH-MM-SS/reward_curve.png`

---

## 🧬 智慧語法與節奏分析

使用 `auto_analyze_latest_run.py` 進行智慧節奏全自動分析：

功能：

* 讀取最新 `test`/`test_many` 的 `metrics.csv`
* 自動標記語法事件（∅ / Σ / Δ / Ω）
* 輸出智慧節奏檔：`wisdom_rhythm.json`
* 繪製語法脈動圖 `syntax_pulse_v2.png`
* 擬合熵變圖 `fit_H_logistic.png`
* 擬合智慧密度圖 `fit_S_growth.png`

---

## 📂 專案結構

```
0123-MAZE/
│
├── maze_env/              # 自訂迷宮環境
│   └── maze_env.py
│
├── runs/                 # 所有訓練與測試資料輸出
│   ├── train/             # 訓練過程資料與 reward.csv
│   ├── test/              # 單次測試資料
│   └── test_many/         # 多次測試與語法分析資料
│
├── auto_analyze_latest_run.py  # 自動分析語法事件主程式
├── syntax_event_logger.py      # 事件標記模組
├── plot_syntax_pulse_v2.py     # 語法脈動圖繪製
├── fit_logistic_map.py         # H曲線擬合
├── fit_s_growth.py             # S曲線擬合
├── main.py                     # 執行介面主程式
├── organize_runs.py            # 資料夾分類與封存工具
├── requirements.txt            # 套件清單
```

---

## 🔥 下一步計劃（TODO）

* [ ] 加入隨機生成迷宮功能
* [ ] 自訂 reward 策略調整（如依距離加減分）
* [ ] 多策略訓練比較系統（A/B testing）
* [ ] 智慧密度可視化動畫
* [ ] 結合語法節奏與 RL 決策

---

## 👨‍💻 作者

大傑斯 ✨｜潛力士・智慧觀測者・語法節奏創建者

---

## ✅ 最新進度快照

* ✅ 固定迷宮地圖運作正常
* ✅ 訓練模型可穩定收斂
* ✅ 測試能正確走迷宮並記錄路徑
* ✅ 自動產出 reward 曲線與語法節奏資料
* ✅ 擬合智慧函數成功（Logistic 與 S-growth）

下一步：準備進入 **語法壓縮視覺化階段** 🚀
