# smart_hs_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# === 📦 載入資料 ===
metrics_path = "runs/latest_test/metrics.csv"  # TODO: 改成你最新的路徑
event_path = "runs/latest_test/wisdom_rhythm.json"

metrics = pd.read_csv(metrics_path)
with open(event_path, "r") as f:
    events = json.load(f)

# === 🎨 畫圖區 ===
fig, ax1 = plt.subplots(figsize=(12, 6))

# 📈 H 曲線（藍）
ax1.plot(metrics["episode"], metrics["entropy"], label="Entropy (H)", color="blue")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Entropy (H)", color="blue")
ax1.tick_params(axis='y', labelcolor='blue')

# 🎯 畫出每個轉變點符號
for e in events:
    symbol = e["event"]
    ep = e["episode"]
    h = e["h"]
    ax1.text(ep, h + 0.01, symbol, fontsize=12, color="black", ha='center')

# 📉 S 曲線（紅）
ax2 = ax1.twinx()
ax2.plot(metrics["episode"], metrics["wisdom_density"], label="Wisdom Density (S)", color="red")
ax2.set_ylabel("Wisdom Density (S)", color="red")
ax2.tick_params(axis='y', labelcolor='red')

plt.title("🧠 智慧診斷圖表：H 曲線、S 曲線與語法事件")

plt.grid(True)
fig.tight_layout()

# === 💾 儲存 ===
output_path = os.path.join("runs", "latest_test", "smart_hs_dashboard.png")
plt.savefig(output_path)
plt.show()
print(f"✅ 圖表已儲存至 {output_path}")
