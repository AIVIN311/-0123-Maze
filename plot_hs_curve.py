import pandas as pd
import matplotlib.pyplot as plt
import os

# === 設定路徑（⚡請改成你的metrics資料夾）===
metrics_path = "runs/test_2025-04-29_21-45-38/metrics.csv"  # 記得改
save_path = "runs/test_2025-04-29_21-45-38/h_s_curve.png"   # 記得改

# === 讀取 metrics.csv ===
data = pd.read_csv(metrics_path)

# === 畫 H-S 曲線 ===
fig, ax1 = plt.subplots()

ax1.plot(data['episode'], data['entropy'], 'b-', label='Entropy (H)')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Entropy', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(data['episode'], data['wisdom_density'], 'r-', label='Wisdom Density (S)')
ax2.set_ylabel('Wisdom Density', color='r')
ax2.tick_params(axis='y', labelcolor='r')

fig.tight_layout()
fig.legend(loc='upper center')
plt.title('0-1-2-3 H-S Curve')
plt.savefig(save_path)

print(f"🎨 H-S 曲線圖儲存到 {save_path}")
plt.show()
