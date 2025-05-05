# detect_transition_points.py
# 🧠 檢測語法轉變點（微觀）：追蹤 entropy 與 wisdom_density 的結構轉換

import pandas as pd
import numpy as np
import os

# === 可調參數 ===
ENTROPY_JUMP = 0.05          # 如果 H 的變化大於這個值，視為轉變點
WISDOM_MIN_JUMP = 1e-5       # 如果 S 突然上升（或下降）
WINDOW_SIZE = 3              # 使用滑動窗口平滑資料

# === 自動尋找最新測試資料夾 ===
def find_latest_test_dir(base_dir="runs"):
    all_dirs = []
    for category in ["test", "test_many", "fast"]:
        cat_path = os.path.join(base_dir, category)
        if not os.path.isdir(cat_path): continue
        for f in os.listdir(cat_path):
            full_path = os.path.join(cat_path, f)
            if os.path.isdir(full_path):
                all_dirs.append(full_path)
    return sorted(all_dirs)[-1] if all_dirs else None

# === 計算移動平均 ===
def smooth(values, window=3):
    return np.convolve(values, np.ones(window)/window, mode='same')

# === 偵測轉變點 ===
def detect_transitions(metrics_path, save_path=None):
    df = pd.read_csv(metrics_path)

    H = df['entropy'].values
    S = df['wisdom_density'].values
    episodes = df['episode'].values

    H_smooth = smooth(H, WINDOW_SIZE)
    S_smooth = smooth(S, WINDOW_SIZE)

    transition_points = []

    for i in range(1, len(H_smooth)):
        h_diff = abs(H_smooth[i] - H_smooth[i-1])
        s_diff = abs(S_smooth[i] - S_smooth[i-1])

        if h_diff > ENTROPY_JUMP or s_diff > WISDOM_MIN_JUMP:
            transition_points.append({
                'episode': int(episodes[i]),
                'h': H[i],
                's': S[i],
                'h_jump': round(h_diff, 5),
                's_jump': round(s_diff, 8),
                'reason': f"{'H' if h_diff > ENTROPY_JUMP else ''}{' + ' if h_diff > ENTROPY_JUMP and s_diff > WISDOM_MIN_JUMP else ''}{'S' if s_diff > WISDOM_MIN_JUMP else ''}"
            })

    result_df = pd.DataFrame(transition_points)
    if save_path:
        result_df.to_csv(save_path, index=False)
        print(f"✅ 轉變點已儲存：{save_path}")
    else:
        print("✅ 偵測完成，未指定儲存路徑。")

    return result_df

# === 主程式入口 ===
if __name__ == "__main__":
    latest_dir = find_latest_test_dir()
    if latest_dir:
        metrics_path = os.path.join(latest_dir, "metrics.csv")
        output_path = os.path.join(latest_dir, "transition_points.csv")
        detect_transitions(metrics_path, save_path=output_path)
    else:
        print("❌ 找不到任何測試資料夾！")
