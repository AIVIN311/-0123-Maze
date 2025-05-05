import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def logistic_map(x, r):
    return r * x * (1 - x)

def fit_logistic(H_sequence):
    x0 = H_sequence[0]
    y_true = H_sequence[1:]

    def loss(r):
        x = x0
        return sum(((x := logistic_map(x, r)) - y)**2 for y in y_true)

    r_values = np.linspace(2.5, 4.0, 5000)
    losses = [loss(r) for r in r_values]
    best_r = r_values[np.argmin(losses)]
    return best_r

def fit_logistic_map(metrics_csv_path, output_path):
    df = pd.read_csv(metrics_csv_path)
    if 'cycle' not in df.columns:
        raise ValueError("❌ 缺少 'cycle' 欄位，請先執行語法標記器加入 cycle 編號。")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_records = []
    plt.figure(figsize=(10, 6))

    valid_cycle_count = 0  # 計數有效 cycle

    for cycle_id, group in df.groupby('cycle'):
        H_seq = group['entropy'].values  # 👈 記得是 entropy，不是 h
        if len(H_seq) < 2:
            print(f"⚠️ Cycle {cycle_id} 太短（{len(H_seq)} 筆），跳過。")
            continue

        try:
            r = fit_logistic(H_seq)
            x_fit = [H_seq[0]]
            for _ in range(len(H_seq) - 1):
                x_fit.append(logistic_map(x_fit[-1], r))

            episodes = group['episode'].values
            plt.plot(episodes, H_seq, 'o-', label=f"Cycle {cycle_id} H")
            plt.plot(episodes, x_fit, '--', label=f"Fit r={r:.3f}")

            result_records.append({
                "cycle": cycle_id,
                "best_r": r,
                "length": len(H_seq),
                "episode_start": episodes[0],
                "episode_end": episodes[-1]
            })

            valid_cycle_count += 1

        except Exception as e:
            print(f"❌ 擬合 Cycle {cycle_id} 發生錯誤：{e}")

    if valid_cycle_count == 0:
        print("⚠️ 無有效 cycle 可擬合，未產生圖像。")
        return

    plt.xlabel("Episode")
    plt.ylabel("Entropy H")
    plt.title("H Curve Logistic Fitting Across Cycles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    summary_path = os.path.join(os.path.dirname(output_path), "logistic_fit_summary.csv")
    pd.DataFrame(result_records).to_csv(summary_path, index=False)
    print(f"✅ Logistic H 擬合完成，圖像儲存於 {output_path}，統計儲存於 {summary_path}")
