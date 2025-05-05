# Final integrated version of plot_syntax_pulse_v2.py with function definition

def plot_syntax_pulse(df, event_path, save_path):
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

    # === 讀取語法事件日誌 ===
    with open(event_path, encoding='utf-8') as f:
        event_log = json.load(f)

    # === 初步處理 ===
    cycles = []
    current_cycle = []
    cycle_id = 1

    for event in event_log:
        event['cycle'] = cycle_id
        current_cycle.append(event)

        if event['event'] == "Ω":
            cycles.append(current_cycle)
            current_cycle = []
            cycle_id += 1

    if current_cycle:
        cycles.append(current_cycle)

    # === 畫圖 ===
    plt.figure(figsize=(12, 6))
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lavender', 'mistyrose', 'honeydew', 'aliceblue']
    color_idx = 0
    last_ep = 0

    for cycle in cycles:
        if not cycle:
            continue
        episodes = [e['episode'] for e in cycle]
        H = [e['h'] for e in cycle]
        S = [e['s'] for e in cycle]

        plt.axvspan(last_ep, episodes[-1], color=colors[color_idx % len(colors)], alpha=0.3)
        last_ep = episodes[-1]
        color_idx += 1

        plt.plot(episodes, H, label=f"Cycle {cycle[0]['cycle']} (H)")
        plt.plot(episodes, S, linestyle='--', alpha=0.5)

    plt.xlabel('Episode')
    plt.ylabel('H (solid) / S (dashed)')
    plt.title('Syntax Pulse Map V2 — Recursive Tracking')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Syntax Pulse Map 儲存到 {save_path}")

    # === 輸出統計 ===
    summary = []
    for cycle in cycles:
        if not cycle:
            continue
        avg_h = np.mean([e['h'] for e in cycle])
        avg_s = np.mean([e['s'] for e in cycle])
        num_delta = sum(1 for e in cycle if e['event'] == "Δ")
        num_omega = sum(1 for e in cycle if e['event'] == "Ω")
        num_break = sum(1 for e in cycle if e['event'] == "⊖")
        num_suspect = sum(1 for e in cycle if e['event'] == "≈")
        episodes = [e['episode'] for e in cycle]

        summary.append({
            "cycle_id": cycle[0]['cycle'],
            "start_episode": episodes[0],
            "end_episode": episodes[-1],
            "length": episodes[-1] - episodes[0] + 1,
            "avg_H": avg_h,
            "avg_S": avg_s,
            "Δ_count": num_delta,
            "Ω_count": num_omega,
            "⊖_count": num_break,
            "≈_count": num_suspect
        })

    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(os.path.dirname(save_path), "cycle_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Cycle Summary 儲存到 {summary_csv}")
