# summary_report_generator.py
# ✅ 產出智慧訓練與語法事件的分析總結報告（純文字）

import os
import json
import pandas as pd

def generate_summary_report(run_dir):
    lines = []
    lines.append("\n📘 智慧節奏分析報告")
    lines.append(f"\n📂 測試資料夾：{os.path.basename(run_dir)}")

    metrics_path = os.path.join(run_dir, "metrics.csv")
    rhythm_path = os.path.join(run_dir, "wisdom_rhythm.json")
    transition_path = os.path.join(run_dir, "transition_points.csv")
    q_event_path = os.path.join(run_dir, "q_events.json")

    # Step 1: metrics 檢查與事件統計
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        df = df.rename(columns={"entropy": "h", "wisdom_density": "s"})  # ⬅️ 欄位轉換以避免錯誤
        lines.append(f"\n1. 語法事件統計：")
        lines.append(f"   - 總回合數：{len(df)}")
        if "cycle" in df.columns:
            lines.append(f"   - cycle 欄位：✅ 已加入")
            lines.append(f"   - 總週期數：{df['cycle'].nunique()}")
        else:
            lines.append("   - cycle 欄位：❌ 尚未標記")
    else:
        lines.append("\n1. 語法事件統計：❌ 找不到 metrics.csv")

    # Step 2: wisdom_rhythm 統計
    if os.path.exists(rhythm_path):
        with open(rhythm_path, encoding="utf-8") as f:
            events = json.load(f)
        counts = {}
        for e in events:
            counts[e["event"]] = counts.get(e["event"], 0) + 1
        lines.append("\n2. 語法節奏分佈：")
        for k in ["∅", "Σ", "Δ", "Ω", "⊖", "≈"]:
            v = counts.get(k, 0)
            lines.append(f"   - {k}：{v} 回合")
    else:
        lines.append("\n2. 語法節奏分佈：❌ 找不到 wisdom_rhythm.json")

    # Step 3: transition point
    if os.path.exists(transition_path):
        df_trans = pd.read_csv(transition_path)
        lines.append(f"\n3. 結構轉變點：{len(df_trans)} 個轉折點偵測到 ✅")
    else:
        lines.append("\n3. 結構轉變點：❌ 找不到 transition_points.csv")

    # Step 4: Q事件
    if os.path.exists(q_event_path):
        with open(q_event_path, encoding="utf-8") as f:
            q_events = json.load(f)
        lines.append(f"\n4. 模組化 Q 值事件：{len(q_events)} 筆 ✅")
    else:
        lines.append("\n4. 模組化 Q 值事件：❌ 無資料或未執行")

    # Step 5: 圖像產出情況
    lines.append("\n5. 圖片輸出：")
    for fig in ["syntax_pulse_v2.png", "fit_H_logistic.png", "fit_S_growth.png"]:
        fig_path = os.path.join(run_dir, fig)
        if os.path.exists(fig_path):
            lines.append(f"   - {fig}：✅")
        else:
            lines.append(f"   - {fig}：❌ 未生成")

    # 輸出報告
    report_path = os.path.join(run_dir, "summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n📝 分析報告已儲存：{report_path}")
    return report_path
