# ✅ 完整自動化：載入最新 test run，標記語法事件、畫圖、曲線擬合 + cycle/transition 分析 + 產出總結報告

import os
import pandas as pd
from syntax_event_logger import SyntaxEventLogger
from plot_syntax_pulse_v2 import plot_syntax_pulse
from fit_logistic_map import fit_logistic_map
from fit_s_growth import fit_s_growth
from cycle_marker_utils import add_cycle_markers
from detect_transition_points import detect_transitions
from cycle_transition_detector import detect_cycle_transitions
from summary_report_generator import generate_summary_report
from plot_hs_curve import plot_hs_curve  # ✅ 新增：繪製 H-S 曲線

# === 自動尋找最新的 test 結果資料夾 ===
def find_latest_test_dir(base_dir="runs"):
    all_test_dirs = []
    for category in ["test", "test_many", "fast"]:
        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path):
            continue
        for f in os.listdir(category_path):
            subdir = os.path.join(category_path, f)
            if os.path.isdir(subdir) and f.startswith(category):
                all_test_dirs.append(subdir)

    if not all_test_dirs:
        raise FileNotFoundError("❌ 找不到任何 test、test_many 或 fast 類型的測試資料夾。")

    latest_dir = sorted(all_test_dirs)[-1]
    print("📂 最新測試資料夾：", latest_dir)
    return latest_dir

# === 主流程 ===
def main():
    try:
        test_dir = find_latest_test_dir()
        metrics_path = os.path.join(test_dir, "metrics.csv")

        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"❌ 找不到 metrics.csv：{metrics_path}")
        print(f"✅ 載入 metrics 檔案：{metrics_path}")

        # Step 0: 加入智慧週期欄位
        df_check = pd.read_csv(metrics_path)
        if "cycle" not in df_check.columns:
            print("🔁 加入 cycle 欄位...")
            add_cycle_markers(metrics_path)
            df_check = pd.read_csv(metrics_path)

        # Step 1: 語法事件標記
        logger = SyntaxEventLogger()
        for _, row in df_check.iterrows():
            logger.log_event(
                episode=row['episode'],
                h=row['entropy'],
                s=row['wisdom_density'],
                q=row.get('modularity', 0.0),
                terminated=bool(row.get('terminated', False))
            )
        rhythm_path = os.path.join(test_dir, "wisdom_rhythm.json")
        logger.save_log(rhythm_path)

        # Step 2: 繪製語法脈動圖
        plot_syntax_pulse(df_check, event_path=rhythm_path,
                          save_path=os.path.join(test_dir, "syntax_pulse_v2.png"))

        # Step 3: 擬合 H 曲線
        try:
            fit_logistic_map(metrics_path, output_path=os.path.join(test_dir, "fit_H_logistic.png"))
        except Exception as e:
            print("⚠️ 擬合 H 曲線失敗：", e)

        # Step 4: 擬合 S 曲線
        try:
            fit_s_growth(metrics_path, output_path=os.path.join(test_dir, "fit_S_growth.png"))
        except Exception as e:
            print("⚠️ 擬合 S 曲線失敗：", e)

        # Step 4.5: 繪製 H-S 曲線
        try:
            hs_curve_path = os.path.join(test_dir, "hs_curve_many.png")
            plot_hs_curve(df_check, save_path=hs_curve_path)
            print(f"📈 H-S 曲線圖儲存：{hs_curve_path}")
        except Exception as e:
            print("⚠️ 無法繪製 H-S 曲線：", e)

        # Step 5: 偵測轉變點（微觀）
        transition_path = os.path.join(test_dir, "transition_points.csv")
        detect_transitions(metrics_path, save_path=transition_path)

        # Step 6: 偵測週期切換（宏觀）
        cycle_transitions = detect_cycle_transitions(metrics_path)
        print(f"🌀 共偵測到 {len(cycle_transitions)} 個 cycle transition")

        # Step 7: 產出文字總結報告
        report_path = generate_summary_report(test_dir)
        print(f"📝 分析報告已儲存：{report_path}")

        print("🎉 全部分析完成 ✅ 結果儲存於：", test_dir)

    except Exception as e:
        print("❌ 發生錯誤：", e)

if __name__ == "__main__":
    main()
