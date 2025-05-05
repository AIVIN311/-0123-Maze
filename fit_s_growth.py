import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# === S 成長模型 ===
def s_growth(S, t, k, C, epsilon):
    return k * S * (1 - S / C) - epsilon * S

def s_growth_wrapper(t, k, C, epsilon):
    S0 = S_data[0]
    result = odeint(s_growth, S0, t, args=(k, C, epsilon)).flatten()
    return result

# === 主函數：用於 auto_analyze ===
def fit_s_growth(metrics_path, output_path="fit_S_growth.png"):
    df = pd.read_csv(metrics_path)
    global S_data
    S_data = df["wisdom_density"].values
    t = np.arange(len(S_data))

    try:
        popt, _ = curve_fit(s_growth_wrapper, t, S_data, bounds=([0, 0, 0], [10, 1, 0.5]))
        k, C, epsilon = popt
    except:
        print("⚠️ 擬合失敗")
        return

    fitted_S = s_growth_wrapper(t, *popt)

    plt.figure(figsize=(8, 5))
    plt.plot(t, S_data, 'bo-', label="Actual S")
    plt.plot(t, fitted_S, 'r--', label="Fitted S")
    plt.xlabel("Steps")
    plt.ylabel("Wisdom Density (S)")
    plt.title("S Growth Fitting")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

    print(f"✅ 成功擬合並儲存圖檔：{output_path}")

