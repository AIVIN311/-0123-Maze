# test_many.py
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from maze_env.maze_env import MazeEnv

def calculate_entropy(actions):
    counts = np.bincount(actions)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def calculate_wisdom_density(successes, total, params=1e5):
    return successes / (total * params)

def find_latest_model():
    train_dirs = sorted(glob.glob("runs/train_*"), reverse=True)
    for dir in train_dirs:
        for name in ["ppo_maze.zip"]:
            model_path = os.path.join(dir, name)
            if os.path.exists(model_path):
                print(f"✅ 載入模型：{model_path}")
                return model_path
    raise FileNotFoundError("❌ 找不到模型。")

def run_test_episode(model, env):
    obs, _ = env.reset()
    actions = []
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(int(action))
        if terminated:
            return actions, True
    return actions, False

def test_many(n=10):
    model_path = find_latest_model()
    model = PPO.load(model_path)
    env = MazeEnv()

    records = []

    for i in range(n):
        actions, success = run_test_episode(model, env)
        H = calculate_entropy(actions)
        S = calculate_wisdom_density(1 if success else 0, 1)
        print(f"🧪 測試 {i+1}: 成功={success}, H={H:.4f}, S={S:.6e}")
        records.append({"episode": i + 1, "entropy": H, "wisdom_density": S})

    df = pd.DataFrame(records)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("runs", f"test_many_{now}")
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)

    # 畫圖
    fig, ax1 = plt.subplots()
    ax1.plot(df['episode'], df['entropy'], 'b-', label='Entropy (H)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Entropy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(df['episode'], df['wisdom_density'], 'r-', label='Wisdom Density (S)')
    ax2.set_ylabel('Wisdom Density', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()
    fig.legend(loc='upper center')
    plt.title("H-S Curve（多次測試）")
    plt.grid(True)
    hs_path = os.path.join(save_dir, "hs_curve_many.png")
    plt.savefig(hs_path)
    plt.show()

    print(f"📈 多次 H-S 曲線儲存到 {hs_path}")

if __name__ == "__main__":
    test_many(n=10)  # ✅ 改次數可以在這裡
