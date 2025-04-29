import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from maze_env.maze_env import MazeEnv

# === Reward Logger（訓練時記錄總回報）===
class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.current_rewards += rewards[0]
        dones = self.locals.get("dones")
        if dones is not None and dones[0]:
            self.episode_rewards.append(self.current_rewards)
            self.current_rewards = 0.0
        return True

# === 每10萬步儲存 checkpoint ===
class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"checkpoint_{self.num_timesteps}.zip")
            self.model.save(model_path)
            print(f"✅ 自動儲存檢查點：{model_path}")
        return True

def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_rewards(rewards, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, label="Raw Rewards")
    plt.plot(moving_average(rewards), label="Smoothed Rewards (window=50)")
    plt.title("Reward Curve (Maze)")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"🎨 Reward 曲線圖儲存到 {save_path}")
    plt.show()

def save_rewards_to_csv(rewards, save_path):
    df = pd.DataFrame({"episode": np.arange(1, len(rewards) + 1), "reward": rewards})
    df.to_csv(save_path, index=False)
    print(f"📄 Rewards CSV 儲存到 {save_path}")

def create_run_folder(prefix="train"):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("runs", f"{prefix}_{now}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def plot_path(path, save_dir):
    maze_size = (8, 8)
    maze = np.zeros(maze_size)

    for x, y in path:
        if 0 <= x < maze_size[0] and 0 <= y < maze_size[1]:
            maze[x, y] = 0.5

    if path:
        start_x, start_y = path[0]
        end_x, end_y = path[-1]
        maze[start_x, start_y] = 0.8
        maze[end_x, end_y] = 1.0

    plt.figure(figsize=(6, 6))
    plt.imshow(maze, cmap="gray_r", origin="upper")
    plt.title("Maze Path Taken")
    plt.axis("off")
    path_plot_path = os.path.join(save_dir, "path_taken.png")
    plt.savefig(path_plot_path)
    print(f"🛤️ 路徑圖儲存到 {path_plot_path}")
    plt.show()

def find_latest_model():
    train_dirs = sorted(glob.glob("runs/train_*"), reverse=True)
    for dir in train_dirs:
        model_path = os.path.join(dir, "ppo_maze.zip")  # ⬅ 改這裡
        if os.path.exists(model_path):
            print(f"✅ 載入最新模型：{model_path}")
            return model_path
    raise FileNotFoundError("❌ 找不到模型 ppo_maze.zip")


def train_maze_agent():
    env = MazeEnv()
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5
    )

    run_dir = create_run_folder(prefix="train")
    reward_logger = RewardLoggerCallback()
    checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path=run_dir, verbose=1)

    model.learn(total_timesteps=2_000_000, callback=[reward_logger, checkpoint_callback])
    model.save(os.path.join(run_dir, "ppo_maze.zip"))
    print(f"✅ 訓練完成，模型儲存到 {run_dir}")

    save_rewards_to_csv(reward_logger.episode_rewards, os.path.join(run_dir, "rewards.csv"))
    plot_rewards(reward_logger.episode_rewards, save_path=os.path.join(run_dir, "reward_curve.png"))

def test_maze_agent(fast_mode=False):
    model_path = find_latest_model()
    model = PPO.load(model_path)
    env = MazeEnv()
    obs, _ = env.reset()

    test_log = []
    path = []

    for step in range(10000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        x, y = int(obs[0]), int(obs[1])
        path.append((x, y))

        print(f"第 {step+1} 步：位置 ({x},{y})，動作 {int(action)}，獎勵 {reward}")

        test_log.append({
            "step": step + 1,
            "position_x": x,
            "position_y": y,
            "action": int(action),
            "reward": reward,
            "terminated": terminated
        })

        if not fast_mode:
            time.sleep(0.03)
        env.render()

        if terminated:
            print("🎉 成功到達終點！")
            break

    run_dir = create_run_folder(prefix="test")
    pd.DataFrame(test_log).to_csv(os.path.join(run_dir, "test_log.csv"), index=False)
    print(f"📄 測試紀錄儲存到 {run_dir}")
    plot_path(path, save_dir=run_dir)

# === 批次測試（multi 模式）===
def test_multiple_times(n=10):
    for i in range(n):
        print(f"\n——— 第 {i+1}/{n} 次測試 ———")
        test_maze_agent(fast_mode=True)

# === 主程式入口 ===
if __name__ == "__main__":
    mode = input("輸入 'train' 開始訓練，輸入 'test' 單次測試，輸入 'fast' 快速測試，輸入 'multi' 多次測試：").strip().lower()

    if mode == "train":
        train_maze_agent()
    elif mode == "test":
        test_maze_agent(fast_mode=False)
    elif mode == "fast":
        test_maze_agent(fast_mode=True)
    elif mode == "multi":
        n = input("要測試幾次？輸入一個數字：").strip()
        if n.isdigit():
            test_multiple_times(int(n))
        else:
            print("請輸入有效數字！")
    else:
        print("請正確輸入 'train'、'test'、'fast' 或 'multi'！")
