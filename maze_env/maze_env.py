import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MazeEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(4)  # 上下左右四個動作
        self.observation_space = spaces.Box(low=0, high=7, shape=(2,), dtype=np.float32)  # 因為是8x8，所以最高index是7

        self.maze = np.array([
            [2, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 1, 1, 0],
            [1, 0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 3]
        ])
        # 0: 空地, 1: 障礙物, 2: 起點, 3: 終點

        self.start_pos = np.argwhere(self.maze == 2)[0]
        self.goal_pos = np.argwhere(self.maze == 3)[0]
        self.state = self.start_pos.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_pos.copy()
        return self.state.astype(np.float32), {}

    def step(self, action):
        next_state = self.state.copy()

        max_row, max_col = self.maze.shape[0] - 1, self.maze.shape[1] - 1

        if action == 0:  # 上
            next_state[0] = max(self.state[0] - 1, 0)
        elif action == 1:  # 下
            next_state[0] = min(self.state[0] + 1, max_row)
        elif action == 2:  # 左
            next_state[1] = max(self.state[1] - 1, 0)
        elif action == 3:  # 右
            next_state[1] = min(self.state[1] + 1, max_col)

        if self.maze[next_state[0], next_state[1]] != 1:  # 如果不是牆壁，才移動
            self.state = next_state

        terminated = np.array_equal(self.state, self.goal_pos)
        reward = 1.0 if terminated else -0.01

        return self.state.astype(np.float32), reward, terminated, False, {"position": self.state.copy()}

    def render(self):
        display = ""
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if (i, j) == tuple(self.state):
                    display += "A"
                elif (i, j) == tuple(self.goal_pos):
                    display += "G"
                elif self.maze[i, j] == 1:
                    display += "#"
                else:
                    display += "."
            display += "\n"
        print(display)
