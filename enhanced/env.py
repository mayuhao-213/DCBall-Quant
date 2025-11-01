import gym
import numpy as np
import pandas as pd
from gym import spaces
from enhanced.scoring import compute_reward
from enhanced.strategy import  load_history, get_recent_30
from enhanced.heuristic import *
class LotteryEnv(gym.Env):
    """
    强化学习双色球选号环境：
    - 状态：红球/蓝球频率向量（49维）
    - 动作：从预生成的候选组合池中选多个索引（默认5注）
    - 奖励：多个组合的期望收益总和 - 成本
    """
    def __init__(self, history_df: pd.DataFrame, pool_size: int = 1000, num_actions_per_step: int = 5):
        super().__init__()
        self.history_df = history_df.reset_index(drop=True)
        self.pool_size = pool_size
        self.num_actions = num_actions_per_step
        self.current_index = 33
        self.state_window = 30
        self.action_pool = []

        self.observation_space = spaces.Box(low=0, high=100, shape=(49,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.pool_size] * self.num_actions)

        self._generate_action_pool()

        self.ev_table = {
            '一等奖': (1/17721088, 5000000),
            '二等奖': (1/296618, 30000),
            '三等奖': (1/54201, 3000),
            '四等奖': (1/1032, 200),
            '五等奖': (1/56, 10),
            '六等奖': (1/16, 5)
        }

    def _generate_action_pool(self):
        recent_df = get_recent_30(self.history_df.iloc[:self.current_index])
        seen = set()
        pool = []
        while len(pool) < self.pool_size:
            red, blue = generate_enhanced_recommendation(self.history_df, recent_df)
            key = tuple(red) + (blue,)
            if key not in seen:
                pool.append((red, blue))
                seen.add(key)
        self.action_pool = pool

    def _get_state(self):
        df_recent = get_recent_30(self.history_df.iloc[self.current_index - self.state_window:self.current_index])
        reds = df_recent[['红球1', '红球2', '红球3', '红球4', '红球5', '红球6']].values.flatten().astype(int)
        blues = df_recent['蓝球'].astype(int).values

        red_freq = np.bincount(reds, minlength=34)[1:]
        blue_freq = np.bincount(blues, minlength=17)[1:]

        return np.concatenate([red_freq, blue_freq]).astype(np.float32)

    def reset(self):
        self.current_index = 33
        self._generate_action_pool()
        return self._get_state()

    def _expected_value(self, reward):
        for level, (prob, value) in self.ev_table.items():
            if reward == value:
                return prob * value
        return 0.0

    def step(self, actions):
        total_ev = 0
        actual = self.history_df.iloc[self.current_index]
        true_red = [actual[f'红球{i}'] for i in range(1, 7)]
        true_blue = actual['蓝球']

        for action in actions:
            red, blue = self.action_pool[action]
            reward = compute_reward(red, blue, true_red, true_blue)
            total_ev += self._expected_value(reward)

        net_reward = total_ev - self.num_actions * 0.2

        self.current_index += 1
        done = self.current_index >= len(self.history_df)

        if not done:
            self._generate_action_pool()
            next_state = self._get_state()
        else:
            next_state = np.zeros((49,), dtype=np.float32)

        return next_state, net_reward, done, {}

    def render(self, mode='human'):
        print(f"当前期号：{self.history_df.iloc[self.current_index]['期号']}")
