import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from tqdm import trange

from env import LotteryEnv
from strategy import load_history
from policy_model import PolicyNet

# 初始化环境与策略
history_df = load_history('history.csv')
env = LotteryEnv(history_df, pool_size=50, num_actions_per_step=5)
policy = PolicyNet(state_dim=49, hidden_dim=128, pool_size=50, num_actions=5)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

epochs = 1000
print_interval = 10
all_rewards = []
all_losses = []

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

for episode in trange(epochs):
    state = env.reset()
    log_probs_list = []
    rewards_list = []
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, 49)
        actions, log_probs = policy.sample_actions(state_tensor)              # (1, 5), (1, 5)

        next_state, reward, done, _ = env.step(actions[0].tolist())

        log_probs_list.append(log_probs.sum())
        rewards_list.append(reward)
        state = next_state

    returns = compute_returns(rewards_list)
    loss = 0
    for log_prob, R in zip(log_probs_list, returns):
        loss += -log_prob * R

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    all_rewards.append(sum(rewards_list))
    all_losses.append(loss.item())

    if episode % print_interval == 0:
        print(f"Episode {episode:4d} | Total Return: {sum(rewards_list):.4f} | Loss: {loss.item():.4f}")

# 保存模型
torch.save(policy.state_dict(), 'policy_model.pt')

# 可视化结果
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(all_rewards)
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.subplot(1, 2, 2)
plt.plot(all_losses)
plt.title('Loss per Episode')
plt.xlabel('Episode')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()
