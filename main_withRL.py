import torch
from enhanced.strategy import load_history, get_recent_30
from enhanced.env import LotteryEnv
from enhanced.policy_model import PolicyNet
from enhanced.tracker import log_predictions, update_prediction_rewards
from update_data import update_history
# 调用
update_history()
# 0. 自动更新历史数据并评估已有预测
df_all = load_history('history.csv')


# 1. 构建环境和状态
df_recent = get_recent_30(df_all)
env = LotteryEnv(df_all, pool_size=50, num_actions_per_step=5)
state = env.reset()

# 2. 加载策略模型
model_path = 'policy_model.pt'
policy = PolicyNet(state_dim=49, hidden_dim=128, pool_size=50, num_actions=5)
policy.load_state_dict(torch.load(model_path))
policy.eval()

# 3. 获取状态并生成推荐组合
state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    actions, _ = policy.sample_actions(state_tensor)
actions = actions[0].tolist()

# 4. 从动作池中查找推荐组合
candidates = [env.action_pool[i] for i in actions]

# 5. 输出推荐结果
latest_period = df_recent['期号'].astype(int).max()
next_period = str(latest_period + 1).zfill(5)

print(f'\n🎯 推荐期号：{next_period}')
for i, (red, blue) in enumerate(candidates):
    print(f"第{i+1}注：红球 {red} | 蓝球 {blue}")

# 6. 记录预测结果
log_predictions(candidates, next_period)
# 7. 更新预测奖金
update_prediction_rewards(df_recent)
