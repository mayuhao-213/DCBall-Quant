import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, state_dim=49, hidden_dim=128, pool_size=1000, num_actions=5):
        super().__init__()
        self.state_dim = state_dim
        self.pool_size = pool_size
        self.num_actions = num_actions

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 输出每注组合的 logits（可重复选）
        self.head = nn.Linear(hidden_dim, pool_size)

    def forward(self, state):
        """
        输入：state shape = (B, 49)
        输出：logits shape = (B, num_actions, pool_size)
        """
        x = self.backbone(state)              # (B, H)
        logits = self.head(x)                 # (B, pool_size)
        logits = logits.unsqueeze(1).repeat(1, self.num_actions, 1)  # (B, num_actions, pool_size)
        return logits

    def sample_actions(self, state):
        """
        从策略中抽样 num_actions 个不重复动作
        返回：(B, num_actions)，(B, num_actions)
        """
        logits = self.head(self.backbone(state))  # shape: (B, pool_size)
        probs = torch.softmax(logits, dim=-1)     # shape: (B, pool_size)

        # ⚠️ torch.multinomial 自动处理 batch（但只能处理二维）
        actions = torch.multinomial(probs, num_samples=self.num_actions, replacement=False)  # (B, num_actions)
        log_probs = torch.log(probs.gather(1, actions))  # 获取对应 log_prob (B, num_actions)

        return actions, log_probs

