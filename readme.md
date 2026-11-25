
# DCBall-Quant — 基于强化学习与冷热号的双色球量化预测

## 项目简介
DCBall-Quant 使用 **历史开奖数据**，结合 **传统统计规则（冷热号、频率偏移、组合覆盖）** 与 **强化学习（RL）** 探索选号策略。训练环境以 **30期窗口** 构造状态；当前 `main_withRL.py` 加载已有权重，输出 **5组** 实验号码，不会在每次运行时重新训练。奖励函数与训练代码见 `enhanced/`。

> 免责声明：本项目仅用于研究与教学。请理性娱乐。

---

## 中奖规则与期望建模（简要）
- 号码结构：**6 个红球（1–33） + 1 个蓝球（1–16）**；每注 **2 元**。
- 奖级（摘要）：一等奖与二等奖为浮动奖金（单注封顶 500 万元，按奖池及销售额分配）；三等奖起为固定奖金：三等 **3000 元**、四等 **200 元**、五等 **10 元**、六等 **5 元**。
- 组合总数：\( \binom{33}{6} \times 16 = 17,721,088 \)；对应一等奖概率约为 \(1/17{,}721{,}088\)。  
- 训练时，本项目将各奖级的**奖金 × 概率**作为期望回报，构造 RL 的奖励信号。

> 当前 `enhanced/env.py` 使用固定奖金与概率表，并按每注 `0.2` 扣减训练奖励；这是实验中的简化设定，不能直接解释为按每注 2 元计算的实际净收益。

---

## 目录结构
```
.
├── __pycache__/
├── back_up/                 # 备份脚本与历史版本（如有）
├── enhanced/                # 增强策略模块（冷热分层、规则约束、打分器等）
├── history.csv              # 历史开奖数据（本地可持续更新）
├── predictions.csv          # 历次预测记录（自动追加保存）
├── policy_model.pt          # RL 策略模型权重（训练后生成/加载）
├── main_normal.py           # 仅基于统计与规则的 baseline 预测
├── main_withRL.py           # 加载已有 RL 权重的推断入口（不训练）
├── update_data.py           # 刷新/追加最新开奖到 history.csv
├── utils.py                 # 工具函数（选号、约束、验证、评估等）
├── visualize.py             # 可视化（红蓝球频次与蓝球走势）
├── test.ipynb               # 交互式实验/调参笔记本
└── readme.md                # （本文件）
```

> 说明：可以在本地维护一份会随问随更的 `history.csv` 与 `predictions.csv`，本项目会优先读取本地最新数据。

---

## 快速开始

### 1) 环境
- Python 3.9+（建议 3.10/3.11）
- 依赖：`pandas`, `numpy`, `torch`, `matplotlib`, `gym`, `requests`, `lxml`；训练脚本还使用 `tqdm`。

```bash
pip install pandas numpy torch matplotlib gym requests lxml tqdm
```

### 2) 更新数据
```bash
python -c "from update_data import update_history; update_history()"

# 方式B：直接将最新开奖写入 history.csv（追加一行）
# 保持现有列名：期号,红球1,红球2,红球3,红球4,红球5,红球6,蓝球
```

### 3) 运行预测
```bash
# Baseline（仅规则/统计）
python main_normal.py

# 强化学习版本（加载 policy_model.pt 后推断）
python main_withRL.py
```

以上命令在仓库根目录运行。`main_withRL.py` 会先联网更新 `history.csv`，再加载 `policy_model.pt` 并写入 `predictions.csv`；`main_normal.py` 只读取本地数据并打印结果。训练循环位于 `enhanced/train_rl.py`，其导入路径和数据路径仍需单独配置，不属于上述推断命令。

默认策略：
- 训练环境以 **30 期窗口** 构造状态；当前推断调用 `env.reset()`，状态从历史索引 `33` 开始，尚未切换到最新窗口；
- 输出 **5 组** 下一期开奖推荐；
- 将 **奖金×概率** 的期望作为 RL 的 reward；
- 结合冷热号、覆盖度与去重约束进行采样与筛选；
- RL 入口的结果会 **自动写入 `predictions.csv`**。

---

## 强化学习与规则融合（摘要）
- **状态（State）**：最近 30 期的统计特征（冷热分布、出现频率、间隔、蓝球分布等）。  
- **动作（Action）**：生成一注号码（6 红 + 1 蓝），或一次性生成 5 注；
- **奖励（Reward）**：按官方奖级，将该注的奖金期望作为奖励（训练时可采用蒙特卡洛估计或近似期望）；
- **策略（Policy）**：
  - Baseline：规则采样 + 打分器（冷热/覆盖/惩罚项）；
  - RL：策略网络（如 MLP/Transformer 小模型），用策略梯度或近似 bandit 更新；
- **约束/惩罚**（可选）：避免与历史期次的蓝球完全一致；减少与历史号码 **≥5 个红球相同** 的组合；限制重复率与相似度，提升多样性与覆盖。

> 上述为方法概览；当前实现以 `enhanced/env.py`、`enhanced/policy_model.py` 和 `enhanced/train_rl.py` 为准，尚不代表经过独立回测验证的预测能力。

---

## 数据文件
- **history.csv**：历史开奖，列名为 `期号,红球1,红球2,红球3,红球4,红球5,红球6,蓝球`。
- **predictions.csv**：RL 入口输出的期号、6 个红球、蓝球与奖金列；当前不记录时间戳或策略配置。

---

## 结果可视化（示例）
```bash
python -c "from enhanced.strategy import load_history, get_recent_30; from visualize import visualize_history; visualize_history(get_recent_30(load_history()))"
```

当前图表展示红蓝球频次与蓝球走势，不包含回测收益曲线。


## License
This repository is for **research and educational use only**. No warranty. Please play responsibly.
