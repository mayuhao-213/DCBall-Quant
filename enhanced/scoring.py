# enhanced/scoring.py

def classify_win_level(pred_reds, pred_blue, true_reds, true_blue):
    """
    判断预测组合属于哪一中奖等级
    输入：预测红球（6个）、蓝球，真实红球、蓝球
    输出：等级名称（字符串）
    """
    red_hit = len(set(pred_reds) & set(true_reds))
    blue_hit = (pred_blue == true_blue)

    if red_hit == 6 and blue_hit:
        return '一等奖'
    elif red_hit == 6:
        return '二等奖'
    elif red_hit == 5 and blue_hit:
        return '三等奖'
    elif (red_hit == 5) or (red_hit == 4 and blue_hit):
        return '四等奖'
    elif (red_hit == 4) or (red_hit == 3 and blue_hit):
        return '五等奖'
    elif blue_hit:
        return '六等奖'
    else:
        return '未中奖'

def reward_mapping(level):
    """根据中奖等级返回对应奖金"""
    reward_dict = {
        '一等奖': 5000000,
        '二等奖': 30000,
        '三等奖': 3000,
        '四等奖': 200,
        '五等奖': 10,
        '六等奖': 5,
        '未中奖': 0
    }
    return reward_dict.get(level, 0)

def compute_reward(pred_reds, pred_blue, true_reds, true_blue):
    """
    核心接口：计算预测组合的实际奖励（用于强化学习或训练监督标签）
    """
    level = classify_win_level(pred_reds, pred_blue, true_reds, true_blue)
    return reward_mapping(level)

def compute_net_reward(reward, num_bets=1, cost_per_bet=2):
    """
    扣除成本后的净收益
    """
    total_cost = num_bets * cost_per_bet
    return reward - total_cost
