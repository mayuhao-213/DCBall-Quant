import random
import pandas as pd

def compute_red_hot_cold(df_30: pd.DataFrame):
    """统计最近30期红球频率，返回热号集、冷号集"""
    reds = df_30[['红球1', '红球2', '红球3', '红球4', '红球5', '红球6']].values.flatten()
    red_counts = pd.Series(reds).value_counts().sort_index()
    avg = red_counts.mean()
    hot = red_counts[red_counts > avg].index.tolist()
    cold = red_counts[red_counts <= avg].index.tolist()
    return hot, cold

def compute_blue_cold(df_30: pd.DataFrame):
    """统计最近30期蓝球频率，返回冷号集（<=平均）"""
    blue_counts = df_30['蓝球'].astype(int).value_counts().sort_index()
    full = blue_counts.reindex(range(1, 17), fill_value=0)
    avg = full.mean()
    cold_blues = full[full <= avg].index.tolist()
    return cold_blues

def generate_balanced_reds(hot_pool, cold_pool):
    """
    生成一组红球，满足增强策略：
    - 奇偶数比例在 2:4 ~ 4:2
    - 涉及至少两个号段（1~11, 12~22, 23~33）
    - 热号+冷号混合生成
    """
    all_pool = hot_pool + cold_pool
    tries = 0
    while tries < 1000:
        candidate = sorted(random.sample(all_pool, 6))
        odds = [n for n in candidate if n % 2 == 1]
        evens = [n for n in candidate if n % 2 == 0]
        sections = set()
        for n in candidate:
            if n <= 11:
                sections.add(1)
            elif n <= 22:
                sections.add(2)
            else:
                sections.add(3)
        if 2 <= len(odds) <= 4 and len(sections) >= 2:
            return candidate
        tries += 1
    raise ValueError("无法生成满足增强策略的红球组合")

def generate_blue_ball(cold_blues):
    """从冷号中随机选出一个蓝球"""
    return random.choice(cold_blues)

def is_conflict(new_reds, new_blue, history_df) -> bool:
    """判断该组号码是否与历史重复或冲突"""
    for _, row in history_df.iterrows():
        hist_red = [row[f'红球{i}'] for i in range(1, 7)]
        hist_blue = row['蓝球']
        same_red_count = len(set(new_reds) & set(hist_red))
        if same_red_count == 6:
            return True
        if same_red_count == 5 and new_blue == hist_blue:
            return True
        if same_red_count == 6 and new_blue == hist_blue:
            return True
    return False


def is_sum_in_range(red, min_sum=80, max_sum=120):
    return min_sum <= sum(red) <= max_sum
def count_consecutive(red):
    red = sorted(red)
    return sum(1 for i in range(5) if red[i+1] - red[i] == 1)

def is_consecutive_ok(red, max_pairs=2):
    return count_consecutive(red) <= max_pairs


def generate_enhanced_recommendation(df_full, df_30, max_retry=1000):
    """生成不与历史冲突的组合，使用增强策略"""
    hot, cold = compute_red_hot_cold(df_30)
    cold_blues = compute_blue_cold(df_30)
    
    for _ in range(max_retry):
        red = generate_balanced_reds(hot, cold)
        blue = generate_blue_ball(cold_blues)
        if (
            is_sum_in_range(red) and
            is_consecutive_ok(red) #and
            #not is_conflict(red, blue, df_full)
        ):
            return red, blue

    raise ValueError("无法在限制内生成不冲突的增强组合")