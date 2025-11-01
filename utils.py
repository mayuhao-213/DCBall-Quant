import pandas as pd  # 导入pandas库，用于数据处理
def load_history(path='history.csv') -> pd.DataFrame:
    """加载历史双色球数据，并返回 DataFrame"""
    df = pd.read_csv(path, dtype=str)
    red_cols = ['红球1', '红球2', '红球3', '红球4', '红球5', '红球6']
    for col in red_cols + ['蓝球']:
        df[col] = df[col].astype(int)
    return df
def get_recent_30(df: pd.DataFrame) -> pd.DataFrame:
    """返回最近30期的数据"""
    return df.tail(33).reset_index(drop=True)
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
import random

def generate_red_ball(hot, cold):
    """生成1组红球：从热号中选3个，冷号中选3个，组合打乱"""
    red = random.sample(hot, 3) + random.sample(cold, 3)
    return sorted(red)
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
def generate_unique_recommendation(df_full, df_30, max_retry=1000):
    """生成不与历史冲突的推荐号码组合"""
    hot, cold = compute_red_hot_cold(df_30)
    cold_blues = compute_blue_cold(df_30)
    
    for _ in range(max_retry):
        red = generate_red_ball(hot, cold)
        blue = generate_blue_ball(cold_blues)
        if not is_conflict(red, blue, df_full):
            return red, blue
    raise ValueError("无法在限制内生成不冲突的组合")
