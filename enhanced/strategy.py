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
    return df.tail(30).reset_index(drop=True)