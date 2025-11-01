def analyze_sum_distribution(df):
    """计算历史红球和值的分布"""
    red_cols = ['红球1', '红球2', '红球3', '红球4', '红球5', '红球6']
    red_sum = df[red_cols].astype(int).sum(axis=1)
    return red_sum.describe(), red_sum.value_counts().sort_index()

def count_consecutive_pairs(row):
    """给定一组红球，返回有多少对连续号码（如11和12）"""
    red = sorted([int(row[f'红球{i}']) for i in range(1, 7)])
    return sum(1 for i in range(5) if red[i+1] - red[i] == 1)

def analyze_consecutive_distribution(df):
    """分析历史期号中连号对的分布"""
    return df.apply(count_consecutive_pairs, axis=1).value_counts().sort_index()


