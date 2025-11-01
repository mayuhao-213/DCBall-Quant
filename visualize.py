import pandas as pd
import matplotlib.pyplot as plt

def visualize_history(df):
    # —— 2. 红球频次统计 —— 
    # 将红球1~6列合并为一个 Series，并转为整数
    red_columns = ['红球1', '红球2', '红球3', '红球4', '红球5', '红球6']
    reds_flat = df[red_columns].values.flatten().astype(int)
    red_freq = pd.Series(reds_flat).value_counts().sort_index()
    avg_red = red_freq.mean()

    # —— 3. 红球可视化 —— 
    plt.figure(figsize=(12, 4))
    colors_red = ['red' if f > avg_red else 'blue' for f in red_freq.values]
    plt.bar(red_freq.index, red_freq.values, color=colors_red)
    plt.axhline(avg_red, color='green', linestyle='--', label=f'Avg={avg_red:.2f}')
    plt.xlabel('Red Ball Number')
    plt.ylabel('Frequency')
    plt.title('Red Ball Hot/Cold Distribution (Last 30 Draws)')
    plt.xticks(range(1, 34))
    plt.legend()
    plt.tight_layout()
    plt.show()

    # —— 4. 蓝球走势与频次 —— 
    blue_nums = df['蓝球'].astype(int)
    blue_freq = blue_nums.value_counts().sort_index()
    blue_freq_full = blue_freq.reindex(range(1, 17), fill_value=0)
    avg_blue = blue_freq_full.mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 4.1 折线图：蓝球走势（纵轴是期号）
    axes[0].plot(
        blue_nums.values,
        df['期号'],
        marker='o',
        linestyle='-',
        color='blue'
    )
    axes[0].set_xlabel('Blue Ball Number')
    axes[0].set_ylabel('Draw Number')
    axes[0].set_title('Blue Ball Trend vs Draw')
    axes[0].get_yaxis().set_visible(False)

    # 4.2 柱状图：蓝球频率分布
    colors_blue = ['red' if f > avg_blue else 'blue' for f in blue_freq_full.values]
    axes[1].bar(blue_freq_full.index, blue_freq_full.values, color=colors_blue)
    axes[1].axhline(avg_blue, color='green', linestyle='--', label=f'Avg={avg_blue:.2f}')
    axes[1].set_xlabel('Blue Ball Number')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Blue Ball Hot/Cold Distribution')
    axes[1].set_xticks(range(1, 17))
    axes[1].legend()

    plt.tight_layout()
    plt.show()


