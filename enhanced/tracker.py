import csv
import os
import pandas as pd
from enhanced.scoring import compute_reward

def log_predictions(predictions, next_period, path='predictions.csv'):
    """
    记录预测组合到 CSV 文件中，奖金初始为空 ''
    """
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['期号', '红球1', '红球2', '红球3', '红球4', '红球5', '红球6', '蓝球', '奖金'])
        for red, blue in predictions:
            writer.writerow([next_period] + red + [blue, ''])


def update_prediction_rewards(history_df, pred_path='predictions.csv'):
    """
    遍历 predictions.csv 中尚未开奖的记录，根据 history.csv 填入奖金
    """
    pred_df = pd.read_csv(pred_path)
    updated = False

    for i, row in pred_df.iterrows():
        if pd.isna(row['奖金']) or str(row['奖金']).strip() == '':
            period = str(int(row['期号'])).zfill(5)
            matched = history_df[history_df['期号'] == period]
            if not matched.empty:
                true = matched.iloc[0]
                pred_reds = [int(row[f'红球{i}']) for i in range(1, 7)]
                pred_blue = int(row['蓝球'])
                true_reds = [true[f'红球{i}'] for i in range(1, 7)]
                true_blue = true['蓝球']

                reward = compute_reward(pred_reds, pred_blue, true_reds, true_blue)
                pred_df.at[i, '奖金'] = reward
                updated = True

                if reward > 0:
                    print(f"🎉 第 {period} 期命中！中奖金额：¥{reward} 元")
                    print(f"  👉 预测号码：红球 {pred_reds} | 蓝球 {pred_blue}")
                    print(f"  ✅ 开奖号码：红球 {true_reds} | 蓝球 {true_blue}")

                

    if updated:
        pred_df.to_csv(pred_path, index=False, encoding='utf-8')
        print("✅ predictions.csv 已更新中奖信息。")
    else:
        print("🔍 无需更新，暂无新开奖匹配预测期号。")
