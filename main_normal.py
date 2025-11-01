from utils import *
from update_data import update_history
#from visualize import visualize_history

df_all = load_history('history.csv')
df_recent = get_recent_30(df_all)

for i in range(5):
    red, blue = generate_unique_recommendation(df_all, df_recent)
    print(f"第{i+1}注：红球 {red} | 蓝球 {blue}")
