import requests
from lxml import etree
import csv
import os
from collections import deque

def update_history():
    csv_path = 'history.csv'

    # Step 1: 仅加载CSV中最近30期的期号
    existing_periods = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # 跳过表头
            recent_rows = deque(f, maxlen=50)  # 读取文件尾部最后50行
            for row in recent_rows:
                period = row.strip().split(',')[0].strip()
                existing_periods.add(period)

    # Step 2: 抓取页面中最近 30 期数据
    url = 'https://datachart.500.com/ssq/'
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers)
    resp.encoding = 'gbk'
    e = etree.HTML(resp.text)

    trs = e.xpath('//tbody[@id="tdata"]/tr[not(contains(@class,"tdbck"))]')
    new_rows = []

    for tr in trs:
        period = tr.xpath('./td[1]/text()')[0].strip().zfill(5)
        reds = tr.xpath('./td[contains(@class,"chartBall01")]/text()')
        blue = tr.xpath('./td[contains(@class,"chartBall02")]/text()')[0]
        if period not in existing_periods:
            new_rows.append([period] + reds + [blue])

    # Step 3: 追加写入新数据
    if new_rows:
        print(f"发现 {len(new_rows)} 条新数据，正在追加...")
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in reversed(new_rows):  # 保持时间顺序一致
                writer.writerow(row)
        print("更新完成 ✅")
    else:
        print("无新增记录，无需更新。")
        # 打印最新一期内容
    
    # Step 5: 始终打印最新一期开奖信息（trs[-1] 为最新期）
    latest_tr = trs[-1]
    latest_period = latest_tr.xpath('./td[1]/text()')[0].strip().zfill(5)
    latest_reds = latest_tr.xpath('./td[contains(@class,"chartBall01")]/text()')
    latest_blue = latest_tr.xpath('./td[contains(@class,"chartBall02")]/text()')[0]
    print(f"最新一期：{latest_period}，红球：{' '.join(latest_reds)}，蓝球：{latest_blue}")

