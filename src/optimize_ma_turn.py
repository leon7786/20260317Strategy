"""
MA 方向拐头策略优化
均线方向开始向上时买入，向下时卖出
测试 13日 ~ 56日 均线，找最佳周期
对 PFE、601857.SS、0857.HK 三只股票分别测试
"""
import pandas as pd
import numpy as np
import os
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
START_DATE = pd.Timestamp("2012-05-18")
END_DATE = pd.Timestamp("2025-12-01")
INITIAL = 10000.0

STOCKS = [
    ("PFE", "PFE", "Pfizer 辉瑞"),
    ("601857_SS", "601857_SS", "中国石油 A股"),
    ("0857_HK", "0857_HK", "中国石油 H股"),
]

def load(file_prefix):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{file_prefix}.csv"), index_col=0, parse_dates=True)
    df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]
    prices = df["Close"].dropna()
    
    divs_dict = {}
    div_path = os.path.join(DATA_DIR, f"{file_prefix}_dividends.csv")
    if os.path.exists(div_path):
        ddf = pd.read_csv(div_path, index_col=0, parse_dates=True)
        s = ddf.iloc[:, 0]
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index, utc=True)
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        for d, v in s.items():
            if START_DATE <= d <= END_DATE:
                divs_dict[d] = v
    return prices, divs_dict

def backtest_ma_turn(prices_arr, dates, divs_dict, ma_period):
    """
    MA 方向拐头策略:
    - 当 MA 今天 > MA 昨天（方向向上）→ 买入
    - 当 MA 今天 < MA 昨天（方向向下）→ 卖出
    """
    n = len(prices_arr)
    if ma_period + 1 >= n:
        return INITIAL, 0, 0
    
    ma = pd.Series(prices_arr).rolling(ma_period).mean().values
    
    cash = INITIAL
    shares = 0.0
    in_pos = False
    buy_count = 0
    sell_count = 0
    
    for i in range(ma_period + 1, n):
        # DRIP
        if in_pos and dates[i] in divs_dict:
            div = shares * divs_dict[dates[i]]
            shares += div / prices_arr[i]
        
        ma_rising = ma[i] > ma[i-1]  # 均线方向向上
        ma_falling = ma[i] < ma[i-1]  # 均线方向向下
        
        if not in_pos and ma_rising:
            shares = cash / prices_arr[i]
            cash = 0
            in_pos = True
            buy_count += 1
        elif in_pos and ma_falling:
            cash = shares * prices_arr[i]
            shares = 0
            in_pos = False
            sell_count += 1
    
    final = cash + shares * prices_arr[-1]
    return final, buy_count, sell_count

print("=" * 80)
print("📊 MA 方向拐头策略优化 (13日 ~ 56日)")
print("   策略: 均线方向开始向上买入，向下卖出")
print("=" * 80)

for file_prefix, _, stock_name in STOCKS:
    prices, divs_dict = load(file_prefix)
    prices_arr = prices.values
    dates = prices.index
    
    # 买入持有基准
    sh_bh = INITIAL / prices_arr[0]
    for d in dates:
        if d in divs_dict:
            sh_bh += sh_bh * divs_dict[d] / prices.loc[d]
    bh_final = sh_bh * prices_arr[-1]
    
    print(f"\n{'─'*80}")
    print(f"🔍 {stock_name}")
    print(f"   数据: {len(prices_arr)} 天 | 价格: {prices_arr[0]:.2f} → {prices_arr[-1]:.2f}")
    print(f"   基准(买入持有+DRIP): ${bh_final:,.0f} ({(bh_final/INITIAL-1)*100:+.1f}%)")
    print(f"{'─'*80}")
    print(f"   {'MA周期':>6}  {'最终价值':>12}  {'回报率':>10}  {'vs买入持有':>10}  {'买入次数':>8}  {'卖出次数':>8}")
    print(f"   {'─'*70}")
    
    results = []
    for period in range(13, 57):
        final, buys, sells = backtest_ma_turn(prices_arr, dates, divs_dict, period)
        ret = (final / INITIAL - 1) * 100
        vs_bh = (final - bh_final) / bh_final * 100
        results.append((period, final, ret, vs_bh, buys, sells))
        
        marker = ""
        if final == max(r[1] for r in results):
            marker = " ← 当前最佳"
        
        print(f"   {period:>4}日  ${final:>11,.0f}  {ret:>+9.1f}%  {vs_bh:>+9.1f}%  {buys:>6}  {sells:>6}{marker}")
    
    # 排序找 Top 5
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n   🏆 Top 5 最佳周期:")
    for i, (period, final, ret, vs_bh, buys, sells) in enumerate(results[:5]):
        medal = "🥇🥈🥉"[i] if i < 3 else "  "
        print(f"   {medal} {period}日均线: ${final:,.0f} ({ret:+.1f}%) | vs买入持有 {vs_bh:+.1f}% | {buys}买/{sells}卖")
    
    worst = results[-1]
    print(f"   📉 最差: {worst[0]}日均线: ${worst[1]:,.0f} ({worst[2]:+.1f}%)")

print(f"\n{'='*80}")
print("✅ 优化完成!")
print("=" * 80)
