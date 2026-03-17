"""
MA 方向拐头策略 — 全范围优化 (1日 ~ 365日)
"""
import pandas as pd
import numpy as np
import os, time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
START_DATE = pd.Timestamp("2012-05-18")
END_DATE = pd.Timestamp("2025-12-01")
INITIAL = 10000.0

STOCKS = [
    ("PFE", "Pfizer 辉瑞"),
    ("601857_SS", "中国石油 A股"),
    ("0857_HK", "中国石油 H股"),
]

def load(fp):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{fp}.csv"), index_col=0, parse_dates=True)
    df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]
    prices = df["Close"].dropna()
    divs = {}
    dp = os.path.join(DATA_DIR, f"{fp}_dividends.csv")
    if os.path.exists(dp):
        ddf = pd.read_csv(dp, index_col=0, parse_dates=True)
        s = ddf.iloc[:, 0]
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index, utc=True)
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        for d, v in s.items():
            if START_DATE <= d <= END_DATE:
                divs[d] = v
    return prices, divs

def bt(prices_arr, dates, divs, period):
    n = len(prices_arr)
    if period < 1 or period + 1 >= n:
        return INITIAL, 0
    if period == 1:
        # 1日"均线"就是价格本身，拐头 = 价格上涨买/下跌卖
        cash, sh, inp, trades = INITIAL, 0.0, False, 0
        for i in range(2, n):
            if inp and dates[i] in divs:
                sh += sh * divs[dates[i]] / prices_arr[i]
            if not inp and prices_arr[i] > prices_arr[i-1]:
                sh = cash / prices_arr[i]; cash = 0; inp = True; trades += 1
            elif inp and prices_arr[i] < prices_arr[i-1]:
                cash = sh * prices_arr[i]; sh = 0; inp = False; trades += 1
        return cash + sh * prices_arr[-1], trades
    
    ma = pd.Series(prices_arr).rolling(period).mean().values
    cash, sh, inp, trades = INITIAL, 0.0, False, 0
    for i in range(period + 1, n):
        if inp and dates[i] in divs:
            sh += sh * divs[dates[i]] / prices_arr[i]
        if not inp and ma[i] > ma[i-1]:
            sh = cash / prices_arr[i]; cash = 0; inp = True; trades += 1
        elif inp and ma[i] < ma[i-1]:
            cash = sh * prices_arr[i]; sh = 0; inp = False; trades += 1
    return cash + sh * prices_arr[-1], trades

print("=" * 80)
print("📊 MA 方向拐头策略 — 全范围优化 (1日 ~ 365日)")
print("=" * 80)

for fp, name in STOCKS:
    prices, divs = load(fp)
    pa = prices.values
    dates = prices.index
    n = len(pa)
    
    # 基准
    sh_bh = INITIAL / pa[0]
    for d in dates:
        if d in divs:
            sh_bh += sh_bh * divs[d] / prices.loc[d]
    bh = sh_bh * pa[-1]
    
    print(f"\n{'─'*80}")
    print(f"🔍 {name} | {n} 天 | {pa[0]:.2f} → {pa[-1]:.2f}")
    print(f"   基准(买入持有+DRIP): ${bh:,.0f} ({(bh/INITIAL-1)*100:+.1f}%)")
    print(f"{'─'*80}")
    
    results = []
    t0 = time.time()
    for p in range(1, 366):
        if p + 2 >= n:
            break
        val, tr = bt(pa, dates, divs, p)
        results.append((p, val, tr))
    elapsed = time.time() - t0
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   ⏱ {elapsed:.1f}s | 测试 {len(results)} 个周期")
    print(f"\n   🏆 Top 10 最佳周期:")
    for i, (p, val, tr) in enumerate(results[:10]):
        ret = (val/INITIAL-1)*100
        vs = (val-bh)/bh*100
        medal = "🥇🥈🥉"[i] if i < 3 else f" {i+1}"
        print(f"   {medal} {p:>3}日均线: ${val:>11,.0f} ({ret:>+8.1f}%) | vs买入持有 {vs:>+7.1f}% | {tr} 笔交易")
    
    print(f"\n   📉 Bottom 5:")
    for p, val, tr in results[-5:]:
        ret = (val/INITIAL-1)*100
        print(f"      {p:>3}日均线: ${val:>11,.0f} ({ret:>+8.1f}%) | {tr} 笔交易")
    
    # 按区间统计最佳
    ranges = [(1,10,"极短期"), (11,30,"短期"), (31,60,"中短期"), (61,120,"中期"), (121,250,"中长期"), (251,365,"长期")]
    print(f"\n   📊 各区间冠军:")
    for lo, hi, label in ranges:
        subset = [(p,v,t) for p,v,t in results if lo <= p <= hi]
        if subset:
            bp, bv, bt_ = subset[0]
            ret = (bv/INITIAL-1)*100
            vs = (bv-bh)/bh*100
            print(f"      {label:>6} ({lo:>3}-{hi:>3}日): {bp}日 ${bv:,.0f} ({ret:+.1f}%) vs买入持有 {vs:+.1f}%")

print(f"\n{'='*80}")
print("✅ 完成!")
