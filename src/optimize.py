"""
PFE 辉瑞 — 暴力参数优化，找最大收益率策略
"""
import pandas as pd
import numpy as np
import os
import time
from itertools import product

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
START_DATE = pd.Timestamp("2012-05-18")
END_DATE = pd.Timestamp("2025-12-01")
INITIAL = 10000.0

# 加载数据
df = pd.read_csv(os.path.join(DATA_DIR, "PFE.csv"), index_col=0, parse_dates=True)
df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]
prices = df["Close"].dropna()

# 加载分红
div_path = os.path.join(DATA_DIR, "PFE_dividends.csv")
divs_dict = {}
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

print(f"PFE 数据: {len(prices)} 天, {len(divs_dict)} 次分红")
print(f"回测区间: {prices.index[0].date()} → {prices.index[-1].date()}")
print(f"价格范围: ${prices.min():.2f} ~ ${prices.max():.2f}")
print("=" * 70)

# ============================================================
# 通用回测引擎（高性能版）
# ============================================================

def backtest_ma_cross(prices_arr, dates, short_w, long_w):
    """均线交叉：短均线上穿长均线买，下穿卖"""
    n = len(prices_arr)
    if long_w >= n:
        return INITIAL
    
    ma_short = pd.Series(prices_arr).rolling(short_w).mean().values
    ma_long = pd.Series(prices_arr).rolling(long_w).mean().values
    
    cash = INITIAL
    shares = 0.0
    in_pos = False
    
    for i in range(long_w, n):
        # DRIP
        if in_pos and dates[i] in divs_dict:
            div = shares * divs_dict[dates[i]]
            shares += div / prices_arr[i]
        
        if not in_pos and ma_short[i] > ma_long[i] and ma_short[i-1] <= ma_long[i-1]:
            shares = cash / prices_arr[i]
            cash = 0
            in_pos = True
        elif in_pos and ma_short[i] < ma_long[i] and ma_short[i-1] >= ma_long[i-1]:
            cash = shares * prices_arr[i]
            shares = 0
            in_pos = False
    
    return cash + shares * prices_arr[-1]

def backtest_ma_single(prices_arr, dates, ma_w):
    """单均线突破：价格上穿均线买，下穿卖"""
    n = len(prices_arr)
    if ma_w >= n:
        return INITIAL
    
    ma = pd.Series(prices_arr).rolling(ma_w).mean().values
    
    cash = INITIAL
    shares = 0.0
    in_pos = False
    
    for i in range(ma_w, n):
        if in_pos and dates[i] in divs_dict:
            div = shares * divs_dict[dates[i]]
            shares += div / prices_arr[i]
        
        if not in_pos and prices_arr[i] > ma[i] and prices_arr[i-1] <= ma[i-1]:
            shares = cash / prices_arr[i]
            cash = 0
            in_pos = True
        elif in_pos and prices_arr[i] < ma[i] and prices_arr[i-1] >= ma[i-1]:
            cash = shares * prices_arr[i]
            shares = 0
            in_pos = False
    
    return cash + shares * prices_arr[-1]

def backtest_macd(prices_arr, dates, fast, slow, signal):
    """MACD 策略"""
    n = len(prices_arr)
    if slow >= n:
        return INITIAL
    
    ps = pd.Series(prices_arr)
    ema_fast = ps.ewm(span=fast, adjust=False).mean().values
    ema_slow = ps.ewm(span=slow, adjust=False).mean().values
    macd = ema_fast - ema_slow
    sig = pd.Series(macd).ewm(span=signal, adjust=False).mean().values
    
    cash = INITIAL
    shares = 0.0
    in_pos = False
    
    for i in range(slow, n):
        if in_pos and dates[i] in divs_dict:
            div = shares * divs_dict[dates[i]]
            shares += div / prices_arr[i]
        
        if not in_pos and macd[i] > sig[i] and macd[i-1] <= sig[i-1]:
            shares = cash / prices_arr[i]
            cash = 0
            in_pos = True
        elif in_pos and macd[i] < sig[i] and macd[i-1] >= sig[i-1]:
            cash = shares * prices_arr[i]
            shares = 0
            in_pos = False
    
    return cash + shares * prices_arr[-1]

def backtest_rsi(prices_arr, dates, period, buy_thresh, sell_thresh):
    """RSI 策略：超卖买入，超买卖出"""
    n = len(prices_arr)
    if period >= n:
        return INITIAL
    
    ps = pd.Series(prices_arr)
    delta = ps.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    rsi = (100 - 100 / (1 + rs)).values
    
    cash = INITIAL
    shares = 0.0
    in_pos = False
    
    for i in range(period + 1, n):
        if in_pos and dates[i] in divs_dict:
            div = shares * divs_dict[dates[i]]
            shares += div / prices_arr[i]
        
        if not in_pos and rsi[i-1] < buy_thresh and rsi[i] >= buy_thresh:
            shares = cash / prices_arr[i]
            cash = 0
            in_pos = True
        elif in_pos and rsi[i-1] < sell_thresh and rsi[i] >= sell_thresh:
            cash = shares * prices_arr[i]
            shares = 0
            in_pos = False
    
    return cash + shares * prices_arr[-1]

def backtest_bollinger(prices_arr, dates, period, num_std):
    """布林带策略：跌破下轨买，突破上轨卖"""
    n = len(prices_arr)
    if period >= n:
        return INITIAL
    
    ps = pd.Series(prices_arr)
    ma = ps.rolling(period).mean().values
    std = ps.rolling(period).std().values
    
    cash = INITIAL
    shares = 0.0
    in_pos = False
    
    for i in range(period, n):
        if in_pos and dates[i] in divs_dict:
            div = shares * divs_dict[dates[i]]
            shares += div / prices_arr[i]
        
        lower = ma[i] - num_std * std[i]
        upper = ma[i] + num_std * std[i]
        
        if not in_pos and prices_arr[i] < lower and prices_arr[i-1] >= (ma[i-1] - num_std * std[i-1]):
            shares = cash / prices_arr[i]
            cash = 0
            in_pos = True
        elif in_pos and prices_arr[i] > upper and prices_arr[i-1] <= (ma[i-1] + num_std * std[i-1]):
            cash = shares * prices_arr[i]
            shares = 0
            in_pos = False
    
    return cash + shares * prices_arr[-1]

# ============================================================
# 开始优化
# ============================================================

prices_arr = prices.values
dates = prices.index

# 基准：买入持有
shares_bh = INITIAL / prices_arr[0]
for d in dates:
    if d in divs_dict:
        shares_bh += shares_bh * divs_dict[d] / prices.loc[d]
buy_hold_final = shares_bh * prices_arr[-1]
print(f"📊 基准 (买入持有+DRIP): ${buy_hold_final:,.0f} ({(buy_hold_final/INITIAL - 1)*100:+.1f}%)")
print("=" * 70)

best_results = []
total_combos = 0
t0 = time.time()

# --- 1. 均线交叉 ---
print("\n🔍 搜索均线交叉 (MA Cross)...")
short_range = range(3, 51, 1)
long_range = range(10, 201, 2)
count = 0
best_mac = (0, "", "")
for s in short_range:
    for l in long_range:
        if s >= l:
            continue
        val = backtest_ma_cross(prices_arr, dates, s, l)
        count += 1
        if val > best_mac[0]:
            best_mac = (val, f"MA Cross {s}/{l}", f"short={s}, long={l}")

total_combos += count
ret = (best_mac[0]/INITIAL - 1)*100
print(f"  ✅ {count} 组合 | 最佳: {best_mac[1]} → ${best_mac[0]:,.0f} ({ret:+.1f}%)")
best_results.append(("MA Cross", best_mac[1], best_mac[2], best_mac[0], ret))

# --- 2. 单均线突破 ---
print("\n🔍 搜索单均线突破 (MA Single)...")
count = 0
best_mas = (0, "", "")
for w in range(3, 301):
    val = backtest_ma_single(prices_arr, dates, w)
    count += 1
    if val > best_mas[0]:
        best_mas = (val, f"MA Single {w}", f"period={w}")

total_combos += count
ret = (best_mas[0]/INITIAL - 1)*100
print(f"  ✅ {count} 组合 | 最佳: {best_mas[1]} → ${best_mas[0]:,.0f} ({ret:+.1f}%)")
best_results.append(("MA Single", best_mas[1], best_mas[2], best_mas[0], ret))

# --- 3. MACD ---
print("\n🔍 搜索 MACD 参数...")
count = 0
best_macd = (0, "", "")
for fast in range(5, 25, 1):
    for slow in range(15, 50, 1):
        if fast >= slow:
            continue
        for sig in range(3, 20, 1):
            val = backtest_macd(prices_arr, dates, fast, slow, sig)
            count += 1
            if val > best_macd[0]:
                best_macd = (val, f"MACD({fast},{slow},{sig})", f"fast={fast}, slow={slow}, signal={sig}")

total_combos += count
ret = (best_macd[0]/INITIAL - 1)*100
print(f"  ✅ {count} 组合 | 最佳: {best_macd[1]} → ${best_macd[0]:,.0f} ({ret:+.1f}%)")
best_results.append(("MACD", best_macd[1], best_macd[2], best_macd[0], ret))

# --- 4. RSI ---
print("\n🔍 搜索 RSI 参数...")
count = 0
best_rsi = (0, "", "")
for period in range(5, 30):
    for buy_t in range(15, 45, 1):
        for sell_t in range(55, 85, 1):
            val = backtest_rsi(prices_arr, dates, period, buy_t, sell_t)
            count += 1
            if val > best_rsi[0]:
                best_rsi = (val, f"RSI({period}, buy<{buy_t}, sell>{sell_t})", f"period={period}, buy={buy_t}, sell={sell_t}")

total_combos += count
ret = (best_rsi[0]/INITIAL - 1)*100
print(f"  ✅ {count} 组合 | 最佳: {best_rsi[1]} → ${best_rsi[0]:,.0f} ({ret:+.1f}%)")
best_results.append(("RSI", best_rsi[1], best_rsi[2], best_rsi[0], ret))

# --- 5. 布林带 ---
print("\n🔍 搜索布林带参数...")
count = 0
best_bb = (0, "", "")
for period in range(10, 50, 1):
    for std_mult in [x/10 for x in range(10, 35, 1)]:
        val = backtest_bollinger(prices_arr, dates, period, std_mult)
        count += 1
        if val > best_bb[0]:
            best_bb = (val, f"Bollinger({period}, {std_mult:.1f}σ)", f"period={period}, std={std_mult}")

total_combos += count
ret = (best_bb[0]/INITIAL - 1)*100
print(f"  ✅ {count} 组合 | 最佳: {best_bb[1]} → ${best_bb[0]:,.0f} ({ret:+.1f}%)")
best_results.append(("Bollinger", best_bb[1], best_bb[2], best_bb[0], ret))

# ============================================================
# 汇总
# ============================================================

elapsed = time.time() - t0
print("\n" + "=" * 70)
print(f"🏁 搜索完成！共 {total_combos:,} 种参数组合，耗时 {elapsed:.1f} 秒")
print("=" * 70)

print(f"\n📊 基准 (买入持有+DRIP): ${buy_hold_final:,.0f} ({(buy_hold_final/INITIAL-1)*100:+.1f}%)")
print(f"\n{'排名':<4} {'策略类型':<12} {'最佳参数':<35} {'最终价值':>12} {'回报率':>10}")
print("-" * 80)

best_results.sort(key=lambda x: x[3], reverse=True)
for i, (stype, name, params, val, ret) in enumerate(best_results, 1):
    marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    print(f"{marker} {i}  {stype:<12} {name:<35} ${val:>11,.0f} {ret:>+9.1f}%")

print(f"\n🏆 冠军策略: {best_results[0][1]}")
print(f"   参数: {best_results[0][2]}")
print(f"   $10,000 → ${best_results[0][3]:,.0f} ({best_results[0][4]:+.1f}%)")
vs_bh = (best_results[0][3] - buy_hold_final) / buy_hold_final * 100
print(f"   对比买入持有: {vs_bh:+.1f}%")
