"""
中国石油 601857.SS — 暴力参数优化
"""
import pandas as pd
import numpy as np
import os
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
START_DATE = pd.Timestamp("2012-05-18")
END_DATE = pd.Timestamp("2025-12-01")
INITIAL = 10000.0

for ticker_file, ticker_name in [("601857_SS", "中国石油 A股"), ("0857_HK", "中国石油 H股")]:
    print(f"\n{'='*70}")
    print(f"🔍 {ticker_name}")
    print(f"{'='*70}")
    
    df = pd.read_csv(os.path.join(DATA_DIR, f"{ticker_file}.csv"), index_col=0, parse_dates=True)
    df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]
    prices = df["Close"].dropna()
    
    # 分红
    divs_dict = {}
    div_path = os.path.join(DATA_DIR, f"{ticker_file}_dividends.csv")
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
    
    prices_arr = prices.values
    dates = prices.index
    n = len(prices_arr)
    
    print(f"数据: {n} 天, {len(divs_dict)} 次分红")
    print(f"价格: {prices_arr[0]:.2f} → {prices_arr[-1]:.2f}")
    
    # 买入持有
    shares_bh = INITIAL / prices_arr[0]
    for d in dates:
        if d in divs_dict:
            shares_bh += shares_bh * divs_dict[d] / prices.loc[d]
    bh_final = shares_bh * prices_arr[-1]
    print(f"基准(买入持有+DRIP): ${bh_final:,.0f} ({(bh_final/INITIAL-1)*100:+.1f}%)")
    
    def bt_macd(fast, slow, sig):
        ps = pd.Series(prices_arr)
        ef = ps.ewm(span=fast, adjust=False).mean().values
        es = ps.ewm(span=slow, adjust=False).mean().values
        macd = ef - es
        sl = pd.Series(macd).ewm(span=sig, adjust=False).mean().values
        cash, sh, inp = INITIAL, 0.0, False
        for i in range(slow, n):
            if inp and dates[i] in divs_dict:
                sh += sh * divs_dict[dates[i]] / prices_arr[i]
            if not inp and macd[i] > sl[i] and macd[i-1] <= sl[i-1]:
                sh = cash / prices_arr[i]; cash = 0; inp = True
            elif inp and macd[i] < sl[i] and macd[i-1] >= sl[i-1]:
                cash = sh * prices_arr[i]; sh = 0; inp = False
        return cash + sh * prices_arr[-1]
    
    def bt_ma_cross(sw, lw):
        mas = pd.Series(prices_arr).rolling(sw).mean().values
        mal = pd.Series(prices_arr).rolling(lw).mean().values
        cash, sh, inp = INITIAL, 0.0, False
        for i in range(lw, n):
            if inp and dates[i] in divs_dict:
                sh += sh * divs_dict[dates[i]] / prices_arr[i]
            if not inp and mas[i] > mal[i] and mas[i-1] <= mal[i-1]:
                sh = cash / prices_arr[i]; cash = 0; inp = True
            elif inp and mas[i] < mal[i] and mas[i-1] >= mal[i-1]:
                cash = sh * prices_arr[i]; sh = 0; inp = False
        return cash + sh * prices_arr[-1]
    
    def bt_ma_single(w):
        ma = pd.Series(prices_arr).rolling(w).mean().values
        cash, sh, inp = INITIAL, 0.0, False
        for i in range(w, n):
            if inp and dates[i] in divs_dict:
                sh += sh * divs_dict[dates[i]] / prices_arr[i]
            if not inp and prices_arr[i] > ma[i] and prices_arr[i-1] <= ma[i-1]:
                sh = cash / prices_arr[i]; cash = 0; inp = True
            elif inp and prices_arr[i] < ma[i] and prices_arr[i-1] >= ma[i-1]:
                cash = sh * prices_arr[i]; sh = 0; inp = False
        return cash + sh * prices_arr[-1]
    
    def bt_rsi(period, buy_t, sell_t):
        ps = pd.Series(prices_arr)
        delta = ps.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss
        rsi = (100 - 100 / (1 + rs)).values
        cash, sh, inp = INITIAL, 0.0, False
        for i in range(period+1, n):
            if inp and dates[i] in divs_dict:
                sh += sh * divs_dict[dates[i]] / prices_arr[i]
            if not inp and rsi[i-1] < buy_t and rsi[i] >= buy_t:
                sh = cash / prices_arr[i]; cash = 0; inp = True
            elif inp and rsi[i-1] < sell_t and rsi[i] >= sell_t:
                cash = sh * prices_arr[i]; sh = 0; inp = False
        return cash + sh * prices_arr[-1]
    
    def bt_bb(period, num_std):
        ps = pd.Series(prices_arr)
        ma = ps.rolling(period).mean().values
        std = ps.rolling(period).std().values
        cash, sh, inp = INITIAL, 0.0, False
        for i in range(period, n):
            if inp and dates[i] in divs_dict:
                sh += sh * divs_dict[dates[i]] / prices_arr[i]
            lo = ma[i] - num_std * std[i]
            hi = ma[i] + num_std * std[i]
            lo_p = ma[i-1] - num_std * std[i-1]
            hi_p = ma[i-1] + num_std * std[i-1]
            if not inp and prices_arr[i] < lo and prices_arr[i-1] >= lo_p:
                sh = cash / prices_arr[i]; cash = 0; inp = True
            elif inp and prices_arr[i] > hi and prices_arr[i-1] <= hi_p:
                cash = sh * prices_arr[i]; sh = 0; inp = False
        return cash + sh * prices_arr[-1]
    
    t0 = time.time()
    best_all = []
    
    # MACD
    print("\n  MACD...", end=" ", flush=True)
    best = (0, "")
    for f in range(5, 25):
        for s in range(15, 50):
            if f >= s: continue
            for sg in range(3, 20):
                v = bt_macd(f, s, sg)
                if v > best[0]: best = (v, f"MACD({f},{s},{sg})")
    print(f"${best[0]:,.0f} | {best[1]}")
    best_all.append(("MACD", best[1], best[0]))
    
    # MA Cross
    print("  MA Cross...", end=" ", flush=True)
    best = (0, "")
    for sw in range(3, 51):
        for lw in range(10, 201, 2):
            if sw >= lw: continue
            v = bt_ma_cross(sw, lw)
            if v > best[0]: best = (v, f"MA Cross {sw}/{lw}")
    print(f"${best[0]:,.0f} | {best[1]}")
    best_all.append(("MA Cross", best[1], best[0]))
    
    # MA Single
    print("  MA Single...", end=" ", flush=True)
    best = (0, "")
    for w in range(3, 301):
        v = bt_ma_single(w)
        if v > best[0]: best = (v, f"MA Single {w}")
    print(f"${best[0]:,.0f} | {best[1]}")
    best_all.append(("MA Single", best[1], best[0]))
    
    # RSI
    print("  RSI...", end=" ", flush=True)
    best = (0, "")
    for p in range(5, 30):
        for bt in range(15, 45):
            for st in range(55, 85):
                v = bt_rsi(p, bt, st)
                if v > best[0]: best = (v, f"RSI({p}, buy<{bt}, sell>{st})")
    print(f"${best[0]:,.0f} | {best[1]}")
    best_all.append(("RSI", best[1], best[0]))
    
    # Bollinger
    print("  Bollinger...", end=" ", flush=True)
    best = (0, "")
    for p in range(10, 50):
        for sm in [x/10 for x in range(10, 35)]:
            v = bt_bb(p, sm)
            if v > best[0]: best = (v, f"Bollinger({p}, {sm:.1f}σ)")
    print(f"${best[0]:,.0f} | {best[1]}")
    best_all.append(("Bollinger", best[1], best[0]))
    
    elapsed = time.time() - t0
    best_all.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n  ⏱ {elapsed:.0f}s | 🏆 冠军: {best_all[0][1]} → ${best_all[0][2]:,.0f} ({(best_all[0][2]/INITIAL-1)*100:+.1f}%)")
    for i, (tp, nm, v) in enumerate(best_all):
        r = (v/INITIAL-1)*100
        medal = "🥇🥈🥉"[i] if i < 3 else "  "
        print(f"  {medal} {tp:<12} {nm:<35} ${v:>11,.0f} {r:>+9.1f}%")
