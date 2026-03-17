"""
PFE 辉瑞 — 多策略回测对比
端口: 5006
"""
import os
import json
import math
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template, request

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"),
            static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "static"))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
START_DATE = pd.Timestamp("2012-05-18")
END_DATE = pd.Timestamp("2025-12-01")
INITIAL_CAPITAL = 10000.0

# ============================================================
# 数据加载
# ============================================================

def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "PFE.csv"), index_col=0, parse_dates=True)
    df = df[(df.index >= START_DATE) & (df.index <= END_DATE)].copy()
    
    div_path = os.path.join(DATA_DIR, "PFE_dividends.csv")
    divs = pd.Series(dtype=float)
    if os.path.exists(div_path):
        ddf = pd.read_csv(div_path, index_col=0, parse_dates=True)
        if isinstance(ddf, pd.DataFrame) and len(ddf.columns) > 0:
            divs = ddf.iloc[:, 0]
        if not isinstance(divs.index, pd.DatetimeIndex):
            divs.index = pd.to_datetime(divs.index, utc=True)
        if divs.index.tz is not None:
            divs.index = divs.index.tz_localize(None)
        divs = divs[(divs.index >= START_DATE) & (divs.index <= END_DATE)]
    
    return df, divs

# ============================================================
# 辅助函数
# ============================================================

def _divs_dict(divs):
    return dict(divs.items()) if len(divs) > 0 else {}

def apply_dividends(shares, date, price, dd):
    if date in dd:
        shares += shares * dd[date] / price
    return shares

def _run(prices, dd, signals_fn):
    """通用回测引擎。signals_fn(i) 返回 'BUY'/'SELL'/None"""
    cash = INITIAL_CAPITAL
    shares = 0.0
    in_pos = False
    portfolio = []
    trades = []
    
    for i, (date, price) in enumerate(prices.items()):
        if in_pos:
            shares = apply_dividends(shares, date, price, dd)
        
        sig = signals_fn(i)
        if sig == "BUY" and not in_pos:
            shares = cash / price
            cash = 0
            in_pos = True
            trades.append({"date": date, "action": "BUY", "price": price, "shares": shares})
        elif sig == "SELL" and in_pos:
            cash = shares * price
            trades.append({"date": date, "action": "SELL", "price": price, "shares": shares})
            shares = 0
            in_pos = False
        
        portfolio.append({"date": date, "value": cash + shares * price})
    
    return pd.DataFrame(portfolio).set_index("date"), trades

# ============================================================
# 策略函数
# ============================================================

def strategy_buy_hold(df, divs):
    """买入持有 + DRIP"""
    prices = df["Close"].dropna()
    dd = _divs_dict(divs)
    shares = INITIAL_CAPITAL / prices.iloc[0]
    portfolio = []
    trades = [{"date": prices.index[0], "action": "BUY", "price": prices.iloc[0], "shares": shares}]
    for date, price in prices.items():
        shares = apply_dividends(shares, date, price, dd)
        portfolio.append({"date": date, "value": shares * price})
    return pd.DataFrame(portfolio).set_index("date"), trades

def strategy_macd(df, divs, fast=12, slow=26, signal=9):
    """MACD 金叉死叉"""
    prices = df["Close"].dropna()
    dd = _divs_dict(divs)
    ef = prices.ewm(span=fast, adjust=False).mean()
    es = prices.ewm(span=slow, adjust=False).mean()
    macd = ef - es
    sl = macd.ewm(span=signal, adjust=False).mean()
    def sig(i):
        if i < slow: return None
        if macd.iloc[i] > sl.iloc[i] and macd.iloc[i-1] <= sl.iloc[i-1]: return "BUY"
        if macd.iloc[i] < sl.iloc[i] and macd.iloc[i-1] >= sl.iloc[i-1]: return "SELL"
        return None
    return _run(prices, dd, sig)

def strategy_rsi(df, divs, period=28, buy_t=42, sell_t=69):
    """RSI 策略"""
    prices = df["Close"].dropna()
    dd = _divs_dict(divs)
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    def sig(i):
        if i <= period: return None
        if rsi.iloc[i-1] < buy_t and rsi.iloc[i] >= buy_t: return "BUY"
        if rsi.iloc[i-1] < sell_t and rsi.iloc[i] >= sell_t: return "SELL"
        return None
    return _run(prices, dd, sig)

def strategy_bollinger(df, divs, period=36, num_std=1.6):
    """布林带 (跌破下轨买，突破上轨卖)"""
    prices = df["Close"].dropna()
    dd = _divs_dict(divs)
    ma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    def sig(i):
        if i < period: return None
        lo = ma.iloc[i] - num_std * std.iloc[i]
        hi = ma.iloc[i] + num_std * std.iloc[i]
        lo_p = ma.iloc[i-1] - num_std * std.iloc[i-1]
        hi_p = ma.iloc[i-1] + num_std * std.iloc[i-1]
        if prices.iloc[i] < lo and prices.iloc[i-1] >= lo_p: return "BUY"
        if prices.iloc[i] > hi and prices.iloc[i-1] <= hi_p: return "SELL"
        return None
    return _run(prices, dd, sig)

def strategy_ma_crossover(df, divs, short_w=20, long_w=60):
    """均线交叉"""
    prices = df["Close"].dropna()
    dd = _divs_dict(divs)
    ms = prices.rolling(short_w).mean()
    ml = prices.rolling(long_w).mean()
    def sig(i):
        if i < long_w: return None
        if ms.iloc[i] > ml.iloc[i] and ms.iloc[i-1] <= ml.iloc[i-1]: return "BUY"
        if ms.iloc[i] < ml.iloc[i] and ms.iloc[i-1] >= ml.iloc[i-1]: return "SELL"
        return None
    return _run(prices, dd, sig)

def strategy_ma_single(df, divs, period=50):
    """单均线突破"""
    prices = df["Close"].dropna()
    dd = _divs_dict(divs)
    ma = prices.rolling(period).mean()
    def sig(i):
        if i < period: return None
        if prices.iloc[i] > ma.iloc[i] and prices.iloc[i-1] <= ma.iloc[i-1]: return "BUY"
        if prices.iloc[i] < ma.iloc[i] and prices.iloc[i-1] >= ma.iloc[i-1]: return "SELL"
        return None
    return _run(prices, dd, sig)

def strategy_ma_turn(df, divs, period=6):
    """均线拐头 (均线方向向上买，向下卖)"""
    prices = df["Close"].dropna()
    dd = _divs_dict(divs)
    ma = prices.rolling(period).mean()
    def sig(i):
        if i <= period: return None
        if ma.iloc[i] > ma.iloc[i-1] and ma.iloc[i-1] <= ma.iloc[i-2]: return "BUY"
        if ma.iloc[i] < ma.iloc[i-1] and ma.iloc[i-1] >= ma.iloc[i-2]: return "SELL"
        return None
    return _run(prices, dd, sig)

def strategy_cci(df, divs, period=20, buy_level=100, sell_level=-100):
    """CCI 策略 (上穿+100买，下穿-100卖)"""
    prices = df["Close"].dropna()
    high = df["High"].reindex(prices.index).fillna(prices)
    low = df["Low"].reindex(prices.index).fillna(prices)
    dd = _divs_dict(divs)
    
    tp = (high + low + prices) / 3
    ma_tp = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - ma_tp) / (0.015 * md)
    
    def sig(i):
        if i < period + 1: return None
        if cci.iloc[i] > buy_level and cci.iloc[i-1] <= buy_level: return "BUY"
        if cci.iloc[i] < sell_level and cci.iloc[i-1] >= sell_level: return "SELL"
        return None
    return _run(prices, dd, sig)

def strategy_kdj(df, divs, k_period=9, d_period=3):
    """KDJ 金叉死叉"""
    prices = df["Close"].dropna()
    high = df["High"].reindex(prices.index).fillna(prices)
    low = df["Low"].reindex(prices.index).fillna(prices)
    dd = _divs_dict(divs)
    
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    rsv = (prices - lowest) / (highest - lowest) * 100
    rsv = rsv.fillna(50)
    
    k_vals = [50.0]
    d_vals = [50.0]
    for i in range(1, len(rsv)):
        k = 2/3 * k_vals[-1] + 1/3 * rsv.iloc[i]
        d = 2/3 * d_vals[-1] + 1/3 * k
        k_vals.append(k)
        d_vals.append(d)
    k_line = pd.Series(k_vals, index=prices.index)
    d_line = pd.Series(d_vals, index=prices.index)
    
    def sig(i):
        if i < k_period + d_period: return None
        if k_line.iloc[i] > d_line.iloc[i] and k_line.iloc[i-1] <= d_line.iloc[i-1]: return "BUY"
        if k_line.iloc[i] < d_line.iloc[i] and k_line.iloc[i-1] >= d_line.iloc[i-1]: return "SELL"
        return None
    return _run(prices, dd, sig)

def strategy_donchian(df, divs, buy_n=20, sell_m=10):
    """唐奇安突破 (突破N日高点买，跌破M日低点卖)"""
    prices = df["Close"].dropna()
    high = df["High"].reindex(prices.index).fillna(prices)
    low = df["Low"].reindex(prices.index).fillna(prices)
    dd = _divs_dict(divs)
    
    upper = high.rolling(buy_n).max().shift(1)
    lower = low.rolling(sell_m).min().shift(1)
    
    def sig(i):
        if i < max(buy_n, sell_m) + 1: return None
        if pd.isna(upper.iloc[i]) or pd.isna(lower.iloc[i]): return None
        if prices.iloc[i] > upper.iloc[i]: return "BUY"
        if prices.iloc[i] < lower.iloc[i]: return "SELL"
        return None
    return _run(prices, dd, sig)

def strategy_turtle(df, divs):
    """海龟交易法则 (突破20日高点买，跌破10日低点卖)"""
    return strategy_donchian(df, divs, buy_n=20, sell_m=10)

def strategy_bb_expand(df, divs, period=20, num_std=2.0):
    """布林带收窄放大突破 (带宽从收窄转放大 + 价格突破上轨买，跌破中轨卖)"""
    prices = df["Close"].dropna()
    dd = _divs_dict(divs)
    
    ma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    bandwidth = (2 * num_std * std) / ma * 100  # 带宽百分比
    bw_ma = bandwidth.rolling(10).mean()  # 带宽的均线
    upper = ma + num_std * std
    
    def sig(i):
        if i < period + 11: return None
        # 带宽开始放大 (当前带宽 > 带宽均线) 且 价格突破上轨
        expanding = bandwidth.iloc[i] > bw_ma.iloc[i] and bandwidth.iloc[i-1] <= bw_ma.iloc[i-1]
        above_upper = prices.iloc[i] > upper.iloc[i]
        if expanding and above_upper: return "BUY"
        # 跌破中轨卖出
        if prices.iloc[i] < ma.iloc[i] and prices.iloc[i-1] >= ma.iloc[i-1]: return "SELL"
        return None
    return _run(prices, dd, sig)

def strategy_atr_trailing(df, divs, entry_period=20, atr_period=14, stop_mult=2.0, trail_mult=3.0):
    """ATR 波动率止损 + 跟踪止盈 (突破N日高点买，ATR止损/跟踪止盈卖)"""
    prices = df["Close"].dropna()
    high = df["High"].reindex(prices.index).fillna(prices)
    low = df["Low"].reindex(prices.index).fillna(prices)
    dd = _divs_dict(divs)
    
    # ATR
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - prices.shift(1)).abs(),
        'lc': (low - prices.shift(1)).abs()
    }).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    upper = high.rolling(entry_period).max().shift(1)
    
    cash = INITIAL_CAPITAL
    shares = 0.0
    in_pos = False
    portfolio = []
    trades = []
    entry_price = 0
    highest_since_entry = 0
    
    for i, (date, price) in enumerate(prices.items()):
        if in_pos:
            shares = apply_dividends(shares, date, price, dd)
            highest_since_entry = max(highest_since_entry, price)
        
        if i < max(entry_period, atr_period) + 1 or pd.isna(atr.iloc[i]):
            portfolio.append({"date": date, "value": cash + shares * price})
            continue
        
        if not in_pos:
            if not pd.isna(upper.iloc[i]) and price > upper.iloc[i]:
                shares = cash / price
                cash = 0
                in_pos = True
                entry_price = price
                highest_since_entry = price
                trades.append({"date": date, "action": "BUY", "price": price, "shares": shares})
        else:
            stop_loss = entry_price - stop_mult * atr.iloc[i]
            trail_stop = highest_since_entry - trail_mult * atr.iloc[i]
            exit_level = max(stop_loss, trail_stop)
            if price < exit_level:
                cash = shares * price
                trades.append({"date": date, "action": "SELL", "price": price, "shares": shares})
                shares = 0
                in_pos = False
        
        portfolio.append({"date": date, "value": cash + shares * price})
    
    return pd.DataFrame(portfolio).set_index("date"), trades

def strategy_ma_alignment(df, divs, ma1=20, ma2=60, ma3=120):
    """均线多头排列 + 回踩买入 (MA20>MA60>MA120 + 回踩MA20买，跌破MA60卖)"""
    prices = df["Close"].dropna()
    dd = _divs_dict(divs)
    
    m1 = prices.rolling(ma1).mean()
    m2 = prices.rolling(ma2).mean()
    m3 = prices.rolling(ma3).mean()
    
    # 回踩检测: 价格曾经接近MA20 (在MA20的1%范围内) 然后反弹
    def sig(i):
        if i < ma3 + 5: return None
        aligned = m1.iloc[i] > m2.iloc[i] > m3.iloc[i]
        # 回踩: 前几天价格接近MA20，现在重新走强
        near_ma20 = False
        for j in range(max(0, i-5), i):
            if abs(prices.iloc[j] - m1.iloc[j]) / m1.iloc[j] < 0.015:
                near_ma20 = True
                break
        bouncing = prices.iloc[i] > m1.iloc[i] and prices.iloc[i-1] <= m1.iloc[i-1]
        if aligned and near_ma20 and bouncing: return "BUY"
        # 跌破 MA60 卖出
        if prices.iloc[i] < m2.iloc[i] and prices.iloc[i-1] >= m2.iloc[i-1]: return "SELL"
        return None
    return _run(prices, dd, sig)

def strategy_combo(df, divs, donchian_n=20, atr_period=14, atr_stop=2.0, atr_trail=3.0, vol_ma=20, exit_n=10):
    """组合策略: 60日线上方 + 突破20日高点 + 放量 + ATR止损/止盈"""
    prices = df["Close"].dropna()
    high = df["High"].reindex(prices.index).fillna(prices)
    low = df["Low"].reindex(prices.index).fillna(prices)
    volume = df["Volume"].reindex(prices.index).fillna(0) if "Volume" in df.columns else pd.Series(1, index=prices.index)
    dd = _divs_dict(divs)
    
    ma60 = prices.rolling(60).mean()
    upper = high.rolling(donchian_n).max().shift(1)
    lower = low.rolling(exit_n).min().shift(1)
    vol_avg = volume.rolling(vol_ma).mean()
    
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - prices.shift(1)).abs(),
        'lc': (low - prices.shift(1)).abs()
    }).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    
    cash = INITIAL_CAPITAL
    shares = 0.0
    in_pos = False
    portfolio = []
    trades = []
    entry_price = 0
    highest_since = 0
    
    warmup = max(60, donchian_n, atr_period, vol_ma, exit_n) + 1
    
    for i, (date, price) in enumerate(prices.items()):
        if in_pos:
            shares = apply_dividends(shares, date, price, dd)
            highest_since = max(highest_since, price)
        
        if i < warmup or pd.isna(atr.iloc[i]):
            portfolio.append({"date": date, "value": cash + shares * price})
            continue
        
        if not in_pos:
            above_ma60 = price > ma60.iloc[i]
            breakout = not pd.isna(upper.iloc[i]) and price > upper.iloc[i]
            vol_ok = vol_avg.iloc[i] > 0 and volume.iloc[i] > vol_avg.iloc[i] if volume.iloc[i] > 0 else True
            if above_ma60 and breakout and vol_ok:
                shares = cash / price
                cash = 0
                in_pos = True
                entry_price = price
                highest_since = price
                trades.append({"date": date, "action": "BUY", "price": price, "shares": shares})
        else:
            stop = entry_price - atr_stop * atr.iloc[i]
            trail = highest_since - atr_trail * atr.iloc[i]
            exit_low = lower.iloc[i] if not pd.isna(lower.iloc[i]) else 0
            exit_level = max(stop, trail, exit_low)
            if price < exit_level:
                cash = shares * price
                trades.append({"date": date, "action": "SELL", "price": price, "shares": shares})
                shares = 0
                in_pos = False
        
        portfolio.append({"date": date, "value": cash + shares * price})
    
    return pd.DataFrame(portfolio).set_index("date"), trades

# ============================================================
# 统计计算
# ============================================================

def calc_stats(portfolio_df, trades):
    values = portfolio_df["value"]
    start_val = values.iloc[0]
    end_val = values.iloc[-1]
    total_return = (end_val - start_val) / start_val * 100
    
    days = (values.index[-1] - values.index[0]).days
    years = days / 365.25
    cagr = ((end_val / start_val) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax
    max_dd = drawdown.min() * 100
    
    buy_count = sum(1 for t in trades if t["action"] == "BUY")
    sell_count = sum(1 for t in trades if t["action"] == "SELL")
    
    wins = 0
    total_trades = 0
    buy_price = None
    for t in trades:
        if t["action"] == "BUY":
            buy_price = t["price"]
        elif t["action"] == "SELL" and buy_price is not None:
            total_trades += 1
            if t["price"] > buy_price:
                wins += 1
            buy_price = None
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    daily_returns = values.pct_change().dropna()
    volatility = daily_returns.std() * (252 ** 0.5) * 100
    sharpe = ((cagr - 2) / volatility) if volatility > 0 else 0
    
    return {
        "final_value": round(end_val, 2),
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "total_round_trips": total_trades,
        "win_rate_pct": round(win_rate, 1),
        "volatility_pct": round(volatility, 2),
        "sharpe_ratio": round(sharpe, 3),
        "years": round(years, 1),
    }

# ============================================================
# 所有策略注册
# ============================================================

ALL_STRATEGIES = [
    {"id": "buy_hold", "name": "买入持有 Buy & Hold", "desc": "持有到底 + 分红再投资 (DRIP)", "fn": strategy_buy_hold, "color": "#888888"},
    # --- 最优参数 (暴力搜索) ---
    {"id": "rsi_best", "name": "🥇 RSI(28,42,69)", "desc": "RSI 28日 买<42 卖>69 | PFE冠军", "fn": lambda df, d: strategy_rsi(df, d, 28, 42, 69), "color": "#ff6b6b"},
    {"id": "bb_best", "name": "🥈 布林带(36,1.6σ)", "desc": "布林带 36日 1.6σ", "fn": lambda df, d: strategy_bollinger(df, d, 36, 1.6), "color": "#ffd93d"},
    {"id": "macd_best", "name": "🥉 MACD(5,48,14)", "desc": "MACD fast=5 slow=48 signal=14", "fn": lambda df, d: strategy_macd(df, d, 5, 48, 14), "color": "#6bcb77"},
    # --- 经典策略 ---
    {"id": "macd_default", "name": "MACD(12,26,9) 经典", "desc": "MACD 经典参数 金叉买/死叉卖", "fn": strategy_macd, "color": "#e17055"},
    {"id": "ma_turn_6", "name": "MA6 拐头 最优", "desc": "6日均线方向向上买，向下卖 | 拐头策略PFE最优", "fn": lambda df, d: strategy_ma_turn(df, d, 6), "color": "#00b894"},
    {"id": "ma_turn_20", "name": "MA20 拐头", "desc": "20日均线方向向上买，向下卖", "fn": lambda df, d: strategy_ma_turn(df, d, 20), "color": "#00cec9"},
    {"id": "cci", "name": "CCI(20) ±100", "desc": "CCI上穿+100买，下穿-100卖", "fn": strategy_cci, "color": "#fd79a8"},
    {"id": "kdj", "name": "KDJ(9,3) 金叉死叉", "desc": "K线上穿D线买，K线下穿D线卖", "fn": strategy_kdj, "color": "#e84393"},
    {"id": "donchian", "name": "唐奇安(20,10)", "desc": "突破20日高点买，跌破10日低点卖", "fn": strategy_donchian, "color": "#0984e3"},
    {"id": "turtle", "name": "海龟交易法则", "desc": "突破20日高点进场，跌破10日低点离场", "fn": strategy_turtle, "color": "#6c5ce7"},
    {"id": "bb_expand", "name": "布林带收窄放大", "desc": "带宽从收窄转放大+突破上轨买，跌回中轨卖", "fn": strategy_bb_expand, "color": "#fdcb6e"},
    {"id": "atr_trail", "name": "ATR止损跟踪止盈", "desc": "突破20日高点买，2ATR止损/3ATR跟踪止盈", "fn": strategy_atr_trailing, "color": "#55efc4"},
    {"id": "ma_align", "name": "均线多头回踩", "desc": "MA20>60>120多头排列+回踩MA20买，跌破MA60卖", "fn": strategy_ma_alignment, "color": "#74b9ff"},
    {"id": "combo", "name": "组合策略 终极版", "desc": "60日线上+突破20日高点+放量+ATR止损止盈", "fn": strategy_combo, "color": "#ff7675"},
    # --- 单均线对比 ---
    {"id": "ma10", "name": "MA10 突破", "desc": "价格突破10日均线买入，跌破卖出", "fn": lambda df, d: strategy_ma_single(df, d, 10), "color": "#b2bec3"},
    {"id": "ma_cross_7_14", "name": "MA Cross 7/14", "desc": "7日上穿14日均线买，下穿卖", "fn": lambda df, d: strategy_ma_crossover(df, d, 7, 14), "color": "#636e72"},
]

# ============================================================
# 图表
# ============================================================

def generate_equity_chart(results):
    fig = go.Figure()
    for r in results:
        vals = r["portfolio"]["value"]
        fig.add_trace(go.Scatter(
            x=r["portfolio"].index, y=vals,
            mode='lines', name=r["name"],
            line=dict(width=1.5, color=r["color"]),
            opacity=0.85,
            hovertemplate=f'{r["name"]}<br>%{{x}}<br>${{y:,.0f}}<extra></extra>'
        ))
    fig.update_layout(
        title=dict(
            text="📈 PFE 辉瑞 — 多策略权益曲线对比<br><sup style='color:#888'>2012-05-18 → 2025-12-01 | $10,000 | DRIP</sup>",
            font=dict(size=16)),
        xaxis_title="日期", yaxis_title="账户价值 ($)",
        template="plotly_dark", height=650,
        legend=dict(font=dict(size=10), orientation="h", y=-0.15),
        hovermode="x unified",
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def generate_drawdown_chart(results):
    fig = go.Figure()
    for r in results:
        vals = r["portfolio"]["value"]
        cm = vals.cummax()
        dd = (vals - cm) / cm * 100
        fig.add_trace(go.Scatter(
            x=r["portfolio"].index, y=dd,
            mode='lines', name=r["name"],
            line=dict(width=1, color=r["color"]),
            fill='tozeroy', opacity=0.4,
            hovertemplate=f'{r["name"]}<br>%{{x}}<br>%{{y:.1f}}%<extra></extra>'
        ))
    fig.update_layout(
        title=dict(text="📉 最大回撤对比<br><sup style='color:#888'>2012-05-18 → 2025-12-01</sup>", font=dict(size=16)),
        xaxis_title="日期", yaxis_title="回撤 (%)",
        template="plotly_dark", height=400,
        legend=dict(font=dict(size=10), orientation="h", y=-0.15),
        hovermode="x unified",
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def generate_bar_comparison(results):
    names = [r["name"] for r in results]
    colors = [r["color"] for r in results]
    fig = make_subplots(rows=1, cols=4,
                        subplot_titles=["总回报率(%)", "CAGR(%)", "最大回撤(%)", "Sharpe"],
                        horizontal_spacing=0.06)
    for vals, col in [
        ([r["stats"]["total_return_pct"] for r in results], 1),
        ([r["stats"]["cagr_pct"] for r in results], 2),
        ([r["stats"]["max_drawdown_pct"] for r in results], 3),
        ([r["stats"]["sharpe_ratio"] for r in results], 4),
    ]:
        fig.add_trace(go.Bar(x=names, y=vals, marker_color=colors,
                             text=[f'{v:.1f}' for v in vals], textposition='outside',
                             showlegend=False), row=1, col=col)
    fig.update_layout(
        title=dict(text="📊 策略指标对比<br><sup style='color:#888'>PFE 辉瑞 | $10,000</sup>", font=dict(size=16)),
        template="plotly_dark", height=450, showlegend=False,
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# ============================================================
# 路由
# ============================================================

@app.route("/")
def index():
    df, divs = load_data()
    
    results = []
    for s in ALL_STRATEGIES:
        try:
            portfolio, trades = s["fn"](df, divs)
            stats = calc_stats(portfolio, trades)
            results.append({
                "id": s["id"],
                "name": s["name"],
                "desc": s["desc"],
                "color": s["color"],
                "portfolio": portfolio,
                "trades": trades,
                "stats": stats,
            })
        except Exception as e:
            print(f"⚠️ Strategy {s['name']} failed: {e}")
            import traceback; traceback.print_exc()
            continue
    
    # 按最终价值排序
    results.sort(key=lambda r: r["stats"]["final_value"], reverse=True)
    
    equity_chart = generate_equity_chart(results)
    drawdown_chart = generate_drawdown_chart(results)
    bar_chart = generate_bar_comparison(results)
    
    best = results[0] if results else None
    worst = results[-1] if results else None
    
    return render_template("index.html",
                           results=results,
                           equity_chart=equity_chart,
                           drawdown_chart=drawdown_chart,
                           bar_chart=bar_chart,
                           best=best,
                           worst=worst,
                           strategy_count=len(results))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5006, debug=False)
