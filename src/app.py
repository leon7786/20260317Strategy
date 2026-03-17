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
from flask import Flask, render_template, jsonify

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
    
    # 加载分红
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
# 策略引擎
# ============================================================

def apply_dividends(shares, date, price, divs_dict):
    """分红再投资"""
    if date in divs_dict:
        div_income = shares * divs_dict[date]
        shares += div_income / price
    return shares

def strategy_buy_hold(df, divs):
    """策略1: 买入持有 + DRIP"""
    prices = df["Close"].dropna()
    divs_dict = dict(divs.items()) if len(divs) > 0 else {}
    
    shares = INITIAL_CAPITAL / prices.iloc[0]
    portfolio = []
    trades = []
    trades.append({"date": prices.index[0], "action": "BUY", "price": prices.iloc[0], "shares": shares})
    
    for date, price in prices.items():
        shares = apply_dividends(shares, date, price, divs_dict)
        portfolio.append({"date": date, "value": shares * price})
    
    return pd.DataFrame(portfolio).set_index("date"), trades

def strategy_macd(df, divs, fast=12, slow=26, signal=9):
    """策略2: MACD 交叉策略"""
    prices = df["Close"].dropna()
    divs_dict = dict(divs.items()) if len(divs) > 0 else {}
    
    # 计算 MACD
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    cash = INITIAL_CAPITAL
    shares = 0.0
    in_position = False
    portfolio = []
    trades = []
    
    for i, (date, price) in enumerate(prices.items()):
        # 分红再投资
        if in_position:
            shares = apply_dividends(shares, date, price, divs_dict)
        
        if i >= slow:  # MACD 需要足够数据
            if not in_position and macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i-1] <= signal_line.iloc[i-1]:
                # 金叉买入
                shares = cash / price
                cash = 0
                in_position = True
                trades.append({"date": date, "action": "BUY", "price": price, "shares": shares})
            elif in_position and macd_line.iloc[i] < signal_line.iloc[i] and macd_line.iloc[i-1] >= signal_line.iloc[i-1]:
                # 死叉卖出
                cash = shares * price
                trades.append({"date": date, "action": "SELL", "price": price, "shares": shares})
                shares = 0
                in_position = False
        
        value = cash + shares * price
        portfolio.append({"date": date, "value": value})
    
    return pd.DataFrame(portfolio).set_index("date"), trades

def strategy_ma_crossover(df, divs, short_ma=20, long_ma=60):
    """策略3: 均线突破策略 (短均线上穿长均线买入，下穿卖出)"""
    prices = df["Close"].dropna()
    divs_dict = dict(divs.items()) if len(divs) > 0 else {}
    
    ma_short = prices.rolling(window=short_ma).mean()
    ma_long = prices.rolling(window=long_ma).mean()
    
    cash = INITIAL_CAPITAL
    shares = 0.0
    in_position = False
    portfolio = []
    trades = []
    
    for i, (date, price) in enumerate(prices.items()):
        if in_position:
            shares = apply_dividends(shares, date, price, divs_dict)
        
        if i >= long_ma:
            if not in_position and ma_short.iloc[i] > ma_long.iloc[i] and ma_short.iloc[i-1] <= ma_long.iloc[i-1]:
                # 短均线上穿长均线 → 买入
                shares = cash / price
                cash = 0
                in_position = True
                trades.append({"date": date, "action": "BUY", "price": price, "shares": shares})
            elif in_position and ma_short.iloc[i] < ma_long.iloc[i] and ma_short.iloc[i-1] >= ma_long.iloc[i-1]:
                # 短均线下穿长均线 → 卖出
                cash = shares * price
                trades.append({"date": date, "action": "SELL", "price": price, "shares": shares})
                shares = 0
                in_position = False
        
        value = cash + shares * price
        portfolio.append({"date": date, "value": value})
    
    return pd.DataFrame(portfolio).set_index("date"), trades

def strategy_ma_single(df, divs, ma_period=50):
    """策略4: 单均线策略 (价格突破N日均线买入，跌破卖出)"""
    prices = df["Close"].dropna()
    divs_dict = dict(divs.items()) if len(divs) > 0 else {}
    
    ma = prices.rolling(window=ma_period).mean()
    
    cash = INITIAL_CAPITAL
    shares = 0.0
    in_position = False
    portfolio = []
    trades = []
    
    for i, (date, price) in enumerate(prices.items()):
        if in_position:
            shares = apply_dividends(shares, date, price, divs_dict)
        
        if i >= ma_period:
            if not in_position and price > ma.iloc[i] and prices.iloc[i-1] <= ma.iloc[i-1]:
                shares = cash / price
                cash = 0
                in_position = True
                trades.append({"date": date, "action": "BUY", "price": price, "shares": shares})
            elif in_position and price < ma.iloc[i] and prices.iloc[i-1] >= ma.iloc[i-1]:
                cash = shares * price
                trades.append({"date": date, "action": "SELL", "price": price, "shares": shares})
                shares = 0
                in_position = False
        
        value = cash + shares * price
        portfolio.append({"date": date, "value": value})
    
    return pd.DataFrame(portfolio).set_index("date"), trades

def strategy_ma_single_200(df, divs):
    """策略5: 200日均线策略"""
    return strategy_ma_single(df, divs, ma_period=200)

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
    
    # 最大回撤
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax
    max_dd = drawdown.min() * 100
    
    # 交易次数
    buy_count = sum(1 for t in trades if t["action"] == "BUY")
    sell_count = sum(1 for t in trades if t["action"] == "SELL")
    
    # 胜率
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
    
    # 年化波动率
    daily_returns = values.pct_change().dropna()
    volatility = daily_returns.std() * (252 ** 0.5) * 100
    
    # Sharpe (假设无风险利率 2%)
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
# 所有策略
# ============================================================

ALL_STRATEGIES = [
    {"id": "buy_hold", "name": "买入持有 Buy & Hold", "desc": "持有到底 + 分红再投资", "fn": strategy_buy_hold, "color": "#888888"},
    {"id": "macd", "name": "MACD 交叉", "desc": "MACD(12,26,9) 金叉买/死叉卖", "fn": strategy_macd, "color": "#ff6b6b"},
    {"id": "ma_20_60", "name": "均线交叉 MA20/60", "desc": "20日均线上穿60日均线买入，下穿卖出", "fn": strategy_ma_crossover, "color": "#4ecdc4"},
    {"id": "ma_50", "name": "50日均线突破", "desc": "价格突破50日均线买入，跌破卖出", "fn": lambda df, d: strategy_ma_single(df, d, 50), "color": "#ffe66d"},
    {"id": "ma_200", "name": "200日均线突破", "desc": "价格突破200日均线买入，跌破卖出", "fn": strategy_ma_single_200, "color": "#a29bfe"},
]

# ============================================================
# 图表
# ============================================================

def generate_equity_chart(results):
    """权益曲线对比图"""
    fig = go.Figure()
    all_vals = []
    
    for r in results:
        vals = r["portfolio"]["value"]
        all_vals.extend(vals.tolist())
        fig.add_trace(go.Scatter(
            x=r["portfolio"].index,
            y=vals,
            mode='lines',
            name=r["name"],
            line=dict(width=1.5, color=r["color"]),
            opacity=0.85,
            hovertemplate=f'{r["name"]}<br>日期: %{{x}}<br>价值: $%{{y:,.0f}}<extra></extra>'
        ))
    
    # 固定 Y 轴
    y_min = max(min(all_vals), 1)
    y_max = max(all_vals)
    
    fig.update_layout(
        title=dict(
            text="📈 PFE 辉瑞 — 多策略权益曲线对比<br><sup style='color:#888'>回测区间: 2012-05-18 → 2025-12-01 | 初始资金 $10,000 | 含分红再投资</sup>",
            font=dict(size=16),
        ),
        xaxis_title="日期",
        yaxis_title="账户价值 ($)",
        template="plotly_dark",
        height=600,
        legend=dict(font=dict(size=11)),
        hovermode="x unified",
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def generate_drawdown_chart(results):
    """回撤对比图"""
    fig = go.Figure()
    
    for r in results:
        vals = r["portfolio"]["value"]
        cummax = vals.cummax()
        dd = (vals - cummax) / cummax * 100
        fig.add_trace(go.Scatter(
            x=r["portfolio"].index,
            y=dd,
            mode='lines',
            name=r["name"],
            line=dict(width=1.2, color=r["color"]),
            fill='tozeroy',
            opacity=0.5,
            hovertemplate=f'{r["name"]}<br>日期: %{{x}}<br>回撤: %{{y:.1f}}%<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text="📉 最大回撤对比<br><sup style='color:#888'>回测区间: 2012-05-18 → 2025-12-01</sup>",
            font=dict(size=16),
        ),
        xaxis_title="日期",
        yaxis_title="回撤 (%)",
        template="plotly_dark",
        height=400,
        legend=dict(font=dict(size=11)),
        hovermode="x unified",
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def generate_bar_comparison(results):
    """策略指标柱状对比"""
    names = [r["name"] for r in results]
    colors = [r["color"] for r in results]
    
    fig = make_subplots(rows=1, cols=4,
                        subplot_titles=["总回报率 (%)", "年化CAGR (%)", "最大回撤 (%)", "Sharpe Ratio"],
                        horizontal_spacing=0.08)
    
    metrics = [
        ([r["stats"]["total_return_pct"] for r in results], 1),
        ([r["stats"]["cagr_pct"] for r in results], 2),
        ([r["stats"]["max_drawdown_pct"] for r in results], 3),
        ([r["stats"]["sharpe_ratio"] for r in results], 4),
    ]
    
    for vals, col in metrics:
        fig.add_trace(go.Bar(
            x=names, y=vals,
            marker_color=colors,
            text=[f'{v:.1f}' for v in vals],
            textposition='outside',
            showlegend=False,
        ), row=1, col=col)
    
    fig.update_layout(
        title=dict(
            text="📊 策略指标对比<br><sup style='color:#888'>PFE 辉瑞 | 2012-05-18 → 2025-12-01 | $10,000 初始资金</sup>",
            font=dict(size=16),
        ),
        template="plotly_dark",
        height=400,
        showlegend=False,
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
    
    equity_chart = generate_equity_chart(results)
    drawdown_chart = generate_drawdown_chart(results)
    bar_chart = generate_bar_comparison(results)
    
    # 找最佳策略
    best = max(results, key=lambda r: r["stats"]["final_value"])
    worst = min(results, key=lambda r: r["stats"]["final_value"])
    
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
