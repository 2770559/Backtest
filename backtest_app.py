import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="全球策略优化系统", layout="wide")
st.title("⚖️ 基金组合策略与参数优化回测系统")

# --- 1. 核心计算函数 ---
def calculate_metrics(nav_series, rebalance_count, risk_free_rate=0.02):
    monthly_returns = nav_series.pct_change().dropna()
    total_return = (nav_series.iloc[-1] / nav_series.iloc[0]) - 1
    days = (nav_series.index[-1] - nav_series.index[0]).days
    years = days / 365.25
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    rolling_max = nav_series.cummax()
    max_dd = ((nav_series - rolling_max) / rolling_max).min()
    ann_vol = monthly_returns.std() * np.sqrt(12)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
    return {
        "最终净值": f"{nav_series.iloc[-1]:,.2f}",
        "总收益率": f"{total_return:.2%}",
        "年化收益率": f"{ann_return:.2%}",
        "最大回撤": f"{max_dd:.2%}",
        "夏普比率": f"{sharpe:.2f}",
        "调仓次数": int(rebalance_count)
    }

# --- 2. 局部再平衡逻辑 ---
def apply_local_rebalance(asset_values, target_weights, threshold):
    total_val = asset_values.sum()
    current_vals = asset_values.copy()
    reset_indices = []
    while True:
        current_weights = current_vals / total_val
        rel_diffs = np.abs(current_weights - target_weights) / target_weights
        to_trigger = (rel_diffs > threshold) & (~current_vals.index.isin(reset_indices))
        if not to_trigger.any(): break
        triggered_indices = to_trigger.index[to_trigger].tolist()
        reset_indices.extend(triggered_indices)
        for idx in triggered_indices:
            current_vals[idx] = target_weights[idx] * total_val
        remaining_indices = [i for i in current_vals.index if i not in reset_indices]
        if not remaining_indices: return total_val * target_weights
        remaining_cash = total_val - current_vals[reset_indices].sum()
        current_remaining_sum = asset_values[remaining_indices].sum()
        ratios = asset_values[remaining_indices] / current_remaining_sum if current_remaining_sum > 0 else target_weights[remaining_indices]/target_weights[remaining_indices].sum()
        current_vals[remaining_indices] = ratios * remaining_cash
    return current_vals

# --- 3. 增强版回测引擎 ---
def run_detailed_backtest(strategy_name, price_df, target_weights, initial_cap, threshold):
    tickers = price_df.columns
    current_shares = (initial_cap * target_weights) / price_df.iloc[0]
    history = []
    last_rebalance_date = price_df.index[0]
    rebalance_count = 0
    
    for i in range(len(price_df)):
        current_date = price_df.index[i]
        current_prices = price_df.iloc[i]
        asset_values = current_shares * current_prices
        total_val = asset_values.sum()
        current_weights = asset_values / total_val
        do_rebalance = False
        new_values = asset_values.copy()
        
        rel_diffs = np.abs(current_weights - target_weights) / target_weights
        if strategy_name == "定期再平衡(年度)":
            if (current_date - last_rebalance_date).days >= 365:
                new_values, do_rebalance = total_val * target_weights, True
        elif strategy_name == "相对差全局再平衡":
            if rel_diffs.max() > threshold:
                new_values, do_rebalance = total_val * target_weights, True
        elif strategy_name == "相对差局部再平衡":
            if rel_diffs.max() > threshold:
                new_values = apply_local_rebalance(asset_values, target_weights, threshold)
                do_rebalance = True
        elif strategy_name == "相对差混合再平衡":
            if ((target_weights >= 0.1) & (rel_diffs > threshold)).any():
                new_values, do_rebalance = total_val * target_weights, True
            elif ((target_weights < 0.1) & (rel_diffs > threshold)).any():
                new_values = apply_local_rebalance(asset_values, target_weights, threshold)
                do_rebalance = True
        
        if do_rebalance:
            rebalance_count += 1
            pre_rec = {"日期": current_date, "类型": "再平衡前", "净值": total_val}
            pre_rec.update({f"{t}占比": f"{current_weights[t]:.2%}" for t in tickers})
            history.append(pre_rec)
            current_shares = new_values / current_prices
            last_rebalance_date = current_date
            post_rec = {"日期": current_date, "类型": "再平衡后", "净值": total_val}
            post_rec.update({f"{t}占比": f"{(new_values/total_val)[t]:.2%}" for t in tickers})
            history.append(post_rec)
        else:
            rec = {"日期": current_date, "类型": "常规", "净值": total_val}
            rec.update({f"{t}占比": f"{current_weights[t]:.2%}" for t in tickers})
            history.append(rec)
    return pd.DataFrame(history), rebalance_count

# --- 4. UI 逻辑 ---
with st.sidebar:
    st.header("🏢 市场与模式")
    market = st.radio("选择市场", ["沪深 (A股)", "美股"], index=1)
    test_mode = st.radio("选择回测维度", ["对比不同策略", "对比不同阈值"])
    
    # 默认值设置
    if market == "沪深 (A股)":
        default_tickers, default_weights, default_bench = "159941.SZ, 513500.SS, 515100.SS, 512400.SS, 515220.SS, 588080.SS, 518880.SS", "0.20, 0.25, 0.2, 0.05, 0.10, 0.05, 0.15", "510300.SS"
    else:
        default_tickers, default_weights, default_bench = "IVV, QQQM, BRK.B, GLDM, XLE, DBMF, KMLM, ETHW", "0.20, 0.20, 0.15, 0.10, 0.10, 0.10, 0.10, 0.05", "SPY"

    st.header("1. 投资组合")
    raw_tickers = st.text_input("代码", default_tickers)
    raw_weights = st.text_input("占比", default_weights)
    benchmark_ticker = st.text_input("基准", default_bench)
    start_date_input = st.date_input("开始日期", datetime(2020, 1, 1))
    
    st.header("2. 策略与参数")
    strategy_options = ["无 (Buy & Hold)", "定期再平衡(年度)", "相对差全局再平衡", "相对差局部再平衡", "相对差混合再平衡"]
    
    if test_mode == "对比不同策略":
        selected_strategies = st.multiselect("对比策略", strategy_options, default=["无 (Buy & Hold)", "相对差混合再平衡"])
        thresholds = [st.slider("固定相对差阈值 (%)", 10, 100, 40) / 100.0]
    else:
        target_strat = st.selectbox("选择要优化的策略", strategy_options[1:])
        selected_strategies = [target_strat]
        raw_thresholds = st.text_input("输入多个阈值 (%, 逗号分隔)", "20, 40, 60")
        thresholds = [float(x.strip())/100 for x in raw_thresholds.split(",")]

    run_btn = st.button("开始深度优化")

# --- 5. 执行回测 ---
if run_btn:
    processed_tickers = [t.strip().upper().replace('.', '-') if 'BRK.B' in t else t.strip().upper() for t in raw_tickers.split(",")]
    bench_ticker = benchmark_ticker.strip().upper()
    weights_list = [float(w.strip()) for w in raw_weights.split(",")]
    if len(processed_tickers) != len(weights_list) or abs(sum(weights_list) - 1.0) > 0.001:
        st.error("验证失败：个数不匹配或总和非 100%"); st.stop()
    target_w = pd.Series(weights_list, index=processed_tickers)

    with st.spinner('正在同步数据并分析参数敏感性...'):
        df_raw = yf.download(list(set(processed_tickers + [bench_ticker])), start=start_date_input - timedelta(days=10), progress=False)
        prices_full = df_raw['Adj Close'] if 'Adj Close' in df_raw.columns.get_level_values(0) else df_raw['Close']
        latest_listing = prices_full[processed_tickers].apply(lambda x: x.first_valid_index()).dropna().max()
        effective_start = max(pd.Timestamp(start_date_input), latest_listing)
        available_prices = prices_full[prices_full.index >= effective_start].dropna(axis=1, how='all')
        
        price_df = pd.concat([available_prices.iloc[[0]], available_prices.iloc[1:].resample('ME').last()]).ffill().dropna()
        price_df = price_df[~price_df.index.duplicated(keep='first')]

        # 回测循环
        comparison_df = pd.DataFrame(index=price_df.index)
        comparison_df[f"基准({bench_ticker})"] = (price_df[bench_ticker] / price_df[bench_ticker].iloc[0]) * 10000
        
        detailed_results = {}
        metrics_list = []
        
        # 处理基准指标
        m_bench = calculate_metrics(comparison_df[f"基准({bench_ticker})"], 0)
        m_bench["测试维度"] = f"基准({bench_ticker})"
        metrics_list.append(m_bench)

        # 核心循环：支持多策略或多阈值
        for strat in selected_strategies:
            for thr in thresholds:
                # 确定显示名称
                label = f"{strat} ({thr*100:.0f}%)" if test_mode == "对比不同阈值" else strat
                
                res_df, count = run_detailed_backtest(strat, price_df[processed_tickers], target_w, 10000, thr)
                detailed_results[label] = res_df
                comparison_df[label] = res_df.drop_duplicates(subset='日期', keep='last').set_index('日期')['净值']
                
                # 计算指标
                m = calculate_metrics(comparison_df[label], count)
                m["测试维度"] = label
                metrics_list.append(m)

        st.line_chart(comparison_df)
        st.subheader("📊 风险收益与敏感性指标")
        st.table(pd.DataFrame(metrics_list).set_index("测试维度"))

        st.divider()
        st.subheader("📋 详细调仓明细对比")
        for label, data in detailed_results.items():
            with st.expander(f"查看明细: {label}"):
                styled_df = data.style.apply(lambda row: ['background-color: #fff3e0']*len(row) if row['类型']=='再平衡前' else (['background-color: #e8f5e9']*len(row) if row['类型']=='再平衡后' else ['']*len(row)), axis=1).format({"净值": "{:,.2f}"})
                st.dataframe(styled_df, use_container_width=True)
