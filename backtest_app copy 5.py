import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="量化回测系统-精准建仓版", layout="wide")
st.title("⚖️ 基金组合回测系统 (精准建仓与自动校准)")

# --- 1. 风险指标计算 ---
def calculate_metrics(nav_series, risk_free_rate=0.02):
    monthly_returns = nav_series.pct_change().dropna()
    total_return = (nav_series.iloc[-1] / nav_series.iloc[0]) - 1
    # 按照数据点间隔计算年数
    days = (nav_series.index[-1] - nav_series.index[0]).days
    years = days / 365.25
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    rolling_max = nav_series.cummax()
    max_dd = ((nav_series - rolling_max) / rolling_max).min()
    ann_vol = monthly_returns.std() * np.sqrt(12)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
    return {
        "最终价值": f"${nav_series.iloc[-1]:,.2f}",
        "总收益率": f"{total_return:.2%}",
        "年化收益率": f"{ann_return:.2%}",
        "最大回撤": f"{max_dd:.2%}",
        "夏普比率": f"{sharpe:.2f}"
    }

# --- 2. 局部再平衡：涟漪重分配算法 ---
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

# --- 3. 核心回测引擎 ---
def run_detailed_backtest(strategy_name, price_df, target_weights, initial_cap, threshold):
    tickers = price_df.columns
    # 初始买入：使用 price_df 的第一行（精准建仓日）
    current_shares = (initial_cap * target_weights) / price_df.iloc[0]
    history = []
    last_rebalance_date = price_df.index[0]
    
    for i in range(len(price_df)):
        current_date = price_df.index[i]
        current_prices = price_df.iloc[i]
        asset_values = current_shares * current_prices
        total_val = asset_values.sum()
        current_weights = asset_values / total_val
        
        do_rebalance = False
        new_values = asset_values.copy()
        
        if strategy_name == "定期再平衡(半年度)":
            if (current_date - last_rebalance_date).days >= 182:
                new_values = total_val * target_weights
                do_rebalance = True
        elif strategy_name == "定期再平衡(年度)":
            if (current_date - last_rebalance_date).days >= 365:
                new_values = total_val * target_weights
                do_rebalance = True
        elif strategy_name == "相对差全局再平衡":
            if (np.abs(current_weights - target_weights) / target_weights).max() > threshold:
                new_values = total_val * target_weights
                do_rebalance = True
        elif strategy_name == "相对差局部再平衡":
            if (np.abs(current_weights - target_weights) / target_weights).max() > threshold:
                new_values = apply_local_rebalance(asset_values, target_weights, threshold)
                do_rebalance = True
        elif strategy_name == "相对差混合再平衡":
            rel_diffs = np.abs(current_weights - target_weights) / target_weights
            if ((target_weights >= 0.1) & (rel_diffs > threshold)).any():
                new_values = total_val * target_weights
                do_rebalance = True
            elif ((target_weights < 0.1) & (rel_diffs > threshold)).any():
                new_values = apply_local_rebalance(asset_values, target_weights, threshold)
                do_rebalance = True
        
        if do_rebalance:
            pre_rec = {"Date": current_date, "Type": "再平衡前", "NAV": total_val}
            pre_rec.update({f"{t}占比": f"{current_weights[t]:.2%}" for t in tickers})
            history.append(pre_rec)
            
            current_shares = new_values / current_prices
            last_rebalance_date = current_date
            post_weights = new_values / total_val
            post_rec = {"Date": current_date, "Type": "再平衡后", "NAV": total_val}
            post_rec.update({f"{t}占比": f"{post_weights[t]:.2%}" for t in tickers})
            history.append(post_rec)
        else:
            rec = {"Date": current_date, "Type": "常规", "NAV": total_val}
            rec.update({f"{t}占比": f"{current_weights[t]:.2%}" for t in tickers})
            history.append(rec)
            
    return pd.DataFrame(history)

# --- 4. 样式 ---
def color_rebalance(row):
    if row['Type'] == '再平衡前': return ['background-color: #fff3e0'] * len(row)
    if row['Type'] == '再平衡后': return ['background-color: #e8f5e9'] * len(row)
    return [''] * len(row)

# --- 5. UI 与 验证 ---
with st.sidebar:
    st.header("1. 投资组合配置")
    raw_tickers = st.text_input("代码", "SPY, QQQ, BRK.B, GLD, XLE, MSTR")
    raw_weights = st.text_input("占比", "0.25, 0.25, 0.2, 0.15, 0.1, 0.05")
    benchmark_ticker = st.text_input("基准基金", "SPY")
    start_date_input = st.date_input("设置起始日期", datetime(2018, 1, 1))
    initial_cap = st.number_input("初始资金", value=10000)
    
    st.header("2. 再平衡策略")
    strategy_options = ["无 (Buy & Hold)", "定期再平衡(年度)", "相对差全局再平衡", "相对差局部再平衡", "相对差混合再平衡"]
    selected_strategies = st.multiselect("对比策略", strategy_options, default=["无 (Buy & Hold)", "相对差混合再平衡"])
    rel_threshold = st.slider("相对差阈值 (%)", 10, 100, 40) / 100.0
    run_btn = st.button("运行回测")

if run_btn:
    # 自动替换 BRK.B 为 BRK-B
    processed_tickers = [t.strip().upper().replace('.', '-') for t in raw_tickers.split(",")]
    bench_ticker = benchmark_ticker.strip().upper().replace('.', '-')
    
    # 个数匹配与 100% 验证
    try:
        weights_list = [float(w.strip()) for w in raw_weights.split(",")]
    except: st.error("占比输入包含非法字符"); st.stop()

    if len(processed_tickers) != len(weights_list):
        st.error(f"❌ 个数不匹配：代码 {len(processed_tickers)} 个，占比 {len(weights_list)} 个。")
        st.stop()
    if abs(sum(weights_list) - 1.0) > 0.001:
        st.error(f"❌ 占比总和应为 100%，当前为 {sum(weights_list)*100:.2f}%。")
        st.stop()

    target_w = pd.Series(weights_list, index=processed_tickers)

    with st.spinner('同步数据中...'):
        all_download = list(set(processed_tickers + [bench_ticker]))
        df_raw = yf.download(all_download, start=start_date_input - timedelta(days=7), progress=False)
        prices_full = df_raw['Adj Close'] if 'Adj Close' in df_raw.columns.get_level_values(0) else df_raw['Close']
        
        # 确定实际起始交易日 (微调点)
        listing_dates = prices_full[processed_tickers].apply(lambda x: x.first_valid_index())
        latest_listing = listing_dates.max()
        effective_start = max(pd.Timestamp(start_date_input), latest_listing)
        
        # 核心逻辑：获取起始日及其后的月度采样
        # 1. 找到有效日期范围内的所有交易日
        available_prices = prices_full[prices_full.index >= effective_start]
        if available_prices.empty: st.error("选定日期无交易数据"); st.stop()
        
        # 2. 提取精准的首个交易日
        entry_date = available_prices.index[0]
        entry_row = available_prices.iloc[[0]]
        
        # 3. 提取之后的月末收盘价
        monthly_rows = available_prices.iloc[1:].resample('ME').last()
        
        # 4. 合并：首日 + 月末序列
        price_df = pd.concat([entry_row, monthly_rows]).ffill().dropna()
        price_df = price_df[~price_df.index.duplicated(keep='first')] # 防止首日恰好是月末导致重复

        st.success(f"✅ 回测开启！建仓日: **{entry_date.date()}** (不限月末) | 后续节点: 月末收盘价")

        # 计算并展示
        comparison_df = pd.DataFrame(index=price_df.index)
        bench_nav = (price_df[bench_ticker] / price_df[bench_ticker].iloc[0]) * initial_cap
        comparison_df[f"基准({bench_ticker})"] = bench_nav
        
        detailed_results = {}
        for strat in selected_strategies:
            res_df = run_detailed_backtest(strat, price_df[processed_tickers], target_w, initial_cap, rel_threshold)
            detailed_results[strat] = res_df
            comparison_df[strat] = res_df.drop_duplicates(subset='Date', keep='last').set_index('Date')['NAV']

        st.subheader("📈 策略净值对比")
        st.line_chart(comparison_df)

        st.subheader("📊 风险收益指标")
        metrics_list = [calculate_metrics(comparison_df[col]) for col in comparison_df.columns]
        for i, col in enumerate(comparison_df.columns): metrics_list[i]["策略"] = col
        st.table(pd.DataFrame(metrics_list).set_index("策略"))

        st.divider()
        st.subheader("📋 详细月度调仓明细")
        for strat, data in detailed_results.items():
            with st.expander(f"查看明细: {strat}"):
                styled_df = data.style.apply(color_rebalance, axis=1).format({"NAV": "${:,.2f}"})
                st.dataframe(styled_df, use_container_width=True, height=400)
