import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="全球策略优化系统 Pro", layout="wide")
st.title("⚖️ 基金组合策略与参数优化系统")

# --- 0. Session State 初始化与市场切换逻辑 ---
if 'expanded' not in st.session_state:
    st.session_state.expanded = True

# 市场预设数据定义
MARKET_PRESETS = {
    "沪深 (A股)": {
        "tickers": "159941.SZ, 513500.SS, 515100.SS, 512400.SS, 515220.SS, 588080.SS, 518880.SS",
        "weights": "0.20, 0.25, 0.2, 0.05, 0.10, 0.05, 0.15",
        "bench": "510300.SS"
    },
    "美股": {
        "tickers": "IVV, QQQM, BRK.B, GLDM, XLE, DBMF, KMLM, ETHW",
        "weights": "0.20, 0.20, 0.15, 0.10, 0.10, 0.10, 0.10, 0.05",
        "bench": "SPY"
    }
}

# 核心：切换市场时的回调函数
def sync_market_defaults():
    new_market = st.session_state.market_radio
    st.session_state.in_tickers = MARKET_PRESETS[new_market]["tickers"]
    st.session_state.in_weights = MARKET_PRESETS[new_market]["weights"]
    st.session_state.in_bench = MARKET_PRESETS[new_market]["bench"]

# 初始化 session_state 的默认值（仅在第一次运行或未设置时）
if "in_tickers" not in st.session_state:
    # 默认展示美股
    st.session_state.in_tickers = MARKET_PRESETS["美股"]["tickers"]
    st.session_state.in_weights = MARKET_PRESETS["美股"]["weights"]
    st.session_state.in_bench = MARKET_PRESETS["美股"]["bench"]

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

# --- 2. 局部再平衡：涟漪分配算法 ---
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

# --- 4. UI 配置面板 ---
with st.expander("🛠️ 操作配置面板", expanded=st.session_state.expanded):
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        # 增加 on_change 回调以恢复自动切换功能
        market = st.radio("选择市场", ["沪深 (A股)", "美股"], index=1, horizontal=True, 
                          key="market_radio", on_change=sync_market_defaults)
    with m_col2:
        test_mode = st.radio("回测维度", ["对比策略", "对比阈值"], horizontal=True, key="mode_radio")
    
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        raw_tickers = st.text_input("代码 (逗号分隔)", key="in_tickers")
        raw_weights = st.text_input("占比 (逗号分隔)", key="in_weights")
    with c2:
        benchmark_ticker = st.text_input("对比基准", key="in_bench")
        start_date_input = st.date_input("设定开始时间", datetime(2020, 1, 1), key="in_start")
    with c3:
        initial_cap = st.number_input("初始资金", value=10000, key="in_cap")

    st.divider()
    strategy_options = ["无 (Buy & Hold)", "定期再平衡(年度)", "相对差全局再平衡", "相对差局部再平衡", "相对差混合再平衡"]
    
    if test_mode == "对比策略":
        selected_strategies = st.multiselect("对比策略", strategy_options, default=["相对差混合再平衡"], key="sel_strat")
        thresholds = [st.slider("固定阈值 (%)", 10, 100, 40, key="slider_val") / 100.0]
    else:
        target_strat = st.selectbox("选择要优化的策略", strategy_options[1:], key="sel_target")
        selected_strategies = [target_strat]
        raw_thresholds = st.text_input("输入多个阈值 (%, 逗号分隔)", "20, 40, 60", key="in_thrs")
        thresholds = [float(x.strip())/100 for x in raw_thresholds.split(",")]
    
    if st.button("确定", type="primary", key="btn_confirm"):
        st.session_state.expanded = False
        st.rerun()

# --- 5. 执行与显示区 ---
if not st.session_state.expanded:
    processed_tickers = [t.strip().upper().replace('.', '-') if 'BRK.B' in t else t.strip().upper() for t in raw_tickers.split(",")]
    bench_ticker = benchmark_ticker.strip().upper()
    
    try:
        weights_list = [float(w.strip()) for w in raw_weights.split(",")]
    except: st.error("占比输入错误"); st.stop()

    with st.spinner('数据分析中...'):
        all_to_download = list(set(processed_tickers + [bench_ticker]))
        df_raw = yf.download(all_to_download, start=start_date_input - timedelta(days=10), progress=False)
        
        if df_raw.empty: st.error("❌ 无法获取数据"); st.stop()

        # 智能价格提取
        available_levels = df_raw.columns.get_level_values(0).unique()
        prices_adj_all = df_raw['Adj Close'] if 'Adj Close' in available_levels else df_raw['Close']
        prices_raw_all = df_raw['Close'] if 'Close' in available_levels else prices_adj_all

        # 缺失检测与修复
        final_tickers, final_weights, missing_report = [], [], []
        req_ts = pd.Timestamp(start_date_input)

        for i, t in enumerate(processed_tickers):
            t_start = prices_adj_all[t].first_valid_index() if t in prices_adj_all.columns else None
            if t_start is None or t_start > req_ts + timedelta(days=180):
                t_start_alt = prices_raw_all[t].first_valid_index() if t in prices_raw_all.columns else None
                if t_start_alt is not None and t_start_alt <= req_ts + timedelta(days=180):
                    prices_adj_all[t] = prices_raw_all[t]
                    final_tickers.append(t)
                    final_weights.append(weights_list[i])
                else:
                    missing_report.append(f"❌ **{t}**: 缺失历史记录 (最早见于 {t_start.date() if t_start else '未知'})")
            else:
                final_tickers.append(t)
                final_weights.append(weights_list[i])

        if missing_report:
            st.error("⚠️ 数据缺失警报")
            for msg in missing_report: st.write(msg)
            if not final_tickers: st.stop()

        current_w = pd.Series(final_weights, index=final_tickers)
        current_w = current_w / current_w.sum()

        listing_dates = prices_adj_all[final_tickers].apply(lambda x: x.first_valid_index())
        latest_listing = listing_dates.max()
        if latest_listing > req_ts + timedelta(days=14):
            st.warning(f"⚠️ **起点顺延**：由于 **{listing_dates.idxmax()}** 数据限制，调整为 **{latest_listing.date()}**。")

        effective_start = max(req_ts, latest_listing)
        available_prices = prices_adj_all[prices_adj_all.index >= effective_start].dropna(axis=1, how='all')
        available_prices[bench_ticker] = prices_adj_all[bench_ticker] if bench_ticker in prices_adj_all.columns else prices_raw_all[bench_ticker]
        
        price_df = pd.concat([available_prices.iloc[[0]], available_prices.iloc[1:].resample('ME').last()]).ffill().dropna()
        price_df = price_df[~price_df.index.duplicated(keep='first')]

        # 回测循环
        comparison_df = pd.DataFrame(index=price_df.index)
        comparison_df[f"基准({bench_ticker})"] = (price_df[bench_ticker] / price_df[bench_ticker].iloc[0]) * initial_cap
        
        detailed_results, metrics_list = {}, []
        for strat in selected_strategies:
            for thr in thresholds:
                label = f"{strat} ({thr*100:.0f}%)" if test_mode == "对比阈值" else strat
                res_df, count = run_detailed_backtest(strat, price_df[final_tickers], current_w, initial_cap, thr)
                detailed_results[label] = res_df
                comparison_df[label] = res_df.drop_duplicates(subset='日期', keep='last').set_index('日期')['净值']
                m = calculate_metrics(comparison_df[label], count)
                m["回测维度"] = label
                metrics_list.append(m)

        st.line_chart(comparison_df)
        st.table(pd.DataFrame(metrics_list).set_index("回测维度"))

        st.divider()
        for label, data in detailed_results.items():
            with st.expander(f"查看明细: {label}"):
                def complex_style(row):
                    if row['类型'] == '再平衡前': return ['background-color: #fff3e0'] * len(row)
                    if row['类型'] == '再平衡后': return ['background-color: #e8f5e9'] * len(row)
                    return [''] * len(row)
                st.dataframe(data.style.apply(complex_style, axis=1).format({"净值": "{:,.2f}"}), use_container_width=True)

    if st.button("重新调整配置", key="btn_reset"):
        st.session_state.expanded = True
        st.rerun()
