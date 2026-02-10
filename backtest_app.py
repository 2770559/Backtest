import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

# --- 页面配置 ---
st.set_page_config(page_title="资产配置实验室 Pro", layout="wide")
st.title("⚖️ 基金组合全维度优化系统")

# --- 0. 基础配置与名称映射 ---
# 需求：代码转中文映射。删除了“占比”字样后，现在支持显示最长 6 个汉字。
TICKER_TO_NAME = {
    "159941.SZ": "纳指ETF",
    "513500.SS": "标普500ETF",
    "512890.SS": "红利低波ETF",
    "512400.SS": "有色金属ETF",
    "515220.SS": "煤炭ETF基金",
    "588080.SS": "科创50ETF",
    "518880.SS": "黄金ETF",
    "510300.SS": "沪深300"
}

if 'expanded' not in st.session_state:
    st.session_state.expanded = True

# 初始化默认组合
if 'portfolios_list' not in st.session_state:
    st.session_state.portfolios_list = [
        {
            "name": "组合 A", 
            "tickers": "IVV, QQQM, BRK.B, GLDM, XLE, DBMF, KMLM, ETH-USD", 
            "weights": "0.20, 0.20, 0.15, 0.10, 0.10, 0.10, 0.10, 0.05", 
            "strat": "相对差混合再平衡", 
            "thr": 40
        },
        {
            "name": "组合 B", 
            "tickers": "159941.SZ, 513500.SS, 512890.SS, 512400.SS, 515220.SS, 588080.SS, 518880.SS", 
            "weights": "0.20, 0.25, 0.2, 0.05, 0.10, 0.05, 0.15", 
            "strat": "相对差混合再平衡", 
            "thr": 40
        }
    ]

# --- 1. 核心工具函数 ---
def clean_ticker(t):
    t = t.strip().upper()
    mapping = {"BRK.B": "BRK-B", "ETHUSD": "ETH-USD", "BTCUSD": "BTC-USD"}
    return mapping.get(t, t)

def calculate_metrics(nav_series, rebalance_count, risk_free_rate=0.02):
    """
    量化指标计算公式：
    $$AnnReturn = (1 + TotalReturn)^{\frac{1}{Years}} - 1$$
    $$Sharpe = \frac{R_p - R_f}{\sigma_p}$$
    """
    monthly_returns = nav_series.pct_change().dropna()
    total_return = (nav_series.iloc[-1] / nav_series.iloc[0]) - 1
    years = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    max_dd = ((nav_series - nav_series.cummax()) / nav_series.cummax()).min()
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

def run_detailed_backtest(strategy_name, price_df, target_weights, initial_cap, threshold):
    tickers = price_df.columns
    current_shares = (initial_cap * target_weights) / price_df.iloc[0]
    history, last_rebalance_date, rebalance_count = [], price_df.index[0], 0
    for i in range(len(price_df)):
        current_date, current_prices = price_df.index[i], price_df.iloc[i]
        asset_values = current_shares * current_prices
        total_val = asset_values.sum()
        current_weights, do_rebalance = asset_values / total_val, False
        new_values = asset_values.copy()
        rel_diffs = np.abs(current_weights - target_weights) / target_weights
        
        if strategy_name == "定期再平衡(年度)":
            if (current_date - last_rebalance_date).days >= 365:
                new_values, do_rebalance = total_val * target_weights, True
        elif strategy_name == "相对差混合再平衡":
            if rel_diffs.max() > threshold:
                if ((target_weights >= 0.1) & (rel_diffs > threshold)).any(): new_values = total_val * target_weights
                else: new_values = apply_local_rebalance(asset_values, target_weights, threshold)
                do_rebalance = True
        elif strategy_name == "相对差局部再平衡":
            if rel_diffs.max() > threshold: new_values, do_rebalance = apply_local_rebalance(asset_values, target_weights, threshold), True
        
        if do_rebalance:
            rebalance_count += 1
            pre_rec = {"日期": current_date, "类型": "再平衡前", "净值": total_val}
            pre_rec.update({f"{t}占比": f"{current_weights[t]:.2%}" for t in tickers})
            history.append(pre_rec)
            current_shares, last_rebalance_date = new_values / current_prices, current_date
            post_rec = {"日期": current_date, "类型": "再平衡后", "净值": total_val}
            post_rec.update({f"{t}占比": f"{(new_values/total_val)[t]:.2%}" for t in tickers})
            history.append(post_rec)
        else:
            rec = {"日期": current_date, "类型": "常规", "净值": total_val}
            rec.update({f"{t}占比": f"{current_weights[t]:.2%}" for t in tickers})
            history.append(rec)
    return pd.DataFrame(history), rebalance_count

# --- 2. UI 配置面板 ---
with st.expander("🛠️ 资产配置实验室 (配置模式)", expanded=st.session_state.expanded):
    c_p1, c_p2, c_p3 = st.columns([2, 2, 1])
    with c_p1:
        benchmark_input = st.text_input("对比基准 (510300.SS / SPY)", "SPY", key="bench_input")
    with c_p2:
        start_date = st.date_input("设定开始时间", datetime(2020, 1, 1), key="start_date")
    with c_p3:
        initial_funds = st.number_input("初始资金", value=10000, key="init_funds")

    st.divider()
    strategy_options = ["无 (Buy & Hold)", "定期再平衡(年度)", "相对差局部再平衡", "相对差混合再平衡"]
    
    for i, port in enumerate(st.session_state.portfolios_list):
        h_col1, h_col2 = st.columns([8, 1])
        with h_col1:
            st.markdown(f"#### 📦 {port['name']}")
        with h_col2:
            if st.button("🗑️", key=f"del_{i}"):
                if len(st.session_state.portfolios_list) > 1:
                    st.session_state.portfolios_list.pop(i)
                    for j, p in enumerate(st.session_state.portfolios_list): p["name"] = f"组合 {chr(65+j)}"
                    st.rerun()

        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        with col1:
            port['tickers'] = st.text_input(f"代码", port['tickers'], key=f"t_{i}")
        with col2:
            port['weights'] = st.text_input(f"占比", port['weights'], key=f"w_{i}")
        with col3:
            port['strat'] = st.selectbox(f"策略", strategy_options, 
                                         index=strategy_options.index(port['strat']), key=f"s_{i}")
        with col4:
            port['thr'] = st.number_input(f"阈值(%)", 1, 200, port['thr'], key=f"thr_{i}")
        st.divider()

    b1, b2, _ = st.columns([1, 1, 3])
    with b1:
        if st.button("➕ 添加新组合"):
            new_idx = len(st.session_state.portfolios_list)
            st.session_state.portfolios_list.append({
                "name": f"组合 {chr(65+new_idx)}", "tickers": st.session_state.portfolios_list[-1]["tickers"],
                "weights": st.session_state.portfolios_list[-1]["weights"], "strat": "相对差混合再平衡", "thr": 40
            })
            st.rerun()
    with b2:
        if st.button("🚀 确定运行", type="primary", key="btn_confirm"):
            st.session_state.expanded = False; st.rerun()

# --- 3. 执行与显示区 ---
if not st.session_state.expanded:
    with st.spinner('同步全球数据中...'):
        bench_ticker = clean_ticker(benchmark_input)
        all_download_tks = [bench_ticker]
        for p in st.session_state.portfolios_list:
            all_download_tks.extend([clean_ticker(t) for t in p['tickers'].split(",")])
        all_download_tks = list(set(all_download_tks))

        df_raw = yf.download(all_download_tks, start=start_date - timedelta(days=10), progress=False)
        if df_raw.empty: st.error("下载失败。"); st.stop()

        avail = df_raw.columns.get_level_values(0).unique()
        pr_adj = df_raw['Adj Close'] if 'Adj Close' in avail else df_raw['Close']
        pr_raw = df_raw['Close'] if 'Close' in avail else pr_adj
        
        final_prices = pd.DataFrame(index=df_raw.index)
        for t in all_download_tks:
            final_prices[t] = pr_adj[t] if not pr_adj[t].dropna().empty else pr_raw[t]

        # 采样对齐
        active_tks = list(set([t for p in st.session_state.portfolios_list for t in [clean_ticker(x) for x in p['tickers'].split(",")]]))
        list_dates = final_prices[active_tks].apply(lambda x: x.first_valid_index())
        eff_st = max(pd.Timestamp(start_date), list_dates.max())
        
        if list_dates.max() > pd.Timestamp(start_date) + timedelta(days=14):
            st.warning(f"⚠️ 起点因数据限制顺延至 {eff_st.date()}")

        df_aligned = final_prices[final_prices.index >= eff_st].dropna(axis=1, how='all')
        price_df = pd.concat([df_aligned.iloc[[0]], df_aligned.iloc[1:].resample('ME').last()]).ffill().dropna()
        price_df = price_df[~price_df.index.duplicated(keep='first')]

        # 运行对比
        comparison_df = pd.DataFrame(index=price_df.index)
        bench_nav = (price_df[bench_ticker] / price_df[bench_ticker].iloc[0]) * initial_funds
        comparison_df[f"基准({benchmark_input})"] = bench_nav
        
        detailed_res, metrics_list = {}, [calculate_metrics(bench_nav, 0)]
        metrics_list[-1]["回测维度"] = f"基准({benchmark_input})"

        for p in st.session_state.portfolios_list:
            p_tks = [clean_ticker(t) for t in p['tickers'].split(",")]
            p_wts = [float(w) for w in p['weights'].split(",")]
            res_df, cnt = run_detailed_backtest(p['strat'], price_df[p_tks], pd.Series(p_wts, index=p_tks), initial_funds, p['thr']/100.0)
            comparison_df[p['name']] = res_df.drop_duplicates(subset='日期', keep='last').set_index('日期')['净值']
            m = calculate_metrics(comparison_df[p['name']], cnt)
            m["回测维度"] = p['name']
            metrics_list.append(m)
            
            # --- 核心改进：删除“占比”字样并扩展名称长度 ---
            new_col_names = {}
            for col in res_df.columns:
                temp_name = col
                for ticker, name in TICKER_TO_NAME.items():
                    if ticker in col:
                        # 1. 将代码替换为中文名称（最多6字） 2. 删除“占比”字样
                        temp_name = col.replace(ticker, name[:6]).replace("占比", "")
                        break
                new_col_names[col] = temp_name
            
            detailed_res[p['name']] = res_df.rename(columns=new_col_names)

        st.line_chart(comparison_df, height=500)
        st.subheader("📊 风险收益对比")
        st.table(pd.DataFrame(metrics_list).set_index("回测维度"))

        st.divider()
        for lbl, data in detailed_res.items():
            with st.expander(f"📋 调仓明细: {lbl}"):
                def style_tr(row):
                    if row['类型'] == '再平衡前': return ['background-color: #fff3e0'] * len(row)
                    if row['类型'] == '再平衡后': return ['background-color: #e8f5e9'] * len(row)
                    return [''] * len(row)
                st.dataframe(data.style.apply(style_tr, axis=1).format({"净值": "{:,.2f}"}), use_container_width=True)

    if st.button("🔙 重新调整配置", key="btn_reset"):
        st.session_state.expanded = True; st.rerun()
