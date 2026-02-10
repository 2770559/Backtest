import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import uuid
import json
from datetime import datetime, timedelta

# --- 1. 页面基本配置 ---
st.set_page_config(page_title="资产配置实验室 Pro", layout="wide")
st.title("⚖️ 基金组合全维度优化系统")

# 代码与中文名映射表 (5字限制)
TICKER_TO_NAME = {
    "159941.SZ": "纳指ETF", "513500.SS": "标普500", "512890.SS": "红利低波",
    "512400.SS": "有色金属", "515220.SS": "煤炭ETF", "588080.SS": "科创50",
    "518880.SS": "黄金ETF", "510300.SS": "沪深300"
}

if 'expanded' not in st.session_state:
    st.session_state.expanded = True

# --- 初始化 Session State ---
if 'bi' not in st.session_state: st.session_state['bi'] = "SPY"
if 'sd' not in st.session_state: st.session_state['sd'] = datetime(2022, 1, 1)
if 'if' not in st.session_state: st.session_state['if'] = 10000

if 'portfolios_list' not in st.session_state:
    st.session_state.portfolios_list = [
        {
            "id": str(uuid.uuid4()), 
            "name": "组合 A", 
            "tickers": "IVV, QQQM, BRK.B, GLDM, XLE, DBMF, KMLM, ETH-USD", 
            "weights": "0.20, 0.20, 0.15, 0.10, 0.10, 0.10, 0.10, 0.05", 
            "strat": "相对差混合再平衡", 
            "thr": 40
        },
        {
            "id": str(uuid.uuid4()), 
            "name": "组合 B", 
            "tickers": "159941.SZ, 513500.SS, 512890.SS, 512400.SS, 515220.SS, 588080.SS, 518880.SS", 
            "weights": "0.20, 0.25, 0.2, 0.05, 0.10, 0.05, 0.15", 
            "strat": "相对差混合再平衡", 
            "thr": 40
        }
    ]

def delete_portfolio(idx):
    if 0 <= idx < len(st.session_state.portfolios_list):
        st.session_state.portfolios_list.pop(idx)

# --- 2. 核心算法 ---
def clean_ticker(t):
    t = t.strip().upper()
    mapping = {"BRK.B": "BRK-B", "ETHUSD": "ETH-USD", "BTCUSD": "BTC-USD"}
    return mapping.get(t, t)

def calculate_metrics(nav_series, rebalance_count, risk_free_rate=0.02):
    if nav_series is None or nav_series.empty or len(nav_series) < 2:
        return {"最终净值": "-", "总收益率": "-", "年化收益率": "-", "最大回撤": "-", "夏普比率": "-", "调仓次数": "-"}
    try:
        nav = nav_series.dropna()
        if len(nav) < 2: return {"最终净值": "-", "总收益率": "-", "年化收益率": "-", "最大回撤": "-", "夏普比率": "-", "调仓次数": "-"}
        total_return = (nav.iloc[-1] / nav.iloc[0]) - 1
        days = (nav.index[-1] - nav.index[0]).days
        if days <= 0: return {"最终净值": "-", "总收益率": "-", "年化收益率": "-", "最大回撤": "-", "夏普比率": "-", "调仓次数": "-"}
        years = days / 365.25
        ann_return = (1 + total_return) ** (1 / years) - 1 
        rolling_max = nav.cummax()
        max_dd = ((nav - rolling_max) / rolling_max).min()
        daily_ret = nav.pct_change().dropna()
        ann_vol = daily_ret.std() * np.sqrt(12) 
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
        return {
            "最终净值": f"{nav.iloc[-1]:,.2f}", 
            "总收益率": f"{total_return:.2%}", 
            "年化收益率": f"{ann_return:.2%}", 
            "最大回撤": f"{max_dd:.2%}", 
            "夏普比率": f"{sharpe:.2f}", 
            "调仓次数": int(rebalance_count)
        }
    except Exception:
        return {"最终净值": "Err", "总收益率": "Err", "年化收益率": "Err", "最大回撤": "Err", "夏普比率": "Err", "调仓次数": "Err"}

def apply_local_rebalance(asset_values, target_weights, threshold):
    total_val = asset_values.sum()
    current_vals = asset_values.copy()
    reset_indices = []
    safe_targets = target_weights.replace(0, 1e-9)
    for _ in range(10): 
        current_weights = current_vals / total_val
        rel_diffs = np.abs(current_weights - target_weights) / safe_targets
        to_trigger = (rel_diffs > threshold) & (~current_vals.index.isin(reset_indices))
        if not to_trigger.any(): break
        triggered_indices = to_trigger.index[to_trigger].tolist()
        reset_indices.extend(triggered_indices)
        for idx in triggered_indices: current_vals[idx] = target_weights[idx] * total_val
        rem_indices = [i for i in current_vals.index if i not in reset_indices]
        if not rem_indices: return total_val * target_weights
        rem_cash = total_val - current_vals[reset_indices].sum()
        current_rem_sum = asset_values[rem_indices].sum()
        if current_rem_sum > 0: ratios = asset_values[rem_indices] / current_rem_sum
        else: ratios = target_weights[rem_indices] / target_weights[rem_indices].sum()
        current_vals[rem_indices] = ratios * rem_cash
    return current_vals

def run_detailed_backtest(strategy_name, price_df, target_weights, initial_cap, threshold):
    tickers = price_df.columns
    if price_df.empty: return pd.DataFrame(), 0
    start_prices = price_df.iloc[0]
    if start_prices.isna().any():
        start_prices = price_df.bfill().iloc[0]
        if start_prices.isna().any(): return pd.DataFrame(), 0

    current_shares = (initial_cap * target_weights) / start_prices
    history = []
    last_rebalance_date = price_df.index[0]
    rebalance_count = 0
    price_df_filled = price_df.ffill()

    for i in range(len(price_df)):
        current_date = price_df.index[i]
        current_prices = price_df_filled.iloc[i]
        asset_values = current_shares * current_prices
        total_val = asset_values.sum()
        if total_val == 0 or np.isnan(total_val): continue
        current_weights = asset_values / total_val
        
        if i == 0:
            rec = {"日期": current_date, "类型": "首次配置", "净值": total_val}
            rec.update({f"{t}": f"{current_weights[t]:.2%}" for t in tickers})
            history.append(rec); continue

        do_rebalance = False
        new_values = asset_values.copy()
        
        if strategy_name == "定期再平衡(年度)":
            if (current_date - last_rebalance_date).days >= 365: 
                new_values, do_rebalance = total_val * target_weights, True
        elif strategy_name == "定期再平衡(半年度)":
            if (current_date - last_rebalance_date).days >= 180: 
                new_values, do_rebalance = total_val * target_weights, True
        elif "相对差" in strategy_name:
            rel_diffs = np.abs(current_weights - target_weights) / target_weights.replace(0, 1e-9)
            if rel_diffs.max() > threshold:
                if strategy_name == "相对差全局再平衡":
                    new_values = total_val * target_weights
                    do_rebalance = True
                elif strategy_name == "相对差混合再平衡":
                    if ((target_weights >= 0.1) & (rel_diffs > threshold)).any():
                        new_values = total_val * target_weights
                    else:
                        new_values = apply_local_rebalance(asset_values, target_weights, threshold)
                    do_rebalance = True
                elif strategy_name == "相对差局部再平衡":
                    new_values = apply_local_rebalance(asset_values, target_weights, threshold)
                    do_rebalance = True
        
        if do_rebalance:
            rebalance_count += 1
            pre_rec = {"日期": current_date, "类型": "再平衡前", "净值": total_val}
            pre_rec.update({f"{t}": f"{current_weights[t]:.2%}" for t in tickers})
            history.append(pre_rec)
            current_shares, last_rebalance_date = new_values / current_prices, current_date
            post_rec = {"日期": current_date, "类型": "再平衡后", "净值": total_val}
            post_rec.update({f"{t}": f"{(new_values/total_val)[t]:.2%}" for t in tickers})
            history.append(post_rec)
        else:
            rec = {"日期": current_date, "类型": "常规", "净值": total_val}
            rec.update({f"{t}": f"{current_weights[t]:.2%}" for t in tickers})
            history.append(rec)
    return pd.DataFrame(history), rebalance_count

# --- 3. UI 界面 ---
with st.sidebar:
    st.header("💾 配置管理")
    st.markdown("将当前的组合、基准、日期等所有设置保存到本地，或从本地加载。")
    
    current_config = {
        "benchmark": st.session_state['bi'],
        "start_date": str(st.session_state['sd']),
        "initial_funds": st.session_state['if'],
        "portfolios": st.session_state.portfolios_list
    }
    json_str = json.dumps(current_config, indent=2, ensure_ascii=False)
    st.download_button(
        label="📥 导出当前配置",
        data=json_str,
        file_name="asset_allocation_config.json",
        mime="application/json"
    )
    
    st.divider()
    
    uploaded_file = st.file_uploader("📤 导入配置", type=["json"])
    if uploaded_file is not None:
        try:
            loaded_config = json.load(uploaded_file)
            if st.button("确认覆盖当前设置"):
                st.session_state.portfolios_list = loaded_config.get("portfolios", [])
                for p in st.session_state.portfolios_list:
                    if 'id' not in p: p['id'] = str(uuid.uuid4())
                
                st.session_state['bi'] = loaded_config.get("benchmark", "SPY")
                st.session_state['sd'] = pd.to_datetime(loaded_config.get("start_date", "2022-01-01")).date()
                st.session_state['if'] = loaded_config.get("initial_funds", 10000)
                
                st.success("配置已加载！页面将自动刷新。")
                st.rerun()
        except Exception as e:
            st.error(f"配置文件解析失败: {e}")

with st.expander("🛠️ 资产配置实验室 (配置模式)", expanded=st.session_state.expanded):
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1: bench_in = st.text_input("对比基准 (决定交易日历)", key="bi")
    with c2: start_d = st.date_input("设定开始时间", key="sd")
    with c3: init_f = st.number_input("初始资金", key="if")
        
    st.divider()
    
    strategy_options = [
        "无 (Buy & Hold)", 
        "定期再平衡(年度)", 
        "定期再平衡(半年度)", 
        "相对差局部再平衡", 
        "相对差混合再平衡",
        "相对差全局再平衡"
    ]
    
    total_portfolios = len(st.session_state.portfolios_list)
    
    for i, port in enumerate(st.session_state.portfolios_list):
        if 'id' not in port: port['id'] = str(uuid.uuid4())

        h1, h2 = st.columns([8, 1])
        with h1: st.markdown(f"#### 📦 {port['name']}")
        with h2: 
            if total_portfolios > 1:
                st.button("🗑️", key=f"del_btn_{port['id']}", on_click=delete_portfolio, args=(i,))

        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        with col1: port['tickers'] = st.text_input(f"代码", port['tickers'], key=f"t_{port['id']}")
        with col2: port['weights'] = st.text_input(f"占比", port['weights'], key=f"w_{port['id']}")
        with col3: port['strat'] = st.selectbox(f"策略", strategy_options, index=strategy_options.index(port['strat']), key=f"s_{port['id']}")
        with col4: port['thr'] = st.number_input(f"阈值(%)", 1, 200, port['thr'], key=f"tr_{port['id']}")
        st.divider()

    b1, b2, _ = st.columns([1, 1, 3])
    with b1:
        if st.button("➕ 添加新组合"):
            existing_names = set(p["name"] for p in st.session_state.portfolios_list)
            new_char_code = 65 
            while f"组合 {chr(new_char_code)}" in existing_names: new_char_code += 1
            
            last_port = st.session_state.portfolios_list[-1]
            st.session_state.portfolios_list.append({
                "id": str(uuid.uuid4()), 
                "name": f"组合 {chr(new_char_code)}", 
                "tickers": last_port["tickers"],
                "weights": last_port["weights"],
                "strat": "相对差混合再平衡", 
                "thr": 40
            })
            st.rerun()
    with b2:
        # --- 核心新增：前置校验逻辑 ---
        if st.button("🚀 确定运行", type="primary"):
            validation_pass = True
            error_msgs = []
            
            for p in st.session_state.portfolios_list:
                # 兼容中文逗号，防止用户手误
                t_str = p['tickers'].replace("，", ",")
                w_str = p['weights'].replace("，", ",")
                
                # 1. 解析数据
                t_list = [x.strip() for x in t_str.split(',') if x.strip()]
                w_list = [x.strip() for x in w_str.split(',') if x.strip()]
                
                # 2. 数量匹配校验
                if len(t_list) != len(w_list):
                    validation_pass = False
                    error_msgs.append(f"❌ **{p['name']}** 配置错误：代码有 {len(t_list)} 个，但占比有 {len(w_list)} 个，请检查逗号分隔。")
                    continue # 跳过该组合的后续检查
                
                # 3. 权重归一校验
                try:
                    w_floats = [float(w) for w in w_list]
                    total_w = sum(w_floats)
                    # 容许 0.01 的浮点误差
                    if abs(total_w - 1.0) > 0.01:
                        validation_pass = False
                        error_msgs.append(f"⚠️ **{p['name']}** 权重异常：当前总和为 **{total_w:.2f}**，请调整至 **1.0**。")
                except ValueError:
                    validation_pass = False
                    error_msgs.append(f"❌ **{p['name']}** 占比格式错误：请确保输入的都是数字。")

            if validation_pass:
                # 只有全通过才收起面板并运行
                st.session_state.expanded = False
                st.rerun()
            else:
                # 有错误，直接弹窗提示，不收起面板
                for msg in error_msgs:
                    st.error(msg)

# --- 4. 执行逻辑 ---
if not st.session_state.expanded:
    with st.spinner('正在基于基准日历同步全球数据...'):
        all_tks = list(set([clean_ticker(bench_in)] + [clean_ticker(t) for p in st.session_state.portfolios_list for t in p['tickers'].replace("，", ",").split(",")]))
        df_raw = yf.download(all_tks, start=start_d - timedelta(days=20), progress=False)
        if df_raw.empty: st.error("下载失败。"); st.stop()

        if 'Adj Close' in df_raw.columns.get_level_values(0): price_data = df_raw['Adj Close']
        elif 'Close' in df_raw.columns.get_level_values(0): price_data = df_raw['Close']
        else: price_data = df_raw
            
        for tk in all_tks:
            if tk not in price_data.columns: price_data[tk] = np.nan

        bench_tk = clean_ticker(bench_in)
        if bench_tk not in price_data.columns: st.error(f"基准 {bench_tk} 数据缺失。"); st.stop()
        
        bench_valid_days = price_data[bench_tk].dropna().index
        future_days = bench_valid_days[bench_valid_days >= pd.Timestamp(start_d)]
        if future_days.empty: st.error(f"在 {start_d} 之后无基准交易数据。"); st.stop()
        market_start_day = future_days[0]
        
        df_aligned = price_data.reindex(bench_valid_days)
        df_aligned = df_aligned[df_aligned.index >= market_start_day]
        df_filled = df_aligned.ffill().bfill()

        all_port_tks = [clean_ticker(t) for p in st.session_state.portfolios_list for t in p['tickers'].replace("，", ",").split(",")]
        raw_aligned = df_aligned[all_port_tks]
        first_valid_idx = raw_aligned.apply(lambda x: x.first_valid_index())
        bottleneck_date = first_valid_idx.max()
        bottleneck_ticker = first_valid_idx.idxmax()
        actual_start_day = market_start_day
        
        if pd.notna(bottleneck_date) and (bottleneck_date - market_start_day).days > 7:
            actual_start_day = bottleneck_date
            st.warning(f"⚠️ **回测起点顺延**：{bottleneck_ticker} 上市较晚 ({bottleneck_date.date()})，起点已调整。")
        else:
            if market_start_day.date() > pd.Timestamp(start_d).date():
                st.info(f"ℹ️ **交易日对齐**：{start_d} 为非交易日，已对齐至基准首个交易日：{market_start_day.date()}。")

        final_data = df_filled[df_filled.index >= actual_start_day]
        if final_data.empty: st.error("数据不足。"); st.stop()
        first_row = final_data.iloc[[0]]
        monthly_rows = final_data.resample('ME').last()
        price_df = pd.concat([first_row, monthly_rows]).sort_index()
        price_df = price_df[~price_df.index.duplicated(keep='first')]

        comp_df = pd.DataFrame(index=price_df.index)
        bench_nav = (price_df[bench_tk] / price_df[bench_tk].iloc[0]) * init_f
        comp_df[f"基准({bench_in})"] = bench_nav
        metrics = [calculate_metrics(bench_nav, 0)]
        metrics[-1]["回测维度"] = f"基准({bench_in})"
        res_list = {}
        
        for p in st.session_state.portfolios_list:
            # 同样兼容中文逗号
            t_str = p['tickers'].replace("，", ",")
            w_str = p['weights'].replace("，", ",")
            
            p_tks = [clean_ticker(t) for t in t_str.split(",")]
            p_wts = [float(w) for w in w_str.split(",")]
            
            valid_p_tks = [t for t in p_tks if t in price_df.columns and not price_df[t].isna().all()]
            if not valid_p_tks: continue
            
            if len(valid_p_tks) < len(p_tks):
                w_series = pd.Series(p_wts[:len(p_tks)], index=p_tks)[valid_p_tks]
                w_series = w_series / w_series.sum()
            else: w_series = pd.Series(p_wts, index=p_tks)

            res_df, cnt = run_detailed_backtest(p['strat'], price_df[valid_p_tks], w_series, init_f, p['thr']/100.0)
            
            if not res_df.empty:
                df_chart = res_df.drop_duplicates(subset='日期', keep='last').copy()
                df_chart['日期'] = pd.to_datetime(df_chart['日期'])
                df_chart = df_chart.set_index('日期')
                comp_df[p['name']] = df_chart['净值']
                
                m = calculate_metrics(comp_df[p['name']], cnt)
                m["回测维度"] = p['name']
                metrics.append(m)
                
                def clean_col(c):
                    target = c.replace("占比", "").strip()
                    for tk, name in TICKER_TO_NAME.items():
                        if tk in target: return name[:5]
                    return target
                res_list[p['name']] = res_df.iloc[::-1].rename(columns=clean_col).reset_index(drop=True)

        comp_df.index.name = '日期'
        chart_data = comp_df.dropna().reset_index().melt('日期', var_name='组合', value_name='净值')
        
        base_chart = alt.Chart(chart_data).mark_line().encode(
            x=alt.X('日期', axis=alt.Axis(format='%Y-%m', title='日期')),
            y=alt.Y('净值', axis=alt.Axis(format=',.0f', title='组合净值')),
            color=alt.Color('组合', legend=alt.Legend(title="组合名称", orient='top')),
            tooltip=[alt.Tooltip('日期', format='%Y-%m-%d'), '组合', alt.Tooltip('净值', format=',.2f')]
        ).properties(
            height=500,
            title="组合净值走势 (静态视图)"
        )
        st.altair_chart(base_chart, use_container_width=True)
        
        if metrics: st.table(pd.DataFrame(metrics).set_index("回测维度"))
        st.divider()
        for lbl, data in res_list.items():
            with st.expander(f"📋 调仓明细: {lbl} (首次配置日: {actual_start_day.date()})"):
                def style_row(row):
                    if row['类型'] == '首次配置': return ['background-color: #e3f2fd; font-weight: bold'] * len(row)
                    if row['类型'] == '再平衡前': return ['background-color: #fff3e0'] * len(row)
                    if row['类型'] == '再平衡后': return ['background-color: #e8f5e9'] * len(row)
                    return [''] * len(row)
                st.dataframe(data.style.apply(style_row, axis=1).format({"净值": "{:,.2f}"}), use_container_width=True)

    if st.button("🔙 重新调整配置"): st.session_state.expanded = True; st.rerun()
