# Portfolio Backtester

基于 Streamlit 的多组合回测与再平衡策略对比工具，含一个独立的情景预测追踪页面。

## 运行

```bash
pip install -r requirements.txt
streamlit run backtest_app.py                  # 主回测应用
streamlit run iran_war_scenario_forecast.py    # 伊朗战争情景预测追踪
python3 -m unittest discover -s tests          # 单元测试
```

## 主回测应用（backtest_app.py）

- 多组合并行回测，与基准（默认 SPY）对比：KPI 卡片、累计收益图、回撤图、年度收益表、再平衡明细
- 数据源 yfinance（复权价，含分红再投资），下载结果缓存 1 小时
- 配置可导出/导入 JSON；`Backtest/` 目录下的已存配置可在侧边栏直接下拉加载
- 支持 CPI 通胀调整（FRED 数据，可手动编辑年度通胀率）
- NAV 序列可导出 CSV

### 再平衡策略

| 策略 | 规则 |
|---|---|
| Buy & Hold | 不再平衡 |
| Periodic (Annual / Semi-Annual) | 每 365 / 180 天再平衡 |
| RelDiff Full | 任一资产相对偏离 > 阈值时全局重置 |
| RelDiff Local | 仅触发资产重置到目标权重，其余按比例分摊 |
| RelDiff Mixed | 主仓（≥10%）触发时全局重置，否则局部重置 |
| Asymmetric RelDiff | 主仓（≥6%）对称阈值；小仓上涨 2.5× 阈值 / 下跌 1.25× 阈值非对称触发 |

相对偏离 = |当前权重 − 目标权重| / 目标权重，阈值 `Thr%` 按百分比输入。

### 设计决策（刻意为之，请勿"修复"）

- **月度采样**：回测期 ≥90 天时按月末取价。这是有意设计——目标是低频关注（每月看一次盘），月内波动视同日内噪声。因此最大回撤、再平衡触发都是月度口径，与日频结果有差异是预期行为，不是 bug。
- **不模拟交易成本**：策略均为低频（年均个位数次再平衡），成本影响可忽略，刻意不引入成本参数。

### 数据清洗

- 上市首日错价（如 Yahoo 上 511130.SS 首日 0.97 vs 正常 ~97 的 100 倍错误）：开头价格与下一笔相差 ≥5 倍即剔除
- 中段孤立毛刺：单笔相对前一笔跳变 ≥5 倍且下一笔即回归的数据点剔除；真实暴涨暴跌跨多笔持续，不会被误伤
- 所有剔除都会在界面上显示警告

## 情景预测追踪（iran_war_scenario_forecast.py）

2026 美伊战争五情景 × 七资产推演 vs 实际市场数据（黑线）对比。预测基准日 2026-03-20 固定锚定，实际数据经 yfinance 每小时刷新。

## 目录结构

```
backtest_app.py                 # 主应用（UI 层）
backtest_core.py                # 核心算法（纯函数，无 Streamlit 依赖）
iran_war_scenario_forecast.py   # 情景预测追踪（独立应用）
Backtest/                       # 已存回测配置（JSON）
tests/                          # 单元测试 + AppTest 冒烟测试
```
