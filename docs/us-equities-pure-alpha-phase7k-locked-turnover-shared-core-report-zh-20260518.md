# Phase7K 锁仓换手约束 Shared-Core 策略报告

生成日期：2026-05-18

范围：validation-only 研究报告。未使用 test lockbox 表现。

## 1. 执行摘要

Phase7K 是一个 shared-capital 多因子多空策略。它不把 NAV 固定切给
不同因子 sleeve，而是维护一套共享的 long-short book；同时，每个持仓
会被拆解成若干 factor-reason lots，也就是“这笔仓位为什么存在”的理由账本。

核心候选策略为：

`shared_core_lambda_0p005_tb0p15`

它的主要设定是：

- 因子：reversal、momentum、small size、low beta、cash quality；
- 每天生成信号；
- 新增或加仓部分按因子理由拆成 lots；
- 每个 factor-reason lot 有自己的最小持有期；
- 初始建仓后，日换手硬预算为 `0.15` NAV；
- 交易成本假设为 `4 bps/side`；
- 回测路径为严格 open-to-open 日度路径。

在验证窗口 `2014-11-13` 至 `2019-12-31` 上，主候选结果如下：

| strategy | 净年化收益 | 净波动 | 净 Sharpe | 最大回撤 | 日均换手 | 日均成本 |
|---|---:|---:|---:|---:|---:|---:|
| `shared_core_lambda_0p005_tb0p15` | 21.35% | 10.84% | 1.97 | -10.77% | 0.153 | 0.61 bps/day |
| `shared_no_momentum_lambda_0p005_tb0p15` | 16.73% | 10.35% | 1.62 | -11.71% | 0.153 | 0.61 bps/day |
| `shared_slow_core_lambda_0p005_tb0p15` | 12.23% | 11.62% | 1.05 | -21.40% | 0.023 | 0.09 bps/day |

这个结果不只是“省交易成本”。相比没有强制持有约束的 Phase7J 严格路径，
`shared_core` 的日均换手从约 `0.658` 降到 `0.153`，净 Sharpe 从约 `0.55`
提升到 `1.97`。这说明 reason-level 持有机制不仅降低成本，也抑制了每日
重优化带来的大量噪声交易。

## 2. 策略思想

一个股票可能同时因为多个理由值得持有。例如一只股票可能既有短期 reversal
机会，又有 momentum 确认，同时还是较小市值、低 beta、现金充裕的公司。

如果我们把 NAV 机械地切成 reversal sleeve、momentum sleeve、size sleeve，
同一只股票可能在多个 sleeve 中重复出现，或者反过来，每份资金只能吃一个
因子暴露。这会降低资金利用效率。

Phase7K 的想法是：

- 资金层面：只维护一个共享的 long-short book；
- 理由层面：每笔仓位被拆成多个 factor-reason lots；
- 持有期约束：作用在 reason lot 上，而不是整只股票上；
- 换手约束：作用在组合整体上。

这样，同一份资金可以同时承担多个因子暴露；但当某个因子的理由过期时，
策略也能只释放那一部分理由，而不是粗暴卖掉整只股票。

## 3. 因子定义

所有股票因子都先在同一交易日、同一 SIC2 行业内做横截面标准化；如果行业内
标准化不可用，则退回到全市场日期标准化。标准化后的分数会裁剪到 `[-3, 3]`。

| score column | 原始方向 | 含义 |
|---|---|---|
| `reversal_score` | `-return_5d` | 过去 5 日跌得多更好 |
| `momentum_score` | `momentum_l120_s20` | 120 日动量，跳过最近 20 日 |
| `small_size_score` | `-market_cap_log` | 市值越小越好 |
| `low_beta_score` | `-beta` | beta 越低越好 |
| `cash_quality_score` | `cash_to_assets` | 现金/资产越高越好 |

主策略 `shared_core` 的因子权重为：

| factor | score weight |
|---|---:|
| `reversal_score` | 0.25 |
| `momentum_score` | 0.10 |
| `small_size_score` | 0.30 |
| `low_beta_score` | 0.20 |
| `cash_quality_score` | 0.15 |

每只股票的 composite score 是这些因子分数的加权平均，再在当日横截面上
重新 z-score。

## 4. 组合构建流程

每个交易日执行以下步骤：

1. 对当前 universe 计算所有因子分数。
2. 对每个 portfolio 计算 composite score。
3. 构造候选池：
   - long 候选：composite score 前 `120` 名，加上已有持仓和锁定持仓；
   - short 候选：composite score 后 `120` 名，加上已有持仓和锁定持仓。
4. 使用线性规划求解 long/short 权重。
5. 对新开仓或加仓部分，按因子支持度拆成 factor-reason lots。
6. 用严格 open-to-open 日度路径计算收益，并按实际换手扣交易成本。

主要组合约束：

| constraint | value |
|---|---:|
| long gross | 1.0 |
| short gross | 1.0 |
| total gross | 2.0 |
| dollar net exposure | 约 0 |
| 单边单票最大权重 | `1/30 = 3.33%` |
| beta 中性 | ex-ante 硬约束 |
| SIC2 行业中性 | soft penalty，`25.0` |
| 日换手预算 | 初始建仓后 `0.15` |
| turnover objective penalty | `0.005` |
| 交易成本 | `4 bps/side` |

因子理由的最小持有期：

| factor | minimum hold |
|---|---:|
| `reversal_score` | 5 sessions |
| `momentum_score` | 10 sessions |
| `small_size_score` | 20 sessions |
| `low_beta_score` | 20 sessions |
| `cash_quality_score` | 20 sessions |

初始建仓日允许完成完整建仓。此后，日换手预算硬约束生效。

## 5. Factor-Reason Lot 机制

当某只股票被新开仓或加仓时，新增的权重会按照因子支持度分配到不同的
factor-reason lots。

对 long 仓位，某因子的支持度为：

```text
factor_weight * factor_score
```

对 short 仓位，某因子的支持度为：

```text
factor_weight * (-factor_score)
```

只保留正支持度，然后归一化为权重分配比例。每个 lot 会记录：

```text
symbol, factor, weight, birth_idx, min_hold
```

如果某个 lot 还没达到最小持有期，则不能被减仓。已经过期的 lot 可以被保留、
减仓或替换。如果过期 lot 被原样保留，不会自动刷新锁仓期；只有新增或加仓的
部分会产生新的锁仓 clock。

需要强调：reason lot 只是记账机制，不是独立资金 sleeve。真实资金仍然只是一套
共享 long-short book。

## 6. 回测口径

严格路径定义如下：

| item | definition |
|---|---|
| signal session | `T` |
| rebalance | `T+1` adjusted open |
| PnL | `T+1` open 至 `T+2` open |
| cost | `sum(abs(target_weight - previous_weight)) * cost_bps_per_side` |
| benchmark | SPY open-to-open return |

这不是 forward-label 诊断，而是真实构建每日目标仓位路径，并按目标仓位变化
收取换手成本。

## 7. 主实验结果

主实验 artifact root：

`artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase7k_locked_turnover_shared_capital_20260518_tb0p15`

主要 row counts：

| artifact | rows |
|---|---:|
| panel | 923,359 |
| positions | 416,359 |
| diagnostics | 3,879 |
| skipped sessions | 0 |
| strict curve | 3,869 |
| strict metrics | 6 |

净收益严格路径结果：

| portfolio | final equity | ann. return | ann. vol | Sharpe | max DD | mean daily bps | turnover | cost bps/day |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `shared_core_lambda_0p005_tb0p15` | 2.691 | 21.35% | 10.84% | 1.970 | -10.77% | 7.91 | 0.153 | 0.61 |
| `shared_no_momentum_lambda_0p005_tb0p15` | 2.206 | 16.73% | 10.35% | 1.616 | -11.71% | 6.35 | 0.153 | 0.61 |
| `shared_slow_core_lambda_0p005_tb0p15` | 1.806 | 12.23% | 11.62% | 1.052 | -21.40% | 4.85 | 0.023 | 0.09 |

`shared_core` 的平均因子暴露：

| factor | mean exposure | positive exposure rate |
|---|---:|---:|
| `reversal_score` | 0.559 | 92.03% |
| `momentum_score` | 0.983 | 100.00% |
| `small_size_score` | 2.702 | 99.15% |
| `low_beta_score` | 0.190 | 98.92% |
| `cash_quality_score` | 1.637 | 100.00% |

可以看到，shared-core 持续保持了对五个目标因子的正暴露。其中 low-beta 暴露
较小，主要是因为优化器有硬 beta 中性约束，会压缩低 beta 维度的表达空间。

## 8. 与 Alpha SOTA 和 SPY 的对比

对比 artifact root：

`artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase7k_shared_core_vs_sota_spy_20260518`

对比窗口为共同窗口 `2014-11-13` 至 `2019-12-31`，三条曲线均 rebased 到 `1.0`。

对比口径：

- shared core：使用 Phase7K 的 `net_return`；
- alpha SOTA：使用 Phase6 daily records，重算为
  `gross_return - turnover * 4 / 10000`；
- SPY：使用 benchmark open-to-open return。

![Shared Core vs Alpha SOTA vs SPY](../artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase7k_shared_core_vs_sota_spy_20260518/shared_core_vs_alpha_sota_spy_equity_drawdown.png)

| series | final equity | ann. return | ann. vol | Sharpe | max drawdown | mean daily bps |
|---|---:|---:|---:|---:|---:|---:|
| Shared Core net 4bps | 2.684 | 21.29% | 10.84% | 1.964 | -10.77% | 7.91 |
| Alpha SOTA net 4bps | 1.712 | 11.08% | 7.49% | 1.480 | -8.69% | 4.28 |
| SPY | 1.774 | 11.86% | 13.24% | 0.896 | -19.02% | 4.83 |

单条曲线的 Newey-West t-stat，使用日收益、lag `10`：

| series | mean daily bps | NW t |
|---|---:|---:|
| Shared Core net 4bps | 7.91 | 4.14 |
| Alpha SOTA net 4bps | 4.28 | 3.02 |
| SPY | 4.83 | 2.35 |

两两收益差的 Newey-West t-stat，lag `10`：

| comparison | mean daily diff bps | NW t |
|---|---:|---:|
| Shared Core - Alpha SOTA | 3.63 | 1.65 |
| Shared Core - SPY | 3.09 | 1.21 |
| Alpha SOTA - SPY | -0.55 | -0.25 |

解读：shared-core 自身收益流的统计强度明显更高。但相对 Alpha SOTA 的
收益差 NW t 约为 `1.6`，因此表述应为“有吸引力的验证集改进候选”，而不是
“统计上已经决定性替代 SOTA”。

## 9. 复现条件

本报告生成时的仓库环境：

- 工作目录：`E:\CodeX\StockMachine-260321`
- git head：`5a0dd80`
- 关键脚本：`src/stockmachine/apps/run_pure_alpha_phase7k.py`
- 注意：生成报告时工作区存在未提交/未跟踪研究文件。仅凭 `5a0dd80`
  的干净 clone 不能完整复现，除非 Phase7K 脚本和所需 artifacts 都存在。

Python/runtime 条件：

- Python 3.10；
- 需要 `numpy`、`pandas`、`scipy`；
- 若重画对比图，需要 `matplotlib`；
- 在 PowerShell 中设置 `PYTHONPATH=src`。

必需输入 artifacts：

| input | path |
|---|---|
| universe membership | `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase1_universe_builder_20260418/phase1_candidate_universe_membership_validation.csv.gz` |
| beta panel | `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase2_beta_panel_20260418/phase2_beta_panel_validation.csv.gz` |
| size panel | `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6j_true_size_mvp_20260508/phase6j_true_size_panel_validation.csv.gz` |
| non-price panel | `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/non_price_phase1_top1000_two_factor_build_20260428/non_price_two_factor_panel.csv.gz` |
| fundamentals panel | `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/non_price_phase4_companyfacts_fundamentals_20260505/companyfacts_fundamental_panel.csv.gz` |
| SEC CIK map | `artifacts/strategy_projects/us_equities_pure_alpha_h5/research/non_price_phase0_two_factor_data_prep_20260419/sec_universe_cik_mapping.csv` |
| SEC submissions | `data/raw/sec/submissions` |
| stock daily bars | `data/silver/daily_bar/phase0_top1000_yahoo_gap_20130805_20151231_chunk*.jsonl`; `data/silver/daily_bar/phase0_top1000_sip_raw_20160104_20260416_chunk*.jsonl` |
| stock adjustment factors | `data/silver/adj_factor/phase0_top1000_yahoo_gap_20130805_20151231_adjfactor_chunk*.jsonl`; `data/silver/adj_factor/phase0_top1000_sip_adjfactor_20160104_20260416_chunk*.jsonl` |
| SPY benchmark bars | `data/silver/benchmark_index/fmf_validation_etf_bootstrap.jsonl` |
| SPY adjustment factors | `data/silver/adj_factor/fmf_validation_etf_bootstrap.jsonl` |

Alpha SOTA 对比需要：

`artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6_sota_robustness_20260507/phase6_sota_backtest_records.csv`

## 10. 复现命令

先做编译检查：

```powershell
python -m py_compile src\stockmachine\apps\run_pure_alpha_phase7k.py
```

运行主实验：

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_pure_alpha_phase7k `
  --turnover-budget-grid 0.15 `
  --cost-bps-per-side 4 `
  --output-root artifacts\strategy_projects\us_equities_pure_alpha_h5\research\phase7k_locked_turnover_shared_capital_20260518_tb0p15
```

预期主要输出：

| output | file |
|---|---|
| rollup | `phase7k_rollup.json` |
| strict curve | `phase7k_strict_daily_curve.csv` |
| strict metrics | `phase7k_strict_daily_metrics.csv` |
| turnover summary | `phase7k_turnover_summary.csv` |
| positions | `phase7k_locked_positions.csv.gz` |
| diagnostics | `phase7k_daily_diagnostics.csv` |
| factor exposure | `phase7k_factor_exposure_summary.csv` |
| memo | `phase7k_locked_turnover_memo.md` |

rollup 中应看到：

```json
{
  "panel": 923359,
  "positions": 416359,
  "diagnostics": 3879,
  "skipped": 0,
  "curve": 3869,
  "strict_metrics": 6
}
```

可选：复现换手预算网格：

```powershell
$env:PYTHONPATH='src'
python -m stockmachine.apps.run_pure_alpha_phase7k `
  --turnover-budget-grid 0.10,0.20 `
  --cost-bps-per-side 4 `
  --output-root artifacts\strategy_projects\us_equities_pure_alpha_h5\research\phase7k_locked_turnover_shared_capital_20260518_tb0p10_tb0p20
```

观察到的净 Sharpe：

| portfolio | tb0.10 | tb0.15 | tb0.20 |
|---|---:|---:|---:|
| shared core | 1.68 | 1.97 | 1.67 |
| no momentum | 1.67 | 1.62 | 1.53 |
| slow core | 1.02 | 1.05 | 0.97 |

## 11. 如何重建对比图

对比图依赖两个文件：

- Phase7K：`phase7k_strict_daily_curve.csv`
- Phase6 SOTA：`phase6_sota_backtest_records.csv`

图像输出为：

`artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase7k_shared_core_vs_sota_spy_20260518/shared_core_vs_alpha_sota_spy_equity_drawdown.png`

机器可读输出：

- `shared_core_vs_alpha_sota_spy_curves.csv`
- `shared_core_vs_alpha_sota_spy_summary.csv`
- `shared_core_vs_alpha_sota_spy_newey_west_t.csv`
- `shared_core_vs_alpha_sota_spy_pairwise_newey_west_t.csv`

对比计算规则：

```text
shared_core_return = Phase7K net_return
alpha_sota_return = Phase6 gross_return - Phase6 turnover * 4 / 10000
spy_return = benchmark open-to-open return
```

三条曲线按日期对齐，并在共同窗口内 rebased 到 `1.0`。

## 12. 风险与限制

1. 本结果只属于验证集研究，不是生产级结论。
2. 这里没有使用 test lockbox。
3. realized beta 仍然偏高。shared-core 对 SPY 的 realized beta 约 `0.15`，
   尽管优化器有 ex-ante beta 中性硬约束。
4. 成本只包含 turnover cost，没有包含 borrow、financing、spread、
   market impact、capacity。
5. 当前数据湖还不是最终 survivorship-bias-free full-market universe。
6. 相对 Alpha SOTA 的 pairwise NW t 为正，但还不够决定性。
7. 策略是在观察验证集行为后逐步形成的，因此在晋级前必须做进一步 robustness。

## 13. 建议的下一步检查

1. 冻结代码和 artifact manifest 后，只在正式晋级时运行一次 test window。
2. 仿照 Phase6 robustness，加入 borrow 与 financing stress。
3. 按年度和 rolling 60/126/252 session 审计 realized beta。
4. 做 `2/4/8/10 bps/side` 的成本压力测试。
5. 加入 ADV participation 和 short borrow availability 的 capacity 检查。
6. 按年度、季度对比 Alpha SOTA，而不只看全样本。
7. 在继续调参前冻结当前候选版本，避免 validation drift。

## 14. 结论

Phase7K 是一个值得认真推进的策略候选。它的机制有清晰金融解释：

- shared capital 让同一份资金同时捕捉多个因子共振；
- factor-reason lot 让持仓理由具备不同生命周期；
- 最小持有期和硬换手预算显著抑制过度交易；
- 在 `4 bps/side` 成本下，验证集表现明显优于此前 alpha SOTA。

但它还不是最终产品版本。下一关是正式 robustness packet、beta drift 审计、
borrow/capacity 成本压力，以及冻结后测试窗口验证。
