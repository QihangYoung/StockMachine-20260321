# Phase7I Shared-Capital Multifactor Results Memo

日期：2026-05-17

## 实验目的

Phase7I 实现了第一版 shared-capital multifactor optimizer：

> 不把 NAV 固定切给不同 sleeve，而是让同一份 long-short gross 同时承载多个 factor exposure。

因子包括：

- `reversal_score`
- `momentum_score`
- `small_size_score`
- `low_beta_score`
- `cash_quality_score`

所有 score 先做 SIC2 sector z-score，再进入 composite alpha。优化器只输出一个共享资金组合。

## 重要限制

这不是 production backtest，也不是 test lockbox 结果。

当前结果是：

- validation-only
- target-portfolio forward-label diagnostic
- beta-residual forward payoff
- costs only approximate as `target_turnover * cost_bps_per_side`
- 未做 strict daily execution path
- 未做 capacity / borrow / realistic fill / financing stress

因此它回答的是：

> 共享资金多因子 optimizer 是否值得继续推进？

而不是：

> 这是不是最终可交易收益？

## 设置

- universe：`top1000_clean_core_beta_full`
- 数据窗口：`2014-08-05` 至 `2019-12-31`
- 验证窗口：`2014-11-11` 至 `2019-12-31`
- panel rows：`923,359`
- diagnostics rows：`3,879`
- skipped sessions：`0`
- candidate pool per side：`120`
- max single-name side weight：`1/30`
- hard beta neutral
- SIC2 sector soft neutral
- turnover penalty：`0.005`
- cost assumption：`2 bps per side`

代码与结果：

- script：`src/stockmachine/apps/run_pure_alpha_phase7i.py`
- artifact root：`artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase7i_shared_capital_multifactor_20260517`
- metrics：`phase7i_metrics.csv`
- daily diagnostics：`phase7i_daily_diagnostics.csv`
- positions：`phase7i_shared_capital_positions.csv.gz`
- factor exposure summary：`phase7i_factor_exposure_summary.csv`

## 组合配置

| portfolio | factor weights |
| --- | --- |
| `shared_core_lambda_0p005` | reversal 0.25, momentum 0.10, size 0.30, low beta 0.20, cash 0.15 |
| `shared_no_momentum_lambda_0p005` | reversal 0.30, size 0.35, low beta 0.20, cash 0.15 |
| `shared_slow_core_lambda_0p005` | size 0.45, low beta 0.30, cash 0.25 |

## 主要结果：net payoff

单位：bps。net 已扣除简化 turnover cost。

| portfolio | h10 mean | h10 NW t | h20 mean | h20 NW t | h60 mean | h60 NW t | mean turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shared core | 77.49 | 5.08 | 155.60 | 5.41 | 388.08 | 4.66 | 0.655 |
| no momentum | 63.81 | 4.43 | 121.66 | 4.69 | 338.38 | 4.60 | 0.738 |
| slow core | 51.86 | 2.85 | 108.84 | 2.82 | 328.20 | 2.78 | 0.022 |

第一读法：

1. `shared_core` 是三者里最强。
2. 加入 momentum 后，h10/h20/h60 均提升。
3. 加入 momentum 后，turnover 反而低于 no-momentum。
4. slow core turnover 极低，但缺少 reversal/momentum 后，h10/h20 的统计强度明显弱。

这说明 momentum 在 shared-capital 框架下不是“占用一份 NAV”，而是在帮助选择更好的共振持仓。

## Factor exposure ledger

| portfolio | reversal | momentum | small size | low beta | cash quality |
| --- | ---: | ---: | ---: | ---: | ---: |
| shared core | 1.88 | 1.05 | 2.55 | 0.19 | 1.53 |
| no momentum | 2.11 | 0.07 | 2.66 | 0.15 | 1.33 |
| slow core | -0.07 | 0.27 | 3.03 | 0.19 | 2.11 |

这张表证明第一版确实实现了“同一份资金吃多个因子”：

- `shared_core` 同时有正 reversal、momentum、size、low beta、cash exposure。
- `no_momentum` 基本没有 momentum exposure。
- `slow_core` 几乎没有 reversal exposure，但 size / cash exposure 极强。

## Momentum 的结论

此前 Phase7F 单独看 momentum，NW t 不算很硬，所以我们把它列为 diversifier。

Phase7I 的新证据更积极：

- `shared_core` 的 momentum exposure 平均为 `1.05`
- momentum exposure positive rate 为 `98.9%`
- 相比 `no_momentum`：
  - h10 net mean：`+13.68 bps`
  - h20 net mean：`+33.94 bps`
  - h60 net mean：`+49.69 bps`
  - mean turnover：`0.655` vs `0.738`

这说明 momentum 虽然单独 sleeve 不够硬，但在共享资金组合里有增益：它帮助过滤出同时满足中期趋势、短期 reversal、size、quality 条件的名字。

更准确地说：

> momentum 不一定适合单独重仓，但适合作为 composite alpha 的慢变量确认项。

## 风险与可能的高估

当前结果很强，必须谨慎解释。

潜在高估来源：

1. 这是 forward-label diagnostic，不是 strict execution path。
2. h20/h60 forward labels 高度重叠，即使 Newey-West 已修正，也不能替代真实路径。
3. small size 暴露很强，真实交易成本和容量可能显著侵蚀收益。
4. sector penalty 当前几乎把 SIC2 exposure 压到 0，这很好，但也可能让 optimizer 在某些行业内做过度精细选择。
5. cost model 太简单，没有 borrow、spread、market impact、ADV participation。

## 当前判断

Phase7I 第一版通过了“值得继续”的门槛：

- construction rate：`100%`
- hard beta neutral：mean abs net beta 约 `1e-16`
- sector exposure：接近 0
- shared factor exposure：成立
- momentum 在 shared-capital 中有增益
- cost-adjusted target payoff：仍强为正

但还不能称为 SOTA 替代。

## 下一步

1. Strict path backtest
   - 把 Phase7I positions 接入类似 Phase4Z/Phase5E 的 daily path 评估。

2. Turnover penalty sweep
   - `0.0025, 0.005, 0.01, 0.02, 0.05`
   - 查看收益/turnover/cost 前沿。

3. Small size cost/capacity stress
   - ADV participation
   - single-name capacity
   - spread / impact proxy

4. Momentum ablation
   - 不只比较 with/without momentum，还测试不同 momentum formation：
     - `l60_s20`
     - `l120_s20`
     - `0.5*l60_s20 + 0.5*l120_s20`

5. Walk-forward factor weights
   - 先不要用全样本最优权重。
   - 用 expanding / rolling validation 来学低自由度权重。

一句话：

> shared-capital 方向成立；momentum 在这里从“弱单因子”变成了“有用确认项”。下一关是 strict path + turnover/capacity stress。
