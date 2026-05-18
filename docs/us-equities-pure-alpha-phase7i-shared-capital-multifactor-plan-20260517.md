# Phase7I Shared-Capital Multifactor Optimizer Plan

日期：2026-05-17

## 背景

此前我们把因子作为不同 sleeve 来理解：

- `reversal`：短周期错配修复
- `momentum`：中慢速趋势延续
- `small size`：横截面 size / liquidity / neglectedness 暴露
- `low beta`：防御性风险偏好暴露
- `cash-to-assets`：公司生存缓冲 / quality 结构证据

但固定 sleeve 预算有资金效率问题。同一份资金可能同时吃到多个因子，例如一只股票同时是 small size、low beta、高 cash-to-assets、短期 reversal 候选。若把 NAV 机械切成几份，会低估这种共振持仓的资本效率。

Phase7I 的目标是实现第一版：

> shared capital, separate factor ledger, one optimizer.

## 核心问题

我们不再问：

```text
reversal 分 30% NAV，size 分 25% NAV，...
```

而是问：

```text
在 beta / sector / single-name / turnover 约束下，
哪些股票能用同一份 long-short gross 同时承载最多可靠因子 exposure？
```

## 第一版设计

### 1. Stock-level factor scores

每日为每只股票计算以下 point-in-time score：

| factor | score direction | normalization | natural speed |
| --- | --- | --- | --- |
| reversal | recent loser high score | sector z-score of `-return_5d` | fast |
| momentum | intermediate winner high score | sector z-score of `momentum_l120_s20` | medium/slow |
| small size | smaller cap high score | sector z-score of `-market_cap_log` | slow |
| low beta | lower beta high score | sector z-score of `-beta` | slow/defensive |
| cash quality | higher cash-to-assets high score | sector z-score of `cash_to_assets` | structural |

使用 sector z-score 是为了让 Phase7H 的 sector-neutral 结论自然进入第一版实现。

### 2. Shared composite alpha

第一版先用固定 prior weights，而不是 walk-forward 学权重：

```text
composite =
    0.25 * reversal_score
  + 0.10 * momentum_score
  + 0.30 * small_size_score
  + 0.20 * low_beta_score
  + 0.15 * cash_quality_score
```

这不是固定 NAV 预算。它只是每只股票的 expected alpha score。优化器仍然只输出一个共享资金组合。

### 3. LP optimizer

沿用 Phase5E/6K 风格：

```text
maximize:
    score_weight * factor_score
  - turnover_penalty * abs(w_t - w_{t-1})
  - sector_penalty * abs(sector_net_exposure)

subject to:
    long weights sum = 1
    short weights sum = 1
    long beta - short beta = 0
    0 <= each side weight <= max_single_name_side_weight
```

实现上用 `scipy.optimize.linprog`：

- long candidates：composite score top pool + previous long incumbents
- short candidates：composite score bottom pool + previous short incumbents
- hard beta neutral
- SIC2 sector soft neutral
- turnover penalty inside objective

### 4. Evaluation

第一版不做完整生产回测，只做 optimizer-level validation diagnostic：

- 每日 target portfolio 的 forward beta-residual payoff：
  - h5
  - h10
  - h20
  - h60
- target turnover
- cost-adjusted diagnostic：
  - `net_payoff_bps = gross_payoff_bps - turnover * cost_bps_per_side`
- factor exposure ledger：
  - reversal exposure
  - momentum exposure
  - size exposure
  - low beta exposure
  - cash exposure
- beta / sector / candidate / nonzero diagnostics

这不是 lockbox test，也不是最终可交易 backtest。

## 第一版实验组合

先跑少量 portfolio configs，避免一次把搜索空间打开：

| portfolio | factor weights | purpose |
| --- | --- | --- |
| `shared_core` | reversal + momentum + size + low beta + cash | 主候选 |
| `shared_no_momentum` | reversal + size + low beta + cash | 检查 momentum 是否增加噪声 |
| `shared_slow_core` | size + low beta + cash | 检查慢因子共振 |

每个 portfolio 跑同一套 LP 约束。

## 判断标准

第一版通过条件不是“收益最高”，而是：

1. 构造率高，optimizer 不频繁 infeasible。
2. h10/h20/h60 payoff 为正。
3. 加入 momentum 后，不显著破坏 h20/h60 的 payoff / t-stat。
4. factor exposure ledger 显示同一持仓确实承载多个正 exposure。
5. turnover 不爆炸，cost-adjusted payoff 仍为正。
6. beta neutral 和 sector soft neutral 约束没有明显失控。

## 明确非目标

本轮不做：

- walk-forward 学 factor weights
- test lockbox 评估
- 完整真实撮合/借券/融资成本回测
- options / analyst revisions / flows 外部数据
- production promotion

如果 Phase7I 第一版成立，下一步才做：

1. turnover penalty sweep
2. cost/capacity stress
3. factor weights walk-forward
4. regime-aware exposure target
5. strict daily path / execution-level backtest
