# Phase7G Non-Momentum Factor Grid Memo

日期：2026-05-17

## 问题

在 Phase7F 里，momentum/trend 已经被证明不是 `-reversal`：中期动量在 h20/h60 上开始更像一个独立慢速 sleeve。Phase7G 继续问：

> 除了 momentum，还有哪些当前数据里已经能测试、且可能和 reversal 低相关的风格因子？

本实验仍然是 validation-only，不使用 test lockbox。

## 实验设置

- universe：`top1000_clean_core_beta_full`
- 数据窗口：`2014-08-05` 至 `2019-12-31`
- 验证窗口：`2014-11-11` 至 `2019-12-31`
- stock-date panel：`923,359` 行
- style/horizon 组合：`124`
- 横截面构造：每日按因子分数做 top 20% long、bottom 20% short
- payoff：未来 h5/h10/h20/h60 的 beta-residual forward return，单位 bps
- t 口径：普通 t-stat 与 Newey-West t-stat，NW lag = `holding - 1`
- 未加入：交易成本、借券、容量、换手、组合优化约束

代码与结果：

- script：`src/stockmachine/apps/run_pure_alpha_phase7g.py`
- artifact root：`artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase7g_non_momentum_factor_grid_20260517`
- metrics：`phase7g_factor_metrics.csv`
- family summary：`phase7g_family_summary.csv`
- combo diagnostics：`phase7g_candidate_combo_diagnostics.csv`
- artifact memo：`phase7g_non_momentum_factor_grid_memo.md`

## 已测试因子族

1. Defensive / Low Risk
   - low beta
   - low realized volatility: 20d / 60d / 120d
   - low downside volatility
   - low drawdown from 60d high

2. Quality / Profitability
   - low fundamental fragility
   - low profit stress
   - low leverage pressure
   - high net margin
   - high operating cash-flow margin
   - high cash-to-assets
   - high equity-to-assets
   - low liabilities-to-assets
   - low debt-to-assets

3. Value
   - book-to-market
   - sales-to-price
   - earnings yield
   - operating cash-flow yield
   - low debt-to-market

4. Growth / Investment
   - revenue change 252d
   - net income change 252d
   - operating cash-flow change 252d
   - low asset growth 252d

5. Filing / Insider
   - low filing red flag score
   - insider net buy score
   - insider buy intensity
   - low insider sell intensity

## 主要结果

最清楚的三个候选 sleeve：

| factor | h10 mean bps | h10 NW t | h20 mean bps | h20 NW t | h60 mean bps | h60 NW t | corr vs reversal, same h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small size | 29.19 | 2.07 | 58.53 | 2.16 | 177.46 | 2.55 | h20: 0.16, h60: 0.02 |
| high cash-to-assets | 27.06 | 2.12 | 50.37 | 1.99 | 132.30 | 1.72 | h20: -0.06, h60: -0.06 |
| low beta | 33.50 | 1.68 | 67.66 | 1.67 | 198.09 | 1.61 | h20: -0.17, h60: 0.01 |

reversal baseline：

| baseline | h5 mean bps | h10 mean bps | h20 mean bps | h60 mean bps |
| --- | ---: | ---: | ---: | ---: |
| reversal 5d | 10.09 | 23.49 | 34.64 | 17.17 |

组合诊断（轻量 ad hoc，同样只在验证窗口）：

| combo | h10 mean bps | h10 NW t | h20 mean bps | h20 NW t | h60 mean bps | h60 NW t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reversal + small size + cash-to-assets + low beta | 28.31 | 6.68 | 52.80 | 6.99 | 131.25 | 5.14 |
| above + selected momentum | 26.70 | 5.48 | 45.52 | 4.08 | 116.61 | 3.52 |
| non-reversal only | 27.50 | 4.53 | 47.69 | 3.32 | 136.50 | 3.36 |

注意：组合诊断只是等权日 payoff 均值，不是可交易组合回测。它的作用是看多个 payoff stream 是否能彼此分散噪声。

## 解读

### 1. Small size 是本轮最强的非 momentum 信号

`size_small_cap` 在 h5/h10/h20/h60 都为正，且 h60 的 NW t 达到 `2.55`。这说明在当前 top1000 liquid universe 中，较小市值股票相对于较大市值股票存在明显的 beta-residual long-short payoff。

但它不应直接被当成无条件 free lunch：

- 它最容易被交易成本、容量、冲击成本侵蚀。
- 它可能混入 liquidity、sector、上市生命周期等结构暴露。
- 它和 reversal 在 h20 有轻微正相关 `0.16`，不是完全独立。

所以 small size 更像一个候选 sleeve，而不是直接上生产的 alpha。

### 2. High cash-to-assets 是最值得进入知识图谱的结构证据

`quality_high_cash_to_assets` 在 h10/h20/h60 都很强，且与 reversal 相关性很低：

- h10：`27.06 bps`，NW t `2.12`
- h20：`50.37 bps`，NW t `1.99`
- h60：`132.30 bps`，NW t `1.72`

这和我们此前讨论的 structural prior 很贴合：现金缓冲更高的公司，在市场状态恶化或估值重定价时，更可能被市场重新定价为“可存活/可选择权”的资产，而不是单纯的 falling knife。

下一步需要做 sector-neutral 版本，因为 cash-to-assets 在不同行业含义不同。

### 3. Low beta / low vol 是防御性收益源头，也能分散 reversal

`defensive_low_beta` 的均值很高，但 NW t 比 small size 和 cash-to-assets 稍弱：

- h10：`33.50 bps`，NW t `1.68`
- h20：`67.66 bps`，NW t `1.67`
- h60：`198.09 bps`，NW t `1.61`

它和 reversal 在 h10/h20 上是负相关：

- h10：`-0.14`
- h20：`-0.17`

这很有价值。它未必是最锋利的单因子，但它可能是分配风险时的稳定器：当 reversal 依赖短期均值回复时，low beta/low vol 更像防御状态下的慢变量。

### 4. Value / Growth / Filing / Insider 在当前口径下偏弱

本轮 value 整体没有跑出来。最好的 `value_low_debt_to_market` 也只是弱正，NW t 大多不到 `0.4`。这不一定说明 value 没用，更可能说明当前 MVP value 定义太粗：

- market cap 不是 enterprise value
- book / earnings / cash-flow 需要行业中性化
- 金融、保险、地产、能源等行业的资产负债表不可直接横比
- companyfacts 的季度/年度口径混合仍然粗糙

filing/insider 也偏弱。尤其 insider intensity 的有效 sessions 很少，更多是事件覆盖问题，不适合在这一版作为稳定 sleeve。

## 对当前研究路线的含义

除 momentum 以外，当前最值得进入下一版 allocator / graph prior 的是：

1. `small size`
2. `high cash-to-assets`
3. `low beta / low vol`

它们和 reversal 的关系不是简单替代：

- reversal：h10/h20 的短周期错配修复
- momentum：h20/h60 的趋势延续
- small size：长期横截面 risk / liquidity / neglectedness 暴露
- high cash-to-assets：公司结构质量与生存选择权
- low beta/low vol：防御性风险偏好状态

这更像一个多收益源头组合，而不是寻找一个万能 alpha。

## 下一步

1. 做 sector-neutral Phase7G-bis：尤其是 cash-to-assets、value、leverage、profitability。
2. 做 cost/capacity sanity check：small size 必须先过交易成本和容量。
3. 把 top candidates 接入 signed allocator：reversal、momentum、small size、cash quality、low beta 分别作为 sleeve，而不是都塞进一个单一预测值。
4. 对外部数据开口：analyst revisions / PEAD、short interest、ETF/fund flows、options skew。这些更可能解释“资金长期向巨头集中”的状态。
