# Phase7H Sector-Neutral Factor Grid Memo

日期：2026-05-17

## 问题

Phase7G 发现三个最像可用 sleeve 的非 momentum 因子：

1. small size
2. high cash-to-assets
3. low beta / low vol

Phase7H 继续问：这些收益是否只是行业暴露、行业会计差异、或者行业权重带出来的影子？

本实验仍然是 validation-only，不使用 test lockbox。

## 实验设置

- universe：`top1000_clean_core_beta_full`
- 数据窗口：`2014-08-05` 至 `2019-12-31`
- 验证窗口：`2014-11-11` 至 `2019-12-31`
- stock-date panel：`923,359` 行
- 验证期 payoff rows：`167,349`
- tested styles：`44`
- 行业：SEC mapping 的 `sic2_sector`
- payoff：未来 h5/h10/h20/h60 beta-residual forward return，单位 bps
- Newey-West lag：`holding - 1`
- 未加入：交易成本、借券、容量、换手、组合优化约束

代码与结果：

- script：`src/stockmachine/apps/run_pure_alpha_phase7h.py`
- artifact root：`artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase7h_sector_neutral_factor_grid_20260517`
- metrics：`phase7h_sector_neutral_metrics.csv`
- robustness：`phase7h_factor_robustness.csv`
- combo diagnostics：`phase7h_candidate_sector_neutral_combo_diagnostics.csv`
- equal-vs-name-weighted sector check：`phase7h_sector_balanced_equal_vs_name_weighted.csv`

## 三种中性化口径

1. `raw`
   - 普通全市场横截面 top/bottom quintile。

2. `sector_rank`
   - 每天、每个 SIC2 行业内先对信号做 rank，再在全市场取 top/bottom quintile。
   - 作用：去掉行业间信号水平差异。

3. `sector_balanced`
   - 每天、每个有效 SIC2 行业内各自取 top/bottom quintile，再把行业 payoff 等权平均。
   - 有效行业要求至少 30 个名字。
   - 作用：更严格地压掉行业暴露。

`sector_balanced` 是诊断口径，不是生产权重建议。

## 关键结果

### 1. Small size 不是行业幻觉，反而更强

| style | raw h20 | raw NW t | sector-rank h20 | rank NW t | sector-balanced h20 | balanced NW t | sector-balanced h60 | balanced NW t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small size | 58.53 | 2.16 | 60.09 | 2.72 | 89.88 | 3.74 | 257.88 | 4.06 |

这说明在当前 universe 里，small size 的收益不是“某些行业天然小市值”带出来的；行业内部的小市值排序本身就有 payoff。

但这也是最需要成本审查的因子：small size 往往对应更高冲击成本、更弱容量、更高换手敏感性。它是强候选 sleeve，不是直接生产结论。

### 2. High cash-to-assets 仍为正，但行业中性后缩水

| style | raw h20 | raw NW t | sector-rank h20 | rank NW t | sector-balanced h20 | balanced NW t | name-weighted balanced h20 | NW t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| high cash-to-assets | 50.37 | 1.99 | 11.12 | 0.93 | 30.12 | 1.66 | 38.02 | 2.05 |

解释：cash-to-assets 不是纯行业幻觉，但它强烈依赖行业语境。行业内 rank 后显著变弱，说明“现金占资产比例”在不同行业的可比性有限；严格行业内 long/short 后仍为正，说明它仍有结构信息。

结论：cash-to-assets 可以进图谱，但要以 industry-aware 形式进入，而不是全市场裸排序。

### 3. Low beta 比 low vol 更可靠

| style | raw h20 | raw NW t | sector-rank h20 | rank NW t | sector-balanced h20 | balanced NW t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low beta | 67.66 | 1.67 | 49.88 | 2.05 | 47.58 | 1.75 |
| low vol 60d | 32.87 | 0.90 | 0.52 | 0.02 | 9.31 | 0.30 |
| low vol 120d | 41.54 | 1.04 | 1.06 | 0.04 | 4.34 | 0.13 |

low beta 在 sector-neutral 后仍然保留；low realized vol 基本消失。也就是说，“防御性 sleeve”目前更应该用 beta 暴露表达，而不是简单 realized volatility 排序。

### 4. Reversal 本身也不是简单行业效应

| baseline | raw h10 | raw NW t | sector-balanced h10 | balanced NW t | raw h20 | raw NW t | sector-balanced h20 | balanced NW t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reversal 5d | 23.49 | 2.27 | 24.70 | 3.08 | 34.64 | 1.98 | 36.55 | 2.79 |

这对 alpha 线很重要：reversal payoff 不只是“某个行业跌了之后反弹”的结果。行业内 reversal 仍然有效。

### 5. Value 仍然不可用

book-to-market、earnings yield、cash-flow yield 在当前 MVP 口径下仍偏弱，很多口径甚至为负。sector-neutral 后也没有恢复。

这更像是数据定义问题，而不是价值因子永远无效：

- market cap 不是 enterprise value
- 金融、地产、能源、保险等行业的会计口径不可直接横比
- trailing accounting tags 仍有季度/年度混合问题
- 缺少更干净的 point-in-time shares、debt、cash、preferred、minority interest、EBITDA 等结构

现阶段不建议把 value 作为 sleeve。

## 组合诊断

sector-balanced 口径下的等权 payoff stream 组合：

| combo | h10 mean | h10 NW t | h20 mean | h20 NW t | h60 mean | h60 NW t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reversal + size + cash + low beta | 27.13 | 7.89 | 51.03 | 8.22 | 128.72 | 7.70 |
| size + cash + low beta | 27.94 | 6.80 | 55.86 | 7.20 | 162.98 | 7.82 |
| size + low beta | 34.18 | 5.77 | 68.73 | 5.75 | 203.45 | 6.08 |

注意：这不是交易组合回测，只是 payoff stream 之间的等权降噪诊断。高 t-stat 来自多个低相关 payoff stream 的平均，不代表不需要交易成本和容量验证。

sector-balanced h20 相关性：

| pair | corr |
| --- | ---: |
| reversal vs size | 0.08 |
| reversal vs cash | -0.05 |
| reversal vs low beta | -0.09 |
| size vs cash | 0.32 |
| size vs low beta | -0.57 |
| cash vs low beta | -0.58 |

这说明它们更像可以分配风险的 sleeve，而不是同一个信号的重复命名。

## 结论

当前优先级应该调整为：

1. `small size`
   - 最强、最稳、sector-neutral 后更强。
   - 下一步必须做 cost/capacity/turnover sanity check。

2. `low beta`
   - 稳定、与 reversal 低相关或负相关。
   - 比 low vol 更适合作为防御 sleeve。

3. `high cash-to-assets`
   - 有结构意义，仍为正，但必须 industry-aware。
   - 适合进知识图谱，作为“公司生存缓冲/选择权”的结构证据。

4. `reversal`
   - 行业内仍然成立，继续作为短周期 sleeve。

暂不推进：

- `value`：当前数据定义不够干净。
- `low vol`：sector-neutral 后弱。
- `filing/insider`：上一轮覆盖和稳定性不足，暂不作为核心 sleeve。

## 下一步

1. 对 small size 做成本/容量/换手压力测试。
2. 把 allocator 从“预测 reversal payoff 一个值”升级为多 sleeve：reversal、momentum、size、low beta、cash quality。
3. 对 cash quality 做行业内标准化版本，避免会计口径跨行业不可比。
4. 为“资金长期向巨头集中”的状态引入外部数据：ETF/fund flows、short interest、analyst revisions、options skew、AI/tech capex/patent/news evidence。
