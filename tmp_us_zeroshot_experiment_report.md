# 美股 Zero-Shot 实验报告

## 1. 实验目标

本次实验的目标是验证：

- 当前在 A 股训练好的 LightGBM 模型，是否可以 **zero-shot 迁移到美股**
- 在 **不重训模型** 的前提下，整套系统是否仍然可以跑通
- 迁移后的表现到底是“完全失效”，还是“还能保留部分 alpha 结构”

这里的 zero-shot 定义为：

- 训练集来自 A 股历史数据
- 美股阶段只做数据抓取、特征重建、直接推理和风控回测
- 不在美股数据上做二次训练或微调

## 2. 实验环境与时间

- 实验执行日期：`2026-03-21`
- 用户时区：`Asia/Shanghai`
- 市场：`美国股票市场`
- 频率：`日线`
- 持有周期：`5 个交易日`

## 3. 数据集与股票池

### 数据源

- 行情数据：`Stooq` 日线 CSV 接口
- 行业元数据：`Wikipedia S&P 500 companies` 表格

### 股票池

- 股票池文件：[stock_universe_us_large_cap_30.txt](C:/Users/Apricity/Desktop/股票/configs/stock_universe_us_large_cap_30.txt)
- 共 `30` 只美股大盘股
- 覆盖示例：`AAPL / MSFT / NVDA / AMZN / GOOGL / META / JPM / TSLA / XOM`

### 时间区间

- 原始抓取区间：`2020-01-01` 到 `2025-12-31`
- zero-shot 评估区间：`2024-01-01` 到 `2025-12-31`
- 实际预测样本日期：`2024-01-02` 到 `2025-12-23`

### 数据文件

- 合并后的美股标准化数据：
  [us_large_cap_30_20200101_20251231_hfq_normalized.csv](C:/Users/Apricity/Desktop/股票/data/interim/stooq/universes/us_large_cap_30_20200101_20251231_hfq_normalized.csv)
- 股票池元数据：
  [us_large_cap_30_metadata.csv](C:/Users/Apricity/Desktop/股票/data/interim/stooq/universes/us_large_cap_30_metadata.csv)

说明：

- 文件名里保留了 `_hfq_` 后缀，这是沿用原 universe 命名逻辑的结果
- 对 `Stooq` 来说，这个后缀只是命名遗留，不代表真的做了 A 股意义上的后复权

## 4. 使用的模型与迁移方式

### A 股来源模型

- 回归模型：
  [large_cap_50_20200101_20241231_hfq_normalized_regression_5d/model.txt](C:/Users/Apricity/Desktop/股票/model_prediction/lightgbm/artifacts/large_cap_50_20200101_20241231_hfq_normalized_regression_5d/model.txt)
- Ranking 模型：
  [large_cap_50_20200101_20241231_hfq_normalized_ranking_5d/model.txt](C:/Users/Apricity/Desktop/股票/model_prediction/lightgbm/artifacts/large_cap_50_20200101_20241231_hfq_normalized_ranking_5d/model.txt)

### Zero-shot 过程

1. 用美股标准化 CSV 重建和 A 股一致的特征
2. 使用 A 股训练出的 `feature_columns` 对齐输入
3. 对美股缺失特征做中性填充
4. 直接调用 LightGBM 模型文件做推理
5. 将预测产物交给白盒风控和回测层

### 本次缺失并中性填充的特征

- `turnover -> 0.0`
- `cs_rank_turnover -> 0.5`

这说明当前美股 zero-shot 并不是“完全同分布迁移”，而是带有一定特征缺口的迁移测试。

## 5. Zero-shot 预测层结果

### 回归模型

文件：
[predict_summary.json](C:/Users/Apricity/Desktop/股票/model_prediction/lightgbm/artifacts/us_zeroshot_regression/predict_summary.json)

关键指标：

- `rows = 14910`
- `symbol_count = 30`
- `directional_accuracy = 0.5364`
- `correlation = 0.0308`
- `mae = 0.0306`
- `rmse = 0.0448`

解释：

- 方向准确率略高于随机
- 相关性很弱，但为正
- 这说明回归模型在美股上仍然保留了一点点可利用的排序能力

### Ranking 模型

文件：
[predict_summary.json](C:/Users/Apricity/Desktop/股票/model_prediction/lightgbm/artifacts/us_zeroshot_ranking/predict_summary.json)

关键指标：

- `rows = 14910`
- `symbol_count = 30`
- `return_correlation = -0.0101`
- `top_decile_mean_return = 0.00509`
- `bottom_decile_mean_return = 0.00692`
- `top_bottom_spread = -0.00182`

解释：

- ranking 分数和未来收益出现了轻微负相关
- top decile 的未来收益反而低于 bottom decile
- 这说明 A 股 ranking 结构迁移到美股后基本失效

## 6. 白盒风控回测设定

本次回测统一使用：

- 调仓步长：`5` 个交易日
- 成本：`10 bps` 单边交易成本
- 最低股价：`5`
- 最低成交额：`100,000,000`
- 行业约束：`industry_group` 每次最多 `1` 只
- 风格约束：`amount_bucket` 每桶最多 `2` 只

批量脚本：
[run_us_zeroshot_suite.py](C:/Users/Apricity/Desktop/股票/risk_management/white_box/scripts/run_us_zeroshot_suite.py)

实验目录：
[us_zeroshot_suite](C:/Users/Apricity/Desktop/股票/risk_management/white_box/experiments/us_zeroshot_suite)

## 7. 回测结果

汇总文件：
[scenario_comparison.csv](C:/Users/Apricity/Desktop/股票/risk_management/white_box/experiments/us_zeroshot_suite/scenario_comparison.csv)

### 结果总览

| 场景 | 总收益 | 基准收益 | 超额收益 | 最大回撤 | 年化收益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| regression_concentrated | 76.49% | 54.29% | 22.19% | -21.20% | 33.54% |
| regression_balanced | 72.74% | 57.17% | 15.57% | -19.06% | 31.72% |
| ranking_balanced | 40.97% | 57.17% | -16.21% | -18.05% | 18.89% |
| ranking_smoothed | 38.29% | 57.17% | -18.88% | -16.46% | 17.75% |

### 结论

- **回归 zero-shot 明显优于 ranking zero-shot**
- `regression_concentrated` 是本次收益最高方案
- `ranking` 两个方案都跑输基准
- `ranking_smoothed` 虽然改善了动作结构和回撤，但仍未解决 alpha 本身不足的问题

## 8. 加仓减仓行为分析

动作汇总文件：
[action_comparison.csv](C:/Users/Apricity/Desktop/股票/risk_management/white_box/experiments/us_zeroshot_suite/action_comparison.csv)

### regression_balanced

- `open = 274`
- `exit = 269`
- `add = 72`
- `reduce = 87`
- `mean_turnover = 0.6459`

解释：

- 回归平衡版仍然以换仓为主
- 但已经开始出现一定数量的加仓和减仓动作

### regression_concentrated

- `open = 187`
- `exit = 184`
- `add = 34`
- `reduce = 38`
- `mean_turnover = 0.7004`

解释：

- 集中持仓提高了收益，也提高了波动和回撤
- 动作上更偏向“高置信度换仓”

### ranking_balanced

- `open = 268`
- `exit = 263`
- `add = 102`
- `reduce = 102`
- `mean_turnover = 0.5733`

解释：

- 虽然动作数量不少，但收益端没有兑现
- 本质问题不在调仓手法，而在信号排序已经弱化

### ranking_smoothed

- `open = 95`
- `exit = 77`
- `add = 301`
- `reduce = 352`
- `hold = 1317`
- `mean_turnover = 0.3786`
- `turnover_budget_binding_rate = 0.99`

解释：

- 这是最像真实“渐进式加减仓”的方案
- 调仓行为被明显平滑
- 但信号本身不够强，所以平滑后只是让曲线更稳，没有把策略变成赢家

## 9. 曲线文件

- 净值曲线：
  [equity_curves.png](C:/Users/Apricity/Desktop/股票/risk_management/white_box/experiments/us_zeroshot_suite/equity_curves.png)
- 回撤曲线：
  [drawdown_curves.png](C:/Users/Apricity/Desktop/股票/risk_management/white_box/experiments/us_zeroshot_suite/drawdown_curves.png)
- 宽表：
  [equity_curve_wide.csv](C:/Users/Apricity/Desktop/股票/risk_management/white_box/experiments/us_zeroshot_suite/equity_curve_wide.csv)

## 10. 核心判断

### 能不能 zero-shot 到美股

能。

更准确地说：

- **系统层面完全可以**
- **模型层面要分算法看**

当前结果说明：

- 数据抓取、标准化、模型推理、信号归一化、白盒风控、回测输出这条链路都能直接复用
- A 股训练的回归模型，在美股上还有一定迁移能力
- A 股训练的 ranking 模型，在美股上当前并不可靠

### 这是不是已经代表可实盘

还不是。

原因包括：

- 股票池只有 `30` 只
- 时间段仍然有限
- 美股阶段没有重新训练或调参
- 特征里存在缺失项的中性填充
- 还没有加入更完整的执行层假设

## 11. 下一步建议

建议优先级如下：

1. 在美股上直接重训一版 `LightGBM regression`
2. 做更大的股票池，比如 `100+` 只大盘股
3. 增加美股专属特征，如 gap、隔夜收益、财报季标签
4. 做 walk-forward 验证，而不是单个评估窗口
5. 再决定 ranking 是否值得单独重构

## 12. 本次实验结论

本次实验的最终结论是：

- **项目架构已经具备跨市场复用能力**
- **zero-shot 到美股并没有完全失效**
- **真正保留下来的是回归模型，不是 ranking 模型**
- **如果继续推进美股方向，最合理的路线是“先用现有回归框架在美股重训，再做更大规模验证”**
