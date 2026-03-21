# 美股 Zero-Shot 实验报告（修订版）

## 1. 实验问题与结论摘要

本实验要回答的问题是：

- `A 股训练的 LightGBM 模型能否 zero-shot 到美股`
- `在不重训、不微调的前提下，整套工程链路是否还能跑通`
- `如果能跑通，它相对于明确基线到底有多少信息增益`

当前证据最多支持的结论应当收敛到下面这一级：

- `本次实验链路可在美股样本上复用`
  数据抓取、标准化、特征重建、模型推理、白盒风控、回测和执行规划这条链路在当前样本和当前窗口里可以跑通。
- `在当前 30 只美股大盘样本上，回归模型未完全失效`
  回归 zero-shot 在当前时间窗里优于 `SPY buy-and-hold` 和 `股票池等权 buy-and-hold`。
- `不能据此证明跨市场泛化已经成立`
  因为样本只覆盖 `30` 只高流动性大盘股，而且简单 `20 日动量 top-k` 基线也能达到很强的结果，说明当前证据更像“存在有限迁移迹象”，还不是“已经证明跨市场 alpha 迁移成立”。

## 2. 测试数据定义

### 数据源

- 行情数据：`Stooq` 日线 CSV 接口
- 行业元数据：`Wikipedia S&P 500 companies`

### 股票池

- 股票池文件：
  [stock_universe_us_large_cap_30.txt](C:/Users/Apricity/Desktop/股票/configs/stock_universe_us_large_cap_30.txt)
- 股票数量：`30`
- 代表性说明：
  主要覆盖 `AAPL / MSFT / NVDA / AMZN / GOOGL / META / JPM / XOM / HD / CVX` 等高市值高流动性个股

### 时间范围

- 原始抓取区间：`2020-01-01` 到 `2025-12-31`
- 实际评估区间：`2024-01-01` 到 `2025-12-31`
- 实际预测样本日期：`2024-01-02` 到 `2025-12-23`
- 频率：`日线`
- 持有周期：`5 个交易日`

### 样本摘要

| item | value |
| --- | --- |
| source | `Stooq + Wikipedia metadata` |
| universe | `us_large_cap_30` |
| prediction rows | `14910` |
| symbols | `30` |
| date_min | `2024-01-02` |
| date_max | `2025-12-23` |

关键文件：

- 数据集：
  [us_large_cap_30_20200101_20251231_hfq_normalized.csv](C:/Users/Apricity/Desktop/股票/data/interim/stooq/universes/us_large_cap_30_20200101_20251231_hfq_normalized.csv)
- 元数据：
  [us_large_cap_30_metadata.csv](C:/Users/Apricity/Desktop/股票/data/interim/stooq/universes/us_large_cap_30_metadata.csv)

样本局限必须明确写清：

- 当前只覆盖 `30` 只大盘、高流动性股票
- 不代表全美股市场
- 对趋势性大盘股更友好，不能直接外推到中小盘、低流动性或更广股票宇宙

## 3. 训练来源与 Zero-Shot 迁移定义

### A 股训练来源

- 回归模型：
  [large_cap_50_20200101_20241231_hfq_normalized_regression_5d/model.txt](C:/Users/Apricity/Desktop/股票/model_prediction/lightgbm/artifacts/large_cap_50_20200101_20241231_hfq_normalized_regression_5d/model.txt)
- Ranking 模型：
  [large_cap_50_20200101_20241231_hfq_normalized_ranking_5d/model.txt](C:/Users/Apricity/Desktop/股票/model_prediction/lightgbm/artifacts/large_cap_50_20200101_20241231_hfq_normalized_ranking_5d/model.txt)

### Zero-shot 定义

- 训练只在 A 股数据上完成
- 美股阶段不做重训
- 不做微调
- 只做：
  1. 数据抓取与标准化
  2. 特征重建
  3. 缺失特征中性填充
  4. 直接推理
  5. 白盒风控回测

这不是严格意义上的“同分布迁移”，而是**存在特征缺口的迁移测试**。

### 当前缺失并中性填充的特征

- `turnover -> 0.0`
- `cs_rank_turnover -> 0.5`

对应摘要文件：

- 回归 zero-shot：
  [predict_summary.json](C:/Users/Apricity/Desktop/股票/model_prediction/lightgbm/artifacts/us_zeroshot_regression/predict_summary.json)
- Ranking zero-shot：
  [predict_summary.json](C:/Users/Apricity/Desktop/股票/model_prediction/lightgbm/artifacts/us_zeroshot_ranking/predict_summary.json)

## 4. 评估与回测协议

### 预测层指标

- 回归：`directional_accuracy / correlation / mae / rmse`
- Ranking：`return_correlation / top-bottom spread`

### 回测层协议

- 调仓步长：`5` 个交易日
- 成本：`10 bps` 单边
- 最低股价：`5`
- 最低成交额：`100,000,000`
- 行业约束：`industry_group` 每次最多 `1` 只
- 风格约束：`amount_bucket` 每桶最多 `2` 只
- 主权重规则：`score_confidence`

### 当前评估边界

- 当前是**单次评估窗口**
- 不是 `walk-forward`
- 也不是多窗口显著性检验

这意味着当前报告更适合解释“当前样本内的表现”，而不是给出高度外推的泛化结论。

## 5. 基线定义与局限

### 5.1 当前内部基准

必须先说明，当前 `risk_summary.json` 里的 `benchmark_total_return` 不是：

- `SPY`
- 市值加权指数
- buy-and-hold ETF
- 官方市场基准

它真正的定义是：

- 在每个调仓日
- 对当前可选股票池中的股票
- 取未来 `5` 日收益的等权平均
- 然后按调仓步长一路复利

所以它更像：

**“同一股票池、同一调仓频率下的等权内部参考线”**

它的局限是：

- 偏理想化
- 受股票池选择强影响
- 会随着筛选后的可选池变化而变化
- 不适合作为真实市场基准

这也是为什么原报告里会出现 `54%+` 到 `57%+` 的“基准收益”。这不是指数在同一口径下的真实表现，而是内部对照线的累计收益。

### 5.2 新增外部与规则基线

为了让 zero-shot 结果更可解释，这次补了 3 条明确基线：

- `SPY buy-and-hold`
  在同一评估窗口内持有 `SPY`，只做初始建仓，不做轮动。
- `股票池等权 buy-and-hold`
  在首个调仓日对可投资股票池等权建仓，然后一路持有到结束。
- `20 日动量 top-k`
  用过去 `20` 日涨幅做简单排序，选前 `5` 只，等权，沿用同样的流动性和行业/风格约束。

基线汇总文件：

- [baseline_comparison.csv](C:/Users/Apricity/Desktop/股票/risk_management/white_box/experiments/us_zeroshot_suite/baseline_comparison.csv)
- [comparison_with_baselines.csv](C:/Users/Apricity/Desktop/股票/risk_management/white_box/experiments/us_zeroshot_suite/comparison_with_baselines.csv)

## 6. 预测层结果

### 回归 zero-shot

来自：
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
- 说明回归分数在当前样本内保留了一点排序信息，但信号强度并不大

### Ranking zero-shot

来自：
[predict_summary.json](C:/Users/Apricity/Desktop/股票/model_prediction/lightgbm/artifacts/us_zeroshot_ranking/predict_summary.json)

关键指标：

- `return_correlation = -0.0101`
- `top_decile_mean_return = 0.00509`
- `bottom_decile_mean_return = 0.00692`
- `top_bottom_spread = -0.00182`

解释：

- ranking 分数与未来收益轻微负相关
- top decile 反而低于 bottom decile
- 当前证据下，ranking zero-shot 基本不成立

## 7. 策略层与基线对比结果

模型场景汇总：
[scenario_comparison.csv](C:/Users/Apricity/Desktop/股票/risk_management/white_box/experiments/us_zeroshot_suite/scenario_comparison.csv)

基线汇总：
[baseline_comparison.csv](C:/Users/Apricity/Desktop/股票/risk_management/white_box/experiments/us_zeroshot_suite/baseline_comparison.csv)

综合对比：
[comparison_with_baselines.csv](C:/Users/Apricity/Desktop/股票/risk_management/white_box/experiments/us_zeroshot_suite/comparison_with_baselines.csv)

### 7.1 模型场景

| scene | total return | annualized | max drawdown |
| --- | ---: | ---: | ---: |
| `regression_concentrated` | `76.49%` | `33.54%` | `-21.20%` |
| `regression_balanced` | `72.74%` | `31.72%` | `-19.06%` |
| `ranking_balanced` | `40.97%` | `18.89%` | `-18.05%` |
| `ranking_smoothed` | `38.29%` | `17.75%` | `-16.46%` |

### 7.2 外部与规则基线

| baseline | total return | annualized | max drawdown |
| --- | ---: | ---: | ---: |
| `momentum_topk_20d` | `74.21%` | `32.28%` | `-14.77%` |
| `universe_equal_weight_buy_hold` | `56.56%` | `25.35%` | `-18.35%` |
| `spy_buy_hold` | `47.04%` | `21.45%` | `-16.88%` |

### 7.3 如何解释这些结果

可以比较稳地说：

- 回归 zero-shot 明显优于 `SPY buy-and-hold`
- 回归 zero-shot 也优于 `股票池等权 buy-and-hold`
- ranking zero-shot 明显偏弱

但也必须一起说清：

- 一个非常简单的 `20 日动量 top-k` 基线已经能做到 `74.21%`
- 它与 `regression_balanced` 非常接近，也只略低于 `regression_concentrated`

这意味着：

- 当前不能把 zero-shot 结果解读成“只有跨市场模型迁移才能做到”
- 更合理的解读是：
  **当前美股时间窗本身对趋势/动量类信号较友好，而 zero-shot 回归模型可能学到了一部分类似结构**

## 8. 当前有效性威胁

当前报告仍然存在这些限制：

- 股票池过小，只覆盖 `30` 只大盘股
- 样本偏向高流动性龙头
- 仍然缺少更广泛股票池验证
- 回测仍然是单窗口，不是 walk-forward
- 缺少 Sharpe、IR、成本敏感性和显著性检验
- 存在特征缺口填充，不是纯同分布迁移
- 还没有完成“从代码到实验报告”的完整时间切片审计说明

因此，这份报告适合支撑的结论是：

- `当前样本上存在有限迁移迹象`

而不适合支撑：

- `已经证明跨市场泛化`
- `已经证明可实盘`
- `已经证明架构级 alpha 复用成立`

## 9. 补充对照：同窗 US 本地训练模型

为了回应“缺少 US 本地训练对照”这一点，后续还补做了同一时间窗下的对齐比较：

- [strategy_comparison_aligned_balanced.csv](C:/Users/Apricity/Desktop/股票/execution/runtime/strategy_comparison_aligned_balanced.csv)

在该对齐窗口下：

- `A股训练 -> 美股 zero-shot`：`15.74%`
- `US full-trained`：`12.90%`

这只能说明：

- 在当前对齐窗口里，zero-shot 并没有天然输给 US 本地训练版本

但仍然不能直接推出：

- zero-shot 一般优于 US 本地训练

因为样本窗口、策略容量和参数空间都还很有限。

## 10. 修订后的正式结论

本次实验更准确的结论应改写为：

- 当前工程链路可以在美股样本上复用
- 在 `30` 只大盘股、`2024-01-02` 到 `2025-12-23` 的评估窗口内，A 股训练的回归模型在美股上表现未完全失效
- ranking 模型在当前 zero-shot 设定下不成立
- 回归 zero-shot 优于 `SPY buy-and-hold` 与 `股票池等权 buy-and-hold`
- 但由于 `20 日动量 top-k` 也能得到接近甚至更强的表现，当前证据更支持“有限迁移迹象”，而不是“跨市场泛化已被证明”

## 附录 A：对审阅意见的回应

审阅文件：
[us_zeroshot_experiment_review.md](C:/Users/Apricity/Desktop/股票/other_codex_reviewer/us_zeroshot_experiment_review.md)

### A.1 关于“未来函数 / 数据泄漏风险”

**意见**  
报告没有充分证明不存在泄漏，因此风险高。

**回应**  
这个担忧是合理的，原报告在“如何排除泄漏”上写得不够充分。  
但就当前代码实现看，特征主要来自滚动统计和过去收益，目标来自 `shift(-horizon)` 的未来收益；现阶段更准确的说法应是：

- `报告没有把审计过程写充分`
- `目前没有直接代码证据表明已经发生前视泄漏`

**当前状态**  
报告层论证不足，这点接受。

**后续动作**  
补充特征生成与时间切片审计说明；后续增加 walk-forward 口径。

### A.2 关于“30 只大盘股样本偏差过强”

**意见**  
30 只大盘股不足以代表美股。

**回应**  
认同。这条批评成立。

**当前状态**  
本报告已经把这一点提升为正式有效性威胁，而不是隐含前提。

**后续动作**  
扩股票池，增加不同市值和行业覆盖。

### A.3 关于“基线不充分”

**意见**  
原始报告缺少明确外部基线，因此无法判断 zero-shot 是否真的有效。

**回应**  
认同，这也是本次修订最核心的补强点。

**当前状态**  
现已补充：

- `SPY buy-and-hold`
- `股票池等权 buy-and-hold`
- `20 日动量 top-k`
- 同窗 `US full-trained` 对照

对应文件：
[baseline_comparison.csv](C:/Users/Apricity/Desktop/股票/risk_management/white_box/experiments/us_zeroshot_suite/baseline_comparison.csv)
[strategy_comparison_aligned_balanced.csv](C:/Users/Apricity/Desktop/股票/execution/runtime/strategy_comparison_aligned_balanced.csv)

### A.4 关于“评估指标偏窄”

**意见**  
缺少 Sharpe、IR、显著性和更完整成本解释。

**回应**  
基本认同。

**当前状态**  
当前主要还是：

- `total_return`
- `annualized_return`
- `max_drawdown`
- `turnover`
- `cost`

**后续动作**  
后续加入 Sharpe、IR、成本敏感性和更正式的统计检验。

### A.5 关于“原报告结论外推过度”

**意见**  
原报告把“有限证据”写成了“架构与策略都成立”。

**回应**  
认同，这是原报告最应该修正的地方。

本次已把以下结论降调：

  - 从“项目架构已经具备跨市场复用能力”改为“当前实验链路可在美股样本上复用”
  - 从“zero-shot 到美股并没有完全失效”改为“在当前样本上观察到有限迁移迹象”
  - 从“真正保留下来的是回归模型”改为“当前实验中回归模型相对 ranking 更稳”

### A.6 关于“工程复现性不足”

**意见**  
原报告更像实验记录，不像可审计实验报告。

**回应**  
认同一半。

- 问题主要成立在“报告组织方式”
- 不完全等于“代码不可复现”

当前修订版已经补充了：

- 输入数据文件
- 股票池文件
- 预测产物
- 场景汇总
- 基线汇总
- 审阅回复

但如果要达到更强的可审计标准，后续还应补：

- 运行命令摘要
- 环境依赖版本
- 参数快照
- 数据快照 hash

更完整的可复现性模板和审计清单见：
[experiment-manifest.md](/E:/CodeX/StockMachine-260321/docs/experiment-manifest.md)

## 11. 可复现性与审计说明

当前版本仍不是审计级可复现报告。为了把这类实验从“可阅读”提升到“可复跑、可核查”，最少应补齐：

- 一条完整的运行命令，包含入口、配置文件、样本窗口、持有周期、成本假设和随机种子
- 代码版本信息，至少包括 commit hash、分支名、运行时间和执行机器标识
- 依赖版本信息，至少包括 Python 版本、关键库版本和安装方式
- 数据快照信息，至少包括原始数据文件、股票池文件、特征输入文件、标签文件和模型文件的路径与 hash
- 结果产物信息，至少包括预测明细、回测汇总、基线对照、风控输出和报告文件
- 时点口径说明，明确哪些字段是 `as_of`，哪些是 `session`，哪些是推理时可见

更完整的清单和模板见：
[experiment-manifest.md](/E:/CodeX/StockMachine-260321/docs/experiment-manifest.md)
