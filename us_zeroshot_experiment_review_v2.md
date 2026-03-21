# 美股 Zero-Shot 实验报告审阅（修订版）

## 结论先行

这版比上一版更诚实，至少把“单窗口、30 只大盘股、不是 walk-forward”写明白了，但**核心证据问题并没有被真正解决，只是被表述得更克制了**。  
当前版本最多能支持“在一个受限样本和一个单次窗口里观察到有限迁移迹象”，**仍然不能支持跨市场泛化、可实盘、或 zero-shot 优于本地训练的强结论**。

## Findings

1. **高危：没有真正证明时点正确和无泄漏，风险只是被承认了，没有被消除。**  
   修订版虽然加入了“不是 walk-forward”“存在特征缺口”等表述，但仍然没有给出 point-in-time 特征、复权因子、行业元数据、样本切片和评估切片之间严格隔离的证据链。尤其是 `A 股训练 -> 美股 zero-shot` 的推理链条仍然依赖历史重建和中性填充，报告只是在 `[第62-76行](E:/CodeX/StockMachine-260321/us_zeroshot_experiment_report%20%281%29.md#L62)` 一带解释“这不是同分布迁移”，却没有回答“会不会把未来信息带进来”。

2. **高危：30 只大盘股仍然严重偏样本，结论外推范围被夸大。**  
   你们已经把限制写得更明确了，但这并不改变样本结构本身的偏差。`us_large_cap_30`、高市值、高流动性、少行业覆盖，这组样本天然更容易让趋势/动量类方法看起来“没那么差”，见 `[第20-25行](E:/CodeX/StockMachine-260321/us_zeroshot_experiment_report%20%281%29.md#L20)` 到 `[第25行](E:/CodeX/StockMachine-260321/us_zeroshot_experiment_report%20%281%29.md#L25)`，以及 `[第53-54行](E:/CodeX/StockMachine-260321/us_zeroshot_experiment_report%20%281%29.md#L53)` 到 `[第54行](E:/CodeX/StockMachine-260321/us_zeroshot_experiment_report%20%281%29.md#L54)`。  
   这最多证明“在大盘流动性友好样本上有有限迁移迹象”，不是“跨美股市场泛化成立”。

3. **高危：基线补强了，但仍然不是足够硬的对照实验。**  
   现在有 `SPY buy-and-hold`、`universe equal weight buy-hold`、`20d top-k`、`US full-trained`，这比上一版强很多，见 `[第122-132行](E:/CodeX/StockMachine-260321/us_zeroshot_experiment_report%20%281%29.md#L122)` 到 `[第132行](E:/CodeX/StockMachine-260321/us_zeroshot_experiment_report%20%281%29.md#L132)` 和 `[第223-242行](E:/CodeX/StockMachine-260321/us_zeroshot_experiment_report%20%281%29.md#L223)` 到 `[第242行](E:/CodeX/StockMachine-260321/us_zeroshot_experiment_report%20%281%29.md#L242)`。  
   但这些基线仍然没有被严格统一到同一套约束下比较，例如同样的换手惩罚、同样的风控阈值、同样的 rebalance 规则、同样的成本口径。结果是“有更多对照”，但还不是“可审计的强对照”。

4. **中高风险：指标体系仍偏窄，结论容易被局部收益表现带偏。**  
   修订版承认当前主要还是 `total_return`、`annualized_return`、`max_drawdown`、`turnover`、`cost`，见 `[第291-305行](E:/CodeX/StockMachine-260321/us_zeroshot_experiment_report%20%281%29.md#L291)` 到 `[第305行](E:/CodeX/StockMachine-260321/us_zeroshot_experiment_report%20%281%29.md#L305)`。  
   但缺少 Sharpe、IR、成本敏感性、显著性检验、窗口稳定性和结果分布描述。于是“回归 zero-shot 优于 SPY”这种话仍然会显得比证据更硬。

5. **中高风险：叙事比证据更满，尤其是“有限迁移迹象”这句话仍容易被误读成有效泛化。**  
   第 10 节把正式结论改成“工程链路可复用”“回归模型未完全失效”“ranking 不成立”，见 `[第243-249行](E:/CodeX/StockMachine-260321/us_zeroshot_experiment_report%20%281%29.md#L243)` 到 `[第249行](E:/CodeX/StockMachine-260321/us_zeroshot_experiment_report%20%281%29.md#L249)`。  
   这个方向是对的，但措辞还是偏向“给结论”，而不是“给限制条件下的观察”。在没有 walk-forward、没有更广股票池、没有严格泄漏证明之前，最稳妥的结论仍应停在“当前样本上存在有限迁移现象，且不足以证明可推广性”。

6. **中风险：工程可复现性仍未达到可审计级别。**  
   修订版增加了“数据文件、股票池文件、预测产物、baseline 结果、审阅回复”等附件，见 `[第319-328行](E:/CodeX/StockMachine-260321/us_zeroshot_experiment_report%20%281%29.md#L319)` 到 `[第328行](E:/CodeX/StockMachine-260321/us_zeroshot_experiment_report%20%281%29.md#L328)`。这比上一版好。  
   但仍缺少运行命令、代码版本、环境依赖、参数快照、数据 hash、以及能重建每个结果的最小复跑说明，所以它仍然更像“研究记录”，不是“可审计实验报告”。

## 可以肯定的地方

修订版至少做对了三件事：把样本偏差说清了，把单窗口和 walk-forward 区分开了，也把基线补进来了。相比上一版，态度更诚实，边界更明确。  
但就审稿标准来说，**诚实不等于证据足够，边界清楚也不等于结论成立**。

## 建议的最终口径

- 不要再写“已经证明跨市场泛化成立”。
- 不要再写“可实盘”或“架构级复用已成立”。
- 可以写“在当前 30 只大盘样本和单窗口设定下，回归型模型观察到有限迁移迹象，ranking 未成立；但由于样本偏小、非 walk-forward、基线不完全统一、且仍未充分排除时点/泄漏风险，暂不能外推到更广美股市场或实盘场景”。

## Reproducibility note

如果要把这份报告再往前推一步，建议不要把“可复现性”散写在正文里，而是单独挂一个最小审计清单，至少包含：

- 运行命令
- 代码版本
- 依赖版本
- 数据快照与 hash
- universe 与参数快照
- 结果产物索引

这样正文只负责结论，审计清单只负责复跑条件，二者分离会更清楚。
