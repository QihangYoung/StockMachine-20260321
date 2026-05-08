# US Equities Pure Alpha Neutrality Memo

Date: 2026-05-08

Scope: validation-window research only. The test lockbox is closed again and must not be used for parameter choice, neutralization design, or accept/reject decisions.

## 1. Why This Memo Exists

The current pure-alpha candidate has become reasonably clean on the most important hygiene dimensions: ex-ante beta, long/short gross, SIC2 sector exposure, concentration, turnover, and hard ADV feasibility. However, it is not yet a fully style-neutral pure-alpha portfolio. In particular, we have not implemented true size neutrality, and several second-layer style exposures are only monitored, not controlled.

The purpose of this memo is to define:

- Which neutrality targets matter most.
- Which ones the current SOTA already satisfies.
- Which gaps should be addressed first.
- What proxy we should use for the next validation-only size/liquidity-neutral experiments.

## 2. Priority Ranking

| Priority | Neutrality / Control Target | Current View |
|---:|---|---|
| P0 | Market beta | Must be hard-neutral. Otherwise the strategy is not pure alpha. |
| P0 | Long/short gross and net dollar exposure | Must be hard-controlled. Otherwise market direction, leverage, and financing structure drift. |
| P0/P1 | Sector / industry | Must be strongly controlled. Sector rotation can easily masquerade as stock selection. |
| P1 | Size / market cap | Important gap. We do not currently have true point-in-time market-cap neutrality. |
| P1 | Liquidity | Partially controlled through universe filters and ADV floors, but not neutralized. |
| P1 | Single-name concentration | Must be controlled to avoid idiosyncratic event domination. Current cap is in place. |
| P1/P2 | Short crowding / borrow squeeze | Important for the short book, but requires PIT short-interest / borrow data we do not currently have. |
| P2 | Momentum / reversal exposure | This is partly the alpha itself, so it should be monitored rather than blindly neutralized. |
| P2 | Volatility / idiosyncratic vol | Should be monitored; high-vol names can dominate left and right tails. |
| P2 | Value / growth | Not currently controlled; needs PIT fundamental or vendor style data. |
| P2 | Quality / profitability | Non-price proxies exist, but they are not in the optimizer. |
| P2 | Leverage / balance-sheet strength | Non-price proxies exist, but they are not in the optimizer. |

## 3. Current SOTA Neutrality Status

Current product candidate checked: `adv_floor_1m`.

Validation-only target-position audit:

| Target | Status | Evidence |
|---|---|---|
| Market beta | Satisfied | Target net beta mean is approximately `-3.9e-18`; max absolute value is approximately `9.6e-16`. |
| Long/short gross | Satisfied | Daily target long gross = `1.0`, short gross = `1.0`, gross = `2.0`, net dollar approximately zero. |
| SIC2 sector | Satisfied | SIC2 exposure is effectively zero; mean L1 exposure approximately `1.1e-16`. |
| SIC4 industry | Not satisfied | SIC4 L1 exposure mean approximately `1.15`; this is not industry-neutral below SIC2. |
| Size / market cap | Not satisfied | No point-in-time market cap or shares outstanding is used in construction. |
| Liquidity | Partially satisfied | Long universe uses top1000 liquidity; short universe uses ADV30M; candidate uses hard `$1M` ADV floor. This is filtering, not neutralization. |
| Single-name concentration | Satisfied | Target max side weight is `1/30 = 3.33%`; h10 aggregate max absolute weight is also capped around that level. |
| Momentum / reversal | Intentionally not neutral | The selector deliberately has strong reversal/overextension exposure. This must be monitored, not automatically removed. |
| Volatility | Not directly neutral | Vol-adjusted momentum is used as a selector feature, but volatility exposure is not a portfolio constraint. |
| Quality / profitability | Diagnostic only | Companyfacts profit-stress proxy exists with roughly 92% position-row coverage, but is not constrained. |
| Leverage | Diagnostic only | Companyfacts leverage-pressure proxy exists with roughly 92% position-row coverage, but is not constrained. |
| Short crowding | Not satisfied | No PIT short-interest, borrow-fee, or lendable-supply history is currently available. |

## 4. The Size Problem: What We Can and Cannot Claim

True size neutrality means controlling exposure to company market capitalization:

```text
market_cap[t, i] = adjusted_price[t, i] * point_in_time_shares_outstanding[t, i]
```

Better still, a tradability-aware version would use float market cap:

```text
float_market_cap[t, i] = adjusted_price[t, i] * point_in_time_float_shares[t, i]
```

At the start of this memo, we did not have a clean point-in-time shares-outstanding or float-shares panel in the pure-alpha construction path. Phase6J has now filled an MVP true-size panel using SEC companyfacts and lagged raw closes. It is good enough for first-pass validation-only market-cap neutral experiments, but it is still not final float-market-cap data.

This naming matters. ADV is correlated with size, but it is not size:

- A mega-cap stock can temporarily have low ADV relative to its capitalization.
- A small-cap event stock can temporarily have very high ADV.
- ADV mixes market cap, turnover, volatility, news attention, and tradability.
- Liquidity rank is more robust than raw ADV, but it is still not market cap.

### Phase6J MVP True Size Update

Phase6J generated a validation-only point-in-time market-cap panel:

```text
app: stockmachine.apps.run_pure_alpha_phase6j
artifact root: artifacts/strategy_projects/us_equities_pure_alpha_h5/research/phase6j_true_size_mvp_20260508
panel: phase6j_true_size_panel_validation.csv.gz
coverage: phase6j_true_size_coverage.csv
```

Method:

```text
shares_outstanding =
    SEC companyfacts EntityCommonStockSharesOutstanding
    fallback to weighted-average basic shares
    fallback to weighted-average diluted shares

market_cap =
    lagged raw close * shares_outstanding
```

Validation-only coverage:

| Scope | Rows | Market-cap coverage | Spot-share rate | Raw-price rate |
|---|---:|---:|---:|---:|
| `adv30m_clean_core_beta_full` | 737,370 | 93.66% | 87.11% | 100.00% |
| `top1000_clean_core_beta_full` | 915,312 | 93.13% | 85.81% | 100.00% |
| All rows | 1,652,682 | 93.37% | 86.39% | 100.00% |

Current SOTA descriptive exposure:

```text
portfolio: adv_floor_1m
position rows: 87,291
market-cap rows: 79,370
market-cap position coverage: 90.93%
spot-share position rate: 83.34%
mean net market_cap_log_z exposure: 0.012
mean abs net market_cap_log_z exposure: 0.226
p90 abs net market_cap_log_z exposure: 0.461
```

Interpretation:

- We now have enough MVP true-size coverage to run a first validation-only market-cap neutral experiment.
- ADV/liquidity proxy neutral remains useful as a robustness comparison, but it is no longer the only possible next step.
- This is still not float-market-cap neutral, and the current CIK mapping remains current-symbol based.

## 5. Proposed Proxy Set

### Primary Proxy: `log_trailing_median_dollar_volume_20`

Definition:

```text
dollar_volume[t, i] = adjusted_close[t, i] * volume[t, i]
trailing_median_dollar_volume_20[t, i] =
    median(dollar_volume[t-20 : t-1, i])
adv_log[t, i] = log(trailing_median_dollar_volume_20[t, i])
adv_log_z[t, i] = cross_sectional_zscore(adv_log[t, i])
```

Why this is the first proxy:

- It is already available in the pure-alpha signal panel.
- It is point-in-time safe because the lookback is lagged.
- It directly relates to capacity and trading cost.
- It is a reasonable first-order proxy for large/liquid versus smaller/less-liquid names.

Main limitation:

- It is not market cap. It should be treated as liquidity-size proxy, not true size.

### Secondary Proxy: `liquidity_rank`

Definition:

```text
liquidity_rank[t, i] =
    rank descending by trailing_median_dollar_volume_20 among eligible names

liquidity_rank_z[t, i] =
    cross_sectional_zscore(liquidity_rank[t, i])
```

Interpretation:

- Lower rank means more liquid.
- Higher rank means less liquid.
- For readability, we can also define:

```text
liquidity_score_z[t, i] = -liquidity_rank_z[t, i]
```

Why use it:

- Rank is less sensitive to extreme ADV outliers than raw dollar volume.
- It captures whether the portfolio is systematically long or short the more-liquid end of the universe.

Main limitation:

- It is ordinal, not economic. Moving from rank 10 to 20 is not the same liquidity difference as moving from rank 900 to 910.

### Tertiary Diagnostic Only: `log_lagged_close`

Definition:

```text
price_log[t, i] = log(lagged_adjusted_close[t, i])
price_log_z[t, i] = cross_sectional_zscore(price_log[t, i])
```

Why keep it diagnostic:

- Nominal price is not size.
- It can still catch low-price-stock microstructure exposure.
- It is useful because low nominal price can interact with borrow, spreads, volatility, and retail crowding.

Recommendation:

- Do not use `price_log` as the main size proxy.
- Use it as a secondary audit column or a weak soft penalty only after ADV/rank experiments are understood.

## 6. Recommended Experiment Sequence

### Experiment A: ADV-Log Soft Neutral

Add one numeric soft-neutral exposure to the existing LP optimizer:

```text
adv_exposure =
    sum_i wL_i * adv_log_z_i
    - sum_j wS_j * adv_log_z_j
```

Add absolute-value slack:

```text
s_adv >=  adv_exposure
s_adv >= -adv_exposure
```

Objective becomes:

```text
objective =
    existing_objective
    + adv_neutral_penalty * s_adv
```

This is my preferred first run because it is simple, point-in-time safe, and economically interpretable.

### Experiment B: Liquidity-Rank Soft Neutral

Repeat the same experiment using `liquidity_score_z = -liquidity_rank_z`.

This checks whether the result is robust to using a rank-based proxy rather than raw ADV magnitude.

### Experiment C: ADV-Bucket Soft Neutral

Bucket eligible names into daily ADV quintiles or deciles, then add soft-neutral bucket exposure similar to sector soft-neutrality:

```text
bucket_exposure_g =
    sum_i wL_i * 1{adv_bucket_i = g}
    - sum_j wS_j * 1{adv_bucket_j = g}
```

Why this matters:

- It avoids assuming the relationship is linear.
- It prevents the optimizer from solving the constraint by matching averages while still concentrating one side in the extremes.

This should come after Experiment A/B because it adds more constraints and can reduce optimizer flexibility.

### Experiment D: Composite Proxy

Only after A/B/C, test a composite:

```text
size_liquidity_proxy_z =
    average(
        adv_log_z,
        liquidity_score_z
    )
```

Do not add `price_log_z` to the first composite. Price can be monitored separately.

## 7. Success Criteria

The first goal is not to improve returns. The first goal is to learn whether current SOTA is materially leaning on size/liquidity exposure.

For each validation-only run, record:

- Gross annualized return, volatility, Sharpe, max drawdown.
- 10-session, 60-session, and yearly positive rates.
- Mean target turnover and h10 aggregate turnover.
- Mean gross, long gross, short gross.
- Net beta.
- SIC2 and SIC4 exposure.
- ADV-log exposure.
- Liquidity-rank exposure.
- Price-log exposure.
- Leg attribution for long and short books.

Good outcome:

- ADV/liquidity exposure falls materially.
- Return and path quality are not severely damaged.
- Turnover does not jump enough to make the strategy economically fragile.

Bad but informative outcome:

- Return collapses after ADV/liquidity neutralization.
- This would suggest the current strategy is partly monetizing a liquidity/size-style tilt rather than pure selection.

Ambiguous outcome:

- Return is stable but exposure barely changes.
- This means the soft penalty is too weak, the proxy is redundant with existing constraints, or candidate pools are too narrow.

## 8. Implementation Recommendation

Start with a true-size MVP experiment and keep ADV/liquidity proxy as the control:

```text
Experiment 0:
    market_cap_log_z soft neutral penalty sweep

Experiment A:
    adv_log_z soft neutral penalty sweep
```

The original proxy-first plan was appropriate before Phase6J existed. Now that Phase6J has produced adequate coverage, the first production-relevant neutrality experiment should use `market_cap_log_z`; the proxy experiments should answer whether ADV/liquidity exposure is a separate effect.

For the ADV proxy run, keep the existing SOTA fixed:

```text
existing SOTA:
    beta hard match
    long gross = 1.0
    short gross = 1.0
    max side weight = 1/30
    SIC2 soft neutral penalty = 25.0
    turnover penalty = 0.005
    ADV floor = $1M

new:
    adv_log_z soft neutral penalty sweep
```

Initial penalty sweep:

```text
adv_neutral_penalty in [0.0, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
```

The penalty should not be compared numerically to the SIC2 penalty without care. SIC2 has many bucket slacks; ADV-log has one scalar slack. We should judge by achieved exposure reduction and performance path, not by the penalty number itself.

## 9. Longer-Term Data Upgrade

For true size neutrality, Phase6J now provides an MVP point-in-time market-cap panel:

```text
market_cap_log_z = zscore(log(adjusted_close * shares_outstanding_pit))
```

For the current MVP implementation, the actual price convention is more conservative:

```text
market_cap_log_z = zscore(log(lagged_raw_close * shares_outstanding_pit))
```

Preferred data source order:

- Primary vendor PIT shares outstanding / float shares, if available.
- SEC Companyfacts `EntityCommonStockSharesOutstanding`, carefully lagged by filing availability.
- Exchange or fundamentals vendor shares outstanding, with corporate-action and restatement QA.

Until float shares or a higher-grade vendor PIT market-cap feed exists, the honest label is:

```text
MVP market-cap neutral
```

not:

```text
float-market-cap neutral
```

## 10. Working Conclusion

We should not rush into test-window analysis. The validation-window next step is a clean theory-driven robustness improvement: add MVP market-cap neutrality and compare it with ADV/liquidity-proxy neutrality to see whether current SOTA is relying on an unintended size/liquidity tilt.

The new preferred first target is `market_cap_log_z` from Phase6J. `log(trailing_median_dollar_volume_20)` and `liquidity_rank` remain the robustness proxies. I would not use `lagged_close_log` as the main size proxy; it is too far from market cap and should remain diagnostic unless later evidence shows it captures a distinct and important microstructure risk.
