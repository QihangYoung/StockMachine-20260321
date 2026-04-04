from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from stockmachine.research.p1_rigor import summarize_backtest_records
from stockmachine.research.robustness_frameworks import RobustnessFrameworkSpec


def build_time_stability_summary(
    records: pd.DataFrame,
    *,
    model_name: str,
    horizon: int,
    framework: RobustnessFrameworkSpec,
    period: str,
) -> pd.DataFrame:
    """Build time-sliced robustness summaries using framework date attribution."""

    if records.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "period",
                "period_label",
                "sessions",
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe",
                "max_drawdown",
                "benchmark_total_return",
                "mean_turnover",
                "mean_cost_bps",
                "attribution_date_column",
            ]
        )

    frame = records.copy()
    date_column = str(framework.attribution_date_column)
    if date_column not in frame.columns:
        raise ValueError(f"records missing attribution date column '{date_column}'")

    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    if period == "year":
        periods = frame[date_column].dt.to_period("Y")
    elif period == "quarter":
        periods = frame[date_column].dt.to_period("Q")
    else:
        raise ValueError("period must be 'year' or 'quarter'.")

    rows: list[dict[str, Any]] = []
    for period_key, group in frame.groupby(periods, sort=True):
        summary = summarize_backtest_records(group, horizon=horizon)
        rows.append(
            {
                "model": model_name,
                "period": period,
                "period_label": str(period_key),
                "attribution_date_column": date_column,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def build_tail_dependence_summary(
    records: pd.DataFrame,
    *,
    model_name: str,
    horizon: int,
    framework: RobustnessFrameworkSpec,
    trim_counts: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Estimate how dependent results are on a small number of extreme trades."""

    if records.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "tail_return_column",
                "tail_side",
                "trim_count",
                "sessions_before",
                "sessions_after",
                "removed_return_sum",
                "removed_return_share_of_abs_sum",
                "baseline_total_return",
                "baseline_annualized_return",
                "baseline_sharpe",
                "baseline_max_drawdown",
                "trimmed_total_return",
                "trimmed_annualized_return",
                "trimmed_sharpe",
                "trimmed_max_drawdown",
                "delta_annualized_return",
                "delta_sharpe",
                "delta_max_drawdown",
            ]
        )

    frame = records.copy()
    return_column = str(framework.tail_return_column)
    if return_column not in frame.columns:
        raise ValueError(f"records missing tail return column '{return_column}'")

    counts = tuple(int(value) for value in (trim_counts or framework.tail_trim_counts) if int(value) > 0)
    baseline = summarize_backtest_records(frame, horizon=horizon)
    total_abs_sum = float(frame[return_column].astype(float).abs().sum())

    rows: list[dict[str, Any]] = []
    sorted_desc = frame.sort_values(return_column, ascending=False, kind="mergesort")
    sorted_asc = frame.sort_values(return_column, ascending=True, kind="mergesort")

    for trim_count in counts:
        for tail_side, ordered in (("top", sorted_desc), ("bottom", sorted_asc)):
            removed = ordered.head(trim_count).copy()
            kept = frame.drop(index=removed.index)
            trimmed = summarize_backtest_records(kept, horizon=horizon) if not kept.empty else {
                "sessions": 0,
                "total_return": np.nan,
                "annualized_return": np.nan,
                "annualized_volatility": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
                "benchmark_total_return": np.nan,
                "mean_turnover": np.nan,
                "mean_cost_bps": np.nan,
            }

            removed_sum = float(removed[return_column].astype(float).sum())
            share_abs_sum = float(abs(removed_sum) / total_abs_sum) if total_abs_sum > 0 else np.nan
            rows.append(
                {
                    "model": model_name,
                    "tail_return_column": return_column,
                    "tail_side": tail_side,
                    "trim_count": int(trim_count),
                    "sessions_before": int(len(frame)),
                    "sessions_after": int(len(kept)),
                    "removed_return_sum": removed_sum,
                    "removed_return_share_of_abs_sum": share_abs_sum,
                    "baseline_total_return": baseline["total_return"],
                    "baseline_annualized_return": baseline["annualized_return"],
                    "baseline_sharpe": baseline["sharpe"],
                    "baseline_max_drawdown": baseline["max_drawdown"],
                    "trimmed_total_return": trimmed["total_return"],
                    "trimmed_annualized_return": trimmed["annualized_return"],
                    "trimmed_sharpe": trimmed["sharpe"],
                    "trimmed_max_drawdown": trimmed["max_drawdown"],
                    "delta_annualized_return": (
                        float(trimmed["annualized_return"]) - float(baseline["annualized_return"])
                        if pd.notna(trimmed["annualized_return"]) and pd.notna(baseline["annualized_return"])
                        else np.nan
                    ),
                    "delta_sharpe": (
                        float(trimmed["sharpe"]) - float(baseline["sharpe"])
                        if pd.notna(trimmed["sharpe"]) and pd.notna(baseline["sharpe"])
                        else np.nan
                    ),
                    "delta_max_drawdown": (
                        float(trimmed["max_drawdown"]) - float(baseline["max_drawdown"])
                        if pd.notna(trimmed["max_drawdown"]) and pd.notna(baseline["max_drawdown"])
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_parameter_stability_summary(
    parameter_surface: pd.DataFrame,
    *,
    metric_column: str,
    parameter_columns: Sequence[str],
    higher_is_better: bool = True,
    neighborhood_radius: int = 1,
    run_label: str | None = None,
) -> pd.DataFrame:
    """Summarize whether the best parameter point is supported by nearby points."""

    if parameter_surface.empty:
        return pd.DataFrame(
            columns=[
                "run_label",
                "metric_column",
                "higher_is_better",
                "leader_rank",
                "leader_metric",
                "leader_parameter_signature",
                "leader_parameter_count",
                "neighborhood_radius",
                "neighborhood_size",
                "neighbor_count_excluding_leader",
                "neighborhood_metric_mean",
                "neighborhood_metric_median",
                "neighborhood_metric_min",
                "neighborhood_metric_max",
                "neighborhood_metric_std",
                "leader_vs_neighbor_mean_gap",
                "leader_vs_neighbor_median_gap",
            ]
        )

    missing = [column for column in [metric_column, *parameter_columns] if column not in parameter_surface.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"parameter surface missing required columns: {joined}")

    frame = parameter_surface.copy()
    frame = frame.loc[pd.notna(frame[metric_column])].copy()
    if frame.empty:
        return build_parameter_stability_summary(
            pd.DataFrame(),
            metric_column=metric_column,
            parameter_columns=parameter_columns,
            higher_is_better=higher_is_better,
            neighborhood_radius=neighborhood_radius,
            run_label=run_label,
        )

    frame["_metric_value"] = frame[metric_column].astype(float)
    frame = frame.sort_values("_metric_value", ascending=not higher_is_better, kind="mergesort").reset_index(drop=True)
    frame["_leader_rank"] = np.arange(1, len(frame) + 1, dtype=int)

    leader = frame.iloc[0]
    leader_signature = ", ".join(f"{column}={leader[column]}" for column in parameter_columns)

    neighborhood_mask = pd.Series(True, index=frame.index)
    for column in parameter_columns:
        ordered_values = sorted(pd.Series(frame[column]).dropna().unique().tolist())
        try:
            leader_position = ordered_values.index(leader[column])
        except ValueError:
            neighborhood_mask &= False
            continue
        min_position = max(0, leader_position - int(neighborhood_radius))
        max_position = min(len(ordered_values) - 1, leader_position + int(neighborhood_radius))
        neighbor_values = set(ordered_values[min_position : max_position + 1])
        neighborhood_mask &= frame[column].isin(neighbor_values)

    neighborhood = frame.loc[neighborhood_mask].copy()
    neighbor_only = neighborhood.iloc[1:].copy() if not neighborhood.empty else neighborhood

    leader_metric = float(leader["_metric_value"])
    neighbor_mean = float(neighbor_only["_metric_value"].mean()) if not neighbor_only.empty else np.nan
    neighbor_median = float(neighbor_only["_metric_value"].median()) if not neighbor_only.empty else np.nan
    neighbor_min = float(neighbor_only["_metric_value"].min()) if not neighbor_only.empty else np.nan
    neighbor_max = float(neighbor_only["_metric_value"].max()) if not neighbor_only.empty else np.nan
    neighbor_std = float(neighbor_only["_metric_value"].std(ddof=1)) if len(neighbor_only) > 1 else np.nan

    return pd.DataFrame(
        [
            {
                "run_label": run_label or "",
                "metric_column": metric_column,
                "higher_is_better": bool(higher_is_better),
                "leader_rank": int(leader["_leader_rank"]),
                "leader_metric": leader_metric,
                "leader_parameter_signature": leader_signature,
                "leader_parameter_count": int(len(parameter_columns)),
                "neighborhood_radius": int(neighborhood_radius),
                "neighborhood_size": int(len(neighborhood)),
                "neighbor_count_excluding_leader": int(len(neighbor_only)),
                "neighborhood_metric_mean": neighbor_mean,
                "neighborhood_metric_median": neighbor_median,
                "neighborhood_metric_min": neighbor_min,
                "neighborhood_metric_max": neighbor_max,
                "neighborhood_metric_std": neighbor_std,
                "leader_vs_neighbor_mean_gap": leader_metric - neighbor_mean if pd.notna(neighbor_mean) else np.nan,
                "leader_vs_neighbor_median_gap": leader_metric - neighbor_median if pd.notna(neighbor_median) else np.nan,
            }
        ]
    )


def build_cost_execution_stress_summary(
    records: pd.DataFrame,
    *,
    model_name: str,
    horizon: int,
    framework: RobustnessFrameworkSpec,
    cost_levels_bps: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Revalue a backtest under multiple cost assumptions with slope diagnostics."""

    if records.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "cost_bps_per_side",
                "excess_total_return",
                "sessions",
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe",
                "max_drawdown",
                "benchmark_total_return",
                "mean_turnover",
                "mean_cost_bps",
                "annualized_return_slope_per_10bps",
                "sharpe_slope_per_10bps",
                "break_even_cost_bps_per_side",
            ]
        )

    levels = tuple(float(value) for value in (cost_levels_bps or framework.cost_stress_levels))
    rows: list[dict[str, Any]] = []
    for cost_bps in levels:
        summary = summarize_backtest_records(records, horizon=horizon, cost_bps_per_side=float(cost_bps))
        rows.append(
            {
                "model": model_name,
                "cost_bps_per_side": float(cost_bps),
                "excess_total_return": (
                    summary["total_return"] - summary["benchmark_total_return"]
                    if np.isfinite(summary["total_return"]) and np.isfinite(summary["benchmark_total_return"])
                    else np.nan
                ),
                **summary,
            }
        )

    frame = pd.DataFrame(rows).sort_values("cost_bps_per_side").reset_index(drop=True)
    cost_values = frame["cost_bps_per_side"].astype(float).to_numpy()
    annualized_values = frame["annualized_return"].astype(float).to_numpy()
    sharpe_values = frame["sharpe"].astype(float).to_numpy()

    annualized_slope = _linear_metric_slope(cost_values, annualized_values, scale_x=10.0)
    sharpe_slope = _linear_metric_slope(cost_values, sharpe_values, scale_x=10.0)
    break_even_cost = _estimate_break_even_cost(cost_values, annualized_values)

    frame["annualized_return_slope_per_10bps"] = annualized_slope
    frame["sharpe_slope_per_10bps"] = sharpe_slope
    frame["break_even_cost_bps_per_side"] = break_even_cost
    return frame


def build_turnover_concentration_summary(
    records: pd.DataFrame,
    *,
    model_name: str,
    trim_counts: Sequence[int] = (1, 5, 10),
) -> pd.DataFrame:
    """Summarize whether turnover and realized cost drag are concentrated in a few days."""

    if records.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "trim_count",
                "top_turnover_share",
                "top_cost_share",
                "active_day_ratio",
                "days_to_reach_50pct_turnover",
                "days_to_reach_80pct_turnover",
                "max_turnover_day",
                "mean_turnover_active_days",
            ]
        )

    frame = records.copy()
    frame["turnover"] = frame["turnover"].astype(float)
    frame["cost_bps"] = frame["cost_bps"].astype(float)
    ordered = frame.sort_values("turnover", ascending=False, kind="mergesort").reset_index(drop=True)

    total_turnover = float(ordered["turnover"].sum())
    total_cost = float(ordered["cost_bps"].sum())
    active_mask = ordered["turnover"] > 0
    active_day_ratio = float(active_mask.mean()) if len(ordered) > 0 else np.nan
    mean_turnover_active = float(ordered.loc[active_mask, "turnover"].mean()) if active_mask.any() else 0.0
    cumulative_turnover = ordered["turnover"].cumsum()
    days_to_50 = _days_to_reach_share(cumulative_turnover, total_turnover, 0.5)
    days_to_80 = _days_to_reach_share(cumulative_turnover, total_turnover, 0.8)

    rows: list[dict[str, Any]] = []
    for trim_count in tuple(int(value) for value in trim_counts if int(value) > 0):
        top = ordered.head(trim_count)
        top_turnover = float(top["turnover"].sum())
        top_cost = float(top["cost_bps"].sum())
        rows.append(
            {
                "model": model_name,
                "trim_count": int(trim_count),
                "top_turnover_share": top_turnover / total_turnover if total_turnover > 0 else np.nan,
                "top_cost_share": top_cost / total_cost if total_cost > 0 else np.nan,
                "active_day_ratio": active_day_ratio,
                "days_to_reach_50pct_turnover": days_to_50,
                "days_to_reach_80pct_turnover": days_to_80,
                "max_turnover_day": float(ordered["turnover"].max()) if not ordered.empty else np.nan,
                "mean_turnover_active_days": mean_turnover_active,
            }
        )
    return pd.DataFrame(rows)


def build_universe_stability_summary(
    universe_surface: pd.DataFrame,
    *,
    label_column: str,
    annualized_return_column: str = "annualized_return",
    sharpe_column: str = "sharpe",
    max_drawdown_column: str = "max_drawdown",
    base_selector: str | int | float | None = None,
    base_label_value: str | int | float | None = None,
    run_label: str | None = None,
) -> pd.DataFrame:
    """Summarize robustness to universe perturbations from a precomputed result table."""

    if universe_surface.empty:
        return pd.DataFrame(
            columns=[
                "run_label",
                "label_column",
                "base_label",
                "perturbation_count",
                "base_annualized_return",
                "base_sharpe",
                "base_max_drawdown",
                "perturbation_annualized_return_mean",
                "perturbation_annualized_return_median",
                "perturbation_sharpe_mean",
                "perturbation_sharpe_median",
                "perturbation_max_drawdown_mean",
                "perturbation_max_drawdown_median",
                "annualized_return_range",
                "sharpe_range",
                "max_drawdown_range",
                "worst_annualized_return_drop_vs_base",
                "worst_sharpe_drop_vs_base",
                "worst_drawdown_deepen_vs_base",
            ]
        )

    required = [label_column, annualized_return_column, sharpe_column, max_drawdown_column]
    missing = [column for column in required if column not in universe_surface.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"universe surface missing required columns: {joined}")

    frame = universe_surface.copy()
    if base_label_value is not None:
        base_frame = frame.loc[frame[label_column] == base_label_value].copy()
    elif base_selector is not None and base_selector in frame.columns:
        base_frame = frame.loc[frame[base_selector].astype(str).isin({"66", "0.04", "50000000", "50000000.0"})].copy()
    else:
        base_frame = frame.head(1).copy()

    if base_frame.empty:
        raise ValueError("Unable to resolve a base universe row for stability comparison.")

    base_row = base_frame.iloc[0]
    perturbations = frame.drop(index=base_row.name).copy()

    base_ann = float(base_row[annualized_return_column])
    base_sharpe = float(base_row[sharpe_column])
    base_dd = float(base_row[max_drawdown_column])

    ann_values = perturbations[annualized_return_column].astype(float) if not perturbations.empty else pd.Series(dtype=float)
    sharpe_values = perturbations[sharpe_column].astype(float) if not perturbations.empty else pd.Series(dtype=float)
    dd_values = perturbations[max_drawdown_column].astype(float) if not perturbations.empty else pd.Series(dtype=float)

    return pd.DataFrame(
        [
            {
                "run_label": run_label or "",
                "label_column": label_column,
                "base_label": str(base_row[label_column]),
                "perturbation_count": int(len(perturbations)),
                "base_annualized_return": base_ann,
                "base_sharpe": base_sharpe,
                "base_max_drawdown": base_dd,
                "perturbation_annualized_return_mean": float(ann_values.mean()) if not ann_values.empty else np.nan,
                "perturbation_annualized_return_median": float(ann_values.median()) if not ann_values.empty else np.nan,
                "perturbation_sharpe_mean": float(sharpe_values.mean()) if not sharpe_values.empty else np.nan,
                "perturbation_sharpe_median": float(sharpe_values.median()) if not sharpe_values.empty else np.nan,
                "perturbation_max_drawdown_mean": float(dd_values.mean()) if not dd_values.empty else np.nan,
                "perturbation_max_drawdown_median": float(dd_values.median()) if not dd_values.empty else np.nan,
                "annualized_return_range": (
                    float(ann_values.max() - ann_values.min()) if len(ann_values) > 0 else np.nan
                ),
                "sharpe_range": float(sharpe_values.max() - sharpe_values.min()) if len(sharpe_values) > 0 else np.nan,
                "max_drawdown_range": float(dd_values.max() - dd_values.min()) if len(dd_values) > 0 else np.nan,
                "worst_annualized_return_drop_vs_base": (
                    float((ann_values - base_ann).min()) if len(ann_values) > 0 else np.nan
                ),
                "worst_sharpe_drop_vs_base": (
                    float((sharpe_values - base_sharpe).min()) if len(sharpe_values) > 0 else np.nan
                ),
                "worst_drawdown_deepen_vs_base": (
                    float((dd_values - base_dd).min()) if len(dd_values) > 0 else np.nan
                ),
            }
        ]
    )


def build_selection_bias_summary(
    search_surface: pd.DataFrame,
    *,
    metric_column: str,
    higher_is_better: bool = True,
    family_column: str | None = None,
    run_label: str | None = None,
) -> pd.DataFrame:
    """Build a lightweight search-bias diagnostic from an existing search surface."""

    if search_surface.empty:
        return pd.DataFrame(
            columns=[
                "run_label",
                "metric_column",
                "trial_count",
                "family_column",
                "family_count",
                "leader_metric",
                "leader_percentile",
                "median_metric",
                "p75_metric",
                "p90_metric",
                "mean_metric",
                "std_metric",
                "leader_vs_median_gap",
                "leader_vs_p90_gap",
                "leader_zscore",
                "leader_family",
                "leader_family_trial_count",
                "leader_family_share_of_trials",
            ]
        )

    if metric_column not in search_surface.columns:
        raise ValueError(f"search surface missing metric column '{metric_column}'")

    frame = search_surface.copy()
    frame = frame.loc[pd.notna(frame[metric_column])].copy()
    if frame.empty:
        return build_selection_bias_summary(
            pd.DataFrame(),
            metric_column=metric_column,
            higher_is_better=higher_is_better,
            family_column=family_column,
            run_label=run_label,
        )

    frame["_metric_value"] = frame[metric_column].astype(float)
    ordered = frame.sort_values("_metric_value", ascending=not higher_is_better, kind="mergesort").reset_index(drop=True)
    leader = ordered.iloc[0]
    values = ordered["_metric_value"].astype(float)
    trial_count = int(len(ordered))
    mean_value = float(values.mean())
    std_value = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    median_value = float(values.median())
    p75_value = float(values.quantile(0.75))
    p90_value = float(values.quantile(0.90))
    leader_metric = float(leader["_metric_value"])
    leader_percentile = float((values <= leader_metric).mean()) if higher_is_better else float((values >= leader_metric).mean())
    leader_zscore = float((leader_metric - mean_value) / std_value) if pd.notna(std_value) and std_value > 0 else np.nan

    family_count = 0
    leader_family = ""
    leader_family_trial_count = np.nan
    leader_family_share = np.nan
    if family_column is not None and family_column in ordered.columns:
        families = ordered[family_column].astype(str)
        family_count = int(families.nunique())
        leader_family = str(leader[family_column])
        leader_family_trial_count = float((families == leader_family).sum())
        leader_family_share = float(leader_family_trial_count / trial_count) if trial_count > 0 else np.nan

    return pd.DataFrame(
        [
            {
                "run_label": run_label or "",
                "metric_column": metric_column,
                "trial_count": trial_count,
                "family_column": family_column or "",
                "family_count": family_count,
                "leader_metric": leader_metric,
                "leader_percentile": leader_percentile,
                "median_metric": median_value,
                "p75_metric": p75_value,
                "p90_metric": p90_value,
                "mean_metric": mean_value,
                "std_metric": std_value,
                "leader_vs_median_gap": leader_metric - median_value,
                "leader_vs_p90_gap": leader_metric - p90_value,
                "leader_zscore": leader_zscore,
                "leader_family": leader_family,
                "leader_family_trial_count": leader_family_trial_count,
                "leader_family_share_of_trials": leader_family_share,
            }
        ]
    )


def _linear_metric_slope(x: np.ndarray, y: np.ndarray, *, scale_x: float) -> float:
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if finite_mask.sum() < 2:
        return np.nan
    x_fit = x[finite_mask].astype(float)
    y_fit = y[finite_mask].astype(float)
    slope, _intercept = np.polyfit(x_fit, y_fit, 1)
    return float(slope * float(scale_x))


def _estimate_break_even_cost(x: np.ndarray, annualized_returns: np.ndarray) -> float:
    finite_mask = np.isfinite(x) & np.isfinite(annualized_returns)
    if finite_mask.sum() < 2:
        return np.nan
    costs = x[finite_mask].astype(float)
    returns = annualized_returns[finite_mask].astype(float)
    for idx in range(len(costs) - 1):
        left_cost, right_cost = costs[idx], costs[idx + 1]
        left_ret, right_ret = returns[idx], returns[idx + 1]
        if left_ret == 0:
            return float(left_cost)
        if left_ret > 0 >= right_ret or left_ret < 0 <= right_ret:
            if right_ret == left_ret:
                return float(left_cost)
            ratio = (0.0 - left_ret) / (right_ret - left_ret)
            return float(left_cost + ratio * (right_cost - left_cost))
    return np.nan


def _days_to_reach_share(cumulative: pd.Series, total: float, share: float) -> int:
    if total <= 0:
        return 0
    threshold = float(total * share)
    mask = cumulative >= threshold
    if not bool(mask.any()):
        return int(len(cumulative))
    return int(mask.idxmax() + 1)
