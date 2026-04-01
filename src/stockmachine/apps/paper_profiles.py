from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from stockmachine.domain.project_paths import build_strategy_project_paths


_PROFILE_REPO_ROOT = Path(__file__).resolve().parents[3]
_BUILTIN_PROFILE_DIR = _PROFILE_REPO_ROOT / "configs" / "strategies"


@dataclass(slots=True, frozen=True)
class PaperStrategyProfile:
    """Config-driven defaults for repeatable paper/demo runs."""

    profile_id: str
    description: str
    run_name: str
    model: str
    top_k: int = 10
    horizon: int = 5
    strategy_family: str = "us_equities"
    strategy_project: str = "us_equities_h5"
    strategy_horizon_bucket: str = "h5"
    strategy_track: str | None = None
    ledger_path: str | None = None
    kill_switch_path: str | None = None
    min_close: float = 10.0
    min_median_dollar_volume_20: float = 50_000_000.0
    max_vol_20: float = 0.04
    max_positions_per_sector: int = 2
    disable_sector_neutral: bool = False
    require_market_open: bool = False
    min_buying_power_buffer: float = 0.0
    max_order_notional: float | None = None
    max_total_notional: float | None = None
    max_total_orders: int | None = None
    execution_equity_cap: float | None = None
    post_submit_poll_seconds: float = 15.0
    post_submit_poll_interval_seconds: float = 2.0
    artifact_model: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PaperStrategyProfile":
        horizon = int(payload.get("horizon", 5))
        strategy_horizon_bucket = _optional_str(payload.get("strategy_horizon_bucket")) or _default_horizon_bucket(horizon)
        strategy_family = _optional_str(payload.get("strategy_family")) or "us_equities"
        strategy_project = _optional_str(payload.get("strategy_project")) or f"{strategy_family}_{strategy_horizon_bucket}"
        workspace = build_strategy_project_paths(strategy_project)
        return cls(
            profile_id=str(payload["profile_id"]),
            description=str(payload.get("description", "")),
            run_name=str(payload.get("run_name", payload["profile_id"])),
            model=str(payload["model"]),
            top_k=int(payload.get("top_k", 10)),
            horizon=horizon,
            strategy_family=strategy_family,
            strategy_project=strategy_project,
            strategy_horizon_bucket=strategy_horizon_bucket,
            strategy_track=_optional_str(payload.get("strategy_track")),
            ledger_path=_optional_str(payload.get("ledger_path")) or str(workspace.ledger_path),
            kill_switch_path=_optional_str(payload.get("kill_switch_path")) or str(workspace.kill_switch_path),
            min_close=float(payload.get("min_close", 10.0)),
            min_median_dollar_volume_20=float(payload.get("min_median_dollar_volume_20", 50_000_000.0)),
            max_vol_20=float(payload.get("max_vol_20", 0.04)),
            max_positions_per_sector=int(payload.get("max_positions_per_sector", 2)),
            disable_sector_neutral=bool(payload.get("disable_sector_neutral", False)),
            require_market_open=bool(payload.get("require_market_open", False)),
            min_buying_power_buffer=float(payload.get("min_buying_power_buffer", 0.0)),
            max_order_notional=_optional_float(payload.get("max_order_notional")),
            max_total_notional=_optional_float(payload.get("max_total_notional")),
            max_total_orders=_optional_int(payload.get("max_total_orders")),
            execution_equity_cap=_optional_float(payload.get("execution_equity_cap")),
            post_submit_poll_seconds=float(payload.get("post_submit_poll_seconds", 15.0)),
            post_submit_poll_interval_seconds=float(payload.get("post_submit_poll_interval_seconds", 2.0)),
            artifact_model=_optional_str(payload.get("artifact_model")),
        )

    def to_arg_defaults(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "model": self.model,
            "top_k": self.top_k,
            "horizon": self.horizon,
            "strategy_family": self.strategy_family,
            "strategy_project": self.strategy_project,
            "strategy_horizon_bucket": self.strategy_horizon_bucket,
            "strategy_track": self.strategy_track,
            "ledger_path": self.ledger_path,
            "kill_switch_path": self.kill_switch_path,
            "min_close": self.min_close,
            "min_median_dollar_volume_20": self.min_median_dollar_volume_20,
            "max_vol_20": self.max_vol_20,
            "max_positions_per_sector": self.max_positions_per_sector,
            "disable_sector_neutral": self.disable_sector_neutral,
            "require_market_open": self.require_market_open,
            "min_buying_power_buffer": self.min_buying_power_buffer,
            "max_order_notional": self.max_order_notional,
            "max_total_notional": self.max_total_notional,
            "max_total_orders": self.max_total_orders,
            "execution_equity_cap": self.execution_equity_cap,
            "post_submit_poll_seconds": self.post_submit_poll_seconds,
            "post_submit_poll_interval_seconds": self.post_submit_poll_interval_seconds,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "description": self.description,
            "run_name": self.run_name,
            "model": self.model,
            "top_k": self.top_k,
            "horizon": self.horizon,
            "strategy_family": self.strategy_family,
            "strategy_project": self.strategy_project,
            "strategy_horizon_bucket": self.strategy_horizon_bucket,
            "strategy_track": self.strategy_track,
            "ledger_path": self.ledger_path,
            "kill_switch_path": self.kill_switch_path,
            "min_close": self.min_close,
            "min_median_dollar_volume_20": self.min_median_dollar_volume_20,
            "max_vol_20": self.max_vol_20,
            "max_positions_per_sector": self.max_positions_per_sector,
            "disable_sector_neutral": self.disable_sector_neutral,
            "require_market_open": self.require_market_open,
            "min_buying_power_buffer": self.min_buying_power_buffer,
            "max_order_notional": self.max_order_notional,
            "max_total_notional": self.max_total_notional,
            "max_total_orders": self.max_total_orders,
            "execution_equity_cap": self.execution_equity_cap,
            "post_submit_poll_seconds": self.post_submit_poll_seconds,
            "post_submit_poll_interval_seconds": self.post_submit_poll_interval_seconds,
            "artifact_model": self.artifact_model,
        }


def list_builtin_strategy_profiles() -> tuple[str, ...]:
    """Return available built-in paper strategy profile names."""

    if not _BUILTIN_PROFILE_DIR.exists():
        return ()
    return tuple(sorted(_profile_path_to_builtin_id(path) for path in _iter_builtin_profile_paths()))


def load_strategy_profile(profile_ref: str | Path) -> PaperStrategyProfile:
    """Load one paper strategy profile from a built-in name or JSON file path."""

    profile_path = resolve_strategy_profile_path(profile_ref)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Strategy profile must be a JSON object: {profile_path}")
    return PaperStrategyProfile.from_mapping(payload)


def resolve_strategy_profile_path(profile_ref: str | Path) -> Path:
    """Resolve a strategy profile reference to a concrete JSON file path."""

    candidate = Path(profile_ref)
    if candidate.exists():
        return candidate.resolve()

    builtin_relative_candidates = [candidate]
    if candidate.suffix != ".json":
        builtin_relative_candidates.append(candidate.with_suffix(".json"))
    for relative_candidate in builtin_relative_candidates:
        builtin_candidate = (_BUILTIN_PROFILE_DIR / relative_candidate).resolve()
        if builtin_candidate.exists():
            return builtin_candidate

    basename = candidate.stem if candidate.suffix else candidate.name
    legacy_candidates = _legacy_builtin_aliases().get(basename, ())
    if len(legacy_candidates) == 1:
        return legacy_candidates[0]
    if len(legacy_candidates) > 1:
        choices = ", ".join(sorted(_profile_path_to_builtin_id(path) for path in legacy_candidates))
        raise FileNotFoundError(
            f"Strategy profile '{profile_ref}' is ambiguous. Use one of: {choices}."
        )

    available = ", ".join(list_builtin_strategy_profiles())
    raise FileNotFoundError(
        f"Strategy profile '{profile_ref}' was not found. Available built-ins: {available or '(none)'}."
    )


def _iter_builtin_profile_paths() -> tuple[Path, ...]:
    return tuple(sorted(_BUILTIN_PROFILE_DIR.rglob("*.json")))


def _profile_path_to_builtin_id(path: Path) -> str:
    return path.relative_to(_BUILTIN_PROFILE_DIR).with_suffix("").as_posix()


def _legacy_builtin_aliases() -> dict[str, tuple[Path, ...]]:
    aliases: dict[str, list[Path]] = {}
    for path in _iter_builtin_profile_paths():
        aliases.setdefault(path.stem, []).append(path)
    return {name: tuple(paths) for name, paths in aliases.items()}


def _default_horizon_bucket(horizon: int) -> str:
    return f"h{max(int(horizon), 1)}"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
