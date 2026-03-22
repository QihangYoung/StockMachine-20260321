from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from stockmachine.research.builders.common import FEATURE_COLUMNS, prepare_model_frame

TORCH_IMPORT_ERROR = (
    "PyTorch is required for the sequence-model builders. "
    "Install the optional 'sequence' dependencies or install torch directly."
)

SEQUENCE_FEATURE_COLUMNS: tuple[str, ...] = (
    "intraday_return",
    "range_pct",
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "close_ma5_gap",
    "close_ma10_gap",
    "close_ma20_gap",
    "close_ma60_gap",
    "volume_ma5_ratio",
    "volume_ma20_ratio",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "breakout_20d",
    "distance_to_low_20d",
    "price_position_20d",
    "volume_zscore_20d",
    "market_return_1d",
    "market_return_5d",
    "relative_return_1d",
    "relative_return_5d",
    "cs_rank_return_1d",
    "cs_rank_return_5d",
    "cs_rank_return_20d",
    "cs_rank_volume_ma5_ratio",
    "cs_rank_dollar_volume",
    "dollar_volume_ma5_ratio",
)

_MAX_SEQUENCE_ROLLING_WINDOW = 60


def _load_torch() -> tuple[object, object]:
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
    except ImportError as exc:  # pragma: no cover - exercised via explicit dependency test
        raise ImportError(TORCH_IMPORT_ERROR) from exc
    return torch, nn


def _coerce_numeric_frame(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    working = frame.copy()
    for column in feature_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    return working


def _ensure_sequence_source_columns(frame: pd.DataFrame) -> pd.DataFrame:
    working = prepare_model_frame(frame.copy())
    fallback_close = pd.to_numeric(working.get("close"), errors="coerce")
    fallback_prev_close = pd.to_numeric(working.get("prev_close"), errors="coerce")
    fallback_gap = pd.to_numeric(working.get("gap_1"), errors="coerce")

    if "open" not in working.columns:
        if fallback_prev_close is not None and fallback_gap is not None:
            working["open"] = fallback_prev_close * (1.0 + fallback_gap.fillna(0.0))
        else:
            working["open"] = fallback_close
    if "high" not in working.columns:
        working["high"] = fallback_close
    if "low" not in working.columns:
        working["low"] = fallback_close
    if "volume" not in working.columns:
        if "dollar_volume" in working.columns:
            dollar_volume = pd.to_numeric(working["dollar_volume"], errors="coerce")
            working["volume"] = dollar_volume / fallback_close.replace(0.0, np.nan)
        else:
            working["volume"] = np.nan
    if "dollar_volume" not in working.columns:
        working["dollar_volume"] = pd.to_numeric(working["close"], errors="coerce") * pd.to_numeric(
            working["volume"], errors="coerce"
        )

    for column in ("open", "high", "low", "close", "volume", "dollar_volume", "ret_1d", "mom_5", "mom_10", "mom_20"):
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    return working


def engineer_sequence_features(frame: pd.DataFrame) -> pd.DataFrame:
    working = _ensure_sequence_source_columns(frame)
    group = working.groupby("symbol", group_keys=False)

    working["intraday_return"] = working["close"] / working["open"].replace(0.0, np.nan) - 1.0
    working["range_pct"] = working["high"] / working["low"].replace(0.0, np.nan) - 1.0
    working["return_1d"] = pd.to_numeric(working.get("ret_1d"), errors="coerce")
    working["return_5d"] = pd.to_numeric(working.get("mom_5"), errors="coerce")
    working["return_10d"] = pd.to_numeric(working.get("mom_10"), errors="coerce")
    working["return_20d"] = pd.to_numeric(working.get("mom_20"), errors="coerce")

    for window in (5, 10, 20, 60):
        rolling_close = group["close"].rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
        working[f"close_ma{window}_gap"] = working["close"] / rolling_close.replace(0.0, np.nan) - 1.0

    for window in (5, 20):
        rolling_volume = group["volume"].rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
        working[f"volume_ma{window}_ratio"] = working["volume"] / rolling_volume.replace(0.0, np.nan)

    for window in (5, 10, 20):
        working[f"volatility_{window}d"] = (
            group["return_1d"].rolling(window, min_periods=window).std(ddof=0).reset_index(level=0, drop=True)
        )

    high_20 = group["high"].rolling(20, min_periods=20).max().reset_index(level=0, drop=True)
    low_20 = group["low"].rolling(20, min_periods=20).min().reset_index(level=0, drop=True)
    working["breakout_20d"] = working["close"] / high_20.replace(0.0, np.nan) - 1.0
    working["distance_to_low_20d"] = working["close"] / low_20.replace(0.0, np.nan) - 1.0
    working["price_position_20d"] = (working["close"] - low_20) / (high_20 - low_20).replace(0.0, np.nan)

    volume_mean_20 = group["volume"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    volume_std_20 = group["volume"].rolling(20, min_periods=20).std(ddof=0).reset_index(level=0, drop=True)
    working["volume_zscore_20d"] = (working["volume"] - volume_mean_20) / volume_std_20.replace(0.0, np.nan)

    dollar_volume_mean_5 = (
        group["dollar_volume"].rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    )
    working["dollar_volume_ma5_ratio"] = working["dollar_volume"] / dollar_volume_mean_5.replace(0.0, np.nan)

    date_group = working.groupby("date", group_keys=False)
    working["market_return_1d"] = date_group["return_1d"].transform("mean")
    working["market_return_5d"] = date_group["return_5d"].transform("mean")
    working["relative_return_1d"] = working["return_1d"] - working["market_return_1d"]
    working["relative_return_5d"] = working["return_5d"] - working["market_return_5d"]
    working["cs_rank_return_1d"] = date_group["return_1d"].rank(pct=True, method="average")
    working["cs_rank_return_5d"] = date_group["return_5d"].rank(pct=True, method="average")
    working["cs_rank_return_20d"] = date_group["return_20d"].rank(pct=True, method="average")
    working["cs_rank_volume_ma5_ratio"] = date_group["volume_ma5_ratio"].rank(pct=True, method="average")
    working["cs_rank_dollar_volume"] = date_group["dollar_volume"].rank(pct=True, method="average")

    working = working.replace([np.inf, -np.inf], np.nan)
    return working


def _compute_feature_stats(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for column in feature_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        mean = float(values.mean()) if values.notna().any() else 0.0
        std = float(values.std(ddof=0)) if values.notna().any() else 1.0
        if not np.isfinite(std) or std <= 1e-12:
            std = 1.0
        means[column] = mean if np.isfinite(mean) else 0.0
        stds[column] = std
    return means, stds


def _normalize_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    means: dict[str, float],
    stds: dict[str, float],
) -> pd.DataFrame:
    working = _coerce_numeric_frame(frame, feature_columns).copy()
    for column in feature_columns:
        mean = float(means.get(column, 0.0))
        std = float(stds.get(column, 1.0))
        if not np.isfinite(std) or std <= 1e-12:
            std = 1.0
        values = pd.to_numeric(working[column], errors="coerce").fillna(mean)
        working[column] = (values - mean) / std
    return working


def _build_training_sequences(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    prepared = prepare_model_frame(frame)
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    for _, group in prepared.groupby("symbol", sort=False):
        group = group.reset_index(drop=True)
        feature_values = group[list(feature_columns)].to_numpy(dtype=np.float32)
        target_values = pd.to_numeric(group["target"], errors="coerce").to_numpy(dtype=np.float32)
        for end_index in range(lookback - 1, len(group)):
            sequences.append(feature_values[end_index - lookback + 1 : end_index + 1])
            targets.append(float(target_values[end_index]))

    if not sequences:
        raise ValueError("No sequence samples were available. Increase the train window or reduce lookback.")

    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _build_future_sequences(
    history_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    lookback: int,
    means: dict[str, float],
    stds: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    history = prepare_model_frame(history_frame.copy())
    history["_prediction_row_id"] = -1
    future = prepare_model_frame(future_frame.copy()).reset_index(drop=True)
    future["_prediction_row_id"] = np.arange(len(future), dtype=int)
    combined = pd.concat([history, future], ignore_index=True)
    combined = engineer_sequence_features(combined)
    combined = _normalize_feature_frame(
        combined,
        feature_columns=feature_columns,
        means=means,
        stds=stds,
    )

    sequences: list[np.ndarray] = []
    prediction_row_ids: list[int] = []
    for _, group in combined.groupby("symbol", sort=False):
        group = group.reset_index(drop=True)
        feature_values = group[list(feature_columns)].to_numpy(dtype=np.float32)
        row_ids = group["_prediction_row_id"].to_numpy(dtype=int)
        for end_index in range(lookback - 1, len(group)):
            prediction_row_id = int(row_ids[end_index])
            if prediction_row_id < 0:
                continue
            sequences.append(feature_values[end_index - lookback + 1 : end_index + 1])
            prediction_row_ids.append(prediction_row_id)

    if not sequences:
        raise ValueError("No prediction sequences were available. The future frame is too short for the lookback.")

    return np.asarray(sequences, dtype=np.float32), np.asarray(prediction_row_ids, dtype=int)


def _subsample_training_data(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_samples: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(X) <= max_samples:
        return X, y
    rng = np.random.default_rng(random_state)
    selected = np.sort(rng.choice(len(X), size=max_samples, replace=False))
    return X[selected], y[selected]


def _history_bars_required(lookback: int) -> int:
    return max(lookback + _MAX_SEQUENCE_ROLLING_WINDOW, _MAX_SEQUENCE_ROLLING_WINDOW + 5)


def _select_history_tail(frame: pd.DataFrame, *, lookback: int) -> pd.DataFrame:
    history_bars = _history_bars_required(lookback)
    return (
        prepare_model_frame(frame.copy())
        .groupby("symbol", sort=False, group_keys=False)
        .tail(history_bars)
        .reset_index(drop=True)
    )


def _build_validation_payload(
    *,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame | None,
    feature_columns: Sequence[str],
    lookback: int,
    means: dict[str, float],
    stds: dict[str, float],
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if validation_frame is None or validation_frame.empty:
        return None, None

    history_tail = _select_history_tail(train_frame, lookback=lookback)
    X_valid, validation_row_ids = _build_future_sequences(
        history_tail,
        validation_frame,
        feature_columns=feature_columns,
        lookback=lookback,
        means=means,
        stds=stds,
    )
    ordered_validation = prepare_model_frame(validation_frame).reset_index(drop=True)
    y_valid_raw = pd.to_numeric(ordered_validation.loc[validation_row_ids, "target"], errors="coerce").to_numpy(dtype=np.float32)
    y_valid = ((y_valid_raw - target_mean) / target_std).astype(np.float32)
    return X_valid, y_valid


class _PositionalEncoding:
    def __init__(self, *, torch_module: object, nn_module: object, d_model: int, max_length: int) -> None:
        self._torch = torch_module
        self._nn = nn_module
        self._d_model = d_model
        self._max_length = max_length

    def build(self) -> object:
        module = self._nn.Module()
        embedding = self._torch.zeros(1, self._max_length, self._d_model, dtype=self._torch.float32)
        module.positional_embedding = self._nn.Parameter(embedding)
        return module


@dataclass(slots=True)
class SequenceRegressorModel:
    feature_columns: tuple[str, ...] = field(default_factory=lambda: SEQUENCE_FEATURE_COLUMNS)
    lookback: int = 20
    epochs: int = 8
    patience: int = 2
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_train_samples: int | None = 20_000
    random_state: int = 7
    _feature_means: dict[str, float] | None = field(default=None, init=False, repr=False)
    _feature_stds: dict[str, float] | None = field(default=None, init=False, repr=False)
    _target_mean: float = field(default=0.0, init=False, repr=False)
    _target_std: float = field(default=1.0, init=False, repr=False)
    _history_tail: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        *,
        validation_frame: pd.DataFrame | None = None,
        history_frame: pd.DataFrame | None = None,
    ) -> "SequenceRegressorModel":
        torch, _ = _load_torch()
        frame = prepare_model_frame(X.copy())
        if "target" not in frame.columns:
            frame["target"] = np.asarray(list(y), dtype=float)

        engineered_train = engineer_sequence_features(frame)
        engineered_train = _coerce_numeric_frame(engineered_train, self.feature_columns)
        feature_means, feature_stds = _compute_feature_stats(engineered_train, self.feature_columns)
        normalized_train = _normalize_feature_frame(
            engineered_train,
            feature_columns=self.feature_columns,
            means=feature_means,
            stds=feature_stds,
        )
        X_train, y_train = _build_training_sequences(
            normalized_train,
            feature_columns=self.feature_columns,
            lookback=self.lookback,
        )
        X_train, y_train = _subsample_training_data(
            X_train,
            y_train,
            max_samples=self.max_train_samples,
            random_state=self.random_state,
        )

        self._feature_means = feature_means
        self._feature_stds = feature_stds
        self._target_mean = float(np.mean(y_train)) if len(y_train) else 0.0
        self._target_std = float(np.std(y_train)) if len(y_train) else 1.0
        if not np.isfinite(self._target_std) or self._target_std <= 1e-12:
            self._target_std = 1.0

        y_train_scaled = ((y_train - self._target_mean) / self._target_std).astype(np.float32)
        X_valid, y_valid = _build_validation_payload(
            train_frame=frame,
            validation_frame=validation_frame,
            feature_columns=self.feature_columns,
            lookback=self.lookback,
            means=feature_means,
            stds=feature_stds,
            target_mean=self._target_mean,
            target_std=self._target_std,
        )
        history_source = history_frame if history_frame is not None else frame
        self._history_tail = _select_history_tail(history_source, lookback=self.lookback)

        torch.manual_seed(self.random_state)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        model = self._build_network(input_size=len(self.feature_columns))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
        X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32) if X_valid is not None else None
        y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32) if y_valid is not None else None

        batch_size = max(1, min(self.batch_size, len(X_tensor)))
        best_state: dict[str, object] | None = None
        best_loss = float("inf")
        patience_left = self.patience

        model.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(len(X_tensor))
            for start in range(0, len(X_tensor), batch_size):
                batch_index = permutation[start : start + batch_size]
                batch_x = X_tensor[batch_index]
                batch_y = y_tensor[batch_index]
                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = loss_fn(predictions, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                if X_valid_tensor is not None and y_valid_tensor is not None and len(X_valid_tensor) > 0:
                    monitor_loss = float(loss_fn(model(X_valid_tensor), y_valid_tensor).item())
                else:
                    monitor_loss = float(loss_fn(model(X_tensor), y_tensor).item())
            model.train()

            if monitor_loss + 1e-9 < best_loss:
                best_loss = monitor_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        self._model = model.eval()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        torch, _ = _load_torch()
        if self._model is None or self._feature_means is None or self._feature_stds is None or self._history_tail is None:
            raise RuntimeError("Sequence model must be fit before prediction.")

        future = prepare_model_frame(X.copy())
        X_pred, prediction_row_ids = _build_future_sequences(
            self._history_tail,
            future,
            feature_columns=self.feature_columns,
            lookback=self.lookback,
            means=self._feature_means,
            stds=self._feature_stds,
        )

        with torch.no_grad():
            predictions = self._model(torch.tensor(X_pred, dtype=torch.float32)).detach().cpu().numpy()
        predictions = predictions.astype(np.float64) * self._target_std + self._target_mean

        output = np.full(len(future), np.nan, dtype=float)
        output[prediction_row_ids] = predictions
        if np.isnan(output).any():
            fill_value = float(np.nanmean(output)) if np.isfinite(np.nanmean(output)) else 0.0
            output = np.where(np.isnan(output), fill_value, output)
        return output

    def _build_network(self, *, input_size: int) -> object:
        raise NotImplementedError


@dataclass(slots=True)
class LSTMRegressorModel(SequenceRegressorModel):
    hidden_size: int = 32
    num_layers: int = 2
    dropout: float = 0.1

    def _build_network(self, *, input_size: int) -> object:
        _, nn = _load_torch()
        hidden_size = self.hidden_size
        num_layers = self.num_layers
        dropout = self.dropout

        class _LSTMNetwork(nn.Module):
            def __init__(self, *, in_features: int) -> None:
                super().__init__()
                lstm_dropout = dropout if num_layers > 1 else 0.0
                self.lstm = nn.LSTM(
                    input_size=in_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=lstm_dropout,
                    batch_first=True,
                )
                head_hidden = max(32, hidden_size // 2)
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, head_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden, 1),
                )

            def forward(self, x):
                encoded, _ = self.lstm(x)
                return self.head(encoded[:, -1, :]).squeeze(-1)

        return _LSTMNetwork(in_features=input_size)


@dataclass(slots=True)
class TransformerRegressorModel(SequenceRegressorModel):
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 1
    dim_feedforward: int = 128
    dropout: float = 0.1

    def _build_network(self, *, input_size: int) -> object:
        torch, nn = _load_torch()
        d_model = self.d_model
        nhead = self.nhead
        num_layers = self.num_layers
        dim_feedforward = self.dim_feedforward
        dropout = self.dropout

        positional_encoding = _PositionalEncoding(
            torch_module=torch,
            nn_module=nn,
            d_model=d_model,
            max_length=self.lookback,
        ).build()

        class _TransformerNetwork(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_projection = nn.Linear(input_size, d_model)
                self.input_norm = nn.LayerNorm(d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.positional_encoding = positional_encoding
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Sequential(
                    nn.LayerNorm(d_model * 2),
                    nn.Linear(d_model * 2, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 1),
                )

            def forward(self, x):
                hidden = self.input_projection(x)
                hidden = self.input_norm(hidden)
                hidden = hidden + self.positional_encoding.positional_embedding[:, : hidden.size(1), :]
                encoded = self.encoder(self.dropout(hidden))
                pooled = torch.cat([encoded[:, -1, :], encoded.mean(dim=1)], dim=-1)
                return self.head(pooled).squeeze(-1)

        return _TransformerNetwork()


def build_lstm_regressor(
    *,
    feature_columns: Iterable[str] = SEQUENCE_FEATURE_COLUMNS,
    lookback: int = 20,
    epochs: int = 8,
    patience: int = 2,
    batch_size: int = 256,
    hidden_size: int = 32,
    num_layers: int = 2,
    dropout: float = 0.1,
    max_train_samples: int | None = 20_000,
) -> LSTMRegressorModel:
    return LSTMRegressorModel(
        feature_columns=tuple(feature_columns),
        lookback=lookback,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        max_train_samples=max_train_samples,
    )


def build_transformer_regressor(
    *,
    feature_columns: Iterable[str] = SEQUENCE_FEATURE_COLUMNS,
    lookback: int = 20,
    epochs: int = 8,
    patience: int = 2,
    batch_size: int = 256,
    d_model: int = 32,
    nhead: int = 4,
    num_layers: int = 1,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    max_train_samples: int | None = 20_000,
) -> TransformerRegressorModel:
    return TransformerRegressorModel(
        feature_columns=tuple(feature_columns),
        lookback=lookback,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_train_samples=max_train_samples,
    )


def get_sequence_model_builders() -> dict[str, object]:
    return {
        "lstm_regressor": build_lstm_regressor,
        "transformer_regressor": build_transformer_regressor,
    }
