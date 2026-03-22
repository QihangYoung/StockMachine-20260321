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


def _build_prediction_sequences(
    history_tail: pd.DataFrame,
    future_frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    history = history_tail.copy()
    history["_prediction_row_id"] = -1
    future = prepare_model_frame(future_frame).copy().reset_index(drop=True)
    future["_prediction_row_id"] = np.arange(len(future), dtype=int)
    combined = (
        pd.concat([history, future], ignore_index=True)
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
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


class _PositionalEncoding:
    def __init__(self, *, torch_module: object, nn_module: object, d_model: int, max_length: int) -> None:
        import math

        positions = torch_module.arange(max_length, dtype=torch_module.float32).unsqueeze(1)
        div_term = torch_module.exp(
            torch_module.arange(0, d_model, 2, dtype=torch_module.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch_module.zeros(max_length, d_model, dtype=torch_module.float32)
        encoding[:, 0::2] = torch_module.sin(positions * div_term)
        encoding[:, 1::2] = torch_module.cos(positions * div_term)
        self._buffer = encoding.unsqueeze(0)
        self._module = nn_module

    def build(self) -> object:
        module = self._module.Module()
        module.register_buffer("encoding", self._buffer, persistent=False)

        def _forward(x):
            length = x.size(1)
            return x + module.encoding[:, :length, :]

        module.forward = _forward  # type: ignore[method-assign]
        return module


@dataclass(slots=True)
class SequenceRegressorModel:
    feature_columns: tuple[str, ...] = field(default_factory=lambda: FEATURE_COLUMNS)
    lookback: int = 20
    epochs: int = 3
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

    def fit(self, X: pd.DataFrame, y: Sequence[float]) -> "SequenceRegressorModel":
        torch, _ = _load_torch()
        frame = prepare_model_frame(X.copy())
        if "target" not in frame.columns:
            frame["target"] = np.asarray(list(y), dtype=float)

        frame = _coerce_numeric_frame(frame, self.feature_columns)
        feature_means, feature_stds = _compute_feature_stats(frame, self.feature_columns)
        normalized = _normalize_feature_frame(
            frame,
            feature_columns=self.feature_columns,
            means=feature_means,
            stds=feature_stds,
        )
        X_train, y_train = _build_training_sequences(
            normalized,
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

        target_scaled = ((y_train - self._target_mean) / self._target_std).astype(np.float32)
        self._history_tail = (
            prepare_model_frame(frame)[["date", "symbol", *self.feature_columns]]
            .groupby("symbol", sort=False, group_keys=False)
            .tail(self.lookback - 1)
            .reset_index(drop=True)
        )
        torch.manual_seed(self.random_state)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        model = self._build_network(input_size=len(self.feature_columns))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(target_scaled, dtype=torch.float32)
        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            predictions = model(X_tensor)
            loss = loss_fn(predictions, y_tensor)
            loss.backward()
            optimizer.step()

        self._model = model.eval()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        torch, _ = _load_torch()
        if self._model is None or self._feature_means is None or self._feature_stds is None or self._history_tail is None:
            raise RuntimeError("Sequence model must be fit before prediction.")

        future = prepare_model_frame(X.copy())
        future = _normalize_feature_frame(
            future,
            feature_columns=self.feature_columns,
            means=self._feature_means,
            stds=self._feature_stds,
        )
        history = _normalize_feature_frame(
            self._history_tail,
            feature_columns=self.feature_columns,
            means=self._feature_means,
            stds=self._feature_stds,
        )
        X_pred, prediction_row_ids = _build_prediction_sequences(
            history,
            future,
            feature_columns=self.feature_columns,
            lookback=self.lookback,
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
    hidden_size: int = 24

    def _build_network(self, *, input_size: int) -> object:
        _, nn = _load_torch()

        class _LSTMNetwork(nn.Module):
            def __init__(self, *, in_features: int, hidden_size: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, batch_first=True)
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, 1),
                )

            def forward(self, x):
                _, (hidden, _) = self.lstm(x)
                return self.head(hidden[-1]).squeeze(-1)

        return _LSTMNetwork(in_features=input_size, hidden_size=self.hidden_size)


@dataclass(slots=True)
class TransformerRegressorModel(SequenceRegressorModel):
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 1
    dim_feedforward: int = 64

    def _build_network(self, *, input_size: int) -> object:
        torch, nn = _load_torch()

        positional_encoding = _PositionalEncoding(
            torch_module=torch,
            nn_module=nn,
            d_model=self.d_model,
            max_length=self.lookback,
        ).build()

        class _TransformerNetwork(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_projection = nn.Linear(input_size, self.d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=self.dim_feedforward,
                    dropout=0.0,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
                self.positional_encoding = positional_encoding
                self.head = nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, 1),
                )

            @property
            def d_model(self) -> int:
                return int(self_outer.d_model)

            @property
            def nhead(self) -> int:
                return int(self_outer.nhead)

            @property
            def num_layers(self) -> int:
                return int(self_outer.num_layers)

            @property
            def dim_feedforward(self) -> int:
                return int(self_outer.dim_feedforward)

            def forward(self, x):
                projected = self.input_projection(x)
                encoded = self.positional_encoding(projected)
                transformed = self.encoder(encoded)
                return self.head(transformed[:, -1, :]).squeeze(-1)

        self_outer = self
        return _TransformerNetwork()


def build_lstm_regressor(
    *,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
    lookback: int = 20,
    epochs: int = 3,
    hidden_size: int = 24,
    max_train_samples: int | None = 20_000,
) -> LSTMRegressorModel:
    return LSTMRegressorModel(
        feature_columns=tuple(feature_columns),
        lookback=lookback,
        epochs=epochs,
        hidden_size=hidden_size,
        max_train_samples=max_train_samples,
    )


def build_transformer_regressor(
    *,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
    lookback: int = 20,
    epochs: int = 3,
    d_model: int = 32,
    nhead: int = 4,
    num_layers: int = 1,
    dim_feedforward: int = 64,
    max_train_samples: int | None = 20_000,
) -> TransformerRegressorModel:
    return TransformerRegressorModel(
        feature_columns=tuple(feature_columns),
        lookback=lookback,
        epochs=epochs,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_train_samples=max_train_samples,
    )


def get_sequence_model_builders() -> dict[str, object]:
    return {
        "lstm_regressor": build_lstm_regressor,
        "transformer_regressor": build_transformer_regressor,
    }
