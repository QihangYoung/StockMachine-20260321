from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class StorageLayout:
    """Filesystem layout for raw snapshots, tables, and checkpoints."""

    root: Path = Path("data")

    def raw_stream_dir(self, source_name: str, stream_name: str) -> Path:
        return self.root / "raw" / source_name / stream_name

    def bronze_stream_dir(self, source_name: str, stream_name: str) -> Path:
        return self.root / "bronze" / source_name / stream_name

    def silver_table_dir(self, table_name: str) -> Path:
        return self.root / "silver" / table_name

    def gold_dataset_dir(self, dataset_name: str) -> Path:
        return self.root / "gold" / dataset_name

    def checkpoint_file(self, source_name: str, stream_name: str) -> Path:
        return self.root / "checkpoints" / source_name / f"{stream_name}.json"
