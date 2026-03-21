import os
from pathlib import Path

from stockmachine.data.vendors.alpaca import _load_local_env_file


def test_load_local_env_file_sets_missing_values(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "ALPACA_API_KEY_ID=test_key\nALPACA_API_SECRET_KEY=test_secret\n# comment\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)

    _load_local_env_file(env_path)

    assert os.getenv("ALPACA_API_KEY_ID") == "test_key"
    assert os.getenv("ALPACA_API_SECRET_KEY") == "test_secret"


def test_load_local_env_file_keeps_existing_environment(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("ALPACA_API_KEY_ID=file_key\n", encoding="utf-8")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "existing_key")

    _load_local_env_file(env_path)

    assert os.getenv("ALPACA_API_KEY_ID") == "existing_key"
