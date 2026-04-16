from __future__ import annotations

from stockmachine.apps import bootstrap_fmf_validation_etfs_silver as app


def test_bootstrap_fmf_validation_etfs_app_forwards_args(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _bootstrap(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(app, "bootstrap_fmf_validation_etfs_yahoo_to_silver", _bootstrap)
    monkeypatch.setattr(
        "sys.argv",
        [
            "bootstrap_fmf_validation_etfs_silver",
            "--start",
            "2013-08-01",
            "--end",
            "2026-12-31",
            "--silver-file-stem",
            "custom_fmf_seed",
        ],
    )

    app.main()

    assert captured["start"] == "2013-08-01"
    assert captured["end"] == "2026-12-31"
    assert captured["silver_file_stem"] == "custom_fmf_seed"
