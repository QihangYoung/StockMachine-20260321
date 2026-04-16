from __future__ import annotations

from stockmachine.apps import bootstrap_multi_asset_proxy_etfs_silver as app


def test_bootstrap_multi_asset_proxy_etfs_app_forwards_args(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _bootstrap(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(app, "bootstrap_multi_asset_proxy_etfs_yahoo_to_silver", _bootstrap)
    monkeypatch.setattr(
        "sys.argv",
        [
            "bootstrap_multi_asset_proxy_etfs_silver",
            "--start",
            "2018-01-01",
            "--end",
            "2026-12-31",
            "--silver-file-stem",
            "custom_proxy_seed",
        ],
    )

    app.main()

    assert captured["start"] == "2018-01-01"
    assert captured["end"] == "2026-12-31"
    assert captured["silver_file_stem"] == "custom_proxy_seed"
