from __future__ import annotations

from datetime import datetime

from stockmachine.domain.datetime_utils import normalize_iso_datetime_text, parse_iso_datetime_like


def test_normalize_iso_datetime_text_pads_short_fractional_seconds() -> None:
    assert (
        normalize_iso_datetime_text("2026-03-23T14:04:46.52648+00:00")
        == "2026-03-23T14:04:46.526480+00:00"
    )


def test_normalize_iso_datetime_text_trims_long_fractional_seconds() -> None:
    assert (
        normalize_iso_datetime_text("2026-03-23T14:04:46.526480123Z")
        == "2026-03-23T14:04:46.526480+00:00"
    )


def test_parse_iso_datetime_like_accepts_nonstandard_fractional_precision() -> None:
    parsed = parse_iso_datetime_like("2026-03-23T14:04:46.52648+00:00")

    assert parsed == datetime.fromisoformat("2026-03-23T14:04:46.526480+00:00")
