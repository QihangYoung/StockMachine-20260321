from stockmachine.domain import CANONICAL_TABLES, get_table_spec


def test_canonical_tables_have_unique_column_names() -> None:
    for table in CANONICAL_TABLES.values():
        assert len(table.column_names) == len(set(table.column_names))


def test_get_table_spec_returns_expected_table() -> None:
    table = get_table_spec("daily_bar")
    assert table.name == "daily_bar"
    assert table.has_column("close")
