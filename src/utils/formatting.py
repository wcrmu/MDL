from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def format_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    table_rows = [[str(value) for value in row] for row in rows]
    header_values = [str(header) for header in headers]
    widths = [
        max([len(header_values[index])] + [len(row[index]) for row in table_rows])
        for index in range(len(header_values))
    ]

    def format_row(values: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    return [format_row(header_values), separator] + [format_row(row) for row in table_rows]
