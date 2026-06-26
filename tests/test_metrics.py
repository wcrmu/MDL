from __future__ import annotations

from src.modules import binary_auc, qauc


def test_binary_auc() -> None:
    assert binary_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75


def test_qauc_skips_single_class_groups() -> None:
    result = qauc(
        labels=[0, 1, 1, 1],
        scores=[0.1, 0.9, 0.2, 0.3],
        query_ids=["a", "a", "b", "b"],
    )
    assert result.valid_groups == 1
    assert result.skipped_groups == 1
    assert result.qauc == 1.0
