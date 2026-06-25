from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class QAUCResult:
    qauc: float
    valid_groups: int
    skipped_groups: int


def binary_auc(labels: Sequence[float], scores: Sequence[float]) -> float | None:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positives = sum(1 for label in labels if label > 0.5)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(ranked):
        tie_end = index + 1
        while tie_end < len(ranked) and ranked[tie_end][0] == ranked[index][0]:
            tie_end += 1
        average_rank = (rank + rank + tie_end - index - 1) / 2.0
        rank_sum += sum(1 for _, label in ranked[index:tie_end] if label > 0.5) * average_rank
        rank += tie_end - index
        index = tie_end

    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def qauc(
    labels: Sequence[float],
    scores: Sequence[float],
    query_ids: Iterable[object],
) -> QAUCResult:
    groups: dict[object, list[tuple[float, float]]] = defaultdict(list)
    for label, score, query_id in zip(labels, scores, query_ids):
        groups[query_id].append((float(label), float(score)))

    aucs: list[float] = []
    skipped = 0
    for rows in groups.values():
        group_labels = [label for label, _ in rows]
        group_scores = [score for _, score in rows]
        auc = binary_auc(group_labels, group_scores)
        if auc is None:
            skipped += 1
        else:
            aucs.append(auc)

    return QAUCResult(
        qauc=sum(aucs) / len(aucs) if aucs else float("nan"),
        valid_groups=len(aucs),
        skipped_groups=skipped,
    )

