"""MovieLens-style data loading and deterministic splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RatingData:
    """Ratings with dense, zero-based user and item indexes."""

    users: np.ndarray
    items: np.ndarray
    ratings: np.ndarray
    n_users: int
    n_items: int

    def __post_init__(self) -> None:
        if not (len(self.users) == len(self.items) == len(self.ratings)):
            raise ValueError("users, items, and ratings must have equal lengths")
        if len(self.ratings) == 0:
            raise ValueError("at least one rating is required")
        if self.users.min() < 0 or self.items.min() < 0:
            raise ValueError("indexes must be non-negative")
        if self.users.max() >= self.n_users or self.items.max() >= self.n_items:
            raise ValueError("indexes exceed declared dimensions")


def load_movielens(path: str | Path) -> RatingData:
    """Load MovieLens ``u.data`` or a whitespace/tab-separated three-column file.

    The first three columns must be user id, item id, and numeric rating. Original
    ids are compacted, so non-contiguous ids do not create unused factor rows.
    """
    rows: list[tuple[int, int, float]] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 3:
                raise ValueError(f"{path}:{line_no}: expected at least three columns")
            try:
                rows.append((int(fields[0]), int(fields[1]), float(fields[2])))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid rating row") from exc
    if not rows:
        raise ValueError(f"{path}: no ratings found")
    raw_users = np.asarray([row[0] for row in rows])
    raw_items = np.asarray([row[1] for row in rows])
    _, users = np.unique(raw_users, return_inverse=True)
    _, items = np.unique(raw_items, return_inverse=True)
    return RatingData(users, items, np.asarray([row[2] for row in rows], dtype=float),
                      int(users.max()) + 1, int(items.max()) + 1)


def _subset(data: RatingData, indexes: np.ndarray) -> RatingData:
    return RatingData(data.users[indexes], data.items[indexes], data.ratings[indexes],
                      data.n_users, data.n_items)


def train_test_split(data: RatingData, test_fraction: float = 0.2,
                     seed: int = 0) -> tuple[RatingData, RatingData]:
    """Random record split, retaining at least one training rating per user/item."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data.ratings))
    target = int(round(len(order) * test_fraction))
    user_counts = np.bincount(data.users, minlength=data.n_users)
    item_counts = np.bincount(data.items, minlength=data.n_items)
    chosen: list[int] = []
    for index in order:
        user, item = data.users[index], data.items[index]
        if len(chosen) == target:
            break
        if user_counts[user] > 1 and item_counts[item] > 1:
            chosen.append(int(index))
            user_counts[user] -= 1
            item_counts[item] -= 1
    test_indexes = np.asarray(chosen, dtype=int)
    mask = np.ones(len(data.ratings), dtype=bool)
    mask[test_indexes] = False
    return _subset(data, np.flatnonzero(mask)), _subset(data, test_indexes)
