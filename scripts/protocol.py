"""Deterministic source-directory holdout; no ML dependencies.

This groups the source directory ID, not every possible face-swap target identity.
Use an official identity split for a full identity-disjoint benchmark.
"""
import random
from collections import defaultdict


def split_subjects(rows, seed=42, test_fraction=0.25):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    by_group = defaultdict(list)
    seen = set()
    for row in rows:
        filename, group, label = row["filename"], row["source_group"], int(row["label"])
        if not filename or not group or label not in (0, 1):
            raise ValueError("Every row needs a filename, source_group and binary label")
        if filename in seen:
            raise ValueError("Duplicate filename; split one quality level at a time")
        seen.add(filename)
        by_group[str(group)].append((str(filename), label))
    groups = sorted(by_group)
    if len(groups) < 2:
        raise ValueError("At least two source groups are required")
    count = max(1, min(len(groups) - 1, round(len(groups) * test_fraction)))
    rng = random.Random(seed)
    best = None
    for _ in range(512):
        test_groups = set(rng.sample(groups, count))
        train = sorted(item for g in groups if g not in test_groups for item in by_group[g])
        test = sorted(item for g in groups if g in test_groups for item in by_group[g])
        if {v for _, v in train} != {0, 1} or {v for _, v in test} != {0, 1}:
            continue
        imbalance = abs(sum(v for _, v in test) / len(test) - sum(v for _, v in train) / len(train))
        score = abs(len(test) / len(rows) - test_fraction) + imbalance
        candidate = (score, sorted(test_groups), train, test)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise ValueError("Cannot form a group holdout with both classes in each partition")
    _, test_groups, train, test = best
    return {
        "seed": seed, "group_key": "source_group", "test_fraction": test_fraction,
        "train_groups": sorted(set(groups) - set(test_groups)), "test_groups": test_groups,
        "train": [name for name, _ in train], "test": [name for name, _ in test],
    }
