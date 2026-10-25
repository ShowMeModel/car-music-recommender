"""Re-run the 2019 negative sampling and show where its F1 of 0.79 came from.

The 2019 notebooks ("balanced machine learning", "AutoML") labelled every
rated (driver, song) pair 1. For each one they invented a 0 row, copying the
driver's columns from the positive row and the song's columns from a random
other row. One of the copied driver columns, ``genre_ratio``, is the share of
the driver's ratings that fall in *the positive song's* genre. On the invented
row it no longer matches the song next to it, and that mismatch alone gives
the label away.

Run from the repository root with the current environment:

    python legacy/check_leak.py

It prints the cross-validated F1 of the same kind of model (scikit-learn's
GradientBoostingClassifier) on different feature subsets, and how many of the
invented "negatives" were songs the driver had in fact rated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

HERE = Path(__file__).resolve().parent


def build_2019_training_set(data: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """The notebooks' loop, minus the deprecated ``DataFrame.append``."""
    rng = np.random.default_rng(seed)
    data = data.assign(label=1)
    item_cols = ["ItemID"] + [c for c in data.columns if c.startswith("category_name_")]
    user_cols = [c for c in data.columns if c not in [*item_cols, "avg_rating", "label"]]
    negatives = []
    for _, row in data.iterrows():
        pool = data[(data.ItemID != row.ItemID) & (data.UserID != row.UserID)]
        other = pool.iloc[rng.integers(len(pool))]
        negatives.append({**{k: row[k] for k in user_cols}, **{k: other[k] for k in item_cols}, "label": 0})
    return pd.concat([data, pd.DataFrame(negatives)], ignore_index=True)


def main() -> None:
    aggregated = pd.read_csv(HERE / "dataset_aggr.csv")
    disliked = (aggregated.avg_rating < 2).mean()
    print(f"pairs labelled 'liked' although their average rating is below 2/5: {disliked:.0%}")

    ds = build_2019_training_set(aggregated)
    y = ds["label"].to_numpy()
    cv = StratifiedKFold(6, shuffle=True, random_state=0)

    def f1(cols: list[str]) -> float:
        X = ds[cols].fillna(0).to_numpy()
        return cross_val_score(GradientBoostingClassifier(random_state=0), X, y, cv=cv, scoring="f1").mean()

    features = [c for c in ds.columns if c not in ("UserID", "ItemID", "label", "avg_rating")]
    genre = [c for c in ds.columns if c.startswith("category_name_")]
    print(f"F1, all 2019 features:                 {f1(features):.3f}")
    print(f"F1, genre_ratio + the song's genre:    {f1(['genre_ratio', *genre]):.3f}")
    print(f"F1, every feature except genre_ratio:  {f1([c for c in features if c != 'genre_ratio']):.3f}")

    rated = set(zip(aggregated.UserID, aggregated.ItemID, strict=True))
    negatives = ds[ds.label == 0]
    pairs = zip(negatives.UserID, negatives.ItemID, strict=True)
    false_negatives = np.mean([(u, i) in rated for u, i in pairs])
    print(f"invented negatives that the driver had actually rated: {false_negatives:.0%}")


if __name__ == "__main__":
    main()
