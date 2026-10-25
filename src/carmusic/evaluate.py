"""Cross-validate models under the protocols in :mod:`carmusic.protocol`."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from .data import InCarMusic
from .metrics import all_metrics
from .models import MODELS, build
from .protocol import folds

METRICS = ["MAE", "RMSE", "nDCG@10", "pair acc."]

#: Context-free model every other model is compared with, fold by fold.
REFERENCE = "mf"


def cross_validate(
    data: InCarMusic,
    protocol: str,
    model_names: Sequence[str] = tuple(MODELS),
    n_folds: int = 5,
    seed: int = 0,
    repeats: int = 1,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Fit and score every model on every fold. One row per (fold, model).

    With ``repeats > 1`` the whole k-fold split is redrawn with seeds
    ``seed, seed + 1, ...``. Every model sees exactly the same folds, so
    differences between models can be compared fold by fold.
    """
    results = []
    for repeat in range(repeats):
        split_seed = seed + repeat
        for fold, (train, test) in enumerate(folds(data.ratings, protocol, n_folds, split_seed)):
            train_rows, test_rows = data.ratings.iloc[train], data.ratings.iloc[test]
            for name in model_names:
                model = build(name, split_seed, protocol).fit(train_rows)
                raw = model.score(test_rows)
                metrics = all_metrics(test_rows, model.predict(test_rows), rank_by=raw)
                fold_id = repeat * n_folds + fold
                results.append({"protocol": protocol, "fold": fold_id, "model": name, **metrics})
                if progress:
                    progress(f"{protocol}: repeat {repeat + 1}/{repeats}, fold {fold + 1}/{n_folds}, {name}")
    return pd.DataFrame(results)


def summarize(per_fold: pd.DataFrame, reference: str = REFERENCE) -> pd.DataFrame:
    """Mean and standard deviation across folds, keeping the models in registry order.

    ``beats <reference>`` counts the folds in which a model's RMSE is lower
    than the reference model's on the very same fold.
    """
    grouped = per_fold.groupby(["protocol", "model"], sort=False)[METRICS]
    summary = grouped.mean().join(grouped.std().add_suffix(" std")).reset_index()
    if reference in set(per_fold["model"]):
        rmse = per_fold.pivot_table(index=["protocol", "fold"], columns="model", values="RMSE")
        wins = rmse.lt(rmse[reference], axis=0).groupby(level="protocol").sum()
        n = rmse.groupby(level="protocol").size()
        summary[f"beats {reference}"] = [
            f"{int(wins.loc[p, m])}/{int(n.loc[p])}" if m != reference else "-"
            for p, m in zip(summary["protocol"], summary["model"], strict=True)
        ]
    return summary


def markdown(summary: pd.DataFrame, protocol: str) -> str:
    """One protocol's summary as a Markdown table (what the README shows)."""
    part = summary[summary["protocol"] == protocol]
    extra = [c for c in part.columns if c.startswith("beats ")]
    lines = [
        "| model | " + " | ".join(METRICS + extra) + " |",
        "|---|" + "---:|" * (len(METRICS) + len(extra)),
    ]
    for _, row in part.iterrows():
        cells = [f"{row[m]:.3f} ± {row[m + ' std']:.3f}" for m in METRICS] + [row[c] for c in extra]
        lines.append(f"| {row['model']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
