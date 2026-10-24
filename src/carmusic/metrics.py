"""Scores for predicted ratings, both as numbers and as an ordering of songs.

Ranking metrics are computed *within the same driver and the same situation*,
because that is the list a car would actually show. A situation that lowers
every rating doesn't change which song should come first. Ties in the
predictions count as a coin flip, so a model that predicts one constant
scores exactly at chance rather than at whatever order the data happened to
be stored in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(truth) - np.asarray(pred))))


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(truth) - np.asarray(pred)) ** 2)))


def _groups(rows: pd.DataFrame) -> list[np.ndarray]:
    """Positional indices of each (user, condition) group with something to rank."""
    keys = rows[["user", "condition"]].reset_index(drop=True)
    ratings = rows["rating"].to_numpy()
    out = []
    for idx in keys.groupby(["user", "condition"], sort=True).indices.values():
        if len(idx) > 1 and np.ptp(ratings[idx]) > 0:
            out.append(np.asarray(idx))
    return out


def pair_accuracy(rows: pd.DataFrame, pred: np.ndarray) -> float:
    """How often the better of two songs is ranked first.

    Within each (driver, situation) list, take every pair of songs the driver
    rated differently and count how often the model orders them correctly.
    Lists are then averaged with equal weight, as nDCG is, so the three drivers
    who rated hundreds of songs do not drown out the rest. 0.5 is a coin flip,
    1.0 a perfect order.
    """
    pred = np.asarray(pred, dtype=float)
    truth = rows["rating"].to_numpy(dtype=float)
    scores = []
    for idx in _groups(rows):
        dr = np.sign(truth[idx][:, None] - truth[idx][None, :])
        dp = np.sign(pred[idx][:, None] - pred[idx][None, :])
        upper = np.triu(dr != 0, k=1)
        agree = np.where(dp == 0, 0.5, (dr == dp).astype(float))
        scores.append(float(agree[upper].mean()))
    return float(np.mean(scores)) if scores else float("nan")


def _dcg(gains_in_order: np.ndarray, pred_in_order: np.ndarray, k: int) -> float:
    """DCG@k with tied predictions sharing the average discount of their positions."""
    positions = np.arange(len(gains_in_order))
    discount = np.where(positions < k, 1.0 / np.log2(positions + 2), 0.0)
    total, start = 0.0, 0
    while start < len(pred_in_order):
        end = start + 1
        while end < len(pred_in_order) and pred_in_order[end] == pred_in_order[start]:
            end += 1
        total += gains_in_order[start:end].sum() * discount[start:end].mean()
        start = end
    return float(total)


def ndcg(rows: pd.DataFrame, pred: np.ndarray, k: int = 10) -> float:
    """Mean nDCG@k over (driver, situation) lists, with gain ``2**rating - 1``."""
    pred = np.asarray(pred, dtype=float)
    gains = 2.0 ** rows["rating"].to_numpy(dtype=float) - 1.0
    scores = []
    for idx in _groups(rows):
        order = np.argsort(-pred[idx], kind="stable")
        ideal = np.sort(gains[idx])[::-1]
        best = _dcg(ideal, -np.arange(len(ideal), dtype=float), k)
        scores.append(_dcg(gains[idx][order], pred[idx][order], k) / best)
    return float(np.mean(scores)) if scores else float("nan")


def all_metrics(rows: pd.DataFrame, pred: np.ndarray, rank_by: np.ndarray | None = None) -> dict[str, float]:
    """All metrics reported in the README, for one test fold.

    ``pred`` are ratings on the 0-5 scale. ``rank_by`` is what the ranking
    metrics order by; it defaults to ``pred``, but unclipped scores are
    better, because clipping turns distinct predictions above 5 into ties.
    """
    truth = rows["rating"].to_numpy(dtype=float)
    order = pred if rank_by is None else rank_by
    return {
        "MAE": mae(truth, pred),
        "RMSE": rmse(truth, pred),
        "nDCG@10": ndcg(rows, order, k=10),
        "pair acc.": pair_accuracy(rows, order),
    }
