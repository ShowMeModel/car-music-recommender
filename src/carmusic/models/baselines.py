"""Reference points that any useful model has to beat."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from ..data import pair_keys
from .base import RatingModel
from .factor import FactorModel


class GlobalMean(RatingModel):
    """Every song, every driver, every situation: the average rating."""

    name = "global-mean"

    def fit(self, ratings: pd.DataFrame) -> Self:
        self.mean_ = float(ratings["rating"].mean())
        return self

    def _predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.full(len(rows), self.mean_)


class SeenBefore(RatingModel):
    """What the driver gave this song before, whatever the situation was.

    Falls back to driver and song averages (:func:`biases`) for a song the
    driver never rated. More than half of all situational ratings are equal to
    the same driver's no-context rating of the same song, so this lookup is
    the bar a context-aware model has to clear on ``new-situation``.
    """

    name = "seen-before"

    def __init__(self, seed: int = 0, protocol: str | None = None) -> None:
        self.seed = seed
        self.protocol = protocol

    def fit(self, ratings: pd.DataFrame) -> Self:
        self.memory_ = ratings.groupby(pair_keys(ratings))["rating"].mean()
        self.fallback_ = biases(self.seed, self.protocol).fit(ratings)
        return self

    def _predict(self, rows: pd.DataFrame) -> np.ndarray:
        remembered = self.memory_.reindex(pair_keys(rows)).to_numpy()
        fallback = self.fallback_.predict(rows)
        return np.where(np.isnan(remembered), fallback, remembered)


def biases(seed: int = 0, protocol: str | None = None) -> FactorModel:
    """Average rating + how generous the driver is + how good the song is. No context."""
    return FactorModel(
        name="biases",
        fields=["user", "item"],
        interactions=[],
        protocol=protocol,
        seed=seed,
    )


def matrix_factorization(seed: int = 0, protocol: str | None = None) -> FactorModel:
    """Biased matrix factorisation (Koren et al., 2009) trained on all ratings, context ignored."""
    return FactorModel(
        name="mf",
        fields=["user", "item"],
        interactions=[("user", "item")],
        protocol=protocol,
        seed=seed,
    )
