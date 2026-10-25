"""The interface every model implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd

from ..data import RATING_MAX, RATING_MIN


class RatingModel(ABC):
    """Predicts the rating a driver gives a song in a situation.

    ``fit`` takes rows of the tidy ratings table (see :mod:`carmusic.data`).
    ``predict`` takes rows with at least the columns ``user, item, genre,
    dimension, condition`` and returns ratings clipped to the 0-5 scale;
    ``score`` returns the same numbers unclipped, for ranking. Users,
    songs or situations unseen in training must not crash ``predict``; a model
    falls back to whatever it does know.
    """

    #: Short identifier used on the command line and in result tables.
    name: str = ""
    #: Whether the model looks at the driving situation at all.
    context_aware: bool = False

    @abstractmethod
    def fit(self, ratings: pd.DataFrame) -> Self: ...

    @abstractmethod
    def _predict(self, rows: pd.DataFrame) -> np.ndarray: ...

    def score(self, rows: pd.DataFrame) -> np.ndarray:
        """Raw model output, not clipped. Use it to rank: clipping would tie every song above 5."""
        return np.asarray(self._predict(rows), dtype=float)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """Predicted ratings on the 0-5 scale."""
        return np.clip(self.score(rows), RATING_MIN, RATING_MAX)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
