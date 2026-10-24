"""Turn a fitted model into a ranked playlist for one driver in one situation."""

from __future__ import annotations

import pandas as pd

from .data import InCarMusic
from .models import RatingModel


def recommend(model: RatingModel, data: InCarMusic, user: int, condition: str, k: int = 5) -> pd.DataFrame:
    """The ``k`` songs with the highest predicted rating, best first.

    Songs are ordered by the model's unclipped score, so two songs predicted
    "5 stars or more" still have an order. Remaining ties are broken by song
    id, so the list is deterministic. ``rated_before``
    marks songs the driver has already rated in some situation. A car would
    happily replay those, so they are not filtered out.
    """
    rows = data.rows_for(user, condition)
    rows["score"] = model.score(rows)
    rows["predicted"] = model.predict(rows)
    rated = set(data.ratings.loc[data.ratings["user"] == user, "item"])
    rows["rated_before"] = rows["item"].isin(rated)
    ranked = rows.sort_values(["score", "item"], ascending=[False, True], ignore_index=True)
    return ranked.head(k)[["item", "artist", "title", "genre", "predicted", "rated_before"]]
