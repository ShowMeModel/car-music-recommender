"""The corrected 2019 solution: gradient boosting on leakage-free context features.

The 2019 notebooks trained a gradient-boosted *classifier* on
"was this song rated or not", using synthetic negatives and features computed
from the whole dataset. That reported F1 0.79, but nearly all of it came from
a leak (see the README). This keeps the idea, a tree ensemble over
engineered features, and repairs the parts that made it unsound.

* **The target is the rating itself, in its situation.** No averaging across
  situations, and no invented negatives: InCarMusic has real low ratings.
* **Every statistic a row sees is computed without that row** (cross-fitting,
  as in target encoding). Training rows get their features from the other
  folds, and test rows from the training set, so both look alike.
* **Two regimes, one model.** "What did this driver give this song before" is
  the strongest single clue, and it only exists for songs the driver has
  rated. Every training row is therefore encoded twice: once cross-fitted by
  rows (the song is known) and once cross-fitted by whole (driver, song) pairs
  (the song is new, so pair features are missing). The trees learn both
  regimes, and at prediction time a missing pair feature routes a row to the
  "new song" branch.
* **Monotone where common sense says so.** A song the driver rated higher
  before, or a situation that lifts ratings, can never push a prediction
  down.
* **Early stopping on held-out training data** that asks the same question
  as the evaluation (:func:`carmusic.protocol.holdout`), never on the test
  fold.

Context features are *shifts*: how much a situational rating differs from the
same driver's no-context rating of the same song. They are averaged per
situation, per genre x situation, per song x situation and per driver x
situation, and each finer average is shrunk towards the coarser one.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold

from ..data import NO_CONTEXT, pair_keys
from ..protocol import holdout
from .base import RatingModel

CATEGORICAL = ["genre", "dimension", "condition"]

#: Numeric features, with the direction a higher value may push the prediction (1 up, 0 free).
NUMERIC = {
    "user_mean": 1,
    "item_mean": 1,
    "user_genre_mean": 1,
    "pair_mean": 1,
    "pair_neutral": 1,
    "pair_count": 0,
    "user_count": 0,
    "cond_shift": 1,
    "genre_cond_shift": 1,
    "item_cond_shift": 1,
    "user_cond_shift": 1,
}


def _stats(frame: pd.DataFrame, cols: list[str], value: str = "rating") -> pd.DataFrame:
    return frame.groupby(cols, sort=False)[value].agg(["sum", "count"])


def _lookup(stats: pd.DataFrame, rows: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of ``stats`` for each row's key; zeros where the key is unseen."""
    key = rows[cols[0]] if len(cols) == 1 else pd.MultiIndex.from_frame(rows[cols])
    found = stats.reindex(key)
    return (
        found["sum"].fillna(0.0).to_numpy(dtype=float),
        found["count"].fillna(0.0).to_numpy(dtype=float),
    )


class ContextGBDT(RatingModel):
    name = "gbdt"
    context_aware = True

    def __init__(
        self,
        prior_strength: float = 5.0,
        n_folds: int = 5,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 15,
        min_samples_leaf: int = 40,
        l2_regularization: float = 1.0,
        max_iter: int = 1000,
        patience: int = 30,
        protocol: str | None = None,
        seed: int = 0,
    ) -> None:
        self.prior_strength = prior_strength
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_iter = max_iter
        self.patience = patience
        self.protocol = protocol
        self.seed = seed

    # -- features ---------------------------------------------------------------

    def features(self, reference: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
        """Describe ``rows`` using only the ratings in ``reference``."""
        m = self.prior_strength
        mu = float(reference["rating"].mean())

        def shrunk(
            cols: list[str], prior: np.ndarray | float, frame: pd.DataFrame = reference, value: str = "rating"
        ) -> tuple[np.ndarray, np.ndarray]:
            total, count = _lookup(_stats(frame, cols, value), rows, cols)
            return (total + m * prior) / (count + m), count

        user_mean, user_n = shrunk(["user"], mu)
        item_mean, _ = shrunk(["item"], mu)
        user_genre_mean, _ = shrunk(["user", "genre"], user_mean)

        pair_sum, pair_n = _lookup(_stats(reference, ["user", "item"]), rows, ["user", "item"])
        neutral = reference[reference["condition"] == NO_CONTEXT]
        neutral_stats = _stats(neutral, ["user", "item"])
        neutral_sum, neutral_n = _lookup(neutral_stats, rows, ["user", "item"])

        # Shift = situational rating minus the same pair's no-context rating.
        situational = reference[reference["condition"] != NO_CONTEXT]
        base_sum, base_n = _lookup(neutral_stats, situational, ["user", "item"])
        with_base = base_n > 0
        shifts = situational.loc[with_base].assign(
            shift=situational["rating"].to_numpy()[with_base] - base_sum[with_base] / base_n[with_base]
        )
        cond_shift, _ = shrunk(["condition"], 0.0, shifts, "shift")
        genre_cond_shift, _ = shrunk(["genre", "condition"], cond_shift, shifts, "shift")
        item_cond_shift, _ = shrunk(["item", "condition"], genre_cond_shift, shifts, "shift")
        user_cond_shift, _ = shrunk(["user", "condition"], cond_shift, shifts, "shift")

        with np.errstate(invalid="ignore", divide="ignore"):
            out = pd.DataFrame(
                {
                    "user_mean": user_mean,
                    "item_mean": item_mean,
                    "user_genre_mean": user_genre_mean,
                    "pair_mean": np.where(pair_n > 0, pair_sum / pair_n, np.nan),
                    "pair_neutral": np.where(neutral_n > 0, neutral_sum / neutral_n, np.nan),
                    "pair_count": pair_n,
                    "user_count": np.log1p(user_n),
                    "cond_shift": cond_shift,
                    "genre_cond_shift": genre_cond_shift,
                    "item_cond_shift": item_cond_shift,
                    "user_cond_shift": user_cond_shift,
                }
            )
        for col in CATEGORICAL:
            values = rows[col].astype(str).to_numpy()
            known = np.isin(values, self.categories_[col])
            out[col] = pd.Categorical(np.where(known, values, None), categories=self.categories_[col])
        return out

    def cross_fitted(self, ratings: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Training matrix: every row twice, once as a known song and once as a new one.

        Returns ``(X, y, origin)``, where ``origin[k]`` is the position in
        ``ratings`` of the row that line ``k`` of ``X`` describes.
        """
        y = ratings["rating"].to_numpy(dtype=float)
        regimes = [
            (KFold(self.n_folds, shuffle=True, random_state=self.seed), None),
            (GroupKFold(self.n_folds, shuffle=True, random_state=self.seed), pair_keys(ratings)),
        ]
        parts, origin = [], []
        for splitter, groups in regimes:
            for reference, held in splitter.split(ratings, groups=groups):
                parts.append(self.features(ratings.iloc[reference], ratings.iloc[held]))
                origin.append(held)
        origin = np.concatenate(origin)
        return pd.concat(parts, ignore_index=True), y[origin], origin

    # -- training ---------------------------------------------------------------

    def _regressor(self, max_iter: int, early_stopping: bool) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            categorical_features="from_dtype",
            monotonic_cst=NUMERIC,
            early_stopping=early_stopping,
            n_iter_no_change=self.patience,
            random_state=self.seed,
        )

    def fit(self, ratings: pd.DataFrame) -> Self:
        self.categories_ = {c: sorted(ratings[c].astype(str).unique()) for c in CATEGORICAL}

        fit_idx, val_idx = holdout(ratings, self.protocol, seed=self.seed)
        fit_part, val_part = ratings.iloc[fit_idx], ratings.iloc[val_idx]
        X, y, _ = self.cross_fitted(fit_part)
        X_val, y_val = self.features(fit_part, val_part), val_part["rating"].to_numpy(dtype=float)
        probe = self._regressor(self.max_iter, early_stopping=True).fit(X, y, X_val=X_val, y_val=y_val)
        curve = [np.mean((y_val - p) ** 2) for p in probe.staged_predict(X_val)]
        self.n_iter_ = int(np.argmin(curve)) + 1

        X, y, _ = self.cross_fitted(ratings)
        self.model_ = self._regressor(self.n_iter_, early_stopping=False).fit(X, y)
        self.reference_ = ratings
        return self

    def _predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self.features(self.reference_, rows))
