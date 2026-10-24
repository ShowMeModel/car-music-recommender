"""How models are evaluated: which ratings are hidden, and what is left to learn from.

Every (driver, song) pair in InCarMusic was rated once without context and then
again in three or more imagined situations. That design allows two
evaluation questions, and they are kept separate on purpose:

``new-situation``
    *A song the driver already rated: how will they rate it in this situation?*
    A random fifth of the situational ratings is hidden. The driver's other
    ratings of the same song, including the no-context one, stay in training.
    This isolates the effect of context.

``new-song``
    *A song the driver has never rated: how will they rate it in this situation?*
    A random fifth of the (driver, song) pairs is hidden, with every rating of
    those pairs. This is the recommendation case.

Both protocols score only the situational ratings, so their numbers are
comparable.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from .data import is_contextual, pair_keys

PROTOCOLS = ("new-situation", "new-song")

Split = tuple[np.ndarray, np.ndarray]


def folds(ratings: pd.DataFrame, protocol: str, n_folds: int = 5, seed: int = 0) -> Iterator[Split]:
    """Yield ``(train, test)`` arrays of positional row indices, one pair per fold.

    The test sets of all folds together cover every situational rating once.
    """
    contextual = np.flatnonzero(is_contextual(ratings))
    if protocol == "new-situation":
        everything = np.arange(len(ratings))
        for _, test_pos in KFold(n_folds, shuffle=True, random_state=seed).split(contextual):
            test = contextual[test_pos]
            yield np.setdiff1d(everything, test), test
    elif protocol == "new-song":
        pairs = pair_keys(ratings)
        splitter = GroupKFold(n_folds, shuffle=True, random_state=seed)
        for train, held_out in splitter.split(ratings, groups=pairs):
            # Hidden pairs lose every rating, but only their situational ones are scored.
            yield train, np.intersect1d(held_out, contextual)
    else:
        raise ValueError(f"unknown protocol {protocol!r}; choose from {PROTOCOLS}")


def holdout(
    ratings: pd.DataFrame, protocol: str | None = None, fraction: float = 0.15, seed: int = 0
) -> Split:
    """Split training ratings into a fit part and a validation part.

    Models tune themselves on this split (penalty strength, number of trees)
    and never on a test fold. The validation part asks the same question as
    ``protocol``. With ``protocol=None`` it mixes both, half situational ratings
    of songs that stay in training and half songs hidden completely, which is
    what a car that knows some songs and not others needs.
    """
    if protocol is not None and protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; choose from {PROTOCOLS}")
    share_songs = {None: 0.5, "new-situation": 0.0, "new-song": 1.0}[protocol]
    rng = np.random.default_rng(seed)
    contextual = is_contextual(ratings)
    pairs = pair_keys(ratings)
    unique_pairs = np.unique(pairs)
    n_pairs = round(len(unique_pairs) * fraction * share_songs)
    hidden_pairs = rng.choice(unique_pairs, size=n_pairs, replace=False)
    in_hidden_pair = np.isin(pairs, hidden_pairs)

    candidates = np.flatnonzero(contextual & ~in_hidden_pair)
    n_rows = round(len(candidates) * fraction * (1.0 - share_songs))
    hidden_rows = rng.choice(candidates, size=n_rows, replace=False)

    validation_mask = np.zeros(len(ratings), dtype=bool)
    validation_mask[hidden_rows] = True
    validation_mask |= in_hidden_pair & contextual
    fit_mask = ~validation_mask & ~in_hidden_pair
    return np.flatnonzero(fit_mask), np.flatnonzero(validation_mask)
