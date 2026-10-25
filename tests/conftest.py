from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from carmusic.data import NO_CONTEXT, load

SITUATIONS = {
    "traffic": ["jam", "free road"],
    "mood": ["sad", "happy"],
    "weather": ["rain", "sun"],
}
GENRES = ["Metal", "Pop", "Jazz", "Blues"]


@pytest.fixture(scope="session")
def data():
    return load()


@pytest.fixture(scope="session")
def ratings(data):
    return data.ratings


def make_synthetic(n_users: int = 30, n_items: int = 40, seed: int = 0) -> pd.DataFrame:
    """Ratings shaped like InCarMusic, with a strong planted context effect.

    Every pair is rated once without context and once in each of three
    situations. A situation shifts one genre hard, e.g. metal in a jam drops
    by two stars, so a model that ignores context cannot fit it.
    """
    rng = np.random.default_rng(seed)
    users = np.arange(1, n_users + 1)
    items = np.arange(1, n_items + 1)
    genre = {i: GENRES[i % len(GENRES)] for i in items}
    taste = rng.normal(0, 0.7, (n_users + 1, 2)) @ rng.normal(0, 0.7, (2, n_items + 1))
    user_bias = rng.normal(0, 0.5, n_users + 1)
    conditions = [(d, c) for d, cs in SITUATIONS.items() for c in cs]
    shift = {(g, c): rng.choice([-2.0, 0.0, 2.0]) for g in GENRES for _, c in conditions}

    rows = []
    for u in users:
        for i in rng.choice(items, size=12, replace=False):
            base = 2.5 + user_bias[u] + taste[u, i]
            rows.append((u, i, base, NO_CONTEXT, NO_CONTEXT))
            for k in rng.choice(len(conditions), size=3, replace=False):
                d, c = conditions[k]
                rows.append((u, i, base + shift[(genre[i], c)], d, c))
    frame = pd.DataFrame(rows, columns=["user", "item", "rating", "dimension", "condition"])
    noise = rng.normal(0, 0.3, len(frame))
    frame["rating"] = np.clip(np.round(frame["rating"] + noise), 0, 5)
    frame["genre"] = frame["item"].map(genre)
    frame["artist"] = "artist " + frame["item"].astype(str)
    frame["title"] = "title " + frame["item"].astype(str)
    return frame


@pytest.fixture(scope="session")
def synthetic():
    return make_synthetic()
