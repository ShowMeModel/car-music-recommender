"""One small engine behind the bias, MF, CAMF and factorisation-machine models.

Every model here predicts

    rating = mean + sum of biases + sum of dot products of latent vectors

and they differ only in *which* biases and *which* dot products they have. A
field is a column, or a combination of columns such as ``item x condition``.
Each value of a field gets a bias, and fields that take part in an interaction
also get a latent vector.

* ``biases``  fields user, item; no interactions.
* ``mf``      fields user, item; interaction user.item.
* ``camf``    adds the situation's biases: condition, genre x condition and
  item x condition, as in CAMF (Baltrunas, Ludwig & Ricci, 2011).
* ``fm``      fields user, item, genre, condition; every pair interacts, as in
  a factorisation machine (Rendle et al., 2011).

A field that involves ``condition`` is *absent* on a rating given without
context, and so is any value never seen in training. An absent field
contributes nothing, which is what makes predictions for new drivers, songs or
situations fall back gracefully.

Training minimises squared error plus an L2 penalty (a MAP estimate). It uses
full-batch Adam with a fixed seed, so a fit is deterministic. The penalty
strength is chosen per fit, on a validation slice of the training data
(:func:`carmusic.protocol.holdout`) that asks the same question as the
evaluation protocol. It is never chosen on the test fold.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd

from ..data import NO_CONTEXT
from ..protocol import holdout
from .base import RatingModel

Field = str | tuple[str, ...]

DEFAULT_REG_GRID = (1.0, 3.0, 10.0, 30.0)


def _columns(field: Field) -> tuple[str, ...]:
    return (field,) if isinstance(field, str) else tuple(field)


def _field_name(field: Field) -> str:
    return " x ".join(_columns(field))


def _keys(rows: pd.DataFrame, field: Field) -> tuple[np.ndarray, np.ndarray]:
    """The value of ``field`` on every row, plus a mask of rows where it is present."""
    cols = _columns(field)
    parts = [rows[c].astype(str).to_numpy() for c in cols]
    keys = parts[0].astype(object)
    for part in parts[1:]:
        keys = keys + "|" + part.astype(object)
    present = np.ones(len(rows), dtype=bool)
    if "condition" in cols:
        present &= (rows["condition"] != NO_CONTEXT).to_numpy()
    return keys, present


def _scatter(index: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    """Sum ``values`` into ``size`` buckets by ``index`` (a fast ``np.add.at``)."""
    if values.ndim == 1:
        return np.bincount(index, weights=values, minlength=size)
    return np.stack(
        [np.bincount(index, weights=values[:, j], minlength=size) for j in range(values.shape[1])],
        axis=1,
    )


class FactorModel(RatingModel):
    """A bias-plus-latent-factor model over arbitrary categorical fields."""

    def __init__(
        self,
        name: str,
        fields: Sequence[Field],
        interactions: Sequence[tuple[Field, Field]],
        n_factors: int = 8,
        reg: float | None = None,
        reg_grid: Sequence[float] = DEFAULT_REG_GRID,
        epochs: int = 400,
        learning_rate: float = 0.05,
        protocol: str | None = None,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.fields = [_columns(f) for f in fields]
        self.interactions = [(_columns(a), _columns(b)) for a, b in interactions]
        for a, b in self.interactions:
            if a not in self.fields or b not in self.fields:
                raise ValueError(f"interaction {a}.{b} uses a field that is not declared")
        self.n_factors = n_factors
        self.reg = reg
        self.reg_grid = tuple(reg_grid)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.protocol = protocol
        self.seed = seed
        self.context_aware = any("condition" in f for f in self.fields)

    # -- encoding ---------------------------------------------------------------

    def _encode(self, rows: pd.DataFrame) -> list[np.ndarray]:
        """Index of each row's value in each field; absent or unseen -> the padding slot."""
        out = []
        for field, vocab in zip(self.fields, self.vocab_, strict=True):
            keys, present = _keys(rows, field)
            idx = pd.Index(vocab).get_indexer(keys)
            idx[(idx < 0) | ~present] = len(vocab)
            out.append(idx)
        return out

    # -- training ---------------------------------------------------------------

    def _train(self, ratings: pd.DataFrame, reg: float) -> None:
        rng = np.random.default_rng(self.seed)
        self.vocab_ = []
        for field in self.fields:
            keys, present = _keys(ratings, field)
            self.vocab_.append(np.unique(keys[present]))
        self.mean_ = float(ratings["rating"].mean())
        idx = self._encode(ratings)
        y = ratings["rating"].to_numpy(dtype=float)
        sizes = [len(v) + 1 for v in self.vocab_]  # +1: the padding slot, kept at zero

        has_vector = {f for pair in self.interactions for f in pair}
        params: list[np.ndarray] = [np.zeros(s) for s in sizes]  # biases
        vector_of: dict[tuple[str, ...], int] = {}
        for f, size in zip(self.fields, sizes, strict=True):
            if f in has_vector:
                vector_of[f] = len(params)
                params.append(rng.normal(0.0, 0.1, size=(size, self.n_factors)))
        for k in vector_of.values():
            params[k][-1] = 0.0
        pos = {f: i for i, f in enumerate(self.fields)}

        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for step in range(1, self.epochs + 1):
            vectors = {f: params[k][idx[pos[f]]] for f, k in vector_of.items()}
            pred = np.full(len(y), self.mean_)
            for i in range(len(self.fields)):
                pred += params[i][idx[i]]
            for a, b in self.interactions:
                pred += np.einsum("nk,nk->n", vectors[a], vectors[b])
            err = pred - y

            grads = [reg * p for p in params]
            for i in range(len(self.fields)):
                grads[i] += _scatter(idx[i], err, sizes[i])
            for a, b in self.interactions:
                grads[vector_of[a]] += _scatter(idx[pos[a]], err[:, None] * vectors[b], sizes[pos[a]])
                grads[vector_of[b]] += _scatter(idx[pos[b]], err[:, None] * vectors[a], sizes[pos[b]])

            for k, (p, g) in enumerate(zip(params, grads, strict=True)):
                g[-1] = 0.0  # the padding slot never learns
                m[k] = beta1 * m[k] + (1 - beta1) * g
                v[k] = beta2 * v[k] + (1 - beta2) * g * g
                m_hat = m[k] / (1 - beta1**step)
                v_hat = v[k] / (1 - beta2**step)
                p -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        self.biases_ = params[: len(self.fields)]
        self.vectors_ = {f: params[k] for f, k in vector_of.items()}
        self.reg_ = reg

    def fit(self, ratings: pd.DataFrame) -> Self:
        if self.reg is not None:
            self._train(ratings, self.reg)
            return self
        fit_idx, val_idx = holdout(ratings, self.protocol, seed=self.seed)
        fit_part, val_part = ratings.iloc[fit_idx], ratings.iloc[val_idx]
        truth = val_part["rating"].to_numpy(dtype=float)
        losses = []
        for reg in self.reg_grid:
            self._train(fit_part, reg)
            losses.append(float(np.mean((self.predict(val_part) - truth) ** 2)))
        self.validation_ = dict(zip(self.reg_grid, losses, strict=True))
        self._train(ratings, self.reg_grid[int(np.argmin(losses))])
        return self

    # -- prediction -------------------------------------------------------------

    def _predict(self, rows: pd.DataFrame) -> np.ndarray:
        idx = self._encode(rows)
        pos = {f: i for i, f in enumerate(self.fields)}
        pred = np.full(len(rows), self.mean_)
        for i in range(len(self.fields)):
            pred += self.biases_[i][idx[i]]
        for a, b in self.interactions:
            pred += np.einsum("nk,nk->n", self.vectors_[a][idx[pos[a]]], self.vectors_[b][idx[pos[b]]])
        return pred

    def bias_table(self, field: Field) -> pd.Series:
        """Learned biases of one field, by value (for inspection and the README)."""
        i = self.fields.index(_columns(field))
        return pd.Series(self.biases_[i][:-1], index=self.vocab_[i], name=_field_name(field))
