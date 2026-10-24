"""Alternative 1: context-aware matrix factorisation (CAMF).

The model the InCarMusic authors themselves built for this data (Baltrunas,
Ludwig & Ricci, *Matrix factorization techniques for context aware
recommendation*, RecSys 2011)::

    rating(u, i, c) = mean + b_u + b_i + B_c + B_genre(i),c + B_i,c + p_u . q_i

``p_u . q_i`` is ordinary matrix factorisation: the driver's taste matched
against the song's character. The situation enters only as additive shifts,
at three levels of detail:

* ``B_c``: the situation itself ("sad drivers rate everything lower"),
* ``B_genre,c``: a genre in that situation ("metal suffers in a traffic jam"),
* ``B_i,c``: one song in that situation (CAMF-CI in the paper).

The finer levels have few ratings each. The L2 penalty pulls them to zero, so
they only move when the data insists, and the coarser shift carries the rest.
This is the paper's CAMF-CI with a CAMF-CC-style genre layer added for
sharing.
"""

from __future__ import annotations

from .factor import FactorModel


def camf(seed: int = 0, protocol: str | None = None) -> FactorModel:
    return FactorModel(
        name="camf",
        fields=[
            "user",
            "item",
            "condition",
            ("genre", "condition"),
            ("item", "condition"),
        ],
        interactions=[("user", "item")],
        protocol=protocol,
        seed=seed,
    )
