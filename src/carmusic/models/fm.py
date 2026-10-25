"""Alternative 2: a context-aware factorisation machine (FM).

Rendle, Gantner, Freudenthaler & Schmidt-Thieme, *Fast context-aware
recommendations with factorization machines*, SIGIR 2011. Every rating is a
sparse vector of four one-hot fields: driver, song, genre and situation. The
model is::

    rating = mean + sum of field biases + sum over field pairs <v_a, v_b>

Six pairs interact: driver.song, driver.genre, driver.situation, song.genre,
song.situation and genre.situation. Compared with CAMF, the new ingredient is
**driver x situation**. CAMF assumes a traffic jam shifts a song the same way
for everyone. The FM can learn that it is *this* driver who reaches for metal
in a traffic jam. Every interaction is low-rank, so a pair that was never
observed (this driver, this situation) still gets an estimate from the
vectors of its two parts.
"""

from __future__ import annotations

from itertools import combinations

from .factor import FactorModel

FIELDS = ["user", "item", "genre", "condition"]


def factorization_machine(seed: int = 0, protocol: str | None = None) -> FactorModel:
    return FactorModel(
        name="fm",
        fields=FIELDS,
        interactions=list(combinations(FIELDS, 2)),
        protocol=protocol,
        seed=seed,
    )
