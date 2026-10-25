"""Every model, by the name used on the command line and in the README."""

from __future__ import annotations

from collections.abc import Callable

from .base import RatingModel
from .baselines import GlobalMean, SeenBefore, biases, matrix_factorization
from .camf import camf
from .fm import factorization_machine
from .gbdt import ContextGBDT

Factory = Callable[[int, str | None], RatingModel]

#: name -> factory(seed, protocol). Baselines first, then the three context-aware solutions.
#: ``protocol`` tells a model which question its internal validation should ask
#: (see :func:`carmusic.protocol.holdout`); ``None`` means "both", for deployment.
MODELS: dict[str, Factory] = {
    "global-mean": lambda seed, protocol: GlobalMean(),
    "biases": biases,
    "mf": matrix_factorization,
    "seen-before": lambda seed, protocol: SeenBefore(seed, protocol),
    "gbdt": lambda seed, protocol: ContextGBDT(protocol=protocol, seed=seed),
    "camf": camf,
    "fm": factorization_machine,
}

#: The model ``carmusic recommend`` uses unless told otherwise.
DEFAULT_MODEL = "gbdt"


def build(name: str, seed: int = 0, protocol: str | None = None) -> RatingModel:
    try:
        factory = MODELS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; choose from {', '.join(MODELS)}") from None
    return factory(seed, protocol)


__all__ = ["DEFAULT_MODEL", "MODELS", "RatingModel", "build"]
