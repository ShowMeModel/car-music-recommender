import numpy as np
import pandas as pd
import pytest

from carmusic.data import NO_CONTEXT, is_contextual, pair_keys
from carmusic.metrics import rmse
from carmusic.models import MODELS, build
from carmusic.models.gbdt import ContextGBDT
from carmusic.protocol import folds


@pytest.fixture(scope="module")
def split(synthetic):
    train, test = next(folds(synthetic, "new-situation", n_folds=5, seed=0))
    return synthetic.iloc[train], synthetic.iloc[test]


@pytest.fixture(scope="module")
def fitted(split):
    train, _ = split
    return {name: build(name, seed=0, protocol="new-situation").fit(train) for name in MODELS}


def test_predictions_are_ratings(fitted, split):
    _, test = split
    for name, model in fitted.items():
        pred = model.predict(test)
        assert pred.shape == (len(test),), name
        assert np.isfinite(pred).all() and (pred >= 0).all() and (pred <= 5).all(), name


def test_fitting_is_deterministic(split):
    train, test = split
    for name in ["mf", "camf", "fm", "gbdt"]:
        a = build(name, seed=3).fit(train).predict(test)
        b = build(name, seed=3).fit(train).predict(test)
        np.testing.assert_array_equal(a, b, err_msg=name)


def test_unseen_driver_song_and_situation_fall_back_gracefully(fitted):
    stranger = pd.DataFrame(
        {
            "user": [999, 999],
            "item": [999, 999],
            "genre": ["Polka", "Polka"],
            "dimension": ["mood", NO_CONTEXT],
            "condition": ["euphoric", NO_CONTEXT],
        }
    )
    for name, model in fitted.items():
        pred = model.predict(stranger)
        assert np.isfinite(pred).all(), name


def test_context_aware_models_find_the_planted_context_effect(fitted, split):
    _, test = split
    truth = test["rating"].to_numpy()
    errors = {name: rmse(truth, model.predict(test)) for name, model in fitted.items()}
    for name in ["gbdt", "camf", "fm"]:
        assert errors[name] < 0.8 * errors["mf"], errors


def test_gbdt_features_never_see_their_own_rating(synthetic):
    model = ContextGBDT(seed=0)
    model.categories_ = {c: sorted(synthetic[c].unique()) for c in ["genre", "dimension", "condition"]}
    X, _, origin = model.cross_fitted(synthetic)

    target = int(np.flatnonzero(is_contextual(synthetic))[7])
    changed = synthetic.copy()
    changed.iloc[target, changed.columns.get_loc("rating")] = 5.0 - changed.iloc[target]["rating"]
    X_changed, y_changed, origin_changed = model.cross_fitted(changed)

    np.testing.assert_array_equal(origin, origin_changed)
    own = origin == target
    assert own.sum() == 2  # once per regime
    pd.testing.assert_frame_equal(X[own], X_changed[own])
    assert (y_changed[own] != synthetic.iloc[target]["rating"]).all()


def test_gbdt_new_song_regime_has_no_pair_memory(synthetic):
    model = ContextGBDT(seed=0)
    model.categories_ = {c: sorted(synthetic[c].unique()) for c in ["genre", "dimension", "condition"]}
    X, _, _ = model.cross_fitted(synthetic)
    half = len(synthetic)
    assert X["pair_mean"].iloc[:half].notna().mean() > 0.9  # known-song regime
    assert X["pair_mean"].iloc[half:].isna().all()  # new-song regime
    assert X["pair_neutral"].iloc[half:].isna().all()


def test_seen_before_remembers_the_pair(split):
    train, test = split
    model = build("seen-before").fit(train)
    memory = train.groupby(pair_keys(train))["rating"].mean()
    np.testing.assert_allclose(model.predict(test), memory.reindex(pair_keys(test)).to_numpy())
