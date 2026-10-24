import numpy as np
import pytest

from carmusic.data import is_contextual, pair_keys
from carmusic.protocol import PROTOCOLS, folds, holdout


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_folds_score_every_situational_rating_exactly_once(ratings, protocol):
    seen = []
    for train, test in folds(ratings, protocol, n_folds=5, seed=0):
        assert np.intersect1d(train, test).size == 0
        assert is_contextual(ratings.iloc[test]).all()
        seen.append(test)
    seen = np.concatenate(seen)
    assert len(seen) == len(np.unique(seen)) == is_contextual(ratings).sum()


def test_new_situation_keeps_the_song_known(ratings):
    pairs = pair_keys(ratings)
    neutral = ~is_contextual(ratings)
    for train, test in folds(ratings, "new-situation"):
        known = set(pairs[train][neutral[train]])
        assert set(pairs[test]) <= known


def test_new_song_hides_the_whole_pair(ratings):
    pairs = pair_keys(ratings)
    for train, test in folds(ratings, "new-song"):
        assert not set(pairs[train]) & set(pairs[test])


@pytest.mark.parametrize("protocol", [None, *PROTOCOLS])
def test_holdout_asks_the_requested_question(ratings, protocol):
    fit, val = holdout(ratings, protocol, seed=1)
    pairs = pair_keys(ratings)
    assert np.intersect1d(fit, val).size == 0
    assert is_contextual(ratings.iloc[val]).all()
    unseen = ~np.isin(pairs[val], pairs[fit])
    if protocol == "new-situation":
        assert not unseen.any()
    elif protocol == "new-song":
        assert unseen.all()
    else:
        assert 0.3 < unseen.mean() < 0.7


def test_unknown_protocol_is_rejected(ratings):
    with pytest.raises(ValueError):
        next(folds(ratings, "random"))
    with pytest.raises(ValueError):
        holdout(ratings, "random")
