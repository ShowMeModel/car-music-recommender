from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from carmusic.metrics import _dcg, ndcg, pair_accuracy


def one_list(ratings):
    return pd.DataFrame({"user": 1, "condition": "jam", "rating": ratings})


def test_pair_accuracy_extremes():
    rows = one_list([5, 3, 1, 0])
    assert pair_accuracy(rows, np.array([4.0, 3.0, 2.0, 1.0])) == 1.0
    assert pair_accuracy(rows, np.array([1.0, 2.0, 3.0, 4.0])) == 0.0
    assert pair_accuracy(rows, np.full(4, 2.5)) == 0.5


def test_pair_accuracy_ignores_pairs_rated_equally_and_other_lists():
    rows = pd.concat([one_list([4, 4, 1]), one_list([2, 3]).assign(condition="sun")], ignore_index=True)
    # list 1 has two informative pairs (4>1, 4>1), list 2 has one (3>2)
    pred = np.array([3.0, 0.0, 1.0, 0.0, 1.0])
    assert pair_accuracy(rows, pred) == pytest.approx(2 / 3)


def test_ndcg_is_one_for_a_perfect_order_and_less_otherwise():
    rows = one_list([5, 3, 1, 0])
    assert ndcg(rows, np.array([4.0, 3.0, 2.0, 1.0])) == pytest.approx(1.0)
    assert ndcg(rows, np.array([1.0, 2.0, 3.0, 4.0])) < 0.6


def test_tied_predictions_score_the_average_over_tie_orders():
    gains = np.array([31.0, 7.0, 1.0, 0.0])
    expected = np.mean([_dcg(gains[list(p)], np.arange(4.0)[::-1], 10) for p in permutations(range(4))])
    assert _dcg(gains, np.zeros(4), 10) == pytest.approx(expected)


def test_lists_with_nothing_to_rank_are_skipped():
    assert np.isnan(ndcg(one_list([3, 3]), np.array([1.0, 2.0])))
    assert np.isnan(pair_accuracy(one_list([4]), np.array([1.0])))
