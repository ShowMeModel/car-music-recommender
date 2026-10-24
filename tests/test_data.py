import pytest

from carmusic.data import NO_CONTEXT, is_contextual, pair_keys


def test_known_shape(data, ratings):
    assert len(ratings) == 4012
    assert ratings["user"].nunique() == 42
    assert len(data.songs) == 139
    assert len(data.situations) == 26
    assert data.situations["dimension"].nunique() == 8
    assert data.songs["genre"].nunique() == 10
    assert ratings["rating"].between(0, 5).all()


def test_context_is_one_condition_or_none(ratings):
    contextual = is_contextual(ratings)
    assert contextual.sum() == 3009
    assert (ratings.loc[~contextual, "dimension"] == NO_CONTEXT).all()
    assert (ratings.loc[contextual, "dimension"] != NO_CONTEXT).all()


def test_every_pair_also_has_a_no_context_rating(ratings):
    pairs = pair_keys(ratings)
    neutral = set(pairs[~is_contextual(ratings)])
    assert set(pairs) == neutral


def test_rows_for_frames_the_whole_catalogue(data):
    rows = data.rows_for(1019, "traffic jam")
    assert len(rows) == len(data.songs)
    assert set(rows["user"]) == {1019}
    assert set(rows["dimension"]) == {"traffic conditions"}
    with pytest.raises(KeyError):
        data.rows_for(1019, "on the moon")


def test_twin_songs_share_one_song_id(data):
    twins = data.songs[data.songs["item"] != data.songs["song"]]
    assert len(twins) == 6
    assert data.songs["song"].nunique() == 133
    by_item = data.songs.set_index("item")
    assert by_item.loc[725, "song"] == by_item.loc[680, "song"] == 680  # Bad Romance (- Radio Edit)
    for item, song in zip(twins["item"], twins["song"], strict=True):
        assert by_item.loc[item, "artist"] == by_item.loc[song, "artist"]
