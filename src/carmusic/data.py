"""Load the InCarMusic dataset into one tidy ratings table.

The Excel file keeps one column per context *dimension* (mood, weather, ...)
and leaves all but one of them empty on every row. Here each rating becomes a
single row with two context columns, ``dimension`` and ``condition``. A rating
given without any context has both set to :data:`NO_CONTEXT`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Default location of the raw data, relative to a source checkout.
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "Data_InCarMusic.xlsx"

#: Marker used in ``dimension`` and ``condition`` for a rating given with no context.
NO_CONTEXT = "none"

#: Ratings in InCarMusic run from 0 to 5.
RATING_MIN, RATING_MAX = 0.0, 5.0

#: Raw column name in the ``ContextualRating`` sheet -> readable dimension name.
CONTEXT_COLUMNS = {
    "DrivingStyle": "driving style",
    "landscape": "landscape",
    "mood": "mood",
    "naturalphenomena": "natural phenomena",
    "RoadType": "road type",
    "sleepiness": "sleepiness",
    "trafficConditions": "traffic conditions",
    "weather": "weather",
}

#: Columns a model may look at when predicting a rating.
KEY_COLUMNS = ["user", "item", "genre", "dimension", "condition"]


@dataclass(frozen=True)
class InCarMusic:
    """The dataset: ratings in context, the song catalogue, and the situations."""

    ratings: pd.DataFrame
    """One row per rating: user, item, rating, dimension, condition, genre, artist, title."""

    songs: pd.DataFrame
    """One row per song: item, artist, title, genre, song (see :func:`load`)."""

    situations: pd.DataFrame
    """One row per context condition: dimension, condition."""

    def rows_for(self, user: int, condition: str) -> pd.DataFrame:
        """Every song in the catalogue, framed as a rating query for ``user`` in ``condition``."""
        if condition == NO_CONTEXT:
            dimension = NO_CONTEXT
        else:
            match = self.situations.loc[self.situations.condition == condition, "dimension"]
            if match.empty:
                raise KeyError(f"unknown situation {condition!r}")
            dimension = match.iloc[0]
        rows = self.songs.copy()
        rows.insert(0, "user", user)
        rows["dimension"] = dimension
        rows["condition"] = condition
        return rows


def _strip_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def _recording_key(artist: str, title: str) -> str:
    """Artist and title without edition suffixes: "Bad Romance - Radio Edit" -> "lady gaga|bad romance"."""
    title = re.sub(r"\(.*?\)|\s+-\s+.*$", "", title.lower())
    clean = [re.sub(r"[^0-9a-z]+", " ", part).strip() for part in (artist.lower(), title)]
    return "|".join(clean)


def load(path: str | os.PathLike | None = None) -> InCarMusic:
    """Read ``Data_InCarMusic.xlsx`` and return it as tidy tables.

    The catalogue lists six songs twice, under two ids: a radio edit next to
    the album version, or the same title entered twice. ``item`` keeps the
    original id. ``song`` is the same for both twins (the lower of the two
    ids), so the evaluation can hide a song from a driver *together with* its
    twin.

    The path can also be set through the ``CARMUSIC_DATA`` environment variable.
    """
    path = Path(path or os.environ.get("CARMUSIC_DATA") or DEFAULT_PATH)
    book = pd.ExcelFile(path)
    raw = _strip_columns(pd.read_excel(book, sheet_name="ContextualRating"))
    tracks = _strip_columns(pd.read_excel(book, sheet_name="Music Track"))
    categories = pd.read_excel(book, sheet_name="Music Category", header=None)

    genre_of_category = {
        int(cid): str(name).strip().removesuffix(" music")
        for cid, name in zip(categories[0], categories[1], strict=True)
    }
    songs = pd.DataFrame(
        {
            "item": tracks["id"].astype(int),
            "artist": tracks["artist"].astype(str).str.strip(),
            "title": tracks["title"].astype(str).str.strip(),
            "genre": tracks["category_id"].map(genre_of_category),
        }
    )
    if songs["genre"].isna().any():
        raise ValueError("some songs have a category that is not in 'Music Category'")
    songs = songs.sort_values("item", ignore_index=True)
    recording = [_recording_key(a, t) for a, t in zip(songs["artist"], songs["title"], strict=True)]
    songs["song"] = songs.groupby(recording)["item"].transform("min")

    context = raw[list(CONTEXT_COLUMNS)]
    filled = context.notna()
    if (filled.sum(axis=1) > 1).any():
        raise ValueError("expected at most one context factor per rating")
    has_context = filled.any(axis=1).to_numpy()
    # Exactly one context column is filled on contextual rows: its name is the
    # dimension, its value the condition.
    dimension = np.where(
        has_context,
        filled.idxmax(axis=1).map(CONTEXT_COLUMNS),
        NO_CONTEXT,
    )
    condition = np.where(
        has_context,
        context.bfill(axis=1).iloc[:, 0].astype(str).str.strip(),
        NO_CONTEXT,
    )

    ratings = pd.DataFrame(
        {
            "user": raw["UserID"].astype(int),
            "item": raw["ItemID"].astype(int),
            "rating": raw["Rating"].astype(float),
            "dimension": dimension.astype(str),
            "condition": condition.astype(str),
        }
    )
    unknown = set(ratings["item"]) - set(songs["item"])
    if unknown:
        raise ValueError(f"ratings refer to unknown songs: {sorted(unknown)[:5]}")
    ratings = ratings.merge(songs, on="item", how="left", validate="many_to_one")

    situations = (
        ratings.loc[ratings.condition != NO_CONTEXT, ["dimension", "condition"]]
        .drop_duplicates()
        .sort_values(["dimension", "condition"], ignore_index=True)
    )
    return InCarMusic(ratings=ratings, songs=songs, situations=situations)


def is_contextual(rows: pd.DataFrame) -> np.ndarray:
    """Boolean mask of the rows that were rated in a driving situation."""
    return (rows["condition"] != NO_CONTEXT).to_numpy()


def pair_keys(rows: pd.DataFrame) -> np.ndarray:
    """One integer per (user, song) pair, usable as a group label.

    Uses the twin-aware ``song`` column when present, so a driver's ratings of
    "Bad Romance" and "Bad Romance - Radio Edit" count as one pair.
    """
    song = rows["song"] if "song" in rows.columns else rows["item"]
    return rows["user"].to_numpy(dtype=np.int64) * 100_000 + song.to_numpy(dtype=np.int64)


def question_keys(rows: pd.DataFrame) -> np.ndarray:
    """One integer per (user, song, condition): the question a single rating answers.

    Some drivers took the survey twice and answered the same question again.
    Grouping by this key keeps both answers on the same side of a split.
    """
    codes = pd.factorize(rows["condition"].astype(str))[0].astype(np.int64)
    return pair_keys(rows) * 100 + codes
