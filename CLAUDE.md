# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Context-aware music recommendation on the InCarMusic dataset (Baltrunas et al., 2011). The task: predict how a driver rates a song in a driving situation (traffic jam, night, sad mood, …), then rank songs for "this driver, right now". `README.md` holds the problem statement, the results tables, the 2019 post-mortem and the literature; keep it true when behaviour changes.

The code is the `carmusic` package in `src/carmusic/`. `legacy/` holds the original 2019 notebooks, which are **superseded and must not be "fixed" in place**. They are kept as a record, next to `legacy/check_2019_leak.py`, which demonstrates their label leak.

## Commands

```shell
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # Python 3.11+, tested on 3.13
.venv/bin/pytest                                            # 30 tests, ~17 s
.venv/bin/pytest tests/test_models.py::test_gbdt_features_never_see_their_own_rating
.venv/bin/ruff check src tests && .venv/bin/ruff format src tests
.venv/bin/carmusic evaluate                                 # 5-fold, both protocols, ~70 s
.venv/bin/carmusic evaluate --repeats 3                     # the README tables, ~3.5 min
.venv/bin/carmusic recommend --user 1033 --situation "traffic jam"
.venv/bin/python legacy/check_2019_leak.py                  # reproduces the 2019 leak numbers
```

The system `python3` is 3.9 and cannot run this. `.venv` was built from `~/miniconda3/envs/py_313/bin/python`.

`tools/make_banner.py` redraws `docs/banner-{light,dark}.svg` from real model output: the `recommend()` playlists, CAMF's learned biases, and the CV results. Pass `--results <csv from carmusic evaluate --repeats 3 --csv>` to skip the 3.5-minute CV. Like the other PuzzleLearning repos, it is **local-only and gitignored**: the banners are committed, the generator is not, so it may be missing from a fresh clone. Do not add it back to git. Rendering must stay byte-identical across runs.

## Architecture

- `data.py`: `load()` turns the Excel file into one tidy table, one row per rating: `user, item, rating, dimension, condition, genre, artist, title, song`. `song` is the twin-aware song id. A rating without context has `dimension == condition == "none"` (`NO_CONTEXT`). Every row has at most one situation.
- `protocol.py`: the two evaluation questions. `folds()` gives the test splits. `holdout()` gives the validation split that models tune on internally.
  - `new-situation` hides random (user, song, condition) questions; the pair stays known.
  - `new-song` hides whole (user, song) pairs, twins included.
  - Both score only situational rows.
- `models/`: every model implements `RatingModel` (`fit`, `_predict`; `score` is unclipped and used for ranking, `predict` is clipped to 0–5). They are registered in `models/__init__.py:MODELS` as `factory(seed, protocol)`. `protocol` only tells a model which question its internal `holdout()` should ask; `None` means mixed, which is what `recommend` uses.
  - `factor.py:FactorModel` is one numpy engine (biases + dot products over declared fields, full-batch Adam, L2 chosen on the holdout). `biases`, `mf`, `camf` and `fm` are only field declarations over it. A field involving `condition` is absent on no-context rows; unseen values map to a zero padding slot.
  - `gbdt.py:ContextGBDT` is the corrected 2019 model. It uses cross-fitted target statistics in two regimes (question folds = known song, pair folds = new song), stacked into one `HistGradientBoostingRegressor` with monotone constraints and early stopping on the holdout.
- `metrics.py`: nDCG@10 and pairwise accuracy are computed **within (user, condition) groups**, tie-aware, then macro-averaged over the groups. Pooling would let three heavy raters dominate.
- `evaluate.py` and `cli.py`: repeated k-fold and Markdown tables. Results go to stdout, progress to stderr.

## Conventions worth preserving

- **No leakage, ever.** No statistic a row is trained on may include that row's own rating (`test_gbdt_features_never_see_their_own_rating`). No hyperparameter may be chosen on a test fold; use `holdout()`. Never aggregate ratings across situations; that was the 2019 bug. Any new split must group by `pair_keys()` (new song) or `question_keys()` (known song), never by raw `item`, or twins and repeated survey answers leak across it.
- Every model must be deterministic for a given seed (`test_fitting_is_deterministic`), and the README numbers depend on it. Use `np.random.default_rng(seed)` or `random_state=seed`, never global RNG state.
- Models must not crash on unseen users, songs or situations. They fall back to whatever they know.
- Rank with `model.score()` (unclipped), not `predict()`. Clipping at 5 creates ties.
- If model behaviour changes, re-run `carmusic evaluate --repeats 3` and update the README tables and their prose. Then regenerate the banner, which embeds the same numbers.

## Checking correctness

Known facts about the data:

- 4,012 ratings, 42 users (`1001`–`1042`), 139 songs, 26 conditions in 8 dimensions, 10 genres.
- 1,003 ratings have no context.
- Six songs are listed twice (album version vs radio edit). `songs.song` is the shared id; `pair_keys()` / `question_keys()` use it.
- 930 (user, item) pairs, which is 911 (user, song) pairs; every one has a no-context rating.
- Six drivers took the survey more than once, so 101 situational ratings repeat an already-answered (user, song, condition) question.

Known results (3 × 5-fold, seed 0):

- `new-situation` RMSE: `mf` 0.959, `camf` 0.936, `gbdt` 0.935.
- `new-song` RMSE: `mf` 1.405, `gbdt` 1.276, `fm` 1.294.
- A constant predictor scores pairwise accuracy exactly 0.5.

## Branches

Work happens on `feats/refactor-during-2026-revisit`; `master` is the default/PR target.
