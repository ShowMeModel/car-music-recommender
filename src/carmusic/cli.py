"""Command line: ``carmusic evaluate``, ``carmusic recommend``, ``carmusic situations``.

Results go to stdout; progress and summaries go to stderr, so tables stay
pipeable.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from .data import NO_CONTEXT, load
from .evaluate import cross_validate, markdown, summarize
from .models import DEFAULT_MODEL, MODELS, build
from .protocol import PROTOCOLS
from .recommend import recommend


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _evaluate(args: argparse.Namespace) -> int:
    data = load(args.data)
    names = args.models.split(",") if args.models else list(MODELS)
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        _say(f"unknown model(s): {', '.join(unknown)}; choose from {', '.join(MODELS)}")
        return 2
    protocols = PROTOCOLS if args.protocol == "both" else (args.protocol,)
    started = time.perf_counter()
    progress = _say if args.verbose else None
    per_fold = pd.concat(
        [cross_validate(data, p, names, args.folds, args.seed, args.repeats, progress) for p in protocols],
        ignore_index=True,
    )
    summary = summarize(per_fold)
    repeats = f" x {args.repeats}" if args.repeats > 1 else ""
    for p in protocols:
        print(f"### {p} ({args.folds}-fold{repeats}, seed {args.seed})\n")
        print(markdown(summary, p))
        print()
    if args.csv:
        per_fold.to_csv(args.csv, index=False)
        _say(f"per-fold scores written to {args.csv}")
    _say(f"evaluated {len(names)} model(s) in {time.perf_counter() - started:.1f} s")
    return 0


def _recommend(args: argparse.Namespace) -> int:
    data = load(args.data)
    drivers = data.ratings["user"]
    if args.user not in set(drivers):
        _say(f"unknown driver {args.user}; drivers are {drivers.min()}..{drivers.max()}")
        return 2
    valid = [NO_CONTEXT, *data.situations["condition"]]
    if args.situation not in valid:
        _say(f"unknown situation {args.situation!r}; run `carmusic situations` to list them")
        return 2
    model = build(args.model, args.seed).fit(data.ratings)
    top = recommend(model, data, args.user, args.situation, args.k)
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        seen = "  (rated before)" if row.rated_before else ""
        print(f"{rank:>2}. {row.predicted:.2f}  {row.artist} - {row.title}  [{row.genre}]{seen}")
    _say(f"driver {args.user}, situation {args.situation!r}, model {args.model}")
    return 0


def _situations(args: argparse.Namespace) -> int:
    data = load(args.data)
    for dimension, group in data.situations.groupby("dimension", sort=True):
        print(f"{dimension}: {', '.join(group['condition'])}")
    print(f"(or {NO_CONTEXT!r} for no particular situation)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="carmusic", description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="path to Data_InCarMusic.xlsx (default: data/ in the repo)")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="cross-validate models and print a results table")
    ev.add_argument("--protocol", choices=[*PROTOCOLS, "both"], default="both")
    ev.add_argument("--models", help=f"comma-separated subset of: {', '.join(MODELS)}")
    ev.add_argument("--folds", type=int, default=5)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--repeats", type=int, default=1, help="redraw the folds this many times (default 1)")
    ev.add_argument("--csv", help="also write per-fold scores to this CSV file")
    ev.add_argument("-v", "--verbose", action="store_true", help="report progress on stderr")
    ev.set_defaults(func=_evaluate)

    rec = sub.add_parser("recommend", help="top songs for one driver in one situation")
    rec.add_argument("--user", type=int, required=True, help="driver id, e.g. 1019")
    rec.add_argument("--situation", required=True, help="e.g. 'traffic jam' (see `carmusic situations`)")
    rec.add_argument("-k", type=int, default=5, help="how many songs (default 5)")
    rec.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL)
    rec.add_argument("--seed", type=int, default=0)
    rec.set_defaults(func=_recommend)

    sit = sub.add_parser("situations", help="list every driving situation in the data")
    sit.set_defaults(func=_situations)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
