# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/axgzero/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/axgzero/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, the percent of the players Exophase tracks for the game who will have earned it at 2026-11-26T00:00:00Z (a week after launch), as five quantiles per trophy, scored by a pinball loss on the logit scale. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every `*.in` case file in `$MAKEFUN_INPUT_DIR` (else `/input/`) and writes one `<id>.out` forecast per case to `$MAKEFUN_OUTPUT_DIR` (else `/output/`). Each output is `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row per trophy of the case, quantiles non-decreasing and in [0.01, 99.99]. The program is deterministic: the same case always produces the same bytes.

Check a local run against the kit's practice cases:

```sh
python3 tools/check.py <kit>/input out
```

## Method

The rate being forecast moves for two reasons: players *progress* (the numerator, cumulative earners, only grows) and new players *join* (the denominator, players tracked, dilutes every rate when Exophase refreshes its counts in a lump). Fitting a trend to the rate itself confuses the two, so the program forecasts the rate two ways and reconciles them:

- **Rate curve.** Each trophy's own rate series is fitted with a saturating curve, `r(t) = R_inf * (1 - exp(-t / tau))`, `t` in hours since release: a launch cohort's progress levels off as the reachable players are exhausted, while a power law would accelerate forever. Tau is fitted by a grid search in log space, shrunk in log space toward a class prior and capped at four times that prior, so a history too short to show saturation cannot extrapolate near-linear growth for days. The level is never fitted: the curve only supplies the growth factor from the last read to the target, applied to the trophy's own last rate.
- **Earners over players.** Cumulative earners `E_i(t) = rate_i(t) * N(t) / 100` and tracked players `N(t)` are fitted with the same saturating curves and divided at the target. This decomposition stays honest when a refresh dilutes the rate: the rate falls even though nobody lost a trophy.

Which view leads depends on the history. When the trophy's observed rate is rising, the rate curve leads (its saturation is visible in the data) and the decomposition only bounds it to within a factor of two. When the observed rate is falling — dilution winning over progression — the decomposition leads, because a saturating curve cannot fall. When the history is too thin to fit a curve at all (fewer than three distinct page snapshots, e.g. the practice cases, whose reads are all the unchanged launch-night page), the program falls back to a power law in `(1 + t)` with the class prior slope, anchored the same way. Runs of identical reads (Exophase's flat stretches between refreshes) are collapsed to one snapshot before fitting, so hours of an unchanged page do not overweight the launch-night level; the anchor is always the last read itself.

Trophy classes (from each trophy's own tier and description, at run time) and their priors live in [`data/priors.json`](data/priors.json): story trophies saturate in about a day, side-strand finales slower, grinds and collectibles take most of the week, the platinum essentially never arrives in week one. The priors are judgement, not per-trophy forecasts — every level is anchored on the trophy's own last observed rate, and an informative history outweighs the prior.

Uncertainty is a logit-scale spread combined in quadrature: the trophy's own fit residual, a model-error term growing with the horizon, a fixed refresh-shock term (Exophase refreshes in lumps hours apart and the answer is the page as of its last refresh before the target), and a staleness term when the last read has been unchanged for hours. There is no random-walk term: the median is a trend extrapolation, not a random walk from the last value.

## Files

- `main.py` — the entrypoint; everything above, standard library only.
- `data/priors.json` — class-level slope and time-constant priors, with notes.
- `tools/check.py` — local validator: runs `main.py` on a directory of cases and checks every output against the kit's format contract (and scores it when a reference answer file is supplied).

---

<sub>Entered on [make.fun](https://make.fun/?ref=cefc1nk2&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
