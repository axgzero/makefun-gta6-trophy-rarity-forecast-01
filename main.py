#!/usr/bin/env python3
"""Forecast Grand Theft Auto VI (PS5) trophy earn rates on Exophase.

Reads every ``*.in`` case file in $MAKEFUN_INPUT_DIR (else /input/) and writes
one ``<id>.out`` forecast per case to $MAKEFUN_OUTPUT_DIR (else /output/).

Method
------
The quantity being forecast is, per trophy, the percent of the players
Exophase tracks for the game who have earned it, at the case's target time.
Two things move that rate: players *progress* (the numerator, cumulative
earners, only ever grows) and new players *join* (the denominator, players
tracked, dilutes every rate when Exophase refreshes its counts in a lump).
So the program forecasts the rate two ways and reconciles them:

* Rate curve. The trophy's own rate series is fitted with a saturating
  curve, r(t) = R_inf * (1 - exp(-t / tau)), t in hours since the game's
  release: a launch cohort's progress levels off as the reachable players
  are exhausted, so a power law (which accelerates forever) is the wrong
  shape. tau is fitted by a grid search in log space (where the page's
  2-decimal rounding noise lives), shrunk in log space toward a class
  prior (story / strand / grind / collectible / activity / platinum,
  classified from the trophy's own description; priors live in
  data/priors.json) and capped at four times that prior, so a history too
  short to show saturation cannot extrapolate linear growth forever. The
  level is never fitted: the curve only supplies the growth factor from
  the last read to the target, applied to the trophy's own last rate.
* Earners over players. Cumulative earners E_i(t) = rate_i(t) * N(t) / 100
  and tracked players N(t) are fitted with the same saturating curves and
  divided at the target: 100 * E_last * gE / (N_last * gN). This is the
  decomposition that stays honest when a refresh dilutes the rate: the
  rate falls even though nobody lost a trophy.

Which one leads depends on what the history shows. When the trophy's
observed rate is rising, the rate curve leads (its saturation is visible
in the data) and the decomposition only bounds it to within a factor of
two. When the observed rate is falling -- dilution winning over
progression -- the decomposition leads, because a saturating curve cannot
fall. When the history is too thin to fit a curve at all (fewer than three
distinct page snapshots, e.g. the practice cases, whose reads are all the
unchanged launch-night page), the program falls back to a power law in
(1 + t) with the class prior slope, anchored the same way.

Uncertainty is a logit-scale spread combined in quadrature: the trophy's
own fit residual, a model-error term growing with the horizon (no
functional form is the truth, and the launch weekend sits inside it), a
fixed refresh-shock term (Exophase refreshes in lumps hours apart and the
answer is the page as of its last refresh before the target), and a
staleness term when the last read has been unchanged for hours. There is no
random-walk term: the median is a trend extrapolation, not a random walk
from the last value, so its uncertainty is the trend's. Quantiles are the
median shifted by the standard normal quantiles times that spread.

Only the standard library is used. The program is deterministic: no
randomness, no network, no clock reads except the case file's own times.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone

# Standard normal quantiles for the forecast levels [0.05, 0.25, 0.5, 0.75, 0.95].
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# Grid of time constants (hours) for the saturating-curve fits.
TAU_GRID = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 26.0, 34.0,
            44.0, 58.0, 76.0, 100.0, 130.0, 170.0, 220.0, 290.0, 380.0,
            500.0, 650.0, 850.0, 1100.0)

# A fitted time constant is never allowed to exceed this multiple of the
# class prior: a launch-week curve cannot saturate slower than that without
# evidence, and an unconstrained fit on a short, still-rising history would
# otherwise extrapolate near-linear growth for days.
TAU_CAP_MULTIPLE = 4.0

# Built-in fallbacks used only if data/priors.json cannot be read; the file
# in the repository is the authority and these mirror its values.
FALLBACK_SLOPES = {
    "platinum": 0.10, "story": 1.00, "strand": 0.70,
    "grind": 0.35, "collectible": 0.30, "activity": 0.60,
}
FALLBACK_TAUS = {
    "platinum": 240.0, "story": 24.0, "strand": 48.0,
    "grind": 120.0, "collectible": 96.0, "activity": 48.0,
}
FALLBACK = {
    "prior_weight_reads": 6,
    "players_growth_max": 10.0,
    "model_sigma_per_sqrt_day": 0.2,
    "refresh_shock_logit": 0.35,
    "unknown_fit_spread": 0.35,
    "fit_residual_floor": 0.12,
    "fit_residual_cap": 0.9,
    "stale_page_sigma": 0.3,
    "stale_page_hours": 12.0,
}


def load_priors():
    """Load data/priors.json next to this script; fall back to built-ins."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "data", "priors.json")
    cfg = dict(FALLBACK)
    slopes = dict(FALLBACK_SLOPES)
    taus = dict(FALLBACK_TAUS)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        for key in ("prior_weight_reads", "players_growth_max"):
            if key in raw:
                cfg[key] = raw[key]
        spread = raw.get("spread", {})
        for key in ("model_sigma_per_sqrt_day", "refresh_shock_logit",
                    "unknown_fit_spread", "fit_residual_floor",
                    "fit_residual_cap", "stale_page_sigma",
                    "stale_page_hours"):
            if key in spread:
                cfg[key] = spread[key]
        for name, spec in raw.get("classes", {}).items():
            if isinstance(spec, dict):
                if "prior_slope" in spec:
                    slopes[name] = float(spec["prior_slope"])
                if "prior_tau_hours" in spec:
                    taus[name] = float(spec["prior_tau_hours"])
    except (OSError, ValueError):
        pass
    return cfg, slopes, taus


CFG, PRIORS, PRIOR_TAUS = load_priors()


def parse_time(value):
    """Parse an ISO 8601 UTC timestamp (or a bare date) to an aware datetime."""
    if isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        value = value + "T00:00:00Z"
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def logit(pct):
    """logit of a percent, clamped to the scorer's [0.01, 99.99] range."""
    p = min(max(pct, 0.01), 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    """Inverse logit as a percent, clamped to [0.01, 99.99]."""
    if x >= 0:
        p = 1.0 / (1.0 + math.exp(-x)) if x < 700 else 1.0
    else:
        e = math.exp(x)
        p = e / (1.0 + e)
    return min(max(p * 100.0, 0.01), 99.99)


def classify(trophy):
    """Classify a trophy from its tier and description for the curve prior."""
    if trophy.get("tier") == "platinum":
        return "platinum"
    desc = (trophy.get("description") or "").strip()
    if re.search(r"Complete the final mission for\b", desc):
        return "strand"
    if re.search(r"^Complete Chapter \d+\b", desc):
        return "story"
    if re.search(r"^Complete the final mission\.?$", desc):
        return "story"
    if re.search(r"^Complete [A-Z][^.]*\.$", desc) and "mission" not in desc.lower():
        # A named story mission ("Complete Breakfast Run."). Count-based
        # "Complete 250 Stunts." is caught by the grind rule below first.
        if not re.search(r"\d", desc):
            return "story"
    if (re.search(r"\d+\s+(Stunts|Challenges|locations|sets|races)", desc)
            or "100% Completion" in desc
            or re.search(r"gold medal in all", desc)):
        return "grind"
    if (re.search(r"every spot|every location|across Leonida|Discover \d+",
                  desc)
            or re.search(r"Complete \d+ Collectible", desc)):
        return "collectible"
    return "activity"


def fit_saturating(times_h, values):
    """Fit y(t) = Y_inf * (1 - exp(-t / tau)) in log space.

    Returns (tau, log_Y_inf, rms_residual, n) for the best tau on the grid,
    or None when fewer than 3 positive points are available. Reads before
    half an hour after release are excluded: they are pre-launch data and
    t ~ 0 makes the saturating form singular.
    """
    pts = [(t, v) for t, v in zip(times_h, values)
           if t >= 0.5 and v is not None and v > 0]
    if len(pts) < 3:
        return None
    best = None
    for tau in TAU_GRID:
        logs = []
        for t, v in pts:
            s = 1.0 - math.exp(-t / tau)
            if s <= 0:
                break
            logs.append(math.log(v) - math.log(s))
        else:
            mean = sum(logs) / len(logs)
            sse = sum((x - mean) ** 2 for x in logs)
            if best is None or sse < best[0]:
                best = (sse, tau, mean)
    if best is None:
        return None
    sse, tau, mean = best
    rms = math.sqrt(sse / len(pts))
    return tau, mean, rms, len(pts)


def fit_powerlaw(times_h, values):
    """Fit log(value) ~ a + b*log(1 + t). Returns (a, b, rms, n) or None."""
    xs, ys = [], []
    for t, v in zip(times_h, values):
        if v is not None and v > 0:
            xs.append(math.log1p(t))
            ys.append(math.log(v))
    if len(xs) < 2:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0, n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    rms = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, rms, n


def distinct_snapshots(history):
    """Collapse runs of identical reads (Exophase's flat stretches) to one."""
    out = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or ()))
        if key != prev_key:
            out.append(read)
            prev_key = key
    return out


def saturating_growth(times_h, values, prior_tau, w_prior, t_from, t_to):
    """Growth factor f(t_to) / f(t_from) of a saturating fit.

    The fitted tau is shrunk in log space toward ``prior_tau`` with weight
    ``w_prior`` reads and capped at TAU_CAP_MULTIPLE times the prior. The
    level cancels in the ratio, so only tau matters. Returns
    (factor, rms, n) or None when the series is too thin to fit.
    """
    fit = fit_saturating(times_h, values)
    if fit is None:
        return None
    tau, _, rms, n = fit
    w_fit = float(min(n, 48))
    log_tau = (w_fit * math.log(tau) + w_prior * math.log(prior_tau)) \
        / (w_fit + w_prior)
    tau_s = min(math.exp(log_tau), TAU_CAP_MULTIPLE * prior_tau)
    s_from = 1.0 - math.exp(-max(t_from, 0.0) / tau_s)
    s_to = 1.0 - math.exp(-max(t_to, 0.0) / tau_s)
    if s_from <= 0:
        return 1.0, rms, n
    return s_to / s_from, rms, n


def powerlaw_growth(times_h, values, prior_slope, w_prior, t_from, t_to):
    """Growth factor ((1+t_to)/(1+t_from))^b for a power-law fit.

    The fitted log-log slope is shrunk toward ``prior_slope``. Returns
    (factor, n) or None when the series is too thin to fit.
    """
    fit = fit_powerlaw(times_h, values)
    if fit is None:
        return None
    _, b, _, n = fit
    w_fit = float(min(n, 48))
    slope = (w_fit * b + w_prior * prior_slope) / (w_fit + w_prior)
    slope = min(max(slope, 0.0), 1.5)
    return ((1.0 + t_to) / (1.0 + t_from)) ** slope, n


def forecast_case(case):
    """Forecast one case. Returns {"trophies": [{"id": n, "q": [...]}]}."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_days = float(case.get("horizon_minutes") or 0) / 1440.0
    release = parse_time((case.get("game") or {}).get("release_date",
                                                       "2026-11-19"))
    target = parse_time(case["target"])
    target_h = (target - release).total_seconds() / 3600.0

    snaps = distinct_snapshots(history)
    snap_h = [max(0.0, (parse_time(r["at"]) - release).total_seconds()
                  / 3600.0) for r in snaps]
    # The anchor is the LAST READ, not the last distinct snapshot: when the
    # page sits unchanged for hours, the distinct snapshots collapse to the
    # first read of the run, and anchoring there would rewind the clock by
    # the whole flat stretch and exaggerate growth from t ~ 0.
    last_read = history[-1] if history else None
    last_h = max(0.0, (parse_time(last_read["at"]) - release).total_seconds()
                 / 3600.0) if last_read else target_h
    last_n = last_read.get("players_tracked") if last_read else None
    # Only the last 36 hours of distinct snapshots drive the fits: older
    # reads are a different regime (pre-launch / launch night).
    fit_idx = [i for i, h in enumerate(snap_h)
               if target_h - h <= 36.0 + 1e-9]
    if len(fit_idx) < 3:
        fit_idx = list(range(len(snaps)))

    # --- Denominator: players tracked. ---
    n_growth = saturating_growth(
        [snap_h[i] for i in fit_idx],
        [snaps[i].get("players_tracked") for i in fit_idx],
        PRIOR_TAUS["activity"], 0.0, last_h, target_h)
    if n_growth is not None:
        g_n = n_growth[0]
    else:
        pl = powerlaw_growth(
            [snap_h[i] for i in fit_idx],
            [snaps[i].get("players_tracked") for i in fit_idx],
            PRIORS["activity"], 0.0, last_h, target_h)
        g_n = pl[0] if pl is not None else 1.0
    g_n = min(max(g_n, 1.0), CFG["players_growth_max"])
    n_target = last_n * g_n if last_n else None

    # --- Staleness: how long the page sat unchanged before the checkpoint. ---
    stale_h = 0.0
    if len(snaps) >= 2:
        last_change = parse_time(snaps[-1]["at"])
        checkpoint = parse_time(case["at"])
        stale_h = max(0.0, (checkpoint - last_change).total_seconds() / 3600.0)
    stale_sigma = CFG["stale_page_sigma"] * min(
        1.0, stale_h / CFG["stale_page_hours"])

    model_sigma = CFG["model_sigma_per_sqrt_day"] * math.sqrt(
        max(horizon_days, 0.0))
    shock = CFG["refresh_shock_logit"]
    w_prior = float(CFG["prior_weight_reads"])

    rows = []
    for j, trophy in enumerate(trophies):
        cls = classify(trophy)
        prior_tau = PRIOR_TAUS.get(cls, PRIOR_TAUS["activity"])
        prior_slope = PRIORS.get(cls, PRIORS["activity"])
        # The trophy's own last observed rate.
        last_rate = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if j < len(rates) and rates[j] is not None:
                last_rate = rates[j]
                break
        # Series over the fit window: rates, and implied earner counts.
        r_times, r_vals, e_times, e_vals = [], [], [], []
        for i in fit_idx:
            read = snaps[i]
            rates = read.get("rates") or []
            n = read.get("players_tracked")
            if j < len(rates) and rates[j] is not None:
                r_times.append(snap_h[i])
                r_vals.append(rates[j])
                if n:
                    e_times.append(snap_h[i])
                    e_vals.append(rates[j] * n / 100.0)
        e_last = (last_rate * last_n / 100.0
                  if last_rate is not None and last_n else None)

        # --- Decomposition forecast (earners over players). ---
        e_growth = saturating_growth(e_times, e_vals, prior_tau, w_prior,
                                     last_h, target_h)
        if e_growth is not None:
            g_e, e_rms, _ = e_growth
            e_fit_sigma = min(max(e_rms, CFG["fit_residual_floor"]),
                              CFG["fit_residual_cap"])
        else:
            pl = powerlaw_growth(e_times, e_vals, prior_slope, w_prior,
                                 last_h, target_h)
            g_e = pl[0] if pl is not None else 1.0
            e_fit_sigma = CFG["unknown_fit_spread"]
        g_e = min(max(g_e, 0.0), 50.0)
        if e_last is not None and e_last > 0 and n_target:
            median_b = 100.0 * e_last * g_e / n_target
        elif last_rate is not None:
            median_b = last_rate
        else:
            median_b = {"platinum": 0.1, "gold": 1.0,
                        "silver": 1.0, "bronze": 1.0}.get(
                            trophy.get("tier"), 1.0)

        # --- Rate-curve forecast, when the observed rate is rising. ---
        # The log-log slope of the rate series decides which view leads:
        # a falling rate means dilution is winning, and a saturating curve
        # (which cannot fall) is the wrong shape for it.
        pl_rate = fit_powerlaw(r_times, r_vals)
        rate_slope = pl_rate[1] if pl_rate is not None else 0.0
        median_a = None
        fit_sigma = e_fit_sigma
        if rate_slope >= 0 and last_rate is not None:
            r_growth = saturating_growth(r_times, r_vals, prior_tau, w_prior,
                                         last_h, target_h)
            if r_growth is not None:
                g_r, r_rms, _ = r_growth
                median_a = last_rate * g_r
                fit_sigma = min(max(r_rms, CFG["fit_residual_floor"]),
                                CFG["fit_residual_cap"])
            else:
                pl = powerlaw_growth(r_times, r_vals, prior_slope, w_prior,
                                     last_h, target_h)
                if pl is not None:
                    median_a = last_rate * pl[0]
                    fit_sigma = CFG["unknown_fit_spread"]
        if median_a is not None:
            # The rate curve leads, bounded to within a factor of two by
            # the decomposition.
            median = min(max(median_a, 0.5 * median_b), 2.0 * median_b)
        else:
            median = median_b
        median = min(max(median, 0.01), 99.99)

        # --- Spread on the logit scale. ---
        sigma = math.sqrt(fit_sigma ** 2 + model_sigma ** 2 + shock ** 2
                          + stale_sigma ** 2)
        centre = logit(median)
        qs = []
        prev = None
        for z in Z:
            q = round(inv_logit(centre + z * sigma), 4)
            q = min(max(q, 0.01), 99.99)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main(argv):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    if not names:
        print("no .in case files in %s" % in_dir, file=sys.stderr)
        return 1
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r",
                  encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = forecast_case(case)
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(forecast, fh, separators=(",", ":"))
            fh.write("\n")
        print("wrote %s (%d trophies)" % (out_path, len(forecast["trophies"])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
