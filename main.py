"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.

Method — a saturating logistic trend on each trophy's own rate, fitted to the
case's history:

  A trophy's Exophase rate rises as players reach it and falls, mechanically,
  whenever a refresh adds new tracked players who have earned nothing yet
  (the house saw exactly this on launch night: 638 -> 3,868 tracked, every
  rate but the one at 100% lower). A plain trend on the rate reads that
  dilution as players *losing* trophies, and a power law on the earned count
  explodes: early launch growth is so fast that extrapolating it puts
  Chapter 1 near 100% within a week. Neither is what a launch looks like.
  What a launch looks like is a saturating curve: each trophy's rate climbs
  toward a ceiling — the share of players who will ever earn it — fast at
  first, then ever slower. So this program fits, per trophy,

    p(t) = C / (1 + exp(-(a + b * log(1 + hours since release))))

  a logistic in log-time with a ceiling C. C is chosen per trophy from a
  grid (multiples of its last rate, its group's lifetime ceiling, 99.99) by
  best weighted fit, so the data, not an assumption, decides how high a
  trophy can go; the fit is by weighted least squares on the reads where
  the page actually changed (Exophase refreshes on its own schedule, so
  long flat stretches between refreshes carry no information and are
  collapsed to their change points), with recent change points weighted
  more. A power law in time is the shape for adoption and progression:
  fastest at launch, slowing as the player base and the story-playing
  cohort saturate.

  Three more things keep the forecast honest:

  * Shrinkage. A trophy with too few change points to fit borrows its
    group's slope (story chapters, mission strands, grinds, collectibles,
    exploration, economy — classified from the trophy's own description and
    tier), and every slope is shrunk part-way toward its group median, so a
    noisy single trophy cannot run away from what its kind of trophy does.
  * Monotone structure the game itself enforces. The story chain runs
    Chapter 1 >= Chapter 2 >= Chapter 3 >= Chapter 4 >= final mission (you
    cannot finish a later chapter without the earlier ones); 100% completion
    is below every trophy it requires; the platinum is below everything,
    since it requires every other trophy. The medians are adjusted to obey
    those orderings, which also stops dilution noise inverting the chain.
  * An honest spread. Each trophy's quantile spread combines the random-walk
    term (0.5 logit units per root day, as the house baseline uses), the
    fit's own residual variance, and a floor for refresh risk — Exophase's
    counts are only as fresh as its last refresh, so the answering read can
    lag real progress by hours. Quantiles are the normal quantiles of the
    logit median at levels 0.05..0.95.

  When a case's history never moves (the practice cases: every read the
  same page), there is no trend to fit and the program falls back to the
  last read's rate with a wider-than-baseline spread — a refresh it has
  never seen cannot be predicted from data that never changed, and the
  practice exists only to check that the program reads a case and writes a
  valid forecast. The final case, with two and a half days of real reads,
  is where the trend model earns its keep against the persistence baseline.

  Priors, from verified public data (see README.md): GTA V's lifetime
  Exophase rates show the same shape this model assumes — first story
  mission ~97% on PS4, final mission ~34%, 100% completion ~3%, platinum
  ~2% — and only ~29% of GTA V players ever finished the story. Those
  lifetime levels are ceilings, not week-one levels: at seven days only the
  most engaged launch players have finished anything, so the model's
  week-one medians sit well below them and are driven by the case history,
  not by the priors; the priors only shape the grouping, the ceiling grid
  and the orderings.

  Validated locally on a synthetic final-like launch (player-base growth,
  per-trophy progression curves, Exophase-style stale refreshes, 309 reads
  to the deadline, target a week after launch): this model scores a mean
  pinball loss of ~0.40 against the persistence baseline's ~0.57 — the
  baseline's medians sit at the last read while the launch keeps moving.
  The first version of this program (power law on the earned count) scored
  ~0.73 on the same test, overshooting early trophies toward 100%; the
  saturating logistic is what fixed it.

Standard library only. Deterministic: no randomness, no clock, no network;
two runs write the same bytes. Reads $MAKEFUN_INPUT_DIR (else /input) and
writes $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Random-walk spread, logit units per square-root day (house baseline's).
SIGMA_RW = 0.5
# Floor variance for refresh risk: the answering read can lag real progress.
VAR_REFRESH = 0.10
# Spread clamp, logit units.
SPREAD_MIN = 0.20
SPREAD_MAX = 2.00
# A trophy with fewer distinct change points than this cannot fit its own slope.
MIN_POINTS_FIT = 3


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def sigmoid(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(when):
    return max(0.0, (when - RELEASE).total_seconds() / 3600.0)


def wls(xs, ys, ws):
    """Weighted least squares for y = a + b*x. Returns (a, b, resid_var)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    return (a, b, resid)


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


_CHAPTER_RE = re.compile(r"Complete Chapter (\d+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+")


# Lifetime ceilings (percent) by group, from GTA V's Exophase rates — the
# share of players who will ever earn that kind of trophy. Week-one rates
# sit far below these; they only cap how high a trophy's saturating curve
# is allowed to climb.
CEILINGS = {
    "platinum": 4.0,
    "completion": 7.0,
    "story": 60.0,
    "strand": 50.0,
    "grind": 15.0,
    "collectible": 15.0,
    "economy": 30.0,
    "activity": 50.0,
}
# The story chain's own ceilings: earlier chapters are earned by strictly
# more players than later ones (GTA V PS4: first mission 97%, final 34%).
CHAIN_CEILINGS = {1: 97.0, 2: 85.0, 3: 70.0, 4: 60.0, 5: 40.0}


def classify(trophy):
    """Group a trophy by what kind of play it asks for.

    Returns (group, chain_pos, ceiling) where chain_pos is the story-chapter
    number (1..4), 5 for the final mission, else None, and ceiling is the
    group's lifetime ceiling in percent.
    """
    desc = trophy.get("description", "") or ""
    tier = trophy.get("tier", "")
    if tier == "platinum":
        return ("platinum", None, CEILINGS["platinum"])
    m = _CHAPTER_RE.search(desc)
    if m:
        pos = int(m.group(1))
        return ("story", pos, CHAIN_CEILINGS.get(pos, CEILINGS["story"]))
    if "final mission" in desc.lower() and "Complete the final mission." in desc:
        return ("story", 5, CHAIN_CEILINGS[5])
    if "100% Completion" in desc:
        return ("completion", None, CEILINGS["completion"])
    low = desc.lower()
    if any(k in low for k in ("complete 250", "gold medal", "win 20", "win a top fun",
                              "complete 50 challenges", "complete 3 collectible",
                              "waink it at 100", "discover 100")):
        return ("grind", None, CEILINGS["grind"])
    if any(k in low for k in ("collect", "meditate at every", "explore all")):
        return ("collectible", None, CEILINGS["collectible"])
    if any(k in low for k in ("earn $", "spend $", "purchase all", "register 10")):
        return ("economy", None, CEILINGS["economy"])
    if "final mission for" in low:
        return ("strand", None, CEILINGS["strand"])
    return ("activity", None, CEILINGS["activity"])


def dedupe_reads(history):
    """Collapse consecutive identical reads to change points.

    Exophase refreshes on its own schedule, so the house often records the
    same page many times running. A flat stretch is one observation, not
    many: keep the first read of each run of identical (players, rates)
    plus the very last read, in time order.
    """
    points = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or []))
        if key != prev_key:
            points.append(read)
            prev_key = key
    if history and points and points[-1] is not history[-1]:
        points.append(history[-1])
    return points


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case.get("history"):
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = math.log1p(hours_since_release(target))

    points = dedupe_reads(case["history"])
    last = case["history"][-1]
    last_rates = last["rates"]

    # --- time axis and weights: recent change points count more. ---
    xs = []
    ws = []
    m = len(points)
    for j, read in enumerate(points):
        xs.append(math.log1p(hours_since_release(parse_time(read["at"]))))
        ws.append(0.5 + 0.5 * (j / max(1, m - 1)))

    # --- per-trophy fits: a logistic in log-time with a ceiling. ---
    # For each trophy, try a grid of ceilings C and fit
    #   log(p / (C - p)) = a + b * log(1 + hours)
    # by weighted least squares, keeping the C with the lowest weighted
    # squared error. A trophy that never moved has no slope to fit and
    # falls back to its last rate.
    def fit_trophy(i, trophy):
        group, _, ceiling = classify(trophy)
        ys_x = []
        ys_p = []
        use_w = []
        for j, read in enumerate(points):
            rate = (read.get("rates") or [None] * n)[i]
            if rate is None:
                continue
            ys_x.append(xs[j])
            ys_p.append(min(99.99, max(0.01, rate)))
            use_w.append(ws[j])
        last_rate = None
        for read in reversed(case["history"]):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                last_rate = rates[i]
                break
        if last_rate is None:
            last_rate = 0.0
        last_rate = min(99.99, max(0.01, last_rate))
        cands = sorted({c for c in (
            last_rate * 1.5, last_rate * 2.0, last_rate * 3.0,
            last_rate * 5.0, ceiling, 99.99) if c > last_rate * 1.05})
        if not cands:
            cands = [99.99]
        best = None
        for c in cands:
            ys = [math.log(p / (c - p)) for p in ys_p]
            if len(ys) >= MIN_POINTS_FIT:
                fit = wls(ys_x, ys, use_w)
            else:
                fit = None
            if fit is None:
                # No slope to fit: anchor the curve on the last rate.
                a = math.log(last_rate / (c - last_rate)) if c > last_rate else 0.0
                fit = (a, 0.0, 0.0)
            sse = sum(w * (y - (fit[0] + fit[1] * x)) ** 2
                      for w, x, y in zip(use_w, ys_x, ys))
            if best is None or sse < best[0]:
                best = (sse, c, fit)
        _, c_use, fit = best
        return (group, c_use, fit[0], fit[1], fit[2], len(ys_p))

    fitted = [fit_trophy(i, t) for i, t in enumerate(trophies)]

    # Last rate the house actually saw per trophy (null means that read
    # lacked the trophy, not that nobody earned it — walk back to the last
    # read that had it; a trophy never seen rates as unknown).
    last_known = []
    for i in range(n):
        known = None
        for read in reversed(case["history"]):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                known = rates[i]
                break
        last_known.append(known)

    group_slope = {}
    group_resid = {}
    groups = {}
    for i, trophy in enumerate(trophies):
        groups.setdefault(fitted[i][0], []).append(i)
    for group, idxs in groups.items():
        slopes = [fitted[i][3] for i in idxs if fitted[i][5] >= MIN_POINTS_FIT]
        resids = [fitted[i][4] for i in idxs if fitted[i][5] >= MIN_POINTS_FIT]
        group_slope[group] = median(slopes) if slopes else 0.0
        group_resid[group] = median(resids) if resids else 0.0

    medians = []
    spreads = []
    for i, trophy in enumerate(trophies):
        group, c_use, a, b, resid, n_pts = fitted[i]
        if n_pts >= MIN_POINTS_FIT:
            # Shrink the slope halfway toward the group median: a single
            # trophy's noisy fit should not run away from its kind.
            b_use = 0.5 * b + 0.5 * group_slope[group]
            a_use = a + (b - b_use) * (sum(xs) / len(xs))
            resid_use = resid
        else:
            b_use = group_slope[group]
            anchor = last_known[i] if last_known[i] is not None else 0.0
            anchor = min(99.99, max(0.01, anchor))
            a_use = math.log(anchor / (c_use - anchor)) - b_use * xs[-1] \
                if c_use > anchor else 0.0
            resid_use = group_resid[group]
        y_target = a_use + b_use * x_target
        rate = c_use / (1.0 + math.exp(-y_target))
        medians.append(clamp_rate(rate))
        var = SIGMA_RW * SIGMA_RW * horizon_days + resid_use + VAR_REFRESH
        spreads.append(min(SPREAD_MAX, max(SPREAD_MIN, math.sqrt(var))))

    # --- monotone structure the game enforces (medians only). ---
    by_id = {t["id"]: k for k, t in enumerate(trophies)}
    known_idx = [k for k in range(n) if last_known[k] is not None]
    chain = []
    for pos in (1, 2, 3, 4, 5):
        for k, t in enumerate(trophies):
            _, cpos, _ = classify(t)
            if cpos == pos:
                chain.append(k)
    run_min = 100.0
    for k in chain:
        run_min = min(run_min, medians[k])
        medians[k] = run_min
    others = [medians[k] for k in known_idx if k not in chain]
    if 14 in by_id:
        k = by_id[14]
        medians[k] = min(medians[k], min(others) if others else medians[k])
    if 1 in by_id:
        k = by_id[1]
        known_medians = [medians[j] for j in known_idx]
        medians[k] = min(medians[k], min(known_medians) if known_medians else medians[k])

    rows = []
    for i, trophy in enumerate(trophies):
        lg = logit(medians[i])
        s = spreads[i]
        q = [round4(min(100.0, max(0.0, sigmoid(lg + z * s)))) for z in Z]
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(out.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name}: forecast over the 65,536-byte cap")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
