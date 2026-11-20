[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/afgil-ainudin/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/afgil-ainudin/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

Forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, the percent
of the players Exophase tracks who will have earned it at
2026-11-26T00:00:00Z — a week after launch. Entry for [make.fun's
`gta6-trophy-rarity-forecast-01` bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) ($250, twenty places).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every
`<id>.in` case in the input directory and writes `<id>.out` for each:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, five
quantiles of the trophy's earn rate at the case's target. Deterministic —
two runs write the same bytes — and runs in ~0.1 s per case.

## Method

A trophy's Exophase rate rises as players reach it and falls, mechanically,
whenever a refresh adds new tracked players who have earned nothing yet (the
house saw this on launch night: 638 → 3,868 tracked, every rate but the one
at 100% lower). A plain trend on the rate reads that dilution as players
*losing* trophies, and a power law on the earned count explodes — early
launch growth is so fast that extrapolating it puts Chapter 1 near 100%
within a week. A launch is a saturating curve instead: each trophy's rate
climbs toward a ceiling, fast at first, then ever slower. So `main.py`
fits, per trophy,

```
p(t) = C / (1 + exp(-(a + b * log(1 + hours since release))))
```

a logistic in log-time with a ceiling `C`, chosen per trophy from a grid
(multiples of its last rate, its group's lifetime ceiling, 99.99) by best
weighted fit, so the data — not an assumption — decides how high a trophy
can go. Fits use only the reads where the page actually changed (Exophase
refreshes on its own schedule; flat stretches carry no information), with
recent change points weighted more.

Three guardrails:

- **Shrinkage.** Trophies with too few change points borrow their group's
  slope (story chapters, mission strands, grinds, collectibles, exploration,
  economy — classified from each trophy's description and tier), and every
  slope is shrunk halfway toward its group median.
- **Monotone structure.** Chapter 1 ≥ Chapter 2 ≥ Chapter 3 ≥ Chapter 4 ≥
  final mission; 100% completion below every trophy it requires; the
  platinum below everything, since it requires every other trophy.
- **Honest spread.** Quantile spread combines the random-walk term (0.5
  logit units per root day), the fit's residual variance, and a floor for
  refresh risk — the answering read can lag real progress by hours.

When a case's history never moves (the practice cases: every read the same
page), there is no trend to fit and the program falls back to the last
read's rate with a wider-than-baseline spread.

## Sources (what is verified, what is assumed, what is estimated)

- **Verified — the case data itself.** Every rate in a case comes from the
  house's recording of Exophase's GTA VI PS5 trophy page
  (https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/). The
  model is fitted to that history alone at run time.
- **Verified — GTA V's lifetime trophy shape** (Exophase, PS4/PS5):
  first story mission ~97% on PS4, final mission ~34%, 100% completion
  ~3%, platinum ~2%; only ~29% of GTA V players ever finished the story.
  These lifetime levels are used only as ceilings for the saturating
  curves and for the trophy grouping — never as week-one levels.
- **Assumption — the saturating logistic in log-time.** Launch adoption and
  progression are fastest at launch and slow down; the ceiling grid lets
  the fit reject this shape per trophy when the data disagrees.
- **Estimate — every forecast number.** The five quantiles are the model's
  belief about the answering read, not a measurement; the spread says how
  uncertain that belief is.

## Validation (local, synthetic)

No ground truth exists before the target, so the model was tested on a
simulated final-like launch (player-base growth, per-trophy progression
curves, Exophase-style stale refreshes, 309 reads to the deadline, target
a week after launch):

| forecast | mean pinball loss (logit) | typical miss |
|---|---|---|
| this program | **0.359** | 3.94 pp |
| house baseline (persistence) | 0.567 | 5.82 pp |

The first version of this program (power law on the earned count) scored
0.727 on the same test, overshooting early trophies toward 100%; the
saturating logistic is what fixed it. On the practice cases (flat history)
it outputs the last read's rates with a wider spread, like a persistence
forecast should, and all 55 rows are valid (non-decreasing quantiles,
0–100, ≤ 65,536 bytes).

## Files

- `main.py` — the whole forecaster (standard library only)
- `LICENSE` — MIT
- `README.md` — this file
