# Phase 2 Summary — Redesigned Evaluation Environment

Phase 0 audit (`docs/audit_phase0.md`) found that the then-current
experiments could not demonstrate any benefit of TAPNA: the twin was
open-loop (actions changed no state), the legacy security utility was
0 for every candidate in 94-99% of attack ticks (so TAPNA's choices
equal Static Weighted's), the twin never recovered after injection
(so gradual/sustained/repeated all reduced to one saturated regime),
seeds barely differed (std across per-seed means = 0.0018), and
"defensive response rate" counted any `isolate`.

Phase 2 fixes the **evaluation environment** before any statistics are
run. Per the rules, all canonical weight logic is unchanged (`w_S`
ramp, exponential recovery, Pareto filter, weighted sum, CIGI,
`tau=0.15`, `decay_lambda=0.30`), legacy modes and CSVs stay
reproducible, the 107 existing tests stay green, no tuning is done on
evaluation seeds (seeds 1000-1009 are used for tuning / smoke only),
no cherry-picking, and negative results are reported.

Stop condition (this is the deliverable): Phase 2 is complete. Phase 1
(statistics on the new data) and Phase 3 are NOT started.

## What changed

### 2a — Twin dynamics (`dynamics="v2"`, legacy default unchanged)

New `digital_twin/twin_v2.py` / `config/twin_v2.yaml`.

- Mean reversion: after a perturbation ends, trust and network load
  relax toward their per-device baselines at `trust_recovery_rate`
  / `load_recovery_rate` (0.10 / 0.10).
- Per-device heterogeneity: each device has its own baseline trust,
  baseline network load, and susceptibility; only a seed-drawn
  fraction of devices (`targeted_fraction_range: [0.3, 0.6]`) is
  targeted, with random per-device susceptibility
  (`[0.5, 1.5]`). Default intensities therefore **cannot** drive all
  devices to trust 0 / network 1.
- Real attack schedules (intensity per tick, all additionally jittered
  per seed by ±0.2):
  - **Gradual**: linear ramp over 30 ticks to a 0.5 plateau.
  - **Short attack spike**: one 8-tick pulse at 0.85.
  - **Repeated attacks**: 3 pulses at 0.75, each 10 ticks, separated
    by 25-tick gaps.
  - **Sustained**: constant 0.45 (no instant saturation).
- Seed-dependent onset (base 10, ±10 ticks) and intensity (±20%);
  seed-dependent device parameters, targeted set, and telemetry noise.
- Onset base set to 10 (not 50) so there are 10 pre-attack normal
  ticks and the mean reversion has room to act before the attack.
- Legacy twin / legacy experiment path / legacy CSVs untouched.

### 2b — Security utility that discriminates (`us_mode="v2"`, legacy kept)

New `sim/scoring_v2.py` `us_mode="v2"` path; legacy
`us_mode="legacy"` delegates to the unchanged legacy scorers exactly.

Risk of one device state:

    risk(trust, network, tss) = tw*(1-trust)^2 + nw*network^2 + tws*tss

    U_S(a) = clip01(0.5 + 0.5 * (risk_before - risk_after(a)) / risk_scale)

with `trust_weight=0.4`, `network_weight=0.3`, `tss_weight=0.3`,
`risk_scale=0.5`. `risk` is evaluated on the action's **target
device** (per-device state from the twin, not fleet averages); the
TSS term cancels in `risk_before - risk_after` (actions do not change
the fleet TSS), so TSS enters decisions through TAPNA's adaptive
`w_S` and ThreatRuleOnly's tau rule, not through U_S.

Properties (unit-tested in `tests/test_phase2_v2.py`):

- bounded in [0, 1];
- `no_op` has U_S = 0.5 exactly (zero predicted effect);
- the same isolate effect yields a **larger** U_S on a compromised
  target (low trust / high network) than on a healthy one, because the
  risk terms are convex in the deficits `(1 - trust)` and `network`;
- candidates differ whenever their predicted effects differ, so U_S no
  longer collapses under severe threat.

Documented deviation from the suggested form: a fixed `risk_scale`
replaces division by `max(risk_before, eps)`. Dividing by the target's
own risk inflates tiny effects on healthy devices (whose
`risk_before` is near zero) enough to break the required ordering
"isolate compromised > isolate healthy"; the fixed scale keeps every
U_S in [0, 1] with one common denominator.

### 2c — Closed-loop action effects (`closed_loop=True`, documented,
`config/closed_loop.yaml`)

`DigitalTwinV2.apply_action` (called after negotiation, effects land
on the next tick):

- `isolate(target)`: quarantines the target — network load pinned to 0,
  threat propagation and attack exposure stop, `compromised` flag
  clears, capacity penalty +0.05, energy saving -10%, cost penalty
  +0.02.
- `migrate`: moves 50% of the target's network load to the other
  non-isolated devices (average), retains only 50% of threat influence
  (`leak_factor`), plus migration energy +0.02 and cost +0.03.
- `scale_down`: energy -15%, cpu -10%, **no** effect on threat.
- `no_op`: nothing.
- Propagation: each tick, every compromised non-isolated device lowers
  `neighbours=2` other non-isolated devices' trust and raises their
  network load by `propagation_rate * propagation_impact * leak_of_source`
  = 0.10 * 0.05 = 0.005 per (source, neighbour) pair per tick, with
  2 random neighbours.

Closed loop runs are **stratified by strategy** (the chosen action
changes the state, so the trace is strategy-dependent); open-loop runs
(replay over a strategy-independent trace) are run separately so the
closed-loop effect can be shown distinctly.

### 2d — Outcome metrics (spec 2d)

Per strategy × scenario × seed:

- `time_to_containment` — ticks from onset until TSS falls below tau;
  both the bite and the containment must be **sustained for 3
  consecutive ticks** (the baseline TSS straddles tau=0.15, so a single
  tick ≥ τ is not an attack and a single dip < τ is not containment).
  If the attack never bites, containment is 0/True; if it never falls
  back below τ, the value is capped at `num_ticks - onset` and
  `contained` is False.
- `contained` (bool).
- `cumulative_compromised_device_ticks`.
- `peak_TSS`, `area_under_TSS`.
- `trust_recovery_time`, `trust_recovered`.
- `cumulative_energy`, `cumulative_cost`.
- `FairScore_mean` (unchanged FairScore aggregation).
- `mean_U_S_attack` (mean U_S of the chosen action during the attack
  phase).
- `attack_*_share` / `normal_*_share` for each action type (isolates
  the action-type distribution per phase).
- `isolate_share` — the relabeled legacy "defensive response rate"
  (share of **all** ticks whose chosen action type was isolate). It is
  a behaviour descriptor, **not** an outcome metric, and must not be
  read as one in the paper.

### TDD / tests

New tests in `tests/test_phase2_v2.py` cover the spec requirements:

- 2a: trust and load recover after the attack ends; the four
  schedules have the intended shapes (gradual ramp + plateau; single
  bounded pulse; three separated pulses; constant sub-saturation
  plateau); the four are all different; same seed reproduces schedule,
  trajectory, and twin; different seed differs; targeting is partial
  (30-60%) and seed-dependent, never the whole fleet.
- 2b: no_op scores 0.5 and candidates differ (identical_us_fraction=0);
  isolate-compromised outscores isolate-healthy with a real margin;
  U_S is stable under TSS change (TSS term cancels); legacy mode
  still produces the same values as the direct legacy scorers;
  `identical_us_fraction` flags collapsed and discriminating ticks.
- 2c: isolate quarantines and keeps the target off the network and
  clears compromise; migrate shifts load and retains only the threat
  leak; scale_down changes energy but not threat; propagation damages
  neighbours only in closed loop.
- 2d: the sustained-run containment rule itself (single-tick dip inside
  an attack is not containment; single-tick spike at onset is not a
  bite).

Also fixed a latent bug found by these tests: `score_actions_v2`'s
legacy path was missing the `score_resource` import, so
`us_mode="legacy"` would crash at runtime. No canonical logic was
changed; only an import was added.

### Performance engineering

A 4x6x20x100 grid is infeasible on this machine under a naive
per-predictor-call-per-candidate design: the legacy per-action path
calls three ML predictors per candidate (~2.6 s/tick at the default
candidate count), and the former `compute_threat_severity`-per-tick
path cost ~6 ms/tick.

Two reuses cut the per-tick cost to well under one tick budget:

- Batched TSS: one `IsolationForest.decision_function` call for the
  whole 100-tick open-loop trace (pre-build one `_telemetry_frame` per
  state, concat, one call, per-row normalise/blend in numpy), plus
  the per-tick `_tss_batch` used in the closed-loop path. Verified
  against `compute_threat_severity` to float tolerance in the smoke
  verification.
- Batched resource and sustainability predictors in `us_mode="v2"`:
  build one predicted-telemetry frame for all candidates (they share
  the fleet state and differ only by predicted effect) and call each
  predictor once.

Measured per-tick components (reference machine, steady state after
warmup):

- `twin.advance`: ~17 ms
- `_telemetry_frame`: ~2 ms; `_tss_batch(1 state)`: ~26 ms
- `_prediction_totals`: ~36 ms; resource predict: ~23 ms; sustain
  predict: ~29 ms
- `score_actions_v2` (v2, all three batched): ~170 ms
  (the sustainability `_prediction_totals` baseline dominates; the
  U_S loop itself is ~0 ms)
- `_negotiate` (TAPNA): ~0.1 ms

Open-loop one run (seed 1000, Gradual): 18.43 s. Closed-loop TAPNA one
run: 17.52 s (175.2 ms/tick — dominated by the same sustainability
baseline, which is paid once per tick; the 400 predictor calls in the
unfused path cost 22.7 s of the 31 s profile, so the fused path's
3 calls/tick are the real floor). The grid still runs at
~155 s/tick-window for the parameters used, so the 80 closed-loop
task-blocks (4 scenarios × 20 seeds) each run one scenario-seed across
all six strategies, and the 80 open-loop task-blocks compute the shared
trace once and replay it. With 8 workers the process-pool gives a
substantial wall-clock reduction (see runtime estimate below); the
result CSVs are byte-identical to a serial run (verified on a
1-scenario 2-seed 2-strategy slice) and resume is supported
(verified: resuming a two-seed grid skipped the already-completed pair
and produced the correct final CSVs).

### Runtime estimate from smoke (tuning seeds 1000-1009 only)

- Smoking one scenario-seed closed-loop TAPNA: ~11.6-12.6 s / 100
  ticks (115-126 ms/tick; the constant-pressure sustainability
  baseline explains why it is dominated by the same cost across
  scenarios).
- Open-loop replay: 0.40 ms/tick (negotiation only).
- Estimated closed-loop grid (8 workers): ~95.8 min baseline serial;
  ~24 min with 8 workers (wall-clock, order-identical).
- Estimated open-loop grid (8 workers): ~16.3 min (the shared trace
  cost is paid once per scenario-seed, then replay is cheap).

The closed-loop grid dominates the combined wall-clock; the open-loop
grid is quick once the trace exists.

### IDF reproduction (2f)

`diag_idf.py` reproduces the canonical legacy experiment
(seed 42, injection at tick 50, severity 0.5, every device) against
the current code:

- Attack-phase isolate: **50/50** — exact match to the IDF claim(§8.2).
- Normal-phase isolate: **49/50** vs claimed 46/50. The 3-tick
  difference is explained by the Pareto filter + normalised-WU change
  to the canonical TAPNA negotiation between when the IDF was written
  and the current code (see `docs/idf_reproduction.md`); it is not a
  seed or U_S change. Seed is 42; U_S is the unchanged legacy scorer.

## Per-scenario TSS curves (seed 1000, open loop, onset=4,
targeted_fraction=0.39, tau=0.15)

Sample every 10 ticks (t0..t90). All four now differ:

| scenario | t0 | t10 | t20 | t30 | t40 | t50 | t60 | t70 | t80 | t90 |
|---|---|---|---|---|---|---|---|---|---|---|
| Gradual attack | 0.137 | 0.138 | 0.168 | 0.659 | 0.667 | 0.678 | 0.677 | 0.677 | 0.683 | 0.684 |
| Short attack spike | 0.137 | 0.670 | 0.171 | 0.200 | 0.200 | 0.132 | 0.109 | 0.223 | 0.146 | 0.213 |
| Repeated attacks | 0.137 | 0.664 | 0.178 | 0.203 | 0.223 | 0.666 | 0.129 | 0.230 | 0.674 | 0.247 |
| Sustained attack | 0.137 | 0.169 | 0.667 | 0.675 | 0.668 | 0.675 | 0.673 | 0.673 | 0.679 | 0.680 |

Shape confirmation (seed 1000):

- **Gradual**: ramp to a plateau; 7 episodes above τ overall, 1 long
  episode from tick 4 to 99, peak 0.693, final 0.690.
- **Short spike**: one burst at ticks 4-11 (peak 0.684), then decay;
  18 episodes above τ (noise straddles tau after decay), final 0.203.
- **Repeated**: three separated attack blocks
  (ticks 4-23, 25-32, 35-59), each decaying between pulses and rising
  again; 11 episodes above τ, peak 0.694, final 0.216.
- **Sustained**: constant high plateau from tick 4 onward; 2 episodes
  (pre-attack noise + long attack), peak 0.689, final 0.686.

The four curves are visually / numerically distinct, confirming 2a.

## Seed heterogeneity (tuning seeds 1000-1009, open loop)

| scenario | per-seed means (min, max) | std (ddof=1) | verdict |
|---|---|---|---|
| Gradual attack | [0.189, 0.568] | 0.1044 | MEANINGFUL |
| Short attack spike | [0.159, 0.233] | 0.0237 | MEANINGFUL |
| Repeated attacks | [0.170, 0.410] | 0.0690 | MEANINGFUL |
| Sustained attack | [0.183, 0.645] | 0.1291 | MEANINGFUL |

vs the Phase 0 legacy value of 0.0018. Seeds now carry real
information. (Spike has the smallest spread because the 8-tick pulse is
short; Repeated sits between; Gradual and Sustained have the largest
spread because their long onsets make the onset jitter matter.)

## U_S flatness (closed loop, TAPNA, seed 1000)

Across all four scenarios, `us_identical_fraction = 0.000` (exact ties
can still occur when two candidates have identical predicted effects on
the same target, but the fraction is 0 in practice for these runs). The
legacy collapse (~0.94-0.99 during attacks) is gone.

`mean_U_S_attack` (TAPNA, seed 1000): 0.548 (Gradual), 0.538 (Spike),
0.550 (Repeated), 0.551 (Sustained) — all in the 0.5-0.55 band, with
real per-tick variation (so TAPNA's security weight still has
signals to act on). `isolate_share` (behaviour descriptor, not an
outcome metric): 0.98 / 0.91 / 0.96 / 0.97.

## First results table (closed loop, 4 scenarios × 6 strategies × 20
seeds × 100 ticks, outcomes.csv computed; mean ± SD across per-seed
means, ddof=1)

Values are means of per-seed means ± SD across the 20 per-seed means.
Full CSVs: `data/experiments_v2/closed_loop/`.

### time_to_containment (ticks from onset, sustained-run rule; capped
at horizon; 0 = never bit / contained)

| strategy | Gradual | Spike | Repeated | Sustained |
|---|---|---|---|---|
| TAPNA | 47.6 ± 24.7 (95% contained) | — | — | — |
| Greedy Cost | 51.3 ± 35.9 (70%) | — | — | — |
| Greedy Latency | 81.2 ± 25.6 (10%) | — | — | — |
| Static Weighted | 59.0 ± 20.2 (90%) | — | — | — |
| ParetoCrowding | 55.6 ± 27.5 (80%) | — | — | — |
| ThreatRuleOnly | 35.7 ± 25.8 (95%) | — | — | — |

Containment rates (closed loop, 20 seeds, collapsed across the
scenario table where the 4 × 6 grid is 24 cells):

| strategy | Gradual | Spike | Repeated | Sustained |
|---|---|---|---|---|
| TAPNA | 0.95 | 1.00 | 1.00 | 0.95 |
| Greedy Cost | 0.70 | 0.90 | 0.75 | 0.75 |
| Greedy Latency | 0.10 | 0.85 | 0.45 | 0.00 |
| Static Weighted | 0.90 | 0.95 | 0.95 | 1.00 |
| ParetoCrowding | 0.80 | 0.90 | 0.85 | 0.95 |
| ThreatRuleOnly | 0.95 | 0.95 | 1.00 | 0.95 |

### FairScore_mean (± SD across per-seed means)

| strategy | Gradual | (representative; full table in CSV) |
|---|---|---|
| TAPNA | 0.332 ± 0.008 | — |
| Greedy Cost | 0.319 ± 0.008 | — |
| Greedy Latency | 0.298 ± 0.007 | — |
| Static Weighted | 0.329 ± 0.008 | — |
| ParetoCrowding | 0.316 ± 0.007 | — |
| ThreatRuleOnly | 0.327 ± 0.006 | — |

(FairScore is higher-is-better; TAPNA leads on the gradual scenario
but only slightly. Full 24-cell table is in
`data/experiments_v2/closed_loop/outcome_summary_v2.csv`.)

### area_under_TSS (± SD; lower is better)

| strategy | Gradual (representative; full table in CSV) |
|---|---|
| TAPNA | 17.80 ± 2.30 |
| Greedy Cost | 22.62 ± 10.03 |
| Greedy Latency | 49.95 ± 5.57 |
| Static Weighted | 17.88 ± 2.63 |
| ParetoCrowding | 17.74 ± 2.33 |
| ThreatRuleOnly | 17.49 ± 1.99 |

### peak_TSS (± SD; lower is better)

| strategy | Gradual (representative; full table in CSV) |
|---|---|
| TAPNA | 0.297 ± 0.125 |
| Greedy Cost | 0.420 ± 0.207 |
| Greedy Latency | 0.691 ± 0.018 |
| Static Weighted | 0.291 ± 0.128 |
| ParetoCrowding | 0.275 ± 0.094 |
| ThreatRuleOnly | 0.274 ± 0.093 |

### cumulative_compromised_device_ticks (± SD; lower is better)

| strategy | Gradual (representative; full table in CSV) |
|---|---|
| TAPNA | 1242 ± 427 |
| Greedy Cost | 1810 ± 833 |
| Greedy Latency | 3844 ± 936 |
| Static Weighted | 1272 ± 414 |
| ParetoCrowding | 1253 ± 424 |
| ThreatRuleOnly | 1130 ± 387 |

### isolate_share (behaviour descriptor, **not** an outcome metric;
closed loop, Gradual, representative)

| strategy | isolate_share (± SD) |
|---|---|
| TAPNA | 0.980 ± 0.025 |
| ThreatRuleOnly | 0.987 ± 0.012 |
| Static Weighted | 0.974 ± 0.016 |
| Greedy Cost | 0.871 ± 0.159 |
| ParetoCrowding | 0.868 ± 0.029 |
| Greedy Latency | 0.078 ± 0.031 |

## Where TAPNA does not win (honest, per spec)

- **On the Gradual scenario**, TAPNA's `FairScore_mean` (0.332 ± 0.008)
  is the highest among the six strategies, but the margin over
  Static Weighted (0.329 ± 0.008) and ThreatRuleOnly (0.327 ± 0.006)
  is small and the per-seed SD is so small that the head-to-head
  differences are well within the variation across individual seeds
  (the full per-seed table is in the CSV). The claim "TAPNA wins" on
  FairScore should be read as "TAPNA is not worse than the best
  heuristic on this scenario", not as a large margin.
- **Greedy Latency is the clear loser on the Gradual scenario** on
  every security metric (containment 10%, AUC_TSS 49.95 ± 5.57,
  peak_TSS 0.691 ± 0.018, compromised-device-ticks 3844 ± 936). The
  reason is structural, not a TAPNA victory: Greedy Latency maximises
  U_R (resource utility) and ignores the security signal, so it never
  isolates under attack. This is the one cell where TAPNA's advantage
  is unambiguous and large, and it is a negative result about Greedy
  Latency rather than a positive result about TAPNA.
- **On containment, ThreatRuleOnly ties or beats TAPNA** on several
  scenarios (e.g. ThreatRuleOnly 0.95 containment on Gradual vs TAPNA
  0.95; ThreatRuleOnly and TAPNA both 1.00 on Repeated). The simple
  "isolate-while-TSS>tau" rule is competitive because the closed-loop
  dynamics make isolation genuinely effective (isolation clears
  compromise and stops propagation). That is a **good property of the
  closed-loop design** (which the paper needs), not an indictment of
  TAPNA; it also means the paper's central claim "TAPNA is better than
  any static or heuristic rule" must be tested carefully, and the data
  so far say: TAPNA is competitive with the best simple rule and
  better than the worst, with no large margin on the outcome metrics
  where it matters most.
- **On the action-distribution metric `isolate_share`**, every strategy
  that isolates much (TAPNA, ThreatRuleOnly, Static Weighted) scores
  near 0.97-0.99 on the Gradual scenario, so the renamed legacy
  "defensive response rate" is not discriminating here — exactly the
  failure the Phase 0 audit predicted. The paper must not use it as an
  outcome metric; it is reported only as a behaviour descriptor.
- **Negative result to report**: the closed-loop design makes the
  simple "isolate when TSS>tau" rule (ThreatRuleOnly) competitive with
  TAPNA on containment and on peak/AUC TSS, because isolation is
  genuinely effective once threat propagation and recovery are present.
  If the paper's thesis depends on TAPNA decisively beating all simple
  rules on the new environment, the data here do **not** yet support
  that. The honest framing is: the new environment shows that
  closed-loop action effects change the comparison (isolation now has
  teeth), that the four scenarios are genuinely different, that seeds
  now carry information, and that TAPNA is competitive with the best
  simple rule — but a decisive TAPNA-over-heuristics claim is not yet
  demonstrated by these outcomes.

## Files delivered

- New code: `sim/attack_patterns_v2.py`, `sim/scoring_v2.py`,
  `digital_twin/twin_v2.py`, `config/twin_v2.yaml`,
  `config/closed_loop.yaml`.
- New tests: `tests/test_phase2_v2.py` (18 new tests, all passing).
- New diagnostics: `diag_idf.py` (read-only IDF 8.2 reproduction).
- New docs: `docs/idf_reproduction.md` (this file's companion),
  `docs/phase2_summary.md`.
- New data: `data/experiments_v2/closed_loop/` and
  `data/experiments_v2/open_loop/` (80 × 6 × 4 closed-loop
  outcome rows + 80 × 6 × 4 open-loop outcome rows, each with the
  full 30+ outcome columns; raw per-tick CSV + outcome metrics CSV +
  per-(strategy,scenario) summary CSV in each). No legacy
  `data/experiments/` CSVs touched.
- Reuse: `sim.attack_patterns_v2 --grid both` runs both grids
  (default parallel, idempotent, resumable); `--smoke` runs the
  calibration checks; `--sensitivity` runs the 3-param × 3-value ×
  5-seed OFAT sensitivity (same API, not run in this phase because the
  deliverable stop condition is Phase 2 complete and the sensitivity
  is Phase-1-ish; the runner is in place and verified to produce
  `data/experiments_v2/sensitivity/*.csv`).

## Tests

Baseline: **107 tests pass** (full `tests/` suite, `--tb=short`).

New: **18 Phase-2 tests pass** (`tests/test_phase2_v2.py`).

Total reported: **125 tests pass** (≥107 plus 18 new).

Note: the two long suites in `tests/test_tapna_sensitivity.py` and
`tests/test_tapna_ablations.py` and `tests/test_experiment_runner.py`
dominate the baseline wall-clock (the baseline run was ~543 s). The
new 18 tests add ~79 s. No existing test was weakened.

## Not done (by design, stop condition)

- Phase 1 (statistics on the new data): not started.
- Phase 3: not started.
- Sensitivity grid (3 params × 3 values × 5 seeds): the runner
  `run_sensitivity` is implemented and shipped with the same
  parallel/resume API, and is verified to write
  `data/experiments_v2/sensitivity/*.csv` (rankings + Kendall tau),
  but the actual run was deferred to Phase 1 per the stop condition.
  The spec 2c sensitivity is callable as
  `python -m sim.attack_patterns_v2 --sensitivity`.
