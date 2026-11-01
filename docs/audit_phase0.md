# Phase 0 Audit — TAPNA attack-pattern evidence (read-only)

Date: 2026-10-07. Scope: audit only. No canonical logic, experiment code, or
existing CSVs were touched. Evidence = code reads with line references plus
small single-seed diagnostics (seed 42 unless stated; scripts
`diag_phase0.py` and `diag_followup.py` in the repo root reproduce every
number below; they write no CSVs and are untracked scratch). Full 20-seed
experiments were NOT re-run in Phase 0.

The quoted experiment numbers from the reviewer discussion (gradual mean TSS
0.848, sustained 0.860, TAPNA defensive rate ~0.39, mean wS 0.536 at
tau=0.40) are consistent with the diagnostics below; where a number differs
it is because these diagnostics use seed 42 only and/or shorter runs.

## (a) Metric definitions

All three metrics live in `sim/attack_patterns.py`.

**Which actions count as "defensive"** — exactly the `isolate` type, by type
string only, regardless of strategy or motive:

- `DEFENSIVE_ACTION_TYPES = ("isolate",)` — [sim/attack_patterns.py:67](../sim/attack_patterns.py#L67)
- `is_defensive_response = chosen_action["type"] in DEFENSIVE_ACTION_TYPES` — [sim/attack_patterns.py:295](../sim/attack_patterns.py#L295) (written per tick in `_negotiation_row`)

**`defensive_response_rate`** — mean of that boolean over **attack ticks
only** (`under_attack == True`, i.e. ticks whose schedule severity > 0):

- [sim/attack_patterns.py:448-453](../sim/attack_patterns.py#L448-L453):
  `attacked = results[results["under_attack"]]`,
  `response["defensive_response_rate"] = attacked_grouped["is_defensive_response"].mean()`
- `under_attack` itself is `tick_data["severity"] > 0` — [sim/attack_patterns.py:294](../sim/attack_patterns.py#L294)

**`security_response_gain`** — the chosen action's U_S minus the U_S of the
*first* `no_op` candidate in the same tick (there is always exactly one,
generated at index 1 by [digital_twin/action_generator.py:23-34](../digital_twin/action_generator.py#L23-L34));
0.0 if no no_op candidate exists:

- [sim/attack_patterns.py:269-278](../sim/attack_patterns.py#L269-L278). Note it is defined for *every* tick, but the
  summary averages it over attack ticks only ([sim/attack_patterns.py:455-457](../sim/attack_patterns.py#L455-L457)).

**`response_lag`** — per *run* (seed), not per tick: the first tick `t >=
attack_started` whose chosen action is defensive; `lag = t - attack_started`;
recorded on the attack-onset row only; `-1` if the run never responds
defensively:

- [sim/attack_patterns.py:331-341](../sim/attack_patterns.py#L331-L341). `attack_started` = first row with `under_attack`
  ([sim/attack_patterns.py:332-334](../sim/attack_patterns.py#L332-L334)); in `_iterate_scenario_environment` the same quantity
  is tracked as the first nonzero-severity tick ([sim/attack_patterns.py:205-213](../sim/attack_patterns.py#L205-L213)).
- Summary: `mean_response_lag` averages only runs that responded (excludes
  -1 via `_mean_lag`), and `never_responded_runs` counts -1s —
  [sim/attack_patterns.py:459-467](../sim/attack_patterns.py#L459-L467).

**Is "isolate" counted when Greedy Latency picks it for resource reasons?
Yes.** The check is type-string only; it cannot see the strategy or the
motive. Measured (mini run, seeds 42+123, 20-tick schedules, TAPNA / Greedy
Latency / Static Weighted): Greedy Latency picks `isolate` in **92.5% of its
attack-tick choices**, purely because `isolate`'s large `network_load`
reduction (−0.30..−0.10, [digital_twin/action_generator.py:17](../digital_twin/action_generator.py#L17)) lowers the
predicted resource pressure that U_R rewards — it never observes TSS. Its
`defensive_response_rate` is therefore the *highest* of all strategies
(0.850 gradual / 1.000 sustained in the mini run vs 0.425 / 0.275 for
TAPNA). The metric currently rewards the strategy least aware of threats.

| scenario | strategy | defensive_response_rate | mean_security_response_gain | mean_response_lag | never_responded_runs |
|---|---|---|---|---|---|
| Gradual attack | TAPNA | 0.425 | 0.011047 | 0.000 | 0 |
| Gradual attack | Greedy Latency | 0.850 | 0.001900 | 0.150 | 0 |
| Gradual attack | Static Weighted | 0.425 | 0.011047 | 0.000 | 0 |
| Sustained attack | TAPNA | 0.275 | 0.001252 | 0.000 | 0 |
| Sustained attack | Greedy Latency | 1.000 | 0.000957 | 0.000 | 0 |
| Sustained attack | Static Weighted | 0.250 | 0.000632 | 0.075 | 0 |

(20-tick schedules, 2 seeds, diagnostic only.)

## (b) Why is TAPNA's defensive rate ~0.39 when TSS≈0.85 and wS≈0.88?

**Verdict: genuine behaviour caused by a property of `score_security` — not
a code bug, with a metric-definition weakness on top.** The adaptive weight
is not the problem; the security objective it multiplies is *identically
zero for every candidate* in saturated attack ticks.

Mechanism, seed 42, 100-tick gradual schedule ([agents/security_agent.py:119-133](../agents/security_agent.py#L119-L133)):

`U_S = clip01((avg_trust + δ_trust − 0.3·(avg_network + δ_net)) · (1 − TSS))`

Under a saturated attack, `avg_trust → ~0`, `avg_network → ~1`, `TSS ≈
0.86`, so the base term is `≈ (0 − 0.3) < 0` for **every** candidate →
clipped to 0. Measured on the gradual trace (seed 42):

- **94/100 ticks have `max(U_S over candidates) == 0`** (99/100 under
  sustained). The one flat exception early on is tick 0.
- The max-U_S candidate **is** an isolate in 100/100 ticks — the
  security-optimal choice is always an isolate when U_S is informative.
- When `U_S ≡ 0`, TAPNA's weighted sum `wR·U_R + 0 + wC·U_C` with
  `wR = wC = (1−wS)/2` induces **exactly the same candidate ordering** as
  Static Weighted's `(U_R + 0 + U_C)/3`. Measured: TAPNA's chosen action
  **equals Static Weighted's on 98/99 ticks with TSS ≥ 0.5** (gradual,
  100/100 sustained); the pooled action mixes coincide
  (TAPNA isolate/migrate/scale_down = 0.350/0.150/0.500; Static Weighted
  0.338/0.163/0.500).
- TAPNA chose isolate in **30/100 ticks** (rate 0.30 on seed 42; the pooled
  ~0.39 over 20 seeds is the same phenomenon). Those 30 = the first ~5
  ticks (where U_S still has spread and TAPNA tracks the U_S argmax) plus
  saturated ticks where the best `(U_R+U_C)` candidate happens to be an
  isolate — i.e. threat-irrelevant luck.

Per-tick excerpt (seed 42, gradual):

| tick | TSS | wS | branch | n_cand | chosen | max U_S |
|---|---|---|---|---|---|---|
| 0 | 0.2168 | 0.3857 | ramp | 10 | isolate | 0.6108 |
| 1 | 0.6654 | 0.7375 | ramp | 9 | isolate | 0.2069 |
| 2 | 0.6997 | 0.7645 | ramp | 7 | isolate | 0.1326 |
| 3 | 0.7267 | 0.7857 | ramp | 9 | isolate | 0.0792 |
| 4 | 0.7698 | 0.8195 | ramp | 5 | isolate | 0.0326 |
| 5 | 0.8029 | 0.8454 | ramp | 7 | isolate | 0.0139 |
| 6 | 0.8370 | 0.8721 | ramp | 5 | scale_down | 0.0000 |
| 7 | 0.8616 | 0.8915 | ramp | 10 | migrate | 0.0000 |
| … | ~0.86 | ~0.89 | ramp | 5–10 | scale_down/migrate/isolate | 0.0000 |

Full candidate dump, tick 6 (TSS=0.8370, wS=0.8721, wR=wC=0.0639, Pareto
front 4/5; scores relative to the no_op candidate):

| idx | type | U_R | U_S | U_C | wS·ΔU_S | (wR+wC)·Δ(R+C) | score |
|---|---|---|---|---|---|---|---|
| 0 | isolate | 0.3395 | 0.0000 | 0.0012 | 0.0000 | 0.0065 | 0.0218 |
| 1 | no_op | 0.2385 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0153 |
| 2 | scale_down | 0.2680 | 0.0000 | 0.1528 | 0.0000 | 0.0117 | **0.0269** |
| 3 | isolate | 0.3047 | 0.0000 | 0.0898 | 0.0000 | 0.0100 | 0.0252 |
| 4 | migrate | 0.2813 | 0.0000 | 0.0922 | 0.0000 | 0.0086 | 0.0239 |

Even with wS = 0.87, `wS·ΔU_S = 0` for every candidate: the decision is
decided entirely by U_R + U_C, and `scale_down`'s sustainability advantage
(U_C = 0.15 vs isolate's 0.00–0.09) wins. `security_response_gain`
collapses for the same reason (mean 0.011 gradual / 0.0013 sustained for
TAPNA in the mini run — near zero for *all* strategies).

Classification: (i) **genuine behaviour** — the utility collapse under
saturation is real, not an arithmetic error; (ii) **definition weakness** —
`defensive_response_rate` counts type strings, so it conflates
threat-motivated isolates with resource-motivated ones (part a) and cannot
distinguish TAPNA from Static Weighted in saturated phases, where they are
provably the same selector; (iii) **no code bug found** in the metric
plumbing — rates, gains and lags are computed as documented.

## (c) Is the gradual attack actually gradual? No — it saturates in ~6 ticks

Severity ramps 0.1→0.5 linearly every tick
([sim/attack_patterns.py:101-110](../sim/attack_patterns.py#L101-L110)), but TSS is saturated from tick ~6 on
(seed 42, 100 ticks):

| tick | 0 | 5 | 10 | 15 | 20 | 40 | 60 | 80 | 99 |
|---|---|---|---|---|---|---|---|---|---|
| Gradual TSS | 0.217 | 0.803 | 0.863 | 0.864 | 0.843 | 0.866 | 0.856 | 0.859 | 0.859 |
| Sustained TSS | 0.755 | 0.861 | 0.863 | 0.864 | 0.843 | 0.866 | 0.856 | 0.859 | 0.859 |

- Gradual mean TSS **0.8477**, sustained **0.8598** (seed 42; the reviewer's
  0.848 / 0.860 are the 20-seed versions of the same numbers).
- Gradual TSS reaches 95% of its max at **tick 6**; from ~tick 7 the two
  scenarios' TSS values are *identical* because every device sits at the
  trust floor 0 / network ceiling 1 in both (state saturation), and the RNG
  stream is the same.
- Why saturation is unavoidable: `inject_threat` subtracts `severity` from
  **every** device's trust and adds it to network load each attack tick
  ([digital_twin/threat_injector.py:5-9](../digital_twin/threat_injector.py#L5-L9)), and `twin.tick()` is a pure
  σ=0.02 random walk with **no mean reversion**
  ([digital_twin/twin.py:35-39](../digital_twin/twin.py#L35-L39)). Any severity ≥ ~0.1 sustained for 3–6
  ticks drives all 100 devices to the floor/ceiling; "Repeated attacks"
  (severity 0.5 every 10th tick) therefore *also* stays saturated between
  injections (its TSS never drops below 0.40 after the first injection —
  seed-42 mean 0.8445). Effectively three of the four scenarios test the
  same saturated regime; only "Short attack spike" (2 ticks) probes
  partially-saturated dynamics.

## (d) Does the chosen action feed back into the twin? **No — strictly open-loop**

- The environment generator takes only `(threat_schedule, seed, num_ticks)`
  and never sees a chosen action: [sim/attack_patterns.py:189-237](../sim/attack_patterns.py#L189-L237)
  (`_iterate_scenario_environment`); the trace is strategy-independent by
  construction, which is precisely what makes the cached replay equivalent.
- The canonical loop has the same shape: `twin.tick()` → `inject_threat` →
  `generate_telemetry` → `generate_candidate_actions` → `_score_actions` →
  `compute_threat_severity` → negotiate; the chosen action is only recorded
  in the row ([sim/experiment_runner.py:116-149](../sim/experiment_runner.py#L116-L149)).
- Repo-wide, device state is mutated in exactly two places on every
  experiment path: `twin.tick()` random walks ([digital_twin/twin.py:36-39](../digital_twin/twin.py#L36-L39))
  and `inject_threat` ([digital_twin/threat_injector.py:7-8](../digital_twin/threat_injector.py#L7-L8)).
  `dashboard/app.py` and `backend/api.py` also set device fields, but those
  are demo/mock endpoints no experiment calls.
- Numeric probe: re-simulating seed 42 reproduces the gradual trace exactly
  (identical TSS sum), and the cached-vs-uncached equivalence is already
  pinned by `test_cached_replay_matches_uncached_execution`.

**State variables changed by each action type, in the simulation: none.**
`predicted_effect` dictionaries (`cpu_load`, `network_load`,
`trust_score`, `cost_per_hour`, `energy_kwh`) are *predictions* consumed
only by the scoring agents — U_R via
[agents/resource_agent.py:29-49](../agents/resource_agent.py#L29-L49), U_S via
[agents/security_agent.py:119-133](../agents/security_agent.py#L119-L133), U_C via
[agents/sustainability_agent.py:27-62](../agents/sustainability_agent.py#L27-L62). `no_op` = all zeros;
`isolate` predicts network −0.30..−0.10, trust +0.02..−; `migrate` /
`scale_down` have small randomised ranges
([digital_twin/action_generator.py:38-60](../digital_twin/action_generator.py#L38-L60)). Consequence: no choice can
reduce TSS; every "response" metric measures what was *chosen*, never what
it *did*.

## (e) Candidates per tick: 5–10, global (not per-device)

- `total_actions = random.randint(5, 10)` — [digital_twin/action_generator.py:5-7](../digital_twin/action_generator.py#L5-L7).
- Measured over 100 ticks (seed 42, gradual): min 5, max 10, mean 7.35
  (735 candidates). Composition: isolate 283, no_op 100, migrate 184,
  scale_down 168.
- Exactly **one `no_op`** candidate in 100/100 ticks; 1–6 `isolate`
  candidates per tick; the **first isolate always targets the lowest-trust
  device** ([digital_twin/action_generator.py:8-14](../digital_twin/action_generator.py#L8-L14)).
- The list is **global**: one candidate list per tick, each non-guaranteed
  action targeting a *random* device — there is no per-device candidate
  enumeration anywhere ([digital_twin/action_generator.py:5-72](../digital_twin/action_generator.py#L5-L72)).

## (f) tau=0.40: the "ramp never triggers" claim is wrong for the real experiment

`compute_dynamic_weight` ramps iff `TSS > tau`
([negotiation/tapna.py:38-44](../negotiation/tapna.py#L38-L44)). The claim ("tau=0.40 does not trigger the
security-ramp branch") is **not true for the actual 100-tick experiment**;
it holds only for runs that never reach a threat tick. Note the claim is
not in any repo file (repo-wide search finds it nowhere outside
node_modules) — it appears to come from short/empty-schedule test runs,
where the normal-phase TSS ceiling is ≈ 0.25 < 0.40 and wS pins to 1/3.

What actually happens in the canonical 100-tick experiment
(`threat_schedule={50}`, seed 42):

- Pre-injection TSS ≈ 0.10–0.25. The tick-50 injection drops avg trust
  0.83 → 0.33 and TSS jumps to ≈ 0.76.
- **The twin never recovers**: avg trust is 0.3265 at tick 50 and 0.3159 at
  tick 99 (pure σ=0.02 walk, no mean reversion), so TSS stays ≈ 0.76–0.77
  for the entire second half → at tau=0.40 the ramp triggers on **50/100
  ticks** (all post-injection). Mean wS: **0.5375** at tau=0.40 vs 0.5891
  at tau=0.15 (seed 42) — matching the reported pooled 0.536 at tau=0.40.

Per-scenario branch table (seed 42, 100 ticks, Full TAPNA weights):

| schedule | tau | ramp ticks /100 | mean wS | wS on ramp ticks |
|---|---|---|---|---|
| Gradual | 0.40 | 99 | 0.8329 | 0.8379 |
| Gradual | 0.15 | 100 | 0.8806 | 0.8806 |
| Short spike | 0.40 | 51 | 0.5838 | 0.8245 |
| Short spike | 0.15 | 79 | 0.6235 | 0.6966 |
| Repeated | 0.40 | 100 | 0.8272 | 0.8272 |
| Repeated | 0.15 | 100 | 0.8780 | 0.8780 |
| Sustained | 0.40 | 100 | 0.8442 | 0.8442 |
| Sustained | 0.15 | 100 | 0.8900 | 0.8900 |
| Canonical {50} | 0.40 | 50 | 0.5375 | 0.7416 |
| Canonical {50} | 0.15 | 78 | 0.5891 | 0.6568 |

**Correct statement for the doc**: at tau=0.40 the ramp triggers on every
tick with TSS > 0.40 — i.e. all saturated attack ticks in the attack-pattern
scenarios and the entire post-injection phase of the canonical experiment.
tau=0.40 is not degenerate; it is "post-attack-only": it suppresses the
ramp during normal phases (pre-attack TSS ≈ 0.10–0.25 can exceed 0.15 but
never 0.40), while in saturated phases it produces nearly the same wS as
tau=0.15 (≈ 0.83–0.84 vs ≈ 0.88–0.89, because wS = 1/3 + (2/3)·(TSS−τ)/(1−τ)
compensates for the higher threshold when TSS is high).

## (g) The reported std is pooled over all tick × seed rows (ddof=1)

- `build_attack_response_summary` groups by `(scenario, strategy)` and takes
  pandas `mean`/`std` (sample std, **ddof=1**) over **every tick of every
  seed** in the group — [sim/attack_patterns.py:442-445](../sim/attack_patterns.py#L442-L445).
- Same convention in the canonical `build_summary`
  ([sim/experiment_runner.py:221-242](../sim/experiment_runner.py#L221-L242), docstring: "sample standard
  deviation, ddof=1, over every tick of every seed") and the sensitivity
  summary ([sim/sensitivity_runner.py:268-293](../sim/sensitivity_runner.py#L268-L293)).
- Numeric demonstration (TAPNA, gradual, 3 seeds × 20 ticks): per-seed mean
  TSS = 0.8053 / 0.8017 / 0.8037; **pooled std over 60 rows = 0.1494**
  (what the CSVs report) vs **std across the 3 per-seed means = 0.0018**
  vs mean of per-seed stds = 0.1520.
- Consequence: the reported std is dominated by within-run tick-to-tick
  variation (plus schedule/seed effects), **not** by seed-to-seed
  replication variation. It is neither wrong nor a bug, but it cannot be
  read as "run-to-run variability" — that is what Phase 1's per-seed-first
  statistics fix.

## Summary of issues found (input to Phase 1+)

1. `defensive_response_rate` counts `isolate` by type regardless of motive
   — Greedy Latency scores highest (0.85–1.00) while ignoring threats
   entirely. Recommend re-defining "defensive" as threat-reducing, or
   reporting action-type distributions per phase (Phase 1).
2. In saturated attack ticks U_S ≡ 0 for all candidates → TAPNA is
   *provably identical* to Static Weighted there (98–100% choice
   agreement). Any TAPNA-vs-baseline claim should be split normal-phase vs
   attack-phase, and the saturated-regime collapse of `score_security`
   deserves an explicit limitation note (the (1−TSS) multiplier plus
   clip-at-0 kills the security signal exactly when wS is maximal).
3. The gradual scenario saturates in ~6 ticks and "Repeated attacks" is
   effectively sustained (no twin recovery anywhere). Attack-pattern
   conclusions currently rest on one regime.
4. Everything is open-loop: no choice changes any state variable; response
   metrics measure choices, not effects (motivates Phase 2's closed loop).
5. tau=0.40 doc claim incorrect for the real experiment (ramp fires on
   ~50/100 canonical ticks; mean wS ≈ 0.536 is real, not a bug).
6. Reported stds are pooled tick×seed (ddof=1), not across per-seed means.
7. Environment fact worth documenting: the twin has no trust recovery
   (pure random walk + clip), so any attack ≥ ~0.3 severity permanently
   saturates the state — this drives (b), (c) and (f).
