# IDF Section 8.2 Reproduction

Read-only diagnostic (same contract as the legacy `diag_idf.py` used in
Phase 2): the script reproduces the canonical legacy experiment and quotes
the results below verbatim. It writes no CSVs and edits no canonical files.
The IDF is NOT edited.

## What the IDF claims (section 8.2)

> After the tick-50 injection, TAPNA selected `isolate` in 50 of 50
> attack ticks, and in 46 of 50 normal ticks.

Canonical setup, reproduced verbatim from `sim/run_simulation.py`:
`DigitalTwin(100)`, 100 ticks, deterministic threat injection at
tick 50 that pushes **every** device by severity 0.5, legacy
`score_resource` / `score_security` / `score_sustainability` scorers,
and the canonical TAPNA negotiation (`tau=0.15`, `decay_lambda=0.30`).
Seed = 42.

## What the current code gives (seed 42, current seeded behaviour)

Current coded behaviour seeds both the stdlib `random` and `np.random`
with the run seed before the run (`np.random.seed(SEED)`), which is the
same contract the current experiment runners use. Under that contract the
current code gives on seed 42:

- Normal phase (ticks 0-49, n=50): `isolate` 49/50 (0.98), `migrate` 1/50.
- Attack phase (ticks 50-99, n=50): `isolate` 50/50 (1.00).

Per-phase means:

- Normal phase: mean TSS = 0.1655, mean w_S = 0.3606.
- Attack phase: mean TSS = 0.7669, mean w_S = 0.8172.

Per-tick around the injection (ticks 46-56): isolate at every tick
from 46; TSS jumps from 0.2021 (tick 48) and 0.0956 (tick 49) to
0.7627 at tick 50, and w_S jumps from ~0.355-0.405 to 0.8138 at tick 50.

### Determinism check (what the current code actually does)

The current `run_tapna`/`compute_dynamic_weight`/`pareto_filter`/`_best_action`
in `negotiation/tapna.py` produce a SINGLE deterministic normal-phase isolate
count for a given seed under the current seeded contract. A direct check on
seeds 0-5 (each run seeds `np.random.seed(SEED)` exactly as the current code
does) gives:

- seed 0: 50/50 normal, 50/50 attack
- seed 1: 50/50 normal, 50/50 attack
- seed 2: 43/50 normal, 50/50 attack
- seed 3: 48/50 normal, 50/50 attack
- seed 4: 49/50 normal, 50/50 attack
- seed 5: (same deterministic family; the check above stopped while
  reproducing the distribution and the value for seed 5 was not captured
  before the diagnostic timed out, so it is not quoted here)

The two-phase isolation count is therefore SEED-DEPENDENT: different seeds
give different normal-phase isolate counts (here 43, 48, 49, 50), but for a
given seed the current code gives a single deterministic value. The attack
phase is 50/50 on every seed checked.

## Verdict

| Claim (IDF 8.2) | Current code (seed 42, current seeded behaviour) |
|---|---|
| Attack-phase isolate: 50/50 | **50/50 — exact match** |
| Normal-phase isolate: 46/50 | **49/50 — off by 3** |

**Attack phase matches exactly.** The normal-phase difference is 3 ticks.

## Why the previous explanation was wrong

The previous version of `docs/idf_reproduction.md` attributed the 49/50 vs
46/50 difference to a "Pareto filter + normalised-WU change" to
`negotiation/tapna.py`. That is **wrong** and has been deleted. Evidence:

1. `negotiation/tapna.py` currently contains **no normalisation**. The
   `run_tapna_variant` weighted-sum argmax in `_best_action` is the plain
   `wR*U_R + wS*U_S + wC*U_C` over the (possibly Pareto-filtered) pool, with
   earliest-candidate tie-break. There is no per-call normalisation of the
   weights before the argmax, and there never was one in the current file.
2. A Pareto filter **cannot change the argmax of a positive-weight weighted
   sum**. If the current code's chosen action is `isolate` for a tick, and the
   Pareto filter is enabled, the argmax over the filtered pool is a subset of
   the full candidate set; Pareto filtering can only remove candidates, it
   cannot make a lower-scoring candidate win over the current winner. So
   Pareto filtering alone cannot explain a different isolate count between two
   identical-seeded runs of the same function.

The unsupported paragraph has been deleted. The doc now only states what the
evidence supports.

## What the evidence actually shows

The current code's normal-phase isolate count is SEED-DEPENDENT but
DETERMINISTIC PER SEED. So:

- The current seed-42 run gives 49/50, deterministically.
- The IDF doc's 46/50 was **not** produced by the current seed-42 run.
- The most likely cause is that the IDF doc's 46/50 was produced by a
  **different seed**, or by a **different (older) code path**, or by an
  **unseeded or differently-seeded RNG** at the time the IDF numbers were
  produced. The current diagnostic cannot reconstruct the original run log,
  so it cannot identify WHICH of these is the cause; it can only show that
  the discrepancy is seed-dependent and therefore plausibly seed-driven.

One plausible specific mechanism, consistent with the data, is an RNG-source
difference: the current code seeds `np.random` before the run. If at the time
the IDF numbers were produced the legacy canonical experiment did **not**
seed `np.random` (or seeded it differently), then `DigitalTwin.__init__`'s
`np.random.uniform(...)` draws for device creation — and `generate_candidate_actions`'s
stdlib-`random` draws for candidate effects — would have produced a different
device universe and/or candidate-effect universe, yielding a different
normal-phase isolate count even for the same nominal "seed". A direct check
confirms that WITHOUT `np.random.seed(seed)`, `DigitalTwin(100)` produces a
different device universe each call (mean fleet trust varies across runs),
even when the stdlib `random` is seeded identically. So an unseeded or
differently-seeded numpy RNG at IDF-production time is a **plausible**
explanation for the 46/50 value. The diagnostic does not and cannot prove it
was the actual cause, because the original run log is not available; it only
shows the mechanism is plausible and seed-dependent.

For this reason the doc now says only: the normal-phase difference is
seed-dependent and deterministic-per-seed under the current code, the
Pareto/normalisation explanation was wrong and has been removed, and the most
likely remaining cause is a different seed or a different (possibly
unseeded/differently-seeded) RNG contract at IDF-production time, without
claiming to have proven which.
