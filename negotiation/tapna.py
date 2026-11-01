import math


def pareto_filter(actions, scores):
    non_dominated_actions = []

    for i, action in enumerate(actions):
        dominated = False
        current_scores = (
            scores["U_R"][i],
            scores["U_S"][i],
            scores["U_C"][i],
        )

        for j in range(len(actions)):
            if i == j:
                continue

            other_scores = (
                scores["U_R"][j],
                scores["U_S"][j],
                scores["U_C"][j],
            )

            if (
                all(other >= current for other, current in zip(other_scores, current_scores))
                and any(other > current for other, current in zip(other_scores, current_scores))
            ):
                dominated = True
                break

        if not dominated:
            non_dominated_actions.append(action)

    return non_dominated_actions


def compute_dynamic_weight(tss, tau=0.15, prev_wS=1/3, decay_lambda=0.3):
    if tss > tau:
        wS = 1/3 + (2/3) * min(1, (tss - tau) / (1 - tau))
    else:
        wS = 1/3 + (prev_wS - 1/3) * math.exp(-decay_lambda)

    return wS


def compute_adaptive_weight(
    tss,
    tau=0.15,
    prev_wS=1/3,
    use_decay=True,
    decay_lambda=0.3,
):
    """Adaptive security weight, with the decay term switchable.

    The threat-driven ramp above ``tau`` is identical in both modes. The modes
    differ only in the low-threat branch:

    * ``use_decay=True`` reproduces :func:`compute_dynamic_weight`, which
      relaxes the previous weight back toward the neutral 1/3 exponentially
      (the full TAPNA behavior). ``decay_lambda`` sets the decay rate.
    * ``use_decay=False`` drops the memory term entirely and returns the
      neutral 1/3, so the weight depends only on the current tick's TSS.

    ``tau`` is the threat threshold separating the ramp from the decay branch,
    and ``decay_lambda`` is the exponential decay rate. Both default to the
    canonical Full TAPNA values, so calling this with the defaults is exactly
    the original behavior.

    ``prev_wS`` and ``decay_lambda`` are unused when ``use_decay=False``; they
    are kept in the signature so both modes share one calling convention.
    """
    if not use_decay:
        if tss > tau:
            return 1/3 + (2/3) * min(1, (tss - tau) / (1 - tau))
        return 1/3

    return compute_dynamic_weight(
        tss,
        tau=tau,
        prev_wS=prev_wS,
        decay_lambda=decay_lambda,
    )


def _best_action(actions, pool, U_R_list, U_S_list, U_C_list, weights):
    """Highest weighted sum over ``pool``; ties resolve to the earliest action."""
    best_action = None
    best_score = None

    for action in pool:
        index = actions.index(action)
        score = (
            weights["wR"] * U_R_list[index] +
            weights["wS"] * U_S_list[index] +
            weights["wC"] * U_C_list[index]
        )

        if best_score is None or score > best_score:
            best_score = score
            best_action = action

    return best_action


def run_tapna_variant(
    actions,
    U_R_list,
    U_S_list,
    U_C_list,
    tss,
    prev_wS,
    use_pareto=True,
    use_adaptive_weight=True,
    use_decay=True,
    tau=0.15,
    decay_lambda=0.3,
):
    """TAPNA with its three components independently switchable.

    The components are:

    * ``use_pareto`` - Pareto filtering of the candidate set. When False the
      full candidate set is scored.
    * ``use_adaptive_weight`` - threat-driven security weight. When False the
      weights are the fixed equal thirds, independent of TSS.
    * ``use_decay`` - exponential relaxation of the previous security weight in
      the low-threat branch. Only consulted when ``use_adaptive_weight`` is
      True.

    ``tau`` and ``decay_lambda`` parameterize the adaptive weight for
    sensitivity analysis. Both default to the canonical Full TAPNA values, so
    the defaults reproduce the original behavior exactly.

    Returns ``(chosen_action, weights)``. Ties in the weighted sum are broken
    toward the candidate that appears earliest in ``actions``, matching the
    full TAPNA tie-break.
    """
    if not actions:
        raise ValueError("run_tapna_variant requires at least one candidate action")

    if use_pareto:
        pool = pareto_filter(
            actions,
            {"U_R": U_R_list, "U_S": U_S_list, "U_C": U_C_list},
        )
    else:
        pool = list(actions)

    if use_adaptive_weight:
        wS = compute_adaptive_weight(
            tss,
            tau=tau,
            prev_wS=prev_wS,
            use_decay=use_decay,
            decay_lambda=decay_lambda,
        )
        wR = (1 - wS) / 2
        weights = {"wR": wR, "wS": wS, "wC": wR}
    else:
        # Exact equal thirds rather than (1 - 1/3)/2, which is not exactly
        # 1/3 in floating point. This keeps the fixed-weight variants on the
        # same 1/3 evaluation basis as the FairScore column.
        weights = {"wR": 1 / 3, "wS": 1 / 3, "wC": 1 / 3}

    return _best_action(
        actions, pool, U_R_list, U_S_list, U_C_list, weights
    ), weights


def run_tapna(actions, U_R_list, U_S_list, U_C_list, tss, prev_wS):
    """Full TAPNA: Pareto filtering + adaptive weight + exponential decay.

    Equivalent to ``run_tapna_variant`` with every component enabled.
    """
    return run_tapna_variant(
        actions,
        U_R_list,
        U_S_list,
        U_C_list,
        tss,
        prev_wS,
        use_pareto=True,
        use_adaptive_weight=True,
        use_decay=True,
    )