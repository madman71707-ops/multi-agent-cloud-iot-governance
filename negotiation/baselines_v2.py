"""V2-only baseline strategies (Phase 2, spec 2e).

ParetoCrowding is the same NSGA-II-style selection rule as the legacy
"NSGA-II" baseline (negotiation/baselines.py), renamed for the v2
experiments to avoid confusion with a full generational NSGA-II
implementation. It REUSES the frozen selection helpers by import (no
copies): the rule, its tie-breaks and its model-evaluation budget are
identical to the legacy baseline; only the label differs.

ThreatRuleOnly is a non-negotiating reference policy: isolate while the
fleet TSS exceeds tau, otherwise behave exactly like Static Weighted. It
shows what a pure threat rule achieves without any multi-objective
trade-off. Its weights are reported as fixed thirds for CIGI only
(preference-free rule, like the legacy NSGA-II reporting).
"""
from negotiation.baselines import nsga2_select_index, static_weighted_baseline


def paretocrowding_select_index(actions, U_R_list, U_S_list, U_C_list):
    """NSGA-II-style selection: first Pareto front, largest crowding distance.

    Identical to negotiation.baselines.nsga2_select_index (imported, not
    copied); kept as a named v2 entry point so the v2 strategy label
    "ParetoCrowding" maps to one function.
    """
    return nsga2_select_index(actions, U_R_list, U_S_list, U_C_list)


def paretocrowding_baseline(actions, U_R_list, U_S_list, U_C_list):
    """ParetoCrowding baseline: return the action chosen by the selection rule."""
    return actions[
        paretocrowding_select_index(actions, U_R_list, U_S_list, U_C_list)
    ]


def threat_rule_only_baseline(
    actions,
    U_R_list,
    U_S_list,
    U_C_list,
    tss,
    tau=0.15,
):
    """Isolate while TSS > tau; otherwise behave like Static Weighted.

    When several isolate candidates exist, the one with the highest U_S is
    chosen; ties break toward the earliest candidate (the same deterministic
    tie-break convention as every other selection rule in this project).
    If TSS > tau but the candidate set contains no isolate, the Static
    Weighted choice is used as a documented defensive fallback (the canonical
    candidate generator always includes at least one isolate, so the fallback
    is unreachable in the v2 grids).
    """
    if tss > tau:
        isolate_indices = [
            index
            for index, action in enumerate(actions)
            if action.get("type") == "isolate"
        ]
        if isolate_indices:
            best = max(isolate_indices, key=lambda index: (U_S_list[index], -index))
            return actions[best]
    return static_weighted_baseline(actions, U_R_list, U_S_list, U_C_list)
