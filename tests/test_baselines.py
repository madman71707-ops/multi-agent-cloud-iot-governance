from negotiation.baselines import (
    greedy_cost_baseline,
    greedy_latency_baseline,
    nsga2_baseline,
    nsga2_select_index,
    static_weighted_baseline,
)


def _actions(count):
    return [{"id": str(index), "type": "no_op"} for index in range(count)]


def _is_dominated(left, right):
    return all(l >= r for l, r in zip(left, right)) and any(
        l > r for l, r in zip(left, right)
    )


def test_existing_baselines_keep_their_original_behavior():
    actions = _actions(3)
    u_r = [0.1, 0.9, 0.5]
    u_s = [0.5, 0.2, 0.9]
    u_c = [0.8, 0.3, 0.4]

    assert greedy_latency_baseline(actions, u_r) is actions[1]  # argmax U_R
    assert greedy_cost_baseline(actions, u_c) is actions[0]  # argmax U_C
    # argmax of the equal-weighted sum: a=0.467, b=0.467, c=0.600
    assert static_weighted_baseline(actions, u_r, u_s, u_c) is actions[2]


def test_nsga2_returns_an_action_from_the_same_candidate_set():
    actions = _actions(6)
    u_r = [0.9, 0.2, 0.5, 0.7, 0.1, 0.6]
    u_s = [0.1, 0.8, 0.4, 0.3, 0.9, 0.5]
    u_c = [0.4, 0.3, 0.6, 0.2, 0.5, 0.8]

    chosen = nsga2_baseline(actions, u_r, u_s, u_c)

    assert any(chosen is action for action in actions)


def test_nsga2_never_selects_a_strictly_dominated_candidate():
    cases = [
        # (U_R, U_S, U_C) per candidate
        ([0.9, 0.5, 0.4], [0.9, 0.9, 0.5], [0.9, 0.4, 0.95]),
        ([0.3, 0.9, 0.2], [0.3, 0.8, 0.6], [0.3, 0.7, 0.4]),
        ([0.6, 0.6, 0.1, 0.9], [0.6, 0.6, 0.2, 0.2], [0.6, 0.6, 0.3, 0.1]),
    ]

    for u_r, u_s, u_c in cases:
        actions = _actions(len(u_r))
        objectives = list(zip(u_r, u_s, u_c))
        chosen_index = nsga2_select_index(actions, u_r, u_s, u_c)

        chosen = objectives[chosen_index]
        assert not any(
            _is_dominated(other, chosen)
            for position, other in enumerate(objectives)
            if position != chosen_index
        )


def test_nsga2_dominance_beats_candidate_order():
    # Candidate 0 is strictly worse on every objective than candidate 1,
    # so candidate 0 must lose despite appearing first.
    actions = _actions(2)
    u_r = [0.3, 0.9]
    u_s = [0.3, 0.8]
    u_c = [0.3, 0.7]

    assert nsga2_select_index(actions, u_r, u_s, u_c) == 1


def test_nsga2_crowding_beats_lowest_index_then_ties_break_by_index():
    # All three candidates are non-dominated. Candidate 0 sits in the
    # interior of every objective, so crowding distance (inf) must beat its
    # lower index; candidates 1 and 2 then tie at inf and the documented
    # tie-break (lowest original index) selects candidate 1.
    actions = _actions(3)
    u_r = [0.5, 1.0, 0.0]
    u_s = [0.5, 0.0, 1.0]
    u_c = [0.5, 0.0, 1.0]

    assert nsga2_select_index(actions, u_r, u_s, u_c) == 1


def test_nsga2_identical_objectives_select_lowest_index_deterministically():
    actions = _actions(4)
    u_r = [0.5, 0.5, 0.5, 0.5]
    u_s = [0.5, 0.5, 0.5, 0.5]
    u_c = [0.5, 0.5, 0.5, 0.5]

    selections = {
        nsga2_select_index(actions, u_r, u_s, u_c) for _ in range(10)
    }

    assert selections == {0}


def test_nsga2_selection_is_reproducible_across_repeated_calls():
    actions = _actions(7)
    u_r = [0.4, 0.8, 0.2, 0.6, 0.9, 0.3, 0.5]
    u_s = [0.7, 0.4, 0.9, 0.5, 0.3, 0.6, 0.8]
    u_c = [0.5, 0.6, 0.4, 0.9, 0.2, 0.7, 0.3]

    selections = {
        nsga2_select_index(actions, u_r, u_s, u_c) for _ in range(10)
    }

    assert len(selections) == 1


def test_nsga2_works_with_duplicate_action_dicts():
    # nsga2 tracks candidates by index, so it never relies on
    # actions.index(action) dictionary equality.
    shared = {"id": "same", "type": "no_op"}
    actions = [shared, shared, shared]
    u_r = [0.1, 0.2, 0.9]
    u_s = [0.1, 0.2, 0.9]
    u_c = [0.1, 0.2, 0.9]

    assert nsga2_select_index(actions, u_r, u_s, u_c) == 2
    assert nsga2_baseline(actions, u_r, u_s, u_c) is shared


def test_nsga2_rejects_empty_candidate_set():
    try:
        nsga2_select_index([], [], [], [])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an empty candidate set")
