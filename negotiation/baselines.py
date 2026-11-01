def greedy_cost_baseline(actions, U_C_list):
    best_index = max(range(len(actions)), key=lambda index: U_C_list[index])
    return actions[best_index]


def greedy_latency_baseline(actions, U_R_list):
    best_index = max(range(len(actions)), key=lambda index: U_R_list[index])
    return actions[best_index]


def static_weighted_baseline(
    actions,
    U_R_list,
    U_S_list,
    U_C_list
):
    best_index = max(
        range(len(actions)),
        key=lambda index: (
            (1 / 3) * U_R_list[index] +
            (1 / 3) * U_S_list[index] +
            (1 / 3) * U_C_list[index]
        ),
    )
    return actions[best_index]


def _dominates(left, right):
    """Return True when left is >= right on every objective and > on at least one."""
    return all(l >= r for l, r in zip(left, right)) and any(
        l > r for l, r in zip(left, right)
    )


def _non_dominated_fronts(objectives):
    """Fast non-dominated sort over pre-computed objective vectors.

    Returns a list of fronts; each front is a list of candidate indices in
    their original input order. Candidates with identical objective vectors
    do not dominate each other and therefore share the same front.
    """
    count = len(objectives)
    dominates_list = [[] for _ in range(count)]
    domination_count = [0] * count

    for i in range(count):
        for j in range(i + 1, count):
            if _dominates(objectives[i], objectives[j]):
                dominates_list[i].append(j)
                domination_count[j] += 1
            elif _dominates(objectives[j], objectives[i]):
                dominates_list[j].append(i)
                domination_count[i] += 1

    fronts = []
    current_front = [i for i in range(count) if domination_count[i] == 0]

    while current_front:
        current_front.sort()
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominates_list[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


def _crowding_distance(front, objectives):
    """Standard NSGA-II crowding distance for one front.

    Boundary individuals (per objective) receive +inf. An objective with zero
    range inside the front contributes nothing. Ties in the per-objective sort
    are broken by original index so the result is deterministic.
    """
    if len(front) <= 2:
        return {index: float("inf") for index in front}

    distance = {index: 0.0 for index in front}

    for objective_position in range(len(objectives[0])):
        ordered = sorted(
            front,
            key=lambda index: (objectives[index][objective_position], index),
        )
        low = objectives[ordered[0]][objective_position]
        high = objectives[ordered[-1]][objective_position]
        if high == low:
            continue

        distance[ordered[0]] = float("inf")
        distance[ordered[-1]] = float("inf")
        span = high - low
        for position in range(1, len(ordered) - 1):
            index = ordered[position]
            if distance[index] == float("inf"):
                continue
            distance[index] += (
                objectives[ordered[position + 1]][objective_position]
                - objectives[ordered[position - 1]][objective_position]
            ) / span

    return distance


def nsga2_select_index(actions, U_R_list, U_S_list, U_C_list):
    """Discrete NSGA-II selection over an existing candidate set.

    The candidate actions and their U_R / U_S / U_C values are consumed
    exactly as supplied: no new candidates are generated and no objective is
    re-evaluated, so the model-evaluation budget matches the other baselines.

    Procedure:
      1. Fast non-dominated sort of all candidates (Pareto fronts).
      2. Compute NSGA-II crowding distance within the first front.
      3. Pick the first-front candidate with the largest crowding distance.

    Deterministic tie-break (documented, reproducible): candidates are ordered
    by (-crowding_distance, original_index), so exact ties in crowding distance
    (including the +inf boundary values) are always resolved in favour of the
    candidate that appeared earliest in the candidate list. The same inputs
    therefore always yield the same selection.
    """
    if not actions:
        raise ValueError("nsga2_select_index requires at least one candidate action")

    objectives = list(zip(U_R_list, U_S_list, U_C_list))
    first_front = _non_dominated_fronts(objectives)[0]
    crowding = _crowding_distance(first_front, objectives)

    return sorted(
        first_front,
        key=lambda index: (-crowding[index], index),
    )[0]


def nsga2_baseline(actions, U_R_list, U_S_list, U_C_list):
    """NSGA-II baseline: return the action chosen by nsga2_select_index."""
    return actions[nsga2_select_index(actions, U_R_list, U_S_list, U_C_list)]
