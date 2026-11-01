import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import demo
from agents.resource_agent import score_resource
from agents.security_agent import compute_threat_severity, score_security
from agents.sustainability_agent import score_sustainability
from digital_twin.telemetry_generator import generate_telemetry
from digital_twin.threat_injector import inject_threat
from digital_twin.twin import DigitalTwin
from negotiation.cigi import compute_cigi
from negotiation.tapna import pareto_filter, run_tapna


TAU = 0.15
EVENT_TICK = 4
BASELINE_WS = 1 / 3
SCENARIO_OPTIONS = {name: key for key, name in demo.SCENARIOS.items()}

COLOR_NORMAL = "#047857"
COLOR_WARN = "#b45309"
COLOR_THREAT = "#b91c1c"
COLOR_ACCENT = "#0f766e"
COLOR_MUTED = "#64748b"


st.set_page_config(
    page_title="Multi-Agent Cloud-IoT Governance",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2.4rem; max-width: 1550px;}
    h2 {margin-top: 1.1rem; margin-bottom: 0.4rem;}
    .intro-card {
        border: 1px solid #d8dee8;
        border-left: 5px solid #0f766e;
        border-radius: 8px;
        padding: 0.9rem 1.15rem;
        background: #f8fafc;
        margin-bottom: 0.4rem;
    }
    .intro-card p {margin: 0.22rem 0; color: #334155; font-size: 0.94rem;}
    .badge {
        display: inline-block;
        padding: 0.32rem 0.85rem;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.86rem;
        letter-spacing: 0.05rem;
    }
    .badge-normal {color: #047857; background: #e6f4ec; border: 1px solid #a7d8bf;}
    .badge-warn {color: #b45309; background: #fdf3e3; border: 1px solid #f0c886;}
    .badge-threat {color: #b91c1c; background: #fdeaea; border: 1px solid #f0aaaa;}
    .final-card {
        border: 1px solid #b8c4d6;
        border-left: 7px solid #0f766e;
        border-radius: 10px;
        padding: 1.15rem 1.3rem;
        background: #f8fafc;
        margin: 0.5rem 0 0.4rem;
    }
    .final-card.threat {border-left-color: #b91c1c; background: #fdf6f6;}
    .final-card.warn {border-left-color: #b45309; background: #fdfaf3;}
    .final-title {
        color: #475569;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.08rem;
    }
    .final-action {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.15;
        margin: 0.15rem 0 0.35rem;
    }
    .final-sub {color: #334155; font-size: 0.95rem;}
    .flow-box {
        border: 1px solid #d8dee8;
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        background: #ffffff;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d8dee8;
        border-radius: 8px;
        padding: 0.7rem 0.85rem;
    }
    div[data-testid="stMetricLabel"] p {font-size: 0.82rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

_BADGE_CLASS = {"NORMAL": "badge-normal", "WARNING": "badge-warn", "THREAT": "badge-threat"}


def _system_status(telemetry: dict, avg_memory: float, tss: float) -> str:
    if tss > TAU or telemetry["avg_trust_score"] < 0.7 or telemetry["avg_network_load"] > 0.7:
        return "THREAT"
    if telemetry["avg_cpu_load"] > 0.75 or avg_memory > 0.75:
        return "WARNING"
    return "NORMAL"


def _badge(status: str) -> str:
    return f"<span class='badge {_BADGE_CLASS.get(status, 'badge-normal')}'>{status}</span>"


def _agent_status(telemetry: dict, avg_memory: float, tss: float) -> dict:
    return {
        "resource": (
            "High pressure"
            if telemetry["avg_cpu_load"] > 0.75 or avg_memory > 0.75
            else "Within normal range"
        ),
        "security": (
            "Threat condition active"
            if tss > TAU
            else "No elevated threat condition"
        ),
        "sustainability": (
            "Cost and energy constrain this objective"
            if telemetry["total_cost_per_hour"] > 0
            else "Awaiting cost and energy baseline"
        ),
    }


# ---------------------------------------------------------------------------
# Digital Twin scenario control
# ---------------------------------------------------------------------------

def _set_healthy_state(twin: DigitalTwin) -> None:
    for device in twin.devices:
        device.cpu_load = 0.35
        device.memory_load = 0.40
        device.network_load = 0.25
        device.trust_score = 0.95


def _apply_timeline_event(twin: DigitalTwin, choice: int) -> str:
    if choice == 2:
        for device in twin.devices:
            device.network_load = 0.85
            device.trust_score = 0.92
        return "High network load applied"
    if choice == 3:
        for device in twin.devices:
            device.network_load = 0.30
            device.trust_score = 0.45
        return "Trust degradation applied"
    if choice == 4:
        for device in twin.devices:
            inject_threat(twin, device.id, severity=0.5)
        return "Threat injected"
    if choice == 5:
        for device in twin.devices:
            device.cpu_load = 0.90
            device.memory_load = 0.92
            device.network_load = 0.30
            device.trust_score = 0.90
        return "Resource pressure applied"
    if choice == 6:
        for device in twin.devices:
            device.cpu_load = 0.88
            device.memory_load = 0.90
            device.network_load = 0.30
            device.trust_score = 0.90
        for device in twin.devices:
            inject_threat(twin, device.id, severity=0.5)
        return "Threat + resource pressure applied"
    return "Baseline operation"


# ---------------------------------------------------------------------------
# Core pipeline (unchanged computation, reused by both views)
# ---------------------------------------------------------------------------

def _score_tick(twin: DigitalTwin, security_seed: int, prev_wS: float) -> dict:
    telemetry = generate_telemetry(twin)
    resource_state = twin.get_state()
    candidate_actions = demo._scenario_candidate_actions(twin, telemetry)

    with demo._deterministic_security_features(security_seed):
        u_r = [score_resource(action, resource_state) for action in candidate_actions]
        u_s = [score_security(action, telemetry) for action in candidate_actions]
        u_c = [score_sustainability(action, telemetry) for action in candidate_actions]
        tss = compute_threat_severity(telemetry)
        selected_action, weights = run_tapna(
            candidate_actions,
            u_r,
            u_s,
            u_c,
            tss,
            prev_wS=prev_wS,
        )

    selected_index = candidate_actions.index(selected_action)
    cigi = compute_cigi(
        u_r[selected_index],
        u_s[selected_index],
        u_c[selected_index],
        weights,
    )

    return {
        "telemetry": telemetry,
        "resource_state": resource_state,
        "candidate_actions": candidate_actions,
        "U_R_list": u_r,
        "U_S_list": u_s,
        "U_C_list": u_c,
        "TSS": tss,
        "weights": weights,
        "selected_action": selected_action,
        "selected_index": selected_index,
        "CIGI": cigi,
    }


def _run_pipeline(choice: int, tick_count: int = 12) -> dict:
    seed = demo._seed_demo(choice)
    security_seed = seed + 10_000

    twin = DigitalTwin(100)
    description = demo._apply_scenario(twin, choice)

    tick_result = _score_tick(twin, security_seed, prev_wS=BASELINE_WS)
    telemetry = tick_result["telemetry"]
    resource_state = tick_result["resource_state"]
    candidate_actions = tick_result["candidate_actions"]
    u_r = tick_result["U_R_list"]
    u_s = tick_result["U_S_list"]
    u_c = tick_result["U_C_list"]
    selected_action = tick_result["selected_action"]
    selected_index = tick_result["selected_index"]

    scores = {"U_R": u_r, "U_S": u_s, "U_C": u_c}
    survivors = pareto_filter(candidate_actions, scores)
    survivor_ids = {action["id"] for action in survivors}

    candidates = []
    for idx, action in enumerate(candidate_actions):
        candidates.append(
            {
                "candidate": idx + 1,
                "type": action["type"],
                "target": action["target_device_id"][:8],
                "U_R": u_r[idx],
                "U_S": u_s[idx],
                "U_C": u_c[idx],
                "pareto_survivor": action["id"] in survivor_ids,
                "selected": action is selected_action,
            }
        )

    return {
        "description": description,
        "twin": twin,
        "telemetry": telemetry,
        "resource_state": resource_state,
        "candidate_table": pd.DataFrame(candidates),
        "pareto_survivors": len(survivors),
        "selected_action": selected_action,
        "selected_index": selected_index,
        "U_R": u_r[selected_index],
        "U_S": u_s[selected_index],
        "U_C": u_c[selected_index],
        "TSS": tick_result["TSS"],
        "weights": tick_result["weights"],
        "CIGI": tick_result["CIGI"],
        "history": _scenario_history(choice, tick_count),
        "choice": choice,
    }


def _scenario_history(choice: int, tick_count: int) -> pd.DataFrame:
    seed = demo._seed_demo(choice)
    twin = DigitalTwin(100)
    _set_healthy_state(twin)
    prev_wS = BASELINE_WS
    rows = []
    event_label = None

    for tick in range(tick_count):
        if choice != 1 and tick == EVENT_TICK:
            event_label = _apply_timeline_event(twin, choice)
        elif tick:
            twin.tick()

        tick_result = _score_tick(twin, seed + 20_000 + tick, prev_wS)
        telemetry = tick_result["telemetry"]
        resource_state = tick_result["resource_state"]
        selected_action = tick_result["selected_action"]
        weights = tick_result["weights"]
        tss = tick_result["TSS"]
        cigi = tick_result["CIGI"]
        avg_memory = float(resource_state["memory_load"].mean())
        prev_wS = weights["wS"]

        rows.append(
            {
                "tick": tick,
                "system_state": _system_status(telemetry, avg_memory, tss),
                "TSS": tss,
                "wR": weights["wR"],
                "wS": weights["wS"],
                "wC": weights["wC"],
                "cpu_load": telemetry["avg_cpu_load"],
                "memory_load": avg_memory,
                "network_load": telemetry["avg_network_load"],
                "trust_score": telemetry["avg_trust_score"],
                "selected_action": selected_action["type"],
                "CIGI": cigi,
                "event": event_label if tick == EVENT_TICK else "",
            }
        )

    history = pd.DataFrame(rows)
    history.attrs["event_tick"] = EVENT_TICK if choice != 1 else None
    history.attrs["event_label"] = event_label if choice != 1 else None
    return history


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

def _build_insights(result: dict, avg_memory: float) -> list[str]:
    telemetry = result["telemetry"]
    weights = result["weights"]
    high_resource = telemetry["avg_cpu_load"] > 0.75 or avg_memory > 0.75
    high_network = telemetry["avg_network_load"] > 0.7
    low_trust = telemetry["avg_trust_score"] < 0.7
    dominant_weight = max(weights, key=weights.get)
    weakest_utility = min(
        [("resource", result["U_R"]), ("security", result["U_S"]), ("sustainability", result["U_C"])],
        key=lambda item: item[1],
    )

    if result["TSS"] > TAU:
        condition = (
            f"Infrastructure is in a threat-adaptive state: TSS {result['TSS']:.3f}, "
            f"trust {telemetry['avg_trust_score']:.3f}, network {telemetry['avg_network_load']:.3f}."
        )
    elif high_resource:
        condition = (
            f"Infrastructure is resource-constrained: CPU {telemetry['avg_cpu_load']:.3f}, "
            f"memory {avg_memory:.3f}, with TSS {result['TSS']:.3f}."
        )
    else:
        condition = (
            f"Infrastructure is nominal: trust {telemetry['avg_trust_score']:.3f}, "
            f"network {telemetry['avg_network_load']:.3f}, TSS {result['TSS']:.3f}."
        )

    if low_trust:
        concern = f"Dominant concern is reduced trust; security utility is {result['U_S']:.3f}."
    elif high_network:
        concern = (
            f"Dominant concern is network pressure at {telemetry['avg_network_load']:.3f}; "
            f"security utility is {result['U_S']:.3f}."
        )
    elif high_resource:
        concern = f"Dominant concern is resource pressure; resource utility is {result['U_R']:.3f}."
    else:
        concern = f"Dominant constraint is {weakest_utility[0]} utility at {weakest_utility[1]:.3f}."

    response = (
        f"TAPNA response: {dominant_weight} is highest at {weights[dominant_weight]:.3f}, "
        f"with wR {weights['wR']:.3f}, wS {weights['wS']:.3f}, wC {weights['wC']:.3f}."
    )
    action = (
        f"Governance action: `{result['selected_action']['type']}` with CIGI {result['CIGI']:.3f} "
        f"after combining U_R {result['U_R']:.3f}, U_S {result['U_S']:.3f}, and U_C {result['U_C']:.3f}."
    )

    return [condition, concern, response, action]


def _cigi_interpretation(cigi: float) -> str:
    if cigi >= 0.6:
        return "Strong governance alignment across the active objectives."
    if cigi >= 0.3:
        return "Moderate alignment; at least one objective is constraining the decision."
    return "Low alignment; the selected response is operating under significant tradeoffs."


def _agent_contribution_table(result: dict, avg_memory: float) -> pd.DataFrame:
    telemetry = result["telemetry"]
    weights = result["weights"]
    rows = [
        {
            "Agent": "Resource Intelligence Agent",
            "Objective": "Resource efficiency",
            "Utility": result["U_R"],
            "Weight": weights["wR"],
            "Weighted contribution": weights["wR"] * result["U_R"],
            "Interpretation": (
                "Resource pressure is constraining utility."
                if telemetry["avg_cpu_load"] > 0.75 or avg_memory > 0.75
                else "Resource state is within the normal operating range."
            ),
        },
        {
            "Agent": "Security & Trust Agent",
            "Objective": "Security",
            "Utility": result["U_S"],
            "Weight": weights["wS"],
            "Weighted contribution": weights["wS"] * result["U_S"],
            "Interpretation": (
                "Threat conditions reduce security utility."
                if result["TSS"] > TAU
                else "Security utility remains comparatively strong."
            ),
        },
        {
            "Agent": "Sustainability Agent",
            "Objective": "Sustainability",
            "Utility": result["U_C"],
            "Weight": weights["wC"],
            "Weighted contribution": weights["wC"] * result["U_C"],
            "Interpretation": (
                "Cost and energy predictions constrain this objective."
                if result["U_C"] < 0.2
                else "Sustainability objective is not the main constraint."
            ),
        },
    ]
    return pd.DataFrame(rows)


def _tapna_explanation(result: dict) -> str:
    weights = result["weights"]
    utility_part = (
        f"Selected utility mix: wR*U_R={weights['wR'] * result['U_R']:.4f}, "
        f"wS*U_S={weights['wS'] * result['U_S']:.4f}, "
        f"wC*U_C={weights['wC'] * result['U_C']:.4f}, "
        f"giving CIGI={result['CIGI']:.4f} for `{result['selected_action']['type']}`."
    )
    if result["TSS"] > TAU:
        lead = (
            f"Threat severity {result['TSS']:.4f} exceeded tau {TAU:.2f}, "
            f"so TAPNA increased security weight to {weights['wS']:.4f}."
        )
    else:
        lead = (
            f"Threat severity {result['TSS']:.4f} stayed at or below tau {TAU:.2f}, "
            "so TAPNA kept the objective weights balanced."
        )
    return f"{lead} {utility_part}"


def _final_decision_reason(result: dict) -> str:
    if result["TSS"] > TAU:
        return "Security weight increased because threat severity exceeded tau."
    return "Threat severity stayed below tau, so TAPNA used balanced weights over the feasible candidates."


# ---------------------------------------------------------------------------
# Cause-and-effect chain
# ---------------------------------------------------------------------------

def _causal_chain(result: dict, avg_memory: float) -> pd.DataFrame:
    telemetry = result["telemetry"]
    weights = result["weights"]
    status = _system_status(telemetry, avg_memory, result["TSS"])
    tss_status = "THREAT" if result["TSS"] > TAU else "NORMAL"
    ws_status = "THREAT" if weights["wS"] > BASELINE_WS + 0.01 else "NORMAL"
    resource_status = (
        "WARNING"
        if telemetry["avg_cpu_load"] > 0.75 or avg_memory > 0.75
        else "NORMAL"
    )
    weakest = min(
        [("resource", result["U_R"]), ("security", result["U_S"]), ("sustainability", result["U_C"])],
        key=lambda item: item[1],
    )

    rows = [
        {
            "Stage": "1. Infrastructure condition",
            "Observed": (
                f"CPU {telemetry['avg_cpu_load']:.3f} | Memory {avg_memory:.3f} | "
                f"Network {telemetry['avg_network_load']:.3f} | Trust {telemetry['avg_trust_score']:.3f}"
            ),
            "Status": status,
        },
        {
            "Stage": "2. Agent evaluations",
            "Observed": (
                f"U_R {result['U_R']:.4f} | U_S {result['U_S']:.4f} | U_C {result['U_C']:.4f}"
            ),
            "Status": resource_status,
        },
        {
            "Stage": "3. Threat Severity Score",
            "Observed": f"TSS {result['TSS']:.4f} compared against tau {TAU:.2f}",
            "Status": tss_status,
        },
        {
            "Stage": "4. Adaptive security weight",
            "Observed": (
                f"wS {weights['wS']:.4f} (balanced baseline {BASELINE_WS:.4f})"
            ),
            "Status": ws_status,
        },
        {
            "Stage": "5. TAPNA decision",
            "Observed": (
                f"Pareto survivors {result['pareto_survivors']} -> selected `{result['selected_action']['type']}`"
            ),
            "Status": tss_status,
        },
        {
            "Stage": "6. CIGI",
            "Observed": f"CIGI {result['CIGI']:.4f} from weighted objective mix",
            "Status": "NORMAL" if result["CIGI"] >= 0.3 else "WARNING",
        },
        {
            "Stage": "7. Infrastructure insight",
            "Observed": f"Weakest objective: {weakest[0]} utility at {weakest[1]:.4f}",
            "Status": resource_status,
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _infra_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["tick"],
            y=history["trust_score"],
            mode="lines+markers",
            name="Average Trust Score",
            line=dict(color=COLOR_ACCENT),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history["tick"],
            y=history["network_load"],
            mode="lines+markers",
            name="Average Network Load",
            line=dict(color="#b45309"),
        )
    )
    fig.add_hline(y=0.7, line_dash="dot", line_color=COLOR_MUTED, annotation_text="attack threshold 0.7")

    event_tick = history.attrs.get("event_tick")
    event_label = history.attrs.get("event_label")
    if event_tick is not None and event_label:
        fig.add_vline(
            x=event_tick,
            line_color=COLOR_THREAT,
            line_dash="dash",
            annotation_text=event_label,
            annotation_position="top right",
        )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(range=[0, 1], title="Load / trust"),
        xaxis=dict(title="Tick"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _governance_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["tick"],
            y=history["TSS"],
            mode="lines+markers",
            name="TSS",
            line=dict(color=COLOR_THREAT),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history["tick"],
            y=history["wS"],
            mode="lines+markers",
            name="Security weight wS",
            line=dict(color=COLOR_ACCENT),
        )
    )
    fig.add_hline(y=TAU, line_dash="dot", line_color=COLOR_MUTED, annotation_text=f"tau {TAU:.2f}")
    fig.add_hline(y=BASELINE_WS, line_dash="dot", line_color="#94a3b8")

    event_tick = history.attrs.get("event_tick")
    if event_tick is not None:
        fig.add_vline(x=event_tick, line_color=COLOR_THREAT, line_dash="dash")

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(range=[0, 1], title="TSS / weight"),
        xaxis=dict(title="Tick"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _utility_chart(result: dict) -> go.Figure:
    labels = ["U_R (resource)", "U_S (security)", "U_C (sustainability)"]
    values = [result["U_R"], result["U_S"], result["U_C"]]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=[COLOR_ACCENT, COLOR_THREAT, "#b45309"],
            text=[f"{value:.3f}" for value in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(range=[0, 1], title="Utility"),
        showlegend=False,
    )
    return fig


def _weights_chart(weights: dict) -> go.Figure:
    labels = ["wR", "wS", "wC"]
    values = [weights["wR"], weights["wS"], weights["wC"]]
    colors = [COLOR_ACCENT, COLOR_THREAT, "#b45309"]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{value:.4f}" for value in values],
            textposition="outside",
        )
    )
    fig.add_hline(y=BASELINE_WS, line_dash="dot", line_color=COLOR_MUTED)
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(range=[0, 1], title="Weight"),
        showlegend=False,
    )
    return fig


def _tss_gauge(tss: float) -> go.Figure:
    is_threat = tss > TAU
    fig = go.Figure(
        go.Bar(
            x=[tss],
            y=["TSS"],
            orientation="h",
            marker_color=COLOR_THREAT if is_threat else COLOR_ACCENT,
            text=[f"{tss:.4f}"],
            textposition="outside",
        )
    )
    fig.add_vline(x=TAU, line_dash="dash", line_color=COLOR_MUTED)
    fig.update_layout(
        height=130,
        margin=dict(l=20, r=20, t=10, b=10),
        xaxis=dict(range=[0, 1], title=f"TSS vs tau {TAU:.2f}"),
        yaxis=dict(showticklabels=False),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _decision_timeline(history: pd.DataFrame) -> pd.DataFrame:
    display = history[["tick", "system_state", "TSS", "wS", "selected_action"]].copy()
    display["TSS"] = display["TSS"].map(lambda value: f"{value:.4f}")
    display["wS"] = display["wS"].map(lambda value: f"{value:.4f}")
    display.columns = ["Tick", "System State", "TSS", "wS", "Selected Action"]
    return display


def _before_after_threat(history: pd.DataFrame) -> pd.DataFrame | None:
    event_tick = history.attrs.get("event_tick")
    if event_tick is None or event_tick == 0:
        return None

    before = history[history["tick"] == event_tick - 1]
    after = history[history["tick"] == event_tick]
    if before.empty or after.empty:
        return None

    rows = []
    for label, row in [("BEFORE", before.iloc[0]), ("AFTER", after.iloc[0])]:
        rows.append(
            {
                "Phase": label,
                "Trust": f"{row['trust_score']:.4f}",
                "Network": f"{row['network_load']:.4f}",
                "TSS": f"{row['TSS']:.4f}",
                "wS": f"{row['wS']:.4f}",
                "Action": row["selected_action"],
            }
        )
    return pd.DataFrame(rows)


def _format_candidate_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in ["U_R", "U_S", "U_C"]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    display["pareto_survivor"] = display["pareto_survivor"].map({True: "Yes", False: "No"})
    display["selected"] = display["selected"].map({True: "Selected", False: ""})
    display.columns = [
        "Candidate",
        "Type",
        "Target Device",
        "U_R",
        "U_S",
        "U_C",
        "Pareto Survivor",
        "Selection",
    ]
    return display


def _format_contribution_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in ["Utility", "Weight", "Weighted contribution"]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    return display


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Simulation Controls")
    selected_scenario = st.selectbox(
        "Scenario",
        list(SCENARIO_OPTIONS.keys()),
        index=0,
    )
    run_clicked = st.button("Run Simulation", type="primary", width="stretch")
    reset_clicked = st.button("Reset", width="stretch")

    st.divider()
    st.caption(
        "Runs the existing Digital Twin, agent scoring, TAPNA, and CIGI pipeline. "
        "No external APIs or internet access are required."
    )
    st.caption(f"Threat threshold tau = {TAU:.2f}")

if reset_clicked:
    st.session_state.pop("scenario_name", None)
    st.session_state.pop("result", None)

if run_clicked or "result" not in st.session_state:
    choice = SCENARIO_OPTIONS[selected_scenario]
    st.session_state["scenario_name"] = selected_scenario
    st.session_state["result"] = _run_pipeline(choice)

result = st.session_state["result"]
scenario_name = st.session_state["scenario_name"]
telemetry = result["telemetry"]
resource_state = result["resource_state"]
avg_memory = float(resource_state["memory_load"].mean())
status = _system_status(telemetry, avg_memory, result["TSS"])
agent_status = _agent_status(telemetry, avg_memory, result["TSS"])


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Multi-Agent Cloud-IoT Governance")
st.caption("Threat-Adaptive Digital Twin Governance Simulator")
st.markdown(
    """
    <div class="intro-card">
        <p><strong>What this dashboard demonstrates.</strong> A simulated Cloud-IoT
        infrastructure feeds telemetry into three specialized AI agents — Resource
        Intelligence, Security &amp; Trust, and Sustainability — whose evaluations are
        combined by TAPNA with a threat-adaptive security weighting, producing a
        governance decision scored by CIGI.</p>
        <p><strong>Cause and effect.</strong> Infrastructure condition &rarr; agent
        evaluations &rarr; TSS &rarr; adaptive security weight &rarr; TAPNA decision
        &rarr; CIGI &rarr; infrastructure insight.</p>
        <p><strong>Scenario:</strong> {scenario} &nbsp;|&nbsp; {detail}</p>
    </div>
    """.format(scenario=scenario_name, detail=result["description"]),
    unsafe_allow_html=True,
)

st.markdown(
    f"<div style='margin:0.6rem 0 0.2rem;'>{_badge(status)}"
    f"<span style='margin-left:0.7rem;color:#475569;'>System state under this scenario</span></div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 1. Digital Twin
# ---------------------------------------------------------------------------

st.markdown("## Digital Twin")
metric_cols = st.columns(5)
metric_cols[0].metric("Virtual devices", len(result["twin"].devices))
metric_cols[1].metric("Average CPU utilization", f"{telemetry['avg_cpu_load']:.3f}")
metric_cols[2].metric("Average network load", f"{telemetry['avg_network_load']:.3f}")
metric_cols[3].metric("Average trust score", f"{telemetry['avg_trust_score']:.3f}")
metric_cols[4].metric("Average memory load", f"{avg_memory:.3f}")


# ---------------------------------------------------------------------------
# 2. AI Agents
# ---------------------------------------------------------------------------

st.markdown("## AI Agents")

agent_cols = st.columns(3)
with agent_cols[0]:
    st.markdown("#### Resource Intelligence Agent")
    st.metric("U_R", f"{result['U_R']:.4f}")
    st.write(f"Model: `resource_model.joblib`")
    st.caption(agent_status["resource"])
with agent_cols[1]:
    st.markdown("#### Security & Trust Agent")
    st.metric("U_S", f"{result['U_S']:.4f}")
    st.write("Model: `threat_detector.joblib`")
    st.caption(agent_status["security"])
with agent_cols[2]:
    st.markdown("#### Sustainability Agent")
    st.metric("U_C", f"{result['U_C']:.4f}")
    st.write("Model: `sustainability_model.joblib`")
    st.caption(agent_status["sustainability"])

compare_left, compare_right = st.columns([1, 1.25])
with compare_left:
    st.plotly_chart(_utility_chart(result), width="stretch")
with compare_right:
    st.markdown("##### Agent comparison")
    st.dataframe(
        _format_contribution_table(_agent_contribution_table(result, avg_memory)),
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# 3. TAPNA Governance
# ---------------------------------------------------------------------------

st.markdown("## TAPNA Governance")

tapna_left, tapna_right = st.columns([1.1, 1])

with tapna_left:
    st.markdown("##### Threat Severity Score")
    st.plotly_chart(_tss_gauge(result["TSS"]), width="stretch")
    tss_state = "above" if result["TSS"] > TAU else "at or below"
    st.caption(
        f"TSS {result['TSS']:.4f} is {tss_state} tau {TAU:.2f}, "
        f"so the security weight is {'elevated' if result['TSS'] > TAU else 'held near the balanced baseline'}."
    )

    st.markdown("##### Adaptive security weight")
    st.progress(
        min(max(result["weights"]["wS"], 0.0), 1.0),
        text=f"wS = {result['weights']['wS']:.4f}  (baseline {BASELINE_WS:.4f})",
    )
    weight_cols = st.columns(3)
    weight_cols[0].metric("wR", f"{result['weights']['wR']:.4f}")
    weight_cols[1].metric("wS", f"{result['weights']['wS']:.4f}")
    weight_cols[2].metric("wC", f"{result['weights']['wC']:.4f}")

with tapna_right:
    st.markdown("##### Objective weights")
    st.plotly_chart(_weights_chart(result["weights"]), width="stretch")

st.markdown("##### Why did TAPNA make this decision?")
st.write(_tapna_explanation(result))
st.caption(
    f"Threshold tau: {TAU:.2f} | Pareto surviving candidates: "
    f"{result['pareto_survivors']} of {len(result['candidate_table'])}"
)

with st.expander("Candidate actions and Pareto filtering"):
    st.dataframe(
        _format_candidate_table(result["candidate_table"]),
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# 4. Final Decision
# ---------------------------------------------------------------------------

st.markdown("## Final Decision")

decision_class = {"THREAT": " threat", "WARNING": " warn", "NORMAL": ""}[status]
st.markdown(
    f"""
    <div class="final-card{decision_class}">
        <div class="final-title">SELECTED GOVERNANCE ACTION</div>
        <div class="final-action">{result['selected_action']['type'].upper()}</div>
        <div class="final-sub">
            Target device: <code>{result['selected_action']['target_device_id'][:8]}</code>
            &nbsp;|&nbsp; TSS <strong>{result['TSS']:.4f}</strong>
            &nbsp;|&nbsp; CIGI <strong>{result['CIGI']:.4f}</strong>
            &nbsp;|&nbsp; Status {_badge(status)}
        </div>
        <div class="final-sub" style="margin-top:0.6rem;">
            <strong>Reason:</strong> {_final_decision_reason(result)}
        </div>
        <div class="final-sub" style="margin-top:0.35rem;">
            <strong>CIGI interpretation:</strong> {_cigi_interpretation(result['CIGI'])}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

final_cols = st.columns(3)
final_cols[0].metric("CIGI", f"{result['CIGI']:.4f}")
final_cols[1].metric("TSS", f"{result['TSS']:.4f}")
final_cols[2].metric("wS applied", f"{result['weights']['wS']:.4f}")


# ---------------------------------------------------------------------------
# 5. Governance decision flow (cause and effect chain)
# ---------------------------------------------------------------------------

st.markdown("## Governance Decision Flow")
st.caption(
    "Each stage below is computed by the live pipeline — infrastructure condition, "
    "agent evaluations, TSS, adaptive security weight, TAPNA decision, CIGI, insight."
)
st.dataframe(_causal_chain(result, avg_memory), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# 6. AI Infrastructure Insights
# ---------------------------------------------------------------------------

st.markdown("## AI Infrastructure Insights")
for line in _build_insights(result, avg_memory):
    st.write(f"- {line}")


# ---------------------------------------------------------------------------
# 7. Simulation Timeline
# ---------------------------------------------------------------------------

st.markdown("## Simulation Timeline")
timeline_left, timeline_right = st.columns(2)
with timeline_left:
    st.markdown("##### Infrastructure condition")
    st.plotly_chart(_infra_chart(result["history"]), width="stretch")
with timeline_right:
    st.markdown("##### Governance response")
    st.plotly_chart(_governance_chart(result["history"]), width="stretch")


# ---------------------------------------------------------------------------
# 8. Decision Timeline
# ---------------------------------------------------------------------------

st.markdown("## Decision Timeline")
decision_left, decision_right = st.columns([1.15, 1])
with decision_left:
    st.dataframe(_decision_timeline(result["history"]), width="stretch", hide_index=True)
with decision_right:
    before_after = (
        _before_after_threat(result["history"]) if result["choice"] in {2, 3, 4, 6} else None
    )
    st.markdown("##### Before / after transition")
    if before_after is not None:
        st.dataframe(before_after, width="stretch", hide_index=True)
    else:
        st.info("No threat transition was applied for this scenario.")

with st.expander("How was this decision made?"):
    st.write("1. Digital Twin generates simulated infrastructure state.")
    st.write("2. Three agents independently calculate objective utilities.")
    st.write("3. Pareto filtering removes dominated actions.")
    st.write("4. TSS is compared with tau.")
    st.write("5. TAPNA dynamically adjusts the security weight.")
    st.write("6. Remaining weights are calculated.")
    st.write("7. Weighted candidate selection is performed.")
    st.write("8. CIGI is calculated.")
    st.write("9. The decision is displayed and logged.")
