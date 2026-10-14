"""Predictor View Component for XAI-GT Dashboard.

Provides an interactive fraud prediction interface allowing users to:
1. Simulate custom transactions with topological and transactional parameters.
2. Select pre-configured archetypes (peeling chain, mixer, licit retail, etc.).
3. Look up and predict any transaction node from the Elliptic graph with ground truth comparison.
4. Run dual-model verification (Graph Transformer vs GraphSAGE baseline).
5. View explainable forensic reasoning, feature attributions, and AML action plans.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from src.models.predictor import (
    ARCHETYPE_PRESETS,
    predict_custom_transaction,
    predict_node_transaction,
)


def render_predictor_view(
    data: Any,
    model: torch.nn.Module,
    baseline_model: Optional[torch.nn.Module] = None,
    G: Any = None,
) -> None:
    """Renders the Live Fraud Predictor page in Streamlit."""

    st.markdown(
        """
<div class="investigation-header">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
        <div>
            <span style="font-size:1.6rem; font-weight:800; letter-spacing:-0.02em;">
                ⚡ Live Transaction Fraud Predictor
            </span>
            <div style="color:#94A3B8; font-size:0.9rem; margin-top:4px;">
                Enter transaction attributes or test behavioral archetypes through the Graph Transformer &amp; GraphSAGE models.
            </div>
        </div>
        <div>
            <span class="status-badge badge-suspicious">Dual-Model Engine</span>
            <span class="status-badge badge-clean">Active Inference</span>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if model is None:
        st.error("⚠️ Model checkpoint not found or failed to load. Please verify that `models/graph_transformer.pt` exists.")
        return

    # Top Controls Bar (Threshold & Model Choice)
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.5, 1])
    with ctrl_col1:
        threshold = st.slider(
            "Decision Threshold (Fraud Sensitivity)",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Threshold above which a transaction is flagged as fraud. Default: 0.50.",
        )
    with ctrl_col2:
        compare_baseline = st.toggle(
            "Dual-Model Verification (GraphSAGE)",
            value=True,
            help="Run both Graph Transformer and GraphSAGE to verify consensus.",
        )
    with ctrl_col3:
        st.caption("Active Device")
        st.markdown("**CPU Inference Engine**")

    st.markdown("---")

    # Main Input Tabs
    input_tab1, input_tab2, input_tab3 = st.tabs([
        "🎛️ Custom Parameter Simulator",
        "🔍 Blockchain Transaction Lookup",
        "⚡ Counterfactual 'What-If' Analysis",
    ])

    # =========================================================================
    # TAB 1: Custom Parameter Simulator
    # =========================================================================
    with input_tab1:
        st.markdown("### 🛠️ **Transaction Attributes & Topology Simulation**")
        st.caption(
            "Adjust local transaction attributes and graph topology metrics, or select a behavioral archetype to populate values."
        )

        # Quick Archetype Presets
        st.markdown("##### 📌 Behavioral Archetype Presets:")
        preset_cols = st.columns(len(ARCHETYPE_PRESETS))
        selected_preset = None

        for col, (preset_name, preset_vals) in zip(preset_cols, ARCHETYPE_PRESETS.items()):
            short_name = preset_name.split(" ")[1] if " " in preset_name else preset_name
            if col.button(
                f"{preset_name.split(' ')[0]} {short_name}",
                key=f"preset_btn_{preset_name}",
                use_container_width=True,
                help=preset_vals["description"],
            ):
                st.session_state["active_preset"] = preset_name
                st.session_state["sim_in_deg"] = preset_vals["in_degree"]
                st.session_state["sim_out_deg"] = preset_vals["out_degree"]
                st.session_state["sim_clust"] = preset_vals["clustering_coef"]
                st.session_state["sim_vol"] = preset_vals["volume_proxy"]
                st.session_state["sim_fee"] = preset_vals["fee_proxy"]
                st.session_state["sim_step"] = preset_vals["time_step"]
                st.session_state["sim_neighbor"] = preset_vals["neighbor_risk"]
                st.rerun()

        # Check if an active preset is set
        active_preset_name = st.session_state.get("active_preset", "🚨 Peeling Chain Laundering")
        active_preset_meta = ARCHETYPE_PRESETS.get(active_preset_name, {})
        if active_preset_meta:
            st.info(f"**Loaded Archetype:** `{active_preset_name}` — *{active_preset_meta.get('description', '')}*")

        # Form Inputs
        with st.form("custom_predict_form"):
            p_col1, p_col2, p_col3 = st.columns(3)

            with p_col1:
                st.markdown("##### 🌐 Network Topology")
                in_deg = st.number_input(
                    "Inflow Transactions (In-Degree)",
                    min_value=1,
                    max_value=200,
                    value=int(st.session_state.get("sim_in_deg", active_preset_meta.get("in_degree", 1))),
                    step=1,
                    help="Number of funding/incoming transactions feeding into this wallet.",
                )
                out_deg = st.number_input(
                    "Outflow Transactions (Out-Degree)",
                    min_value=1,
                    max_value=200,
                    value=int(st.session_state.get("sim_out_deg", active_preset_meta.get("out_degree", 42))),
                    step=1,
                    help="Number of dispersion/peeling chain outputs exiting this wallet.",
                )
                clust_coef = st.slider(
                    "Local Clustering Coefficient",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.get("sim_clust", active_preset_meta.get("clustering_coef", 0.02))),
                    step=0.01,
                    help="Measure of cyclic mixing loops among neighbor transactions (0 = tree/chain, 1 = dense clique).",
                )

            with p_col2:
                st.markdown("##### 💰 Financial Properties")
                vol_proxy = st.slider(
                    "Normalized Volume Proxy (σ)",
                    min_value=-2.0,
                    max_value=6.0,
                    value=float(st.session_state.get("sim_vol", active_preset_meta.get("volume_proxy", 2.8))),
                    step=0.1,
                    help="Standardized Bitcoin transaction value proxy (0 = mean, +3 = whale/high volume).",
                )
                fee_proxy = st.slider(
                    "Normalized Fee Proxy (σ)",
                    min_value=-2.0,
                    max_value=5.0,
                    value=float(st.session_state.get("sim_fee", active_preset_meta.get("fee_proxy", 1.9))),
                    step=0.1,
                    help="Standardized miner fee proxy paid for faster block inclusion.",
                )
                time_step = st.slider(
                    "Network Time-Step (1–49)",
                    min_value=1,
                    max_value=49,
                    value=int(st.session_state.get("sim_step", active_preset_meta.get("time_step", 36))),
                    step=1,
                    help="Discrete temporal epoch (timesteps 1-34 train, 35-41 val, 42-49 test).",
                )

            with p_col3:
                st.markdown("##### 👥 Counterparty Environment")
                neighbor_risk = st.selectbox(
                    "Ego-Neighborhood Risk Bias",
                    options=["low", "mixed", "high"],
                    index=["low", "mixed", "high"].index(st.session_state.get("sim_neighbor", active_preset_meta.get("neighbor_risk", "high"))),
                    format_func=lambda x: {
                        "low": "🟢 Clean / Low Risk Counterparties",
                        "mixed": "🟡 Mixed / Unverified Counterparties",
                        "high": "🔴 Illicit / Flagged Counterparties",
                    }[x],
                    help="Historical risk composition of connected wallets.",
                )

                st.markdown("<div style='height:38px;'></div>", unsafe_allow_html=True)
                run_sim_btn = st.form_submit_button("⚡ Predict Fraud Risk", use_container_width=True)

        # Execution
        if run_sim_btn or "last_sim_res" not in st.session_state:
            with st.spinner("🧠 Running Graph Transformer forward pass & neighborhood convolution …"):
                pred_res = predict_custom_transaction(
                    model=model,
                    baseline_model=baseline_model if compare_baseline else None,
                    in_degree=in_deg,
                    out_degree=out_deg,
                    clustering_coef=clust_coef,
                    volume_proxy=vol_proxy,
                    fee_proxy=fee_proxy,
                    time_step=time_step,
                    neighbor_risk=neighbor_risk,
                    threshold=threshold,
                )
                st.session_state["last_sim_res"] = pred_res

        # Display Results
        if "last_sim_res" in st.session_state:
            _render_prediction_results(st.session_state["last_sim_res"], compare_baseline=compare_baseline)

    # =========================================================================
    # TAB 2: Blockchain Transaction Lookup
    # =========================================================================
    with input_tab2:
        st.markdown("### 🔎 **Direct Graph Transaction Inference**")
        st.caption(
            "Select or enter an existing node index from the Elliptic graph to inspect actual recorded attributes and evaluate the model."
        )

        if "direct_lookup_id" not in st.session_state:
            st.session_state["direct_lookup_id"] = int(st.session_state.get("wallet_id", 907))

        b_col1, b_col2 = st.columns([2, 1])
        with b_col1:
            test_wallet_id = st.number_input(
                "Transaction ID to Predict",
                min_value=0,
                max_value=int(data.num_nodes) - 1,
                value=int(st.session_state["direct_lookup_id"]),
                step=1,
            )
            if test_wallet_id != st.session_state["direct_lookup_id"]:
                st.session_state["direct_lookup_id"] = int(test_wallet_id)
        with b_col2:
            st.caption("Quick Samples from Graph:")
            qs_col1, qs_col2, qs_col3 = st.columns(3)
            with qs_col1:
                if st.button("🔴 Illicit", key="qs_illicit", use_container_width=True):
                    illicit_pool = torch.where(data.y == 1)[0].tolist()
                    st.session_state["direct_lookup_id"] = random.choice(illicit_pool)
                    st.rerun()
            with qs_col2:
                if st.button("🟢 Licit", key="qs_licit", use_container_width=True):
                    licit_pool = torch.where(data.y == 0)[0].tolist()
                    st.session_state["direct_lookup_id"] = random.choice(licit_pool)
                    st.rerun()
            with qs_col3:
                if st.button("⚪ Unknown", key="qs_unknown", use_container_width=True):
                    unk_pool = torch.where(data.y == -1)[0].tolist()
                    st.session_state["direct_lookup_id"] = random.choice(unk_pool)
                    st.rerun()

        if st.button("🔎 Predict Transaction Node", type="primary", use_container_width=True):
            with st.spinner(f"Extracting 2-hop neighborhood for Transaction #{test_wallet_id} …"):
                node_pred_res = predict_node_transaction(
                    node_id=int(test_wallet_id),
                    graph_data=data,
                    model=model,
                    baseline_model=baseline_model if compare_baseline else None,
                    threshold=threshold,
                )
                st.session_state["last_node_pred_res"] = node_pred_res

        if "last_node_pred_res" in st.session_state:
            res = st.session_state["last_node_pred_res"]
            st.markdown("---")

            # Ground truth comparison badge
            gt_badge_cls = {
                "Verified Illicit": "badge-fraud",
                "Verified Licit": "badge-clean",
            }.get(res["ground_truth"], "badge-unknown")

            st.markdown(
                f"""
            <div style="background:rgba(30,41,59,0.5); padding:1rem; border-radius:10px; margin-bottom:1rem; border:1px solid rgba(255,255,255,0.08);">
                <b>Evaluating Node #{res['node_id']}</b> · Subgraph: <b>{res['subgraph_nodes']}</b> nodes, <b>{res['subgraph_edges']}</b> edges · 
                Ground Truth Label: <span class="status-badge {gt_badge_cls}">{res['ground_truth']}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
            _render_prediction_results(res, compare_baseline=compare_baseline)

    # =========================================================================
    # TAB 3: Counterfactual "What-If" Analysis
    # =========================================================================
    with input_tab3:
        st.markdown("### ⚡ **Counterfactual 'What-If' Explorer**")
        st.caption(
            "Test sensitivity: how does altering a single property (e.g., dispersion degree or volume) shift the model's fraud verdict?"
        )

        cf_col1, cf_col2 = st.columns([1, 1.4])
        with cf_col1:
            cf_feature = st.selectbox(
                "Counterfactual Feature to Sweep",
                options=["Out-Degree (Dispersion)", "In-Degree (Inflow)", "Volume Proxy", "Clustering Coefficient"],
            )
            cf_steps = st.slider("Sweep Steps", 5, 20, 10)
            run_sweep = st.button("🚀 Run Sensitivity Curve", use_container_width=True)

        with cf_col2:
            if run_sweep:
                sweep_x = []
                sweep_y = []
                sweep_y_base = []

                if "Out-Degree" in cf_feature:
                    vals = np.linspace(1, 100, cf_steps).astype(int)
                    x_label = "Outflow Destinations (Out-Degree)"
                    for v in vals:
                        r = predict_custom_transaction(model, baseline_model, 2, int(v), 0.05, 1.0, 0.5, neighbor_risk="mixed")
                        sweep_x.append(int(v))
                        sweep_y.append(r["fraud_risk"] * 100)
                        if r["baseline_fraud_risk"] is not None:
                            sweep_y_base.append(r["baseline_fraud_risk"] * 100)
                elif "In-Degree" in cf_feature:
                    vals = np.linspace(1, 100, cf_steps).astype(int)
                    x_label = "Inflow Inputs (In-Degree)"
                    for v in vals:
                        r = predict_custom_transaction(model, baseline_model, int(v), 2, 0.05, 1.0, 0.5, neighbor_risk="mixed")
                        sweep_x.append(int(v))
                        sweep_y.append(r["fraud_risk"] * 100)
                        if r["baseline_fraud_risk"] is not None:
                            sweep_y_base.append(r["baseline_fraud_risk"] * 100)
                elif "Volume" in cf_feature:
                    vals = np.linspace(-1.5, 5.0, cf_steps)
                    x_label = "Volume Proxy (σ)"
                    for v in vals:
                        r = predict_custom_transaction(model, baseline_model, 2, 2, 0.05, float(v), 0.5, neighbor_risk="mixed")
                        sweep_x.append(float(v))
                        sweep_y.append(r["fraud_risk"] * 100)
                        if r["baseline_fraud_risk"] is not None:
                            sweep_y_base.append(r["baseline_fraud_risk"] * 100)
                else:
                    vals = np.linspace(0.0, 1.0, cf_steps)
                    x_label = "Clustering Coefficient"
                    for v in vals:
                        r = predict_custom_transaction(model, baseline_model, 15, 15, float(v), 1.0, 0.5, neighbor_risk="mixed")
                        sweep_x.append(float(v))
                        sweep_y.append(r["fraud_risk"] * 100)
                        if r["baseline_fraud_risk"] is not None:
                            sweep_y_base.append(r["baseline_fraud_risk"] * 100)

                fig_sweep = go.Figure()
                fig_sweep.add_trace(go.Scatter(
                    x=sweep_x, y=sweep_y,
                    mode="lines+markers",
                    name="Graph Transformer",
                    line={"color": "#F97316", "width": 3},
                    marker={"size": 6},
                ))
                if sweep_y_base:
                    fig_sweep.add_trace(go.Scatter(
                        x=sweep_x, y=sweep_y_base,
                        mode="lines+markers",
                        name="GraphSAGE Baseline",
                        line={"color": "#38BDF8", "width": 2, "dash": "dash"},
                        marker={"size": 5},
                    ))
                fig_sweep.add_hline(y=threshold * 100, line_dash="dot", line_color="#EF4444", annotation_text="Decision Threshold")
                fig_sweep.update_layout(
                    title=f"Sensitivity Curve: Fraud Risk vs {x_label}",
                    xaxis_title=x_label,
                    yaxis_title="Fraud Risk (%)",
                    height=320,
                    margin={"l": 20, "r": 20, "t": 40, "b": 20},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis={"gridcolor": "rgba(255,255,255,0.08)"},
                    yaxis={"gridcolor": "rgba(255,255,255,0.08)", "range": [0, 100]},
                    font={"color": "#CBD5E1"},
                )
                st.plotly_chart(fig_sweep, use_container_width=True)
            else:
                st.info("👈 Select a feature and click **Run Sensitivity Curve** to generate an automated sensitivity analysis.")


def _render_prediction_results(res: Dict[str, Any], compare_baseline: bool = True) -> None:
    """Helper to render standardized risk cards, gauges, and forensic reasoning."""
    st.markdown("---")
    st.markdown("### 🎯 **Model Prediction & Risk Verdict**")

    # Banner Card
    badge_cls = {
        "fraud": "badge-fraud",
        "suspicious": "badge-suspicious",
        "clean": "badge-clean",
    }.get(res.get("badge_type", "clean"), "badge-clean")

    border_color = {
        "fraud": "rgba(239, 68, 68, 0.4)",
        "suspicious": "rgba(234, 179, 8, 0.4)",
        "clean": "rgba(34, 197, 94, 0.4)",
    }.get(res.get("badge_type", "clean"), "rgba(255,255,255,0.1)")

    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
                border: 1px solid {border_color}; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.2rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <span style="font-size:0.85rem; text-transform:uppercase; color:#94A3B8; font-weight:700; letter-spacing:0.05em;">
                    Final AI Verdict:
                </span>
                <div style="font-size:1.8rem; font-weight:800; margin-top:2px;">
                    <span class="status-badge {badge_cls}" style="font-size:1.3rem; padding:0.4rem 1.2rem;">{res['verdict']}</span>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.85rem; color:#94A3B8;">Confidence Level</div>
                <div style="font-size:1.8rem; font-weight:800; color:#F8FAFC;">
                    {res.get('confidence', 0.5):.1%}
                </div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Gauge & Model Breakdown Row
    col_g, col_comp, col_acts = st.columns([1.3, 1.2, 1.3])

    with col_g:
        # Gauge Meter
        risk_pct = res["fraud_risk"] * 100
        bar_color = "#EF4444" if risk_pct >= 70 else ("#EAB308" if risk_pct >= 40 else "#22C55E")

        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                number={"suffix": "%", "font": {"size": 32, "color": "#F8FAFC"}},
                title={"text": "Graph Transformer Fraud Risk", "font": {"size": 14, "color": "#94A3B8"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                    "bar": {"color": bar_color},
                    "bgcolor": "rgba(30, 41, 59, 0.4)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255, 255, 255, 0.1)",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(34, 197, 94, 0.15)"},
                        {"range": [40, 70], "color": "rgba(234, 179, 8, 0.15)"},
                        {"range": [70, 100], "color": "rgba(239, 68, 68, 0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "#EF4444", "width": 3},
                        "thickness": 0.8,
                        "value": res.get("threshold", 0.5) * 100,
                    },
                },
            )
        )
        gauge_fig.update_layout(
            height=200,
            margin={"l": 20, "r": 20, "t": 40, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col_comp:
        st.markdown("##### ⚖️ Model Comparison")
        gt_risk = res["fraud_risk"]
        base_risk = res.get("baseline_fraud_risk")

        st.metric("Graph Transformer (Attention)", f"{gt_risk:.1%}", help="2-layer multi-head TransformerConv with residual skips.")

        if base_risk is not None:
            delta_val = f"{(gt_risk - base_risk):+.1%}"
            st.metric("GraphSAGE Baseline", f"{base_risk:.1%}", delta=delta_val, delta_color="off", help="2-layer SAGEConv baseline.")
            if (gt_risk >= res.get("threshold", 0.5)) == (base_risk >= res.get("threshold", 0.5)):
                st.success("✅ **Consensus Reached:** Both architectures agree on classification.")
            else:
                st.warning("⚠️ **Architectural Discrepancy:** Graph Transformer attention flagged topological patterns missed by standard neighbor mean.")
        else:
            st.caption("Enable baseline toggle above to compare against GraphSAGE.")

    with col_acts:
        st.markdown("##### 🛡️ AML Action Protocol")
        if res["fraud_risk"] >= 0.70:
            st.error("🚨 **Action:** Immediate Transaction Freeze & File SAR/STR.")
            st.markdown("- **Priority:** Critical (Level 1 Escalation)")
            st.markdown("- **KYC Status:** Trigger mandatory wallet re-verification")
            st.markdown("- **Chainalysis / Elliptic Cross-Check:** High risk cluster")
        elif res["fraud_risk"] >= 0.40:
            st.warning("⚠️ **Action:** Request Source of Wealth / EDD Review.")
            st.markdown("- **Priority:** Moderate (Level 2 Analyst Review)")
            st.markdown("- **Hold Time:** 24h compliance verification buffer")
        else:
            st.success("✅ **Action:** Automated Clearance & Fast Settlement.")
            st.markdown("- **Priority:** Normal (Low Risk)")
            st.markdown("- **Audit Record:** Logged to compliance database")

    # Forensic AI Reasoning Callout
    st.markdown("##### 🧠 AI Forensic Reasoning Narrative:")
    st.markdown(
        f"""
    <div class="explanation-box">
        {res['reasoning']}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Feature Impacts Table if available
    if "feature_impacts" in res:
        with st.expander("📊 **Key Contributing Factors Breakdown**", expanded=False):
            f_df = pd.DataFrame(res["feature_impacts"])
            st.dataframe(f_df, use_container_width=True, hide_index=True)
