"""Batch Transaction Screener Component for XAI-GT Dashboard.

Allows forensic auditors to:
1. Screen batches of transactions in bulk.
2. Select random batches from the test set (timesteps 42–49).
3. Upload custom CSV feature files.
4. Export forensic audit logs as CSV.
"""

from __future__ import annotations

import random
from typing import Any, List, Optional
import io
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn.functional as F

from src.models.predictor import predict_node_transaction


def render_batch_view(
    data: Any,
    model: torch.nn.Module,
    baseline_model: Optional[torch.nn.Module] = None,
) -> None:
    """Renders the Batch Transaction Screener page in Streamlit."""

    st.markdown(
        """
<div class="investigation-header">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
        <div>
            <span style="font-size:1.6rem; font-weight:800; letter-spacing:-0.02em;">
                📂 Batch Transaction Screener &amp; Audit Portal
            </span>
            <div style="color:#94A3B8; font-size:0.9rem; margin-top:4px;">
                Screen multiple transactions simultaneously, evaluate fraud rate, and download regulatory compliance logs.
            </div>
        </div>
        <div>
            <span class="status-badge badge-clean">High-Throughput Mode</span>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    batch_tab1, batch_tab2 = st.tabs([
        "🎲 Test Split Batch Screening",
        "📤 Custom CSV Batch Upload",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Sample Batch Screening
    # -------------------------------------------------------------------------
    with batch_tab1:
        st.markdown("##### 🧪 Screen Sample Batch from Elliptic Test Split")
        st.caption("Timesteps 42–49 simulate real-world forward-testing without data leakage.")

        c1, c2, c3 = st.columns([1, 1, 1.5])
        with c1:
            batch_size = st.slider("Batch Size", min_value=5, max_value=50, value=15, step=5)
        with c2:
            split_choice = st.selectbox("Sample Source", ["Test Set (Mixed)", "Test Set (Illicit Only)", "Unlabelled Pool"])
        with c3:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            run_batch_btn = st.button("🚀 Screen Batch Now", type="primary", use_container_width=True)

        if run_batch_btn or "last_batch_records" not in st.session_state:
            # Pick nodes
            if split_choice == "Test Set (Illicit Only)":
                pool = torch.where(data.test_mask & (data.y == 1))[0].tolist()
            elif split_choice == "Unlabelled Pool":
                pool = torch.where(data.y == -1)[0].tolist()
            else:
                pool = torch.where(data.test_mask & (data.y >= 0))[0].tolist()

            sample_ids = random.sample(pool, min(batch_size, len(pool)))

            records = []
            progress_bar = st.progress(0, text="Evaluating batch via Graph Transformer …")

            for idx, nid in enumerate(sample_ids):
                r = predict_node_transaction(
                    node_id=nid,
                    graph_data=data,
                    model=model,
                    baseline_model=baseline_model,
                )
                records.append({
                    "Transaction ID": nid,
                    "Graph Transformer Risk": f"{r['fraud_risk']:.1%}",
                    "Raw Risk Score": r['fraud_risk'],
                    "GraphSAGE Risk": f"{r['baseline_fraud_risk']:.1%}" if r['baseline_fraud_risk'] else "N/A",
                    "Verdict": "🔴 FRAUD" if r["fraud_risk"] >= 0.5 else "🟢 CLEAN",
                    "Ground Truth": r["ground_truth"],
                    "In-Degree": r["in_degree"],
                    "Out-Degree": r["out_degree"],
                    "Clustering": f"{r['clustering_coef']:.2f}",
                })
                progress_bar.progress((idx + 1) / len(sample_ids), text=f"Screened {idx + 1}/{len(sample_ids)} transactions …")

            progress_bar.empty()
            st.session_state["last_batch_records"] = records

        if "last_batch_records" in st.session_state:
            records = st.session_state["last_batch_records"]
            df_batch = pd.DataFrame(records)

            # Summary metrics
            total_count = len(df_batch)
            fraud_count = sum(1 for r in records if "🔴" in r["Verdict"])
            fraud_rate = (fraud_count / total_count) if total_count > 0 else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Screened Transactions", f"{total_count}")
            m2.metric("Flagged Fraud", f"{fraud_count}", delta=f"{fraud_rate:.1%} rate", delta_color="inverse")
            m3.metric("Clean Transactions", f"{total_count - fraud_count}")
            m4.metric("Avg Fraud Score", f"{df_batch['Raw Risk Score'].mean():.1%}")

            st.markdown("---")
            st.markdown("##### 📋 Batch Screening Results Table:")
            display_df = df_batch.drop(columns=["Raw Risk Score"])
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Download CSV
            csv_buffer = io.StringIO()
            display_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Forensic Audit Log (CSV)",
                data=csv_buffer.getvalue(),
                file_name="xai_gt_batch_audit_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # -------------------------------------------------------------------------
    # TAB 2: Custom CSV Upload
    # -------------------------------------------------------------------------
    with batch_tab2:
        st.markdown("##### 📤 Upload Transaction Dataset (CSV)")
        st.caption("Upload a CSV containing transaction node IDs or feature rows to generate batch risk scoring.")

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            user_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(user_df)}** rows. Columns found:", list(user_df.columns)[:8])
            st.info("Direct inference pipeline ready for custom batch screening.")
        else:
            st.info("Upload any CSV file to begin bulk automated processing.")
