"""
Page 4 — Supply Chain Network Operations AI Copilot.

Interactive Agentic AI interface for diagnosing bottlenecks, running what-if
simulations, evaluating FTL vs Carting policies, and drafting operations memos.
"""

import sys
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
from src.agent.ops_copilot import NetworkOpsCopilot

st.set_page_config(page_title="AI Operations Copilot", layout="wide")
st.title("🤖 Network Operations AI Copilot")
st.markdown(
    "Autonomous supply chain assistant equipped with live graph inspection, "
    "what-if capacity simulators, and route-type optimization tools."
)

# ── Initialize Copilot in Session State ─────────────────────────────────────────
if "copilot" not in st.session_state:
    st.session_state.copilot = NetworkOpsCopilot()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am the **Delhivery Network Operations Copilot**.\n\n"
                "I have real-time access to the logistics network graph, centrality indices, and simulation engines. "
                "How can I assist your network operations team today?"
            ),
        }
    ]

# ── Sidebar Quick Actions ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Quick Diagnostic Queries")
    quick_queries = [
        "What is the bottleneck health status of Gurgaon Bilaspur hub?",
        "What happens to network SLAs if we upgrade Kolkata Dankuni hub by 30%?",
        "Should we use FTL or Carting for an 850km evening shipment with high delay?",
        "Simulate a 25% capacity boost for Bangalore hub and show ROI.",
    ]
    for q in quick_queries:
        if st.button(q, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            res = st.session_state.copilot.run_agentic_pipeline(q)
            st.session_state.messages.append({
                "role": "assistant",
                "content": res["response"],
                "steps": res.get("steps", []),
            })
            st.rerun()

    st.markdown("---")
    st.markdown("### 🛠 Active Agent Tools")
    st.markdown("""
    - `query_hub_health`: Graph centrality & dwell audit
    - `simulate_hub_upgrade`: Downstream delay ripple simulator
    - `recommend_route_type`: FTL vs Carting policy engine
    - `generate_incident_memo`: Executive consulting synthesis
    """)

# ── Render Conversation History ────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "steps" in msg and msg["steps"]:
            with st.expander("🔍 View Agent Tool-Calling Trace"):
                for step in msg["steps"]:
                    st.markdown(f"**Thought:** {step.get('thought')}")
                    st.markdown(f"**Tool Invoked:** `{step.get('tool')}`")
                    st.json(step.get("tool_input", {}))
                    if "tool_output" in step:
                        st.markdown("**Tool Execution Output:**")
                        st.json(step.get("tool_output"))

# ── Chat Input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about bottlenecks, what-if simulations, or corridor routing..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing logistics graph & executing operational tools..."):
            res = st.session_state.copilot.run_agentic_pipeline(prompt)
            st.markdown(res["response"])
            if res.get("steps"):
                with st.expander("🔍 View Agent Tool-Calling Trace"):
                    for step in res["steps"]:
                        st.markdown(f"**Thought:** {step.get('thought')}")
                        st.markdown(f"**Tool Invoked:** `{step.get('tool')}`")
                        st.json(step.get("tool_input", {}))
                        if "tool_output" in step:
                            st.markdown("**Tool Execution Output:**")
                            st.json(step.get("tool_output"))

    st.session_state.messages.append({
        "role": "assistant",
        "content": res["response"],
        "steps": res.get("steps", []),
    })
