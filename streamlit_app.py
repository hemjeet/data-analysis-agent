"""
Streamlit UI for the Agentic Data Analysis Agent.

Run with:  streamlit run streamlit_app.py
"""

import streamlit as st
import anthropic
import json
import pathlib
import pandas as pd
from dotenv import load_dotenv

from config import get_system_prompt
from cost_control import (
    CostTracker,
    trim_messages,
    trim_tool_outputs,
    context_is_safe,
    route,
    make_cached_system,
)
from guradrails import validate_input, validate_output, run_python_safe
from retry import run_agent_turn, run_agent_turn_streaming

# ── Load env ───────────────────────────────────────────────
load_dotenv()

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ─────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0ff !important;
    }

    /* ── Status badges ──────────────────────────── */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .badge-haiku {
        background: linear-gradient(135deg, #00c9ff, #92fe9d);
        color: #0a0a0a !important;
    }
    .badge-sonnet {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: #fff !important;
    }

    /* ── Tool call expander ─────────────────────── */
    .tool-call-box {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        font-family: 'Fira Code', 'Cascadia Code', monospace;
        font-size: 0.82rem;
        color: #cdd6f4;
        overflow-x: auto;
    }
    .tool-label {
        color: #89b4fa;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .tool-output {
        color: #a6e3a1;
        white-space: pre-wrap;
    }

    /* ── Metric cards ──────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #888;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00c9ff, #92fe9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ── Data preview table ────────────────────── */
    .dataframe-container {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Chat messages ─────────────────────────── */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 4px;
    }

    /* ── Header hero ───────────────────────────── */
    .hero-title {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle {
        font-size: 0.9rem;
        color: #888;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],          # Anthropic message history
        "chat_display": [],      # UI display messages
        "df": None,
        "client": None,
        "system": None,
        "cached_system": None,
        "tracker": CostTracker(),
        "tools": [
            {
                "name": "run_python",
                "description": (
                    "Run Python and pandas code to analyze the dataframe. "
                    "The dataframe is already loaded as `df`. "
                    "Always use print() to show your results."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python/pandas code to run. Always use print().",
                        }
                    },
                    "required": ["code"],
                },
            }
        ],
        "dataset_loaded": False,
        "tool_logs": [],        # logs of tool calls for current turn
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ── Helper: run tool ───────────────────────────────────────
def run_tool(tool_name, tool_input):
    if tool_name == "run_python":
        code = tool_input["code"]
        result = run_python_safe(code, st.session_state.df)
        return result
    return f"Unknown tool: {tool_name}"


def clean_tool_result(raw: str) -> str:
    """Strip XML wrapper and injection-defence text for display."""
    text = raw
    if "<tool_result>" in text:
        text = text.replace("<tool_result>\n", "").replace("\n</tool_result>", "")
        idx = text.find("\n\nThe above is raw data output.")
        if idx != -1:
            text = text[:idx]
    return text


# ── Helper: load dataset ──────────────────────────────────
def load_dataset(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("salary_info.csv")

    st.session_state.df = df
    st.session_state.client = anthropic.Anthropic()
    system_prompt = get_system_prompt(df)
    st.session_state.system = system_prompt
    st.session_state.cached_system = make_cached_system(system_prompt)
    st.session_state.dataset_loaded = True


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="hero-title">📊 Data Agent</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">AI-powered data analysis with guardrails</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Dataset upload ─────────────────────────────────────
    st.markdown("#### 📁 Dataset")
    uploaded = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Upload your own CSV or use the built-in salary_info.csv",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Load", use_container_width=True):
            load_dataset(uploaded)
            st.rerun()
    with col2:
        if st.button("📋 Default", use_container_width=True, help="Load salary_info.csv"):
            load_dataset(None)
            st.rerun()

    # ── Dataset info ───────────────────────────────────────
    if st.session_state.dataset_loaded and st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("---")
        st.markdown("#### 📐 Dataset Info")

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rows × Columns</div>
            <div class="metric-value">{df.shape[0]} × {df.shape[1]}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🏷️ Columns & Types", expanded=False):
            for col in df.columns:
                dtype_icon = "🔢" if df[col].dtype in ["int64", "float64"] else "📝"
                st.markdown(f"{dtype_icon} **{col}** — `{df[col].dtype}`")

        with st.expander("👀 Preview (first 5 rows)", expanded=False):
            st.dataframe(df.head(), use_container_width=True, hide_index=True)

    # ── Cost tracker ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💰 Session Cost")
    tracker = st.session_state.tracker

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Spend</div>
        <div class="metric-value">${tracker.total_usd:.5f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">API Calls</div>
        <div class="metric-value">{tracker.calls}</div>
    </div>
    """, unsafe_allow_html=True)

    if tracker.by_model:
        with st.expander("📊 Cost Breakdown", expanded=False):
            for model, cost in tracker.by_model.items():
                label = model.split("-")[1].capitalize()
                st.markdown(f"**{label}**: ${cost:.5f}")

    # ── Session controls ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⚙️ Session")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_display = []
        st.session_state.tool_logs = []
        st.session_state.tracker = CostTracker()
        st.rerun()


# ── Main area ──────────────────────────────────────────────
if not st.session_state.dataset_loaded:
    # ── Welcome screen ─────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 80px 20px;">
        <div style="font-size: 4rem; margin-bottom: 16px;">📊</div>
        <h1 style="
            font-size: 2.4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        ">Data Analysis Agent</h1>
        <p style="color: #888; font-size: 1.1rem; max-width: 500px; margin: 0 auto 32px;">
            Upload a CSV in the sidebar or load the default dataset to start
            asking questions about your data.
        </p>
        <div style="
            display: flex;
            gap: 24px;
            justify-content: center;
            flex-wrap: wrap;
        ">
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                border: 1px solid #2a2a4a;
                border-radius: 14px;
                padding: 24px 28px;
                max-width: 200px;
                text-align: left;
            ">
                <div style="font-size: 1.6rem; margin-bottom: 8px;">🛡️</div>
                <div style="font-weight: 600; margin-bottom: 4px;">Guardrails</div>
                <div style="font-size: 0.8rem; color: #888;">Sandboxed code execution with input validation</div>
            </div>
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                border: 1px solid #2a2a4a;
                border-radius: 14px;
                padding: 24px 28px;
                max-width: 200px;
                text-align: left;
            ">
                <div style="font-size: 1.6rem; margin-bottom: 8px;">💸</div>
                <div style="font-weight: 600; margin-bottom: 4px;">Cost Control</div>
                <div style="font-size: 0.8rem; color: #888;">Smart model routing & prompt caching</div>
            </div>
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                border: 1px solid #2a2a4a;
                border-radius: 14px;
                padding: 24px 28px;
                max-width: 200px;
                text-align: left;
            ">
                <div style="font-size: 1.6rem; margin-bottom: 8px;">🔁</div>
                <div style="font-weight: 600; margin-bottom: 4px;">Resilience</div>
                <div style="font-size: 0.8rem; color: #888;">Retry logic, loop detection & error budgets</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Chat display ───────────────────────────────────────────
for msg in st.session_state.chat_display:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"].replace("$", "\\$"))
        # Show tool calls if any
        if "tool_logs" in msg and msg["tool_logs"]:
            for tl in msg["tool_logs"]:
                with st.expander(f"Tool: `{tl['name']}` -- view code & output", expanded=False):
                    st.markdown("**Code:**")
                    st.code(tl["code"], language="python")
                    st.markdown("**Output:**")
                    st.code(clean_tool_result(tl["result"]), language="text")
        # Show model badge
        if "model_badge" in msg:
            st.markdown(msg["model_badge"], unsafe_allow_html=True)


# ── Chat input ─────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your data…"):

    # Display user message immediately
    st.session_state.chat_display.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # ── Validate input ─────────────────────────────────────
    try:
        validated = validate_input(prompt)
    except ValueError as e:
        error_msg = f"⚠️ {e}"
        st.session_state.chat_display.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant", avatar="🤖"):
            st.error(str(e))
        st.stop()

    # ── Route to model ─────────────────────────────────────
    model, max_tok = route(validated)
    model_label = model.split("-")[1].capitalize()
    badge_class = "badge-haiku" if "haiku" in model else "badge-sonnet"
    badge_html = f'<span class="status-badge {badge_class}">⚡ {model_label} · {max_tok} tokens</span>'

    # ── Add to Anthropic messages ──────────────────────────
    st.session_state.messages.append({"role": "user", "content": validated})

    # ── Trim & safety check ────────────────────────────────
    trimmed = trim_tool_outputs(trim_messages(st.session_state.messages))

    client = st.session_state.client
    cached_system = st.session_state.cached_system
    tools = st.session_state.tools

    if not context_is_safe(client, cached_system, tools, trimmed):
        warning = "⚠️ Context too large. Please clear the chat to continue."
        st.session_state.chat_display.append({"role": "assistant", "content": warning})
        with st.chat_message("assistant", avatar="🤖"):
            st.warning("Context too large. Please clear the chat to continue.")
        st.stop()

    # ── Run agent turn (streaming) ──────────────────────────
    with st.chat_message("assistant", avatar="🤖"):
        text_placeholder = st.empty()
        streamed_text = ""
        final_answer = ""
        tool_logs_collected = []
        current_tool = {}  # buffer for tool_start -> tool_end

        event_stream = run_agent_turn_streaming(
            messages=trimmed,
            client=client,
            system=cached_system,
            tools=tools,
            run_tool_fn=run_tool,
            model=model,
            max_tokens=max_tok,
        )

        for event in event_stream:
            etype = event["type"]

            if etype == "text_delta":
                streamed_text += event["text"]
                # Escape $ to prevent LaTeX rendering during streaming
                safe = streamed_text.replace("$", "\\$")
                text_placeholder.markdown(safe + " |")

            elif etype == "thinking_done":
                # Preceding text was intermediate thinking
                if streamed_text.strip():
                    text_placeholder.info(streamed_text.strip())
                else:
                    text_placeholder.empty()
                streamed_text = ""
                # Reserve a new slot for the next text block
                text_placeholder = st.empty()

            elif etype == "tool_start":
                current_tool = {"name": event["name"], "input": event["input"]}

            elif etype == "tool_end":
                code = current_tool.get("input", {}).get("code", "")
                raw_result = event["result"]
                tool_logs_collected.append({
                    "name": event["name"],
                    "code": code,
                    "result": raw_result,
                })
                with st.expander(
                    f"Tool: `{event['name']}` -- view code & output",
                    expanded=False,
                ):
                    st.markdown("**Code:**")
                    st.code(code, language="python")
                    st.markdown("**Output:**")
                    st.code(clean_tool_result(raw_result), language="text")
                # New placeholder for the next streamed text
                text_placeholder = st.empty()

            elif etype == "end":
                final_answer = validate_output(event["answer"])
                st.session_state.messages = event["messages"]
                st.session_state.tracker.record(model, None)
                # Render the final text (replace cursor), escape $ for LaTeX
                safe_answer = final_answer.replace("$", "\\$")
                text_placeholder.markdown(safe_answer)
                break

            elif etype == "error":
                text_placeholder.error(event["text"])
                final_answer = event["text"]
                break

        st.markdown(badge_html, unsafe_allow_html=True)

    # Save to display history
    st.session_state.chat_display.append({
        "role": "assistant",
        "content": final_answer,
        "tool_logs": tool_logs_collected,
        "model_badge": badge_html,
    })
