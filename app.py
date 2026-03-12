"""Research Continuity Agent — Streamlit Interface."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Research Continuity Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ─────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "chat"
if "generating" not in st.session_state:
    st.session_state.generating = False

dark = st.session_state.dark_mode

# ── Colour palette ─────────────────────────────────────────────────────────────
if dark:
    bg         = "#0f1117"
    surface    = "#1a1d27"
    surface2   = "#22263a"
    border     = "#2e3350"
    accent     = "#7c6af7"
    accent2    = "#4ecdc4"
    text       = "#e8eaf0"
    text_muted = "#8b92b0"
    success    = "#4ecdc4"
    tag_bg     = "#2a2d45"
else:
    bg         = "#f5f6fa"
    surface    = "#ffffff"
    surface2   = "#eef0f8"
    border     = "#d4d8f0"
    accent     = "#5b4de0"
    accent2    = "#00a896"
    text       = "#1a1d2e"
    text_muted = "#6b718f"
    success    = "#00a896"
    tag_bg     = "#e8eaf8"

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

/* ── Theme CSS variables (overrides Streamlit's base="dark" defaults) ── */
:root {{
    --background-color: {bg} !important;
    --secondary-background-color: {surface} !important;
    --text-color: {text} !important;
    --primary-color: {accent} !important;
}}

/* ── Reset & Base ── */
html, body {{
    font-family: 'Inter', sans-serif;
    background-color: {bg} !important;
    color: {text};
}}
[class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {text};
}}
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stHeader"] {{
    background-color: {bg} !important;
}}
/* Baseweb input component (light mode fix) */
[data-baseweb="base-input"] {{
    background-color: {surface} !important;
    color: {text} !important;
}}

/* ── Hide Streamlit chrome (keep sidebar toggle intact) ── */
#MainMenu, footer {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background-color: {surface} !important;
    border-right: 1px solid {border};
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 2rem; }}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label {{
    color: {text};
}}

/* ── Main container ── */
.main .block-container {{ padding: 1.5rem 2rem; max-width: 1200px; }}

/* ── Typography ── */
.rca-header {{
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: {accent};
    letter-spacing: -0.02em;
    margin-bottom: 0.1rem;
}}
.rca-subheader {{
    font-size: 0.78rem;
    color: {text_muted};
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 1.5rem;
}}

/* ── Chat messages ── */
.msg-user {{
    background: {surface2};
    border: 1px solid {border};
    border-radius: 12px 12px 4px 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0 0.5rem 15%;
    font-size: 0.92rem;
}}
.msg-agent {{
    background: {surface};
    border: 1px solid {border};
    border-left: 3px solid {accent};
    border-radius: 4px 12px 12px 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 15% 0.5rem 0;
    font-size: 0.92rem;
    line-height: 1.65;
}}
.msg-agent code {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    background: {tag_bg};
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    color: {accent2};
}}

/* ── Citations ── */
.citation-section {{
    margin-top: 0.8rem;
    border-top: 1px solid {border};
    padding-top: 0.7rem;
}}
.citation-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: {text_muted};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}}
.citation-card {{
    background: {tag_bg};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
}}
.citation-title {{
    font-weight: 500;
    color: {accent};
    font-size: 0.82rem;
    margin-bottom: 0.25rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.citation-id {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: {text_muted};
    margin-bottom: 0.3rem;
}}
.citation-excerpt {{
    font-size: 0.78rem;
    color: {text_muted};
    line-height: 1.5;
    font-style: italic;
}}

/* ── Badges ── */
.badge-grounded {{
    display: inline-block;
    background: {success}22;
    color: {success};
    border: 1px solid {success}44;
    border-radius: 20px;
    padding: 0.1rem 0.6rem;
    font-size: 0.68rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-left: 0.5rem;
}}
.badge-ungrounded {{
    display: inline-block;
    background: #ff6b6b22;
    color: #ff6b6b;
    border: 1px solid #ff6b6b44;
    border-radius: 20px;
    padding: 0.1rem 0.6rem;
    font-size: 0.68rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-left: 0.5rem;
}}

/* ── Sidebar section labels ── */
.mode-label {{
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {text_muted};
    margin-bottom: 0.3rem;
}}

/* ── Ingest status ── */
.ingest-success {{
    background: {success}11;
    border: 1px solid {success}33;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin: 0.3rem 0;
    font-size: 0.82rem;
    font-family: 'IBM Plex Mono', monospace;
    color: {success};
}}
.ingest-error {{
    background: #ff6b6b11;
    border: 1px solid #ff6b6b33;
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin: 0.3rem 0;
    font-size: 0.82rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #ff6b6b;
}}

/* ── Stat metrics ── */
.stat-row {{
    display: flex;
    gap: 1.2rem;
    align-items: flex-start;
    margin: 0.2rem 0;
}}
.stat-metric {{
    border-left: 3px solid {accent};
    padding: 0.3rem 0.6rem;
    flex: 1;
    min-width: 0;
}}
.stat-value {{
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: {accent};
    line-height: 1.1;
    white-space: nowrap;
}}
.stat-label {{
    font-size: 0.65rem;
    color: {text_muted};
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

/* ── Inputs ── */
.stTextInput > div > div > input {{
    background-color: {surface} !important;
    border: 1px solid {border} !important;
    border-radius: 8px;
    color: {text} !important;
}}
/* Suppress red validation border after form submit */
.stTextInput > div > div > input:invalid,
.stTextInput > div > div > input:focus:invalid {{
    box-shadow: none !important;
    border-color: {border} !important;
}}
/* Remove form default focus ring on the form container */
[data-testid="stForm"] {{
    border: none !important;
    padding: 0 !important;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] section {{
    background-color: {surface} !important;
    border: 1px dashed {border} !important;
    border-radius: 8px;
}}
[data-testid="stFileUploaderDropzoneInstructions"] {{
    color: {text_muted} !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {{
    color: {text_muted} !important;
}}
[data-testid="stFileUploaderDropzone"] svg {{ fill: {text_muted}; }}
[data-testid="stFileUploader"] label {{ color: {text} !important; }}

/* ── Buttons: default filled accent ── */
.stButton > button {{
    background: {accent};
    color: white !important;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.5rem 1.2rem;
    transition: all 0.15s;
}}
.stButton > button:hover {{
    background: {accent}cc;
    transform: translateY(-1px);
}}

/* ── Sidebar buttons: ghost style ── */
section[data-testid="stSidebar"] .stButton > button {{
    background: transparent;
    color: {text_muted} !important;
    border: 1px solid {border};
    font-size: 0.78rem;
    padding: 0.3rem 0.8rem;
    transform: none;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: {surface2};
    color: {text} !important;
    transform: none;
}}

/* ── Radio pill style ── */
div[data-testid="stRadio"] > label {{ display: none; }}
div[data-testid="stRadio"] > div {{
    gap: 0.35rem;
    flex-wrap: nowrap;
}}
/* Hide the native radio dot/circle indicator */
div[data-testid="stRadio"] > div > label > div:first-child {{
    display: none !important;
}}
div[data-testid="stRadio"] > div > label {{
    background: {surface2};
    border: 1px solid {border};
    border-radius: 20px;
    padding: 0.25rem 0.85rem;
    font-size: 0.78rem;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: {text_muted};
    cursor: pointer;
    transition: all 0.15s;
    display: flex !important;
    align-items: center;
    white-space: nowrap;
}}
div[data-testid="stRadio"] > div > label:has(input:checked) {{
    background: {accent};
    color: white !important;
    border-color: {accent};
}}
/* Sidebar mode radio: vertical stacked pills */
section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {{
    flex-direction: column;
    gap: 0.25rem;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] > div > label {{
    border-radius: 8px;
    width: 100%;
    justify-content: flex-start;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {surface};
    border-radius: 8px;
    padding: 0.2rem;
    gap: 0.2rem;
}}
.stTabs [data-baseweb="tab"] {{
    color: {text_muted};
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.82rem;
    padding: 0.5rem 1rem;
}}
.stTabs [aria-selected="true"] {{
    background-color: {accent};
    color: white !important;
    border-radius: 6px;
}}

/* ── Misc ── */
hr {{ border-color: {border}; margin: 1rem 0; }}
.empty-state {{
    text-align: center;
    padding: 3rem 1rem;
    color: {text_muted};
}}
.empty-state-icon {{ font-size: 2.5rem; margin-bottom: 0.5rem; }}
.empty-state-text {{ font-size: 0.9rem; line-height: 1.6; }}
</style>
""", unsafe_allow_html=True)


# ── Cached resources ───────────────────────────────────────────────────────────
@st.cache_resource
def get_flows():
    from rca.flows.generate_flow import GenerateFlow
    from rca.flows.ingest_flow import IngestFlow
    from rca.store.graph_store import GraphStore
    from rca.config.settings import get_settings
    settings = get_settings()
    return {
        "generate": GenerateFlow(),
        "ingest": IngestFlow(),
        "graph": GraphStore(settings.graph_db_path),
        "settings": settings,
    }


def get_store_stats(flows):
    import sqlite3
    db_path = flows["settings"].graph_db_path
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT kind, COUNT(*) as count FROM nodes GROUP BY kind").fetchall()
        conn.close()
        return {row["kind"]: row["count"] for row in rows}
    except Exception:
        return {}


def format_answer_with_citations(answer_text):
    import re
    pattern = re.compile(r"\[\[((?:src|chk):[^\]]+)\]\]")
    def replace(m):
        sid = m.group(1)
        short = sid.split("/")[-1][:30] + ("…" if len(sid.split("/")[-1]) > 30 else "")
        return f'<code title="{sid}">📎 {short}</code>'
    return pattern.sub(replace, answer_text)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="rca-header">🧠 RCA</div>', unsafe_allow_html=True)
    st.markdown('<div class="rca-subheader">research continuity agent</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="mode-label">Theme</div>', unsafe_allow_html=True)
    theme_choice = st.radio(
        "Theme",
        ["🌙 Dark", "☀️ Light"],
        index=0 if dark else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="theme_radio",
    )
    new_dark = theme_choice == "🌙 Dark"
    if new_dark != dark and not st.session_state.generating:
        st.session_state.dark_mode = new_dark
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="mode-label">Mode</div>', unsafe_allow_html=True)
    mode_choice = st.radio(
        "Mode",
        ["💬 Chat", "🔬 Workspace"],
        index=0 if st.session_state.mode == "chat" else 1,
        horizontal=False,
        label_visibility="collapsed",
        key="mode_radio",
    )
    new_mode = "chat" if mode_choice == "💬 Chat" else "workspace"
    if new_mode != st.session_state.mode and not st.session_state.generating:
        st.session_state.mode = new_mode
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="mode-label">Knowledge Base</div>', unsafe_allow_html=True)
    try:
        flows = get_flows()
        stats = get_store_stats(flows)
        st.markdown(
            f'<div class="stat-row">'
            f'<div class="stat-metric"><div class="stat-value">{stats.get("paper", 0)}</div><div class="stat-label">papers</div></div>'
            f'<div class="stat-metric"><div class="stat-value">{stats.get("chunk", 0)}</div><div class="stat-label">chunks</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.caption(f"Store not ready: {e}")

    st.markdown("---")
    if st.button("🗑 Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ── CHAT MODE ─────────────────────────────────────────────────────────────────
if st.session_state.mode == "chat":
    st.markdown(
        '<div class="rca-header">Research Chat</div>'
        '<div class="rca-subheader">ask questions · get grounded answers · every claim cited</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-icon">🔬</div>'
            '<div class="empty-state-text">Your research knowledge base is ready.<br>'
            'Ask anything about your ingested papers.<br><br>'
            '<em>Try: "What approaches exist for robotic bin packing?"</em>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                answer_html = format_answer_with_citations(msg["content"])
                badge = (
                    '<span class="badge-grounded">✓ grounded</span>'
                    if msg.get("grounded")
                    else '<span class="badge-ungrounded">⚠ unverified</span>'
                )
                citations_html = ""
                if msg.get("citations"):
                    cards = "".join(
                        f'<div class="citation-card">'
                        f'<div class="citation-title">{c["title"][:60]}{"…" if len(c["title"]) > 60 else ""}</div>'
                        f'<div class="citation-id">{c["source_id"]}</div>'
                        f'<div class="citation-excerpt">{c["excerpt"][:180]}…</div>'
                        f'</div>'
                        for c in msg["citations"]
                    )
                    citations_html = (
                        f'<div class="citation-section">'
                        f'<div class="citation-label">📎 Sources ({len(msg["citations"])})</div>'
                        f'{cards}</div>'
                    )
                st.markdown(
                    f'<div class="msg-agent">{badge}'
                    f'<div style="margin-top:0.4rem">{answer_html}</div>'
                    f'{citations_html}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            query = st.text_input(
                "query",
                placeholder="Ask a question about your research…",
                label_visibility="collapsed",
            )
        with col_btn:
            send = st.form_submit_button("Ask →")

    if send and query.strip():
        st.session_state.generating = True
        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner("Searching knowledge base and generating answer…"):
            try:
                flows = get_flows()
                result = flows["generate"].generate_answer(query)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "grounded": result.grounded,
                    "citations": [
                        {"source_id": c.source_id, "title": c.title, "excerpt": c.excerpt}
                        for c in result.citations
                    ],
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {e}",
                    "grounded": False,
                    "citations": [],
                })
        st.session_state.generating = False
        st.rerun()


# ── WORKSPACE MODE ────────────────────────────────────────────────────────────
else:
    st.markdown(
        '<div class="rca-header">Research Workspace</div>'
        '<div class="rca-subheader">ingest papers · explore knowledge map · manage your research store</div>',
        unsafe_allow_html=True,
    )

    tab_ingest, tab_map, tab_store = st.tabs(["📥 Ingest", "🗺 Knowledge Map", "📚 Store"])

    with tab_ingest:
        st.markdown("#### Add papers to your knowledge base")
        ingest_mode = st.radio(
            "Ingest mode",
            ["Single PDF", "Folder of PDFs"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if ingest_mode == "Single PDF":
            uploaded = st.file_uploader(
                "Upload PDF",
                type=["pdf"],
                label_visibility="collapsed",
            )
            if st.button("Ingest PDF") and uploaded is not None:
                tmp_dir = tempfile.mkdtemp()
                tmp_path = Path(tmp_dir) / uploaded.name
                tmp_path.write_bytes(uploaded.getvalue())
                try:
                    with st.spinner(f"Ingesting {uploaded.name}…"):
                        try:
                            flows = get_flows()
                            result = flows["ingest"].ingest_path(tmp_path)
                            st.markdown(
                                f'<div class="ingest-success">✓ {uploaded.name}<br>'
                                f'→ {result.source_id}<br>'
                                f'→ {len(result.chunk_ids)} chunks · {result.node_count} nodes</div>',
                                unsafe_allow_html=True,
                            )
                            get_flows.clear()
                        except Exception as e:
                            st.markdown(
                                f'<div class="ingest-error">✗ {uploaded.name}: {e}</div>',
                                unsafe_allow_html=True,
                            )
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            uploaded_files = st.file_uploader(
                "Upload PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
            if st.button("Ingest All") and uploaded_files:
                tmp_dir = tempfile.mkdtemp()
                try:
                    flows = get_flows()
                    progress = st.progress(0)
                    results_html = ""
                    for i, uf in enumerate(uploaded_files):
                        tmp_path = Path(tmp_dir) / uf.name
                        tmp_path.write_bytes(uf.getvalue())
                        try:
                            result = flows["ingest"].ingest_path(tmp_path)
                            results_html += f'<div class="ingest-success">✓ [{i+1}/{len(uploaded_files)}] {uf.name} → {len(result.chunk_ids)} chunks</div>'
                        except Exception as e:
                            results_html += f'<div class="ingest-error">✗ [{i+1}/{len(uploaded_files)}] {uf.name}: {e}</div>'
                        progress.progress((i + 1) / len(uploaded_files))
                    st.markdown(results_html, unsafe_allow_html=True)
                    get_flows.clear()
                    st.success(f"Done. Ingested {len(uploaded_files)} files.")
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

    with tab_map:
        st.markdown("#### Knowledge graph — papers and their connections")
        try:
            import sqlite3
            import plotly.graph_objects as go
            import networkx as nx

            flows = get_flows()
            db_path = flows["settings"].graph_db_path
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            nodes_rows = conn.execute(
                "SELECT id, kind, title FROM nodes WHERE kind IN ('paper','note','experiment')"
            ).fetchall()
            edges_rows = conn.execute("SELECT source, target, kind FROM edges").fetchall()
            conn.close()

            if not nodes_rows:
                st.markdown(
                    '<div class="empty-state"><div class="empty-state-icon">🗺</div>'
                    '<div class="empty-state-text">No source nodes yet.<br>Ingest some papers first.</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                G = nx.Graph()
                node_ids    = {row["id"] for row in nodes_rows}
                node_kinds  = {row["id"]: row["kind"]  for row in nodes_rows}
                node_titles = {row["id"]: row["title"] for row in nodes_rows}
                for row in nodes_rows:
                    G.add_node(row["id"])
                edge_count = 0
                for row in edges_rows:
                    if row["source"] in node_ids and row["target"] in node_ids:
                        G.add_edge(row["source"], row["target"])
                        edge_count += 1

                pos = nx.spring_layout(G, seed=42, k=2.5)
                kind_colors = {"paper": accent, "note": accent2, "experiment": "#f7b731"}

                edge_x, edge_y = [], []
                for u, v in G.edges():
                    x0, y0 = pos[u]; x1, y1 = pos[v]
                    edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y, mode="lines",
                    line=dict(width=0.8, color=border), hoverinfo="none",
                )
                node_x      = [pos[n][0] for n in G.nodes()]
                node_y      = [pos[n][1] for n in G.nodes()]
                node_colors = [kind_colors.get(node_kinds.get(n, "paper"), accent) for n in G.nodes()]
                node_labels = [
                    node_titles.get(n, n)[:35] + ("…" if len(node_titles.get(n, n)) > 35 else "")
                    for n in G.nodes()
                ]
                node_hover  = [
                    f"<b>{node_titles.get(n, n)[:60]}</b><br>"
                    f"<span style='color:gray'>{node_kinds.get(n, '?')}</span><br>"
                    f"<code>{n}</code>"
                    for n in G.nodes()
                ]
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode="markers+text",
                    hoverinfo="text", hovertext=node_hover,
                    text=node_labels,
                    textposition="top center",
                    textfont=dict(size=8, color=text_muted),
                    marker=dict(size=14, color=node_colors, line=dict(width=1.5, color=surface)),
                )
                fig = go.Figure(
                    data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False, hovermode="closest",
                        paper_bgcolor=bg, plot_bgcolor=bg,
                        margin=dict(l=20, r=20, t=20, b=20), height=520,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        font=dict(color=text),
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"{len(nodes_rows)} source nodes · {edge_count} connections · hover nodes for details")

        except ImportError:
            st.warning("Install plotly and networkx: `uv add plotly networkx`")
        except Exception as e:
            st.error(f"Could not render graph: {e}")

    with tab_store:
        st.markdown("#### Everything in your knowledge base")
        try:
            import sqlite3
            flows = get_flows()
            conn = sqlite3.connect(flows["settings"].graph_db_path)
            conn.row_factory = sqlite3.Row
            source_nodes = conn.execute(
                "SELECT id, kind, title, created_at FROM nodes WHERE kind != 'chunk' ORDER BY created_at DESC"
            ).fetchall()
            conn.close()

            if not source_nodes:
                st.markdown(
                    '<div class="empty-state"><div class="empty-state-icon">📚</div>'
                    '<div class="empty-state-text">Nothing ingested yet.</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                search_term = st.text_input(
                    "Filter",
                    placeholder="Search by title…",
                    label_visibility="collapsed",
                )
                for row in source_nodes:
                    if search_term and search_term.lower() not in row["title"].lower():
                        continue
                    kind_emoji = {"paper": "📄", "note": "📝", "experiment": "🧪"}.get(row["kind"], "•")
                    date = row["created_at"][:10] if row["created_at"] else ""
                    st.markdown(
                        f'<div class="citation-card">'
                        f'<div class="citation-title">{kind_emoji} {row["title"][:70]}</div>'
                        f'<div class="citation-id">{row["id"]}</div>'
                        f'<span style="font-size:0.7rem;color:{text_muted}">{date}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            st.error(f"Could not load store: {e}")
