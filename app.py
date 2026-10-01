"""
app.py - Streamlit UI (v2: redesigned for a cleaner, more polished feel)

Pages (sidebar navigation):
    Dashboard  - at-a-glance stats + quick actions
    Upload     - ingest PDFs/notes
    AI Agent   - one chat that autonomously calls Q&A / Summarizer / Quiz tools
    Summaries  - per-doc or per-topic cheat sheets
    Flashcards - generate + review with spaced repetition
"""

import json
import tempfile
from datetime import date
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

import db
import ingestion
import qa_agent
import summarizer_agent
import quiz_agent
import agent

st.set_page_config(
    page_title="Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)
db.init_db()

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

        #MainMenu, footer, header { visibility: hidden; }

        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px; }

        /* ---- Animated gradient blobs behind everything ---- */
        .stApp {
            background: #0B0C14 !important;
            position: relative;
            overflow-x: hidden;
        }
        .stApp::before, .stApp::after {
            content: ""; position: fixed; border-radius: 50%; filter: blur(90px);
            opacity: 0.35; z-index: 0; pointer-events: none;
        }
        .stApp::before {
            width: 500px; height: 500px; top: -150px; left: -100px;
            background: radial-gradient(circle, #7C6CF6, transparent 70%);
            animation: floatBlob1 16s ease-in-out infinite;
        }
        .stApp::after {
            width: 450px; height: 450px; bottom: -120px; right: -100px;
            background: radial-gradient(circle, #22D3EE, transparent 70%);
            animation: floatBlob2 18s ease-in-out infinite;
        }
        @keyframes floatBlob1 {
            0%, 100% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(60px, 80px) scale(1.15); }
        }
        @keyframes floatBlob2 {
            0%, 100% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(-50px, -60px) scale(1.1); }
        }
        [data-testid="stAppViewContainer"], [data-testid="stMain"], .block-container,
        section[data-testid="stSidebar"] {
            position: relative !important; z-index: 1 !important;
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(14px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes popIn {
            from { opacity: 0; transform: scale(0.95); }
            to   { opacity: 1; transform: scale(1); }
        }
        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 0 0 0 rgba(124,108,246,0.5); }
            50% { box-shadow: 0 0 0 8px rgba(124,108,246,0); }
        }
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        /* ---- Hero / page header ---- */
        .sa-hero {
            background: linear-gradient(120deg, #7C6CF6 0%, #B15CF1 50%, #22D3EE 100%);
            background-size: 200% 200%;
            animation: fadeInUp 0.5s ease, shimmer 8s ease-in-out infinite;
            padding: 2rem 2.2rem;
            border-radius: 20px;
            margin-bottom: 1.8rem;
            box-shadow: 0 12px 32px rgba(124, 108, 246, 0.3);
        }
        .sa-hero h1 {
            color: white; margin: 0; font-size: 2rem; font-weight: 900;
            text-shadow: 0 2px 12px rgba(0,0,0,0.15);
        }
        .sa-hero p { color: rgba(255,255,255,0.9); margin: 0.35rem 0 0 0; font-size: 1rem; }

        /* ---- Stat / metric cards (glass) ---- */
        .sa-card {
            background: rgba(26, 29, 43, 0.65);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 0.8rem;
            transition: transform 0.2s ease, border-color 0.2s ease;
            animation: fadeInUp 0.5s ease;
        }
        .sa-card:hover { border-color: rgba(124,108,246,0.5); transform: translateY(-3px); }
        .sa-stat-num {
            font-size: 2.1rem; font-weight: 900; margin: 0;
            background: linear-gradient(90deg, #B3A8FA, #7CF6E8);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .sa-stat-label { font-size: 0.78rem; color: #9A9CB0; text-transform: uppercase; letter-spacing: 0.06em; margin: 0.2rem 0 0 0; font-weight: 600; }
        .sa-stat-icon { font-size: 1.4rem; margin-bottom: 0.3rem; }

        /* ---- Badges ---- */
        .sa-badge {
            display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px;
            font-size: 0.72rem; font-weight: 700; letter-spacing: 0.02em;
        }
        .sa-badge-purple { background: rgba(124,108,246,0.18); color: #C4B9FF; }
        .sa-badge-green  { background: rgba(56,193,114,0.18); color: #5EEBA0; }
        .sa-badge-amber  { background: rgba(245,158,11,0.18); color: #FBBF24; }
        .sa-badge-pulse  { animation: pulseGlow 2s infinite; }

        /* ---- Doc list rows ---- */
        .sa-doc-row {
            background: rgba(26, 29, 43, 0.65);
            backdrop-filter: blur(8px);
            border-radius: 14px; padding: 0.9rem 1.2rem;
            margin-bottom: 0.6rem; border: 1px solid rgba(255,255,255,0.06);
            display: flex; justify-content: space-between; align-items: center;
            transition: transform 0.15s ease, border-color 0.15s ease;
            animation: fadeInUp 0.4s ease;
        }
        .sa-doc-row:hover { transform: translateX(4px); border-color: rgba(124,108,246,0.4); }

        /* ---- Flashcard ---- */
        .sa-flashcard {
            background: linear-gradient(160deg, #232840 0%, #171A27 100%);
            border: 1px solid rgba(124,108,246,0.35);
            border-radius: 24px;
            padding: 2.6rem 2rem;
            text-align: center;
            min-height: 170px;
            display: flex; flex-direction: column; justify-content: center; align-items: center;
            margin-bottom: 1.2rem;
            box-shadow: 0 16px 40px rgba(80, 60, 200, 0.18);
            animation: popIn 0.35s cubic-bezier(0.2, 0.8, 0.3, 1.2);
        }
        .sa-flashcard .q { font-size: 1.35rem; font-weight: 700; color: #F2F2FA; line-height: 1.4; }
        .sa-flashcard .a {
            font-size: 1.15rem; margin-top: 1rem; font-weight: 700; padding: 0.4rem 1rem;
            background: rgba(94, 235, 160, 0.12); border-radius: 12px; color: #5EEBA0;
            animation: popIn 0.3s ease;
        }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #14161F 0%, #0F111A 100%);
            border-right: 1px solid rgba(124,108,246,0.15);
        }
        .sa-logo-wrap { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.2rem; }
        .sa-logo-circle {
            width: 42px; height: 42px; border-radius: 12px; flex-shrink: 0;
            background: linear-gradient(135deg, #7C6CF6, #22D3EE);
            display: flex; align-items: center; justify-content: center; font-size: 1.3rem;
            box-shadow: 0 4px 14px rgba(124,108,246,0.4);
        }
        .sa-sidebar-title { font-size: 1.15rem; font-weight: 800; color: #EDEDF4; margin: 0; line-height: 1.2; }
        .sa-sidebar-sub { font-size: 0.75rem; color: #8A8CA0; margin: 0 0 1.3rem 0; }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px; padding: 0.55rem 0.8rem; margin-bottom: 0.4rem;
            transition: all 0.15s ease; width: 100%;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(124,108,246,0.12); border-color: rgba(124,108,246,0.4);
        }

        div.stButton > button {
            border-radius: 12px; font-weight: 700; border: 1px solid rgba(124,108,246,0.35);
            transition: all 0.15s ease;
        }
        div.stButton > button:hover {
            border-color: #7C6CF6; color: #fff; background: rgba(124,108,246,0.15);
            transform: translateY(-1px);
        }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #7C6CF6, #22D3EE); border: none;
        }

        /* ---- Dividers ---- */
        hr {
            border: none; height: 2px; margin: 1.2rem 0;
            background: linear-gradient(90deg, transparent, rgba(124,108,246,0.5), transparent);
        }

        /* ---- Alerts (success/info/warning) ---- */
        [data-testid="stAlert"] {
            background: rgba(26, 29, 43, 0.75) !important;
            backdrop-filter: blur(10px);
            border-radius: 14px !important;
            border: 1px solid rgba(124,108,246,0.25) !important;
            animation: fadeInUp 0.35s ease;
        }

        /* ---- File uploader drop zone ---- */
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(124,108,246,0.06) !important;
            border: 2px dashed rgba(124,108,246,0.4) !important;
            border-radius: 16px !important;
            transition: all 0.2s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #7C6CF6 !important; background: rgba(124,108,246,0.1) !important;
        }

        /* ---- Inputs / selects / sliders ---- */
        div[data-baseweb="select"] > div, input[type="text"], textarea {
            background: rgba(255,255,255,0.04) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
        }
        div[data-baseweb="select"] > div:focus-within, input[type="text"]:focus, textarea:focus {
            border-color: #7C6CF6 !important;
            box-shadow: 0 0 0 3px rgba(124,108,246,0.2) !important;
        }
        div[data-testid="stSlider"] [role="slider"] { background: #7C6CF6 !important; }

        /* ---- Chat bubbles ---- */
        [data-testid="stChatMessage"] {
            animation: fadeInUp 0.3s ease;
            border-radius: 16px !important;
            padding: 0.4rem 0.2rem;
        }
        [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
            background: rgba(124,108,246,0.08) !important;
        }
        [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
            background: rgba(34,211,238,0.06) !important;
        }

        /* ---- Flashcard hover tilt ---- */
        .sa-flashcard { transition: transform 0.25s ease, box-shadow 0.25s ease; }
        .sa-flashcard:hover {
            transform: translateY(-4px) scale(1.01);
            box-shadow: 0 20px 48px rgba(124, 108, 246, 0.28);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
docs = db.list_documents()
due_count = len(quiz_agent.get_due_cards(limit=1000))

with st.sidebar:
    st.markdown(
        '<div class="sa-logo-wrap"><div class="sa-logo-circle">📚</div>'
        '<div><p class="sa-sidebar-title">Study Assistant</p>'
        '<p class="sa-sidebar-sub" style="margin:0;">RAG study companion</p></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "📥 Upload", "🤖 AI Agent", "📝 Summaries", "🃏 Flashcards"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    due_badge = (
        f'<span class="sa-badge sa-badge-amber sa-badge-pulse">{due_count} due</span>'
        if due_count > 0 else '<span class="sa-badge sa-badge-green">All caught up</span>'
    )
    st.markdown(
        f'<div class="sa-card" style="padding:0.9rem 1rem;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">'
        f'<span style="color:#9A9CB0; font-size:0.85rem;">📄 Documents</span><b>{len(docs)}</b></div>'
        f'<div style="display:flex; justify-content:space-between; align-items:center;">'
        f'<span style="color:#9A9CB0; font-size:0.85rem;">🃏 Reviews</span>{due_badge}</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"📅 {date.today().strftime('%d %b %Y')}")

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
if page == "🏠 Dashboard":
    st.markdown(
        """
        <div class="sa-hero">
            <h1>Welcome back 👋</h1>
            <p>Upload your notes, ask grounded questions, and review flashcards on a spaced-repetition schedule.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    for col, icon, num, label in [
        (c1, "📄", len(docs), "Documents"),
        (c2, "🔥", due_count, "Cards due today"),
        (c3, "📕", sum(1 for d in docs if d["filetype"] == "pdf"), "PDFs ingested"),
    ]:
        with col:
            st.markdown(
                f'<div class="sa-card"><div class="sa-stat-icon">{icon}</div>'
                f'<p class="sa-stat-num">{num}</p>'
                f'<p class="sa-stat-label">{label}</p></div>',
                unsafe_allow_html=True,
            )

    mastery = db.get_mastery_breakdown()
    total_cards = sum(mastery.values())

    dash_col1, dash_col2 = st.columns([1.3, 1], gap="large")

    with dash_col1:
        st.markdown("#### Recent documents")
        if docs:
            for d in docs[:5]:
                badge_class = "sa-badge-purple" if d["filetype"] == "pdf" else "sa-badge-green"
                st.markdown(
                    f'<div class="sa-doc-row"><span>📄 <b>{d["filename"]}</b></span>'
                    f'<span class="sa-badge {badge_class}">{d["filetype"].upper()}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No documents yet — head to **Upload** to add your first PDF or notes.")

        st.markdown("#### Quick actions")
        qcol1, qcol2, qcol3 = st.columns(3)
        qcol1.button("📥 Upload a document", use_container_width=True)
        qcol2.button("💬 Ask a question", use_container_width=True)
        qcol3.button("🃏 Review flashcards", use_container_width=True)

    with dash_col2:
        st.markdown("#### Flashcard mastery")
        if total_cards > 0:
            colors = ["#7C6CF6", "#22D3EE", "#FBBF24", "#5EEBA0"]
            fig = go.Figure(data=[go.Pie(
                labels=list(mastery.keys()),
                values=list(mastery.values()),
                hole=0.62,
                marker=dict(colors=colors, line=dict(color="#0B0C14", width=3)),
                textinfo="none",
            )])
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, font=dict(color="#EDEDF4", size=11)),
                margin=dict(t=10, b=10, l=10, r=10),
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                annotations=[dict(
                    text=f"<b>{total_cards}</b><br><span style='font-size:11px;color:#9A9CB0'>cards</span>",
                    x=0.5, y=0.5, showarrow=False, font=dict(color="#EDEDF4", size=22),
                )],
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown(
                '<div class="sa-card" style="text-align:center; padding:2rem;">'
                '🃏 <br><span style="color:#9A9CB0;">Generate flashcards to see your mastery breakdown here.</span></div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
elif page == "📥 Upload":
    st.markdown(
        '<div class="sa-hero"><h1>📥 Upload materials</h1>'
        '<p>Add PDFs, notes, or pasted lecture transcripts to your knowledge base.</p></div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### From a file")
        uploaded = st.file_uploader("Choose a file", type=["pdf", "txt", "md"], label_visibility="collapsed")
        if uploaded and st.button("Ingest file", type="primary", use_container_width=True):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp) / uploaded.name
                tmp_path.write_bytes(uploaded.getvalue())
                with st.spinner("Parsing, chunking, and embedding..."):
                    doc_id = ingestion.ingest_file(str(tmp_path))
                st.success(f"✅ Ingested **{uploaded.name}** as document #{doc_id}")
                st.balloons()

    with col2:
        st.markdown("##### From pasted text")
        pasted = st.text_area("Paste notes", height=150, label_visibility="collapsed",
                               placeholder="Paste lecture notes or a transcript here...")
        if pasted.strip() and st.button("Ingest pasted text", type="primary", use_container_width=True):
            with st.spinner("Chunking and embedding..."):
                doc_id = ingestion.ingest_raw_text(pasted)
            st.success(f"✅ Ingested pasted notes as document #{doc_id}")

    st.markdown("#### Your library")
    if docs:
        for d in docs:
            badge_class = "sa-badge-purple" if d["filetype"] == "pdf" else "sa-badge-green"
            st.markdown(
                f'<div class="sa-doc-row"><span>📄 <b>#{d["id"]} {d["filename"]}</b>'
                f'<br><span style="color:#9A9CB0; font-size:0.8rem;">{d["uploaded_at"][:16].replace("T"," ")}</span></span>'
                f'<span class="sa-badge {badge_class}">{d["filetype"].upper()}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("Nothing uploaded yet.")

# ---------------------------------------------------------------------------
# AI Agent
# ---------------------------------------------------------------------------
elif page == "🤖 AI Agent":
    st.markdown(
        '<div class="sa-hero"><h1>🤖 AI Agent</h1>'
        '<p>One conversation, three tools. The agent decides on its own whether to '
        'search your notes, write a summary, or generate flashcards.</p></div>',
        unsafe_allow_html=True,
    )

    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []      # plain {"role","content"} turns for the LLM
    if "agent_display" not in st.session_state:
        st.session_state.agent_display = []       # richer turns for rendering (tool calls, sources)

    if not st.session_state.agent_display:
        st.markdown(
            '<div class="sa-card" style="text-align:center; padding:2.4rem;">'
            '<div style="font-size:2rem;">🤖</div>'
            '<b>Tell the agent what you need — it picks the tool</b><br>'
            '<span style="color:#9A9CB0; font-size:0.88rem;">Try: "Quiz me on my biology notes" or '
            '"Summarize the AI systems doc" or "What is the SM-2 algorithm?"</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    tool_icons = {
        "list_documents": "📂",
        "ask_question": "🔎",
        "summarize_document": "📝",
        "generate_flashcards": "🃏",
    }

    for msg in st.session_state.agent_display:
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            for call in msg.get("tool_calls", []):
                icon = tool_icons.get(call["name"], "🔧")
                st.caption(f"{icon} used **{call['name']}**")
            if msg.get("sources"):
                st.caption(" · ".join(
                    f"{s['filename']}" + (f" p.{s['page']}" if s["page"] else "")
                    for s in msg["sources"]
                ))

    if question := st.chat_input("Ask the agent anything about your materials..."):
        st.session_state.agent_display.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking and choosing the right tool..."):
                result = agent.run_agent(question, conversation_history=st.session_state.agent_history)
            st.markdown(result["answer"])

            sources = None
            for call in result["tool_calls"]:
                icon = tool_icons.get(call["name"], "🔧")
                st.caption(f"{icon} used **{call['name']}**")
                if call["name"] == "ask_question" and call["result"].get("sources"):
                    sources = call["result"]["sources"]
            if sources:
                st.caption(" · ".join(
                    f"{s['filename']}" + (f" p.{s['page']}" if s["page"] else "")
                    for s in sources
                ))

        st.session_state.agent_display.append({
            "role": "assistant", "content": result["answer"],
            "tool_calls": result["tool_calls"], "sources": sources,
        })
        # Keep the LLM-facing history small and plain-text (no tool payloads)
        st.session_state.agent_history.append({"role": "user", "content": question})
        st.session_state.agent_history.append({"role": "assistant", "content": result["answer"]})

# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
elif page == "📝 Summaries":
    st.markdown(
        '<div class="sa-hero"><h1>📝 Summaries & cheat sheets</h1>'
        '<p>Grounded, exam-ready notes generated from your material.</p></div>',
        unsafe_allow_html=True,
    )

    if docs:
        doc_choice = st.selectbox("Document", [f"#{d['id']} {d['filename']}" for d in docs], key="sum_doc")
        doc_id = int(doc_choice.split()[0].lstrip("#"))
        topic = st.text_input("Optional topic (leave blank to summarize the whole document)")
        if st.button("✨ Generate summary", type="primary"):
            with st.spinner("Summarizing..."):
                summary = summarizer_agent.summarize_document(doc_id, topic=topic or None)
            st.markdown('<div class="sa-card">' + "\n\n" + summary + "\n\n" + '</div>', unsafe_allow_html=True)
    else:
        st.info("Upload a document first from the **Upload** page.")

# ---------------------------------------------------------------------------
# Flashcards
# ---------------------------------------------------------------------------
elif page == "🃏 Flashcards":
    st.markdown(
        '<div class="sa-hero"><h1>🃏 Flashcards</h1>'
        '<p>Generate cards, then review them on a spaced-repetition schedule (SM-2).</p></div>',
        unsafe_allow_html=True,
    )

    gen_col, review_col = st.columns([1, 1.4], gap="large")

    with gen_col:
        st.markdown("##### Generate new cards")
        if docs:
            doc_choice = st.selectbox("Document", [f"#{d['id']} {d['filename']}" for d in docs], key="quiz_doc")
            doc_id = int(doc_choice.split()[0].lstrip("#"))
            topic = st.text_input("Optional topic to focus on", key="quiz_topic")
            num_cards = st.slider("Number of cards", 3, 15, 8)
            if st.button("✨ Generate cards", type="primary", use_container_width=True):
                with st.spinner("Generating flashcards..."):
                    created = quiz_agent.generate_cards(doc_id, topic=topic or None, num_cards=num_cards)
                if created:
                    st.success(f"Created {len(created)} cards — due for review now.")
                    st.rerun()
                else:
                    st.warning("No cards were generated — check the document has enough content.")
        else:
            st.caption("Upload a document first.")

    with review_col:
        st.markdown("##### Review")
        due = quiz_agent.get_due_cards(limit=20)
        if not due:
            st.markdown(
                '<div class="sa-card" style="text-align:center; padding:2rem;">'
                '🎉 <b>All caught up!</b><br><span style="color:#9A9CB0;">No cards due right now.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            if "review_idx" not in st.session_state:
                st.session_state.review_idx = 0
            if "revealed" not in st.session_state:
                st.session_state.revealed = False

            idx = st.session_state.review_idx % len(due)
            card = due[idx]

            progress_pct = int((idx / len(due)) * 100)
            st.markdown(
                f'<div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.6rem;">'
                f'<div style="flex:1; height:8px; background:rgba(255,255,255,0.08); border-radius:999px; overflow:hidden;">'
                f'<div style="width:{progress_pct}%; height:100%; background:linear-gradient(90deg,#7C6CF6,#22D3EE); '
                f'border-radius:999px; transition: width 0.3s ease;"></div></div>'
                f'<span style="font-size:0.8rem; color:#9A9CB0; white-space:nowrap;">Card {idx + 1} of {len(due)}</span></div>',
                unsafe_allow_html=True,
            )

            options_html = ""
            if card["card_type"] == "mcq" and card["options"]:
                options = json.loads(card["options"])
                options_html = "<br>".join(f"• {opt}" for opt in options)

            answer_html = f'<div class="a">✅ {card["answer"]}</div>' if st.session_state.revealed else ""
            options_block = f"<div style='color:#9A9CB0; margin-top:0.6rem;'>{options_html}</div>" if options_html else ""

            st.markdown(
                f'<div class="sa-flashcard"><span class="sa-badge sa-badge-purple">{card["card_type"].upper()}</span>'
                f'<div class="q" style="margin-top:0.8rem;">{card["question"]}</div>'
                f'{options_block}'
                f'{answer_html}</div>',
                unsafe_allow_html=True,
            )

            if not st.session_state.revealed:
                if st.button("👁️ Show answer", use_container_width=True):
                    st.session_state.revealed = True
                    st.rerun()
            else:
                st.write("How well did you know it?")
                cols = st.columns(4)
                labels = [("😵 Again", 0), ("😬 Hard", 3), ("🙂 Good", 4), ("😎 Easy", 5)]
                for col, (label, quality) in zip(cols, labels):
                    with col:
                        if st.button(label, key=f"btn_{label}_{card['id']}", use_container_width=True):
                            quiz_agent.review_card(card["id"], quality)
                            st.session_state.revealed = False
                            st.session_state.review_idx += 1
                            st.rerun()
