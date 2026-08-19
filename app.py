import streamlit as st
import html

from src.pipeline import generate_answer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Skin Cancer RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
:root {
    --blue-1: #0759c7;
    --blue-2: #1683e8;
    --blue-border: #b9d5f5;
    --panel: #f8fbff;
    --page-bottom: #eef5ff;
    --text: #111827;
}

.stApp {
    background: #ffffff !important;
    color: var(--text) !important;
}

.block-container {
    width: 92% !important;
    max-width: 1450px !important;
    padding: 12px 0 125px !important;
    margin: 0 auto !important;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, var(--blue-1), var(--blue-2));
    border-radius: 17px;
    padding: 24px 34px 25px;
    margin: 0 0 28px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(0, 89, 199, .18);
}

.app-header h1 {
    color: #fff !important;
    font-size: 33px !important;
    font-weight: 800 !important;
    line-height: 1.25 !important;
    margin: 0 !important;
}

.app-header p {
    color: #fff !important;
    font-size: 17px !important;
    font-weight: 500 !important;
    margin: 8px 0 0 !important;
}

/* USER MESSAGE: blue framed bubble + centered white text */
[data-testid="stChatMessage"]:has(.user-message-marker) {
    flex-direction: row-reverse !important;
    justify-content: flex-start !important;
    align-items: center !important;
}

[data-testid="stChatMessage"]:has(.user-message-marker)
[data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, #0759c7, #1683e8) !important;
    border: 0 !important;
    color: #ffffff !important;

    width: 40% !important;
    max-width: 40% !important;
    flex: 0 0 40% !important;

    margin: 0 !important;
    padding: 11px 18px !important;

    border-radius: 16px !important;

    box-shadow:
        0 5px 15px rgba(0, 89, 199, .18) !important;

    /* Center the message */
    text-align: center !important;
    /* Horizontal + vertical centering */
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Make ALL text inside the user bubble white and centered */
[data-testid="stChatMessage"]:has(.user-message-marker)
[data-testid="stChatMessageContent"] * {
    color: #ffffff !important;
    text-align: center !important;
}

/* Remove unnecessary paragraph spacing */
[data-testid="stChatMessage"]:has(.user-message-marker)
[data-testid="stChatMessageContent"] p {
    color: #ffffff !important;
    text-align: center !important;
    margin: 0 !important;
}

/* ASSISTANT MESSAGE: visible light-blue frame + black text */
[data-testid="stChatMessage"]:has(.assistant-message-marker) {
    flex-direction: row !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
}

[data-testid="stChatMessage"]:has(.assistant-message-marker)
[data-testid="stChatMessageContent"] {
    background: #f8fbff !important;
    border: 1px solid #b9d5f5 !important;
    color: #000000 !important;
    width: calc(100% - 55px) !important;
    max-width: calc(100% - 55px) !important;
    flex: 1 1 auto !important;
    margin: 0 35px 0 0 !important;
    padding: 14px 18px !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 15px rgba(0, 70, 150, .07) !important;
}

[data-testid="stChatMessage"]:has(.assistant-message-marker)
[data-testid="stChatMessageContent"] * {
    color: #000000 !important;
}


[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, var(--blue-1), var(--blue-2)) !important;
    color: #fff !important;
    width: 40% !important;
    max-width: 40% !important;
    flex: 0 0 40% !important;
    margin: 0 !important;
    padding: 11px 18px !important;
    border-radius: 16px !important;
    box-shadow: 0 5px 15px rgba(0, 89, 199, .18) !important;
}



/* Assistant row */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    flex-direction: row !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
    background: var(--panel) !important;
    border: 1px solid var(--blue-border) !important;
    width: calc(100% - 55px) !important;
    max-width: calc(100% - 55px) !important;
    flex: 1 1 auto !important;
    margin: 0 35px 0 0 !important;
    padding: 14px 18px !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 15px rgba(0, 70, 150, .07) !important;
    color: #000000 !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] {
    color: #000000 !important;
    font-size: 18px !important;
    line-height: 1.78 !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] p {
    color: #000000 !important;
    font-size: 18px !important;
    line-height: 1.78 !important;
    margin: 0 0 14px !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] li,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] li {
    color: #000000 !important;
    font-size: 18px !important;
    line-height: 1.78 !important;
    margin-bottom: 6px !important;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {
    color: #000000 !important;
    font-weight: 750 !important;
}

/* ============================================================
   USER MESSAGE
   Right-side blue bubble with perfectly centered white text
   ============================================================ */

/* Entire user chat row */
[data-testid="stChatMessage"]:has(.user-message-marker) {
    width: 100% !important;
    max-width: 100% !important;

    display: flex !important;
    flex-direction: row-reverse !important;

    justify-content: flex-start !important;
    align-items: center !important;

    margin: 20px 0 24px !important;
    padding: 0 !important;

    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

/* Blue bubble */
[data-testid="stChatMessage"]:has(.user-message-marker)
[data-testid="stChatMessageContent"] {
    background: linear-gradient(
        135deg,
        #0759c7,
        #1683e8
    ) !important;

    width: 40% !important;
    max-width: 40% !important;
    flex: 0 0 40% !important;

    margin: 0 !important;
    padding: 14px 20px !important;

    min-height: 52px !important;

    border: 0 !important;
    border-radius: 16px !important;

    box-shadow:
        0 5px 15px rgba(0, 89, 199, 0.18) !important;

    /* Center the inner content */
    display: flex !important;
    flex-direction: column !important;

    align-items: center !important;
    justify-content: center !important;

    text-align: center !important;
}

/* The Markdown container INSIDE the blue bubble */
[data-testid="stChatMessage"]:has(.user-message-marker)
[data-testid="stChatMessageContent"]
[data-testid="stMarkdownContainer"] {
    width: 100% !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    text-align: center !important;

    margin: 0 !important;
    padding: 0 !important;
}

/* The actual user text */
[data-testid="stChatMessage"]:has(.user-message-marker)
[data-testid="stChatMessageContent"]
[data-testid="stMarkdownContainer"] p {
    width: 100% !important;

    margin: 0 !important;
    padding: 0 !important;

    color: #ffffff !important;

    font-size: 25px !important;
    line-height: 1.55 !important;
    font-weight: 500 !important;

    text-align: center !important;
}

/* Explicit user-message-text used for newly submitted messages */
.user-message-text {
    width: 100% !important;

    margin: 0 !important;
    padding: 0 !important;

    color: #ffffff !important;

    font-size: 25px !important;
    line-height: 1.55 !important;
    font-weight: 500 !important;

    text-align: center !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Make every element inside user bubble white */
[data-testid="stChatMessage"]:has(.user-message-marker)
[data-testid="stChatMessageContent"] * {
    color: #ffffff !important;
    text-align: center !important;
}

/* Force assistant body text to black */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] *,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] * {
    color: #000000 !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) h1,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) h1,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) h2,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) h2,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) h3,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) h3 {
    color: var(--blue-1) !important;
    font-weight: 750 !important;
    margin: 18px 0 10px !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) a,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) a {
    color: var(--blue-1) !important;
}

/* Confidence bar */
[data-testid="stChatMessage"] [data-testid="stAlert"] {
    border-radius: 10px !important;
    margin-top: 15px !important;
    font-size: 19px !important;
    font-weight: 600 !important;
}

[data-testid="stChatMessage"] [data-testid="stAlert"] * {
    font-size: 19px !important;
    font-weight: 600 !important;
}

/* Sources */
.sources-box {
    background: #f8fbff;
    border: 1px solid #c7ddf5;
    border-left: 5px solid var(--blue-1);
    border-radius: 12px;
    padding: 15px 13px 13px;
    margin-top: 18px;
}

.source-title {
    color: var(--blue-1) !important;
    font-weight: 750 !important;
    font-size: 16px !important;
}

.source-card {
    background: #fff;
    border: 1px solid #cbdff4;
    border-radius: 10px;
    padding: 12px 15px;
    margin-top: 10px;
}

.source-info {
    color: #000000 !important;
    font-size: 15px !important;
    margin-top: 6px;
    line-height: 1.7;
}

.source-info b {
    color: #334155 !important;
}

.source-id {
    background: #e8f2ff;
    color: var(--blue-1) !important;
    padding: 3px 7px;
    border-radius: 5px;
    font-family: monospace;
    font-size: 13px;
}

/* Disclaimer */
.disclaimer {
    background: #fff9e8;
    border: 1px solid #f4d98a;
    border-left: 5px solid #f59e0b;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 16px;
    color: #674d0a !important;
    font-size: 16px !important;
    line-height: 1.65;
}

.disclaimer b {
    color: #674d0a !important;
}

/* Explicit message text classes — more reliable than Streamlit DOM selectors */
.user-message-text {
    color: #000000 !important;
    font-size: 20px !important;
    line-height: 1.55 !important;
}

.assistant-message-text {
    color: #000000 !important;
}

.assistant-message-text * {
    color: #000000 !important;
}

/* Loading */
.loading-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
    padding: 25px 0;
}

.robot {
    width: 70px;
    height: 55px;
    background: #fff;
    border: 3px solid var(--blue-2);
    border-radius: 15px;
    position: relative;
    animation: robot-bounce 1.2s infinite ease-in-out;
    box-shadow: 0 5px 18px rgba(0, 100, 200, .18);
}

.robot::before {
    content: "";
    position: absolute;
    width: 9px;
    height: 9px;
    background: var(--blue-2);
    border-radius: 50%;
    top: 17px;
    left: 15px;
    box-shadow: 27px 0 var(--blue-2);
}

.robot::after {
    content: "";
    position: absolute;
    width: 24px;
    height: 4px;
    background: var(--blue-2);
    border-radius: 5px;
    bottom: 9px;
    left: 20px;
}

.robot-antenna {
    position: absolute;
    width: 3px;
    height: 15px;
    background: var(--blue-2);
    top: -15px;
    left: 31px;
}

.robot-antenna::after {
    content: "";
    position: absolute;
    width: 8px;
    height: 8px;
    background: var(--blue-2);
    border-radius: 50%;
    top: -5px;
    left: -2px;
}

.loading-text {
    color: var(--blue-1) !important;
    font-size: 16px !important;
    font-weight: 650;
    margin-top: 13px;
}

@keyframes robot-bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-7px); }
}

/* Bottom input area */
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {
    background: var(--page-bottom) !important;
}

[data-testid="stBottomBlockContainer"] {
    border-top: 1px solid #d3e4f8 !important;
    padding: 14px 0 !important;
}

[data-testid="stChatInput"] {
    width: 850px !important;
    max-width: 90% !important;
    margin: 0 auto !important;
    background: #fff !important;
    border: 2px solid var(--blue-2) !important;
    border-radius: 18px !important;
    padding: 5px !important;
    box-shadow: 0 3px 12px rgba(22, 131, 232, .12) !important;
}

[data-testid="stChatInput"] textarea {
    background: #fff !important;
    color: #000000 !important;
    border: 0 !important;
    outline: 0 !important;
    font-size: 21px !important;
    font-weight: 500 !important;
    line-height: 1.6 !important;
    padding: 14px 18px !important;
    min-height: 52px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
    font-size: 21px !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--blue-1) !important;
    box-shadow: 0 0 0 3px rgba(22, 131, 232, .15) !important;
}

[data-testid="stChatInput"] button {
    color: var(--blue-1) !important;
}

/* Mobile */
@media (max-width: 700px) {
    .block-container {
        width: 96% !important;
        padding-left: 10px !important;
        padding-right: 10px !important;
    }

    .app-header {
        padding: 21px 17px;
    }

    .app-header h1 {
        font-size: 25px !important;
    }

    .app-header p {
        font-size: 14px !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        width: 75% !important;
        max-width: 75% !important;
        flex: 0 0 75% !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        width: calc(100% - 45px) !important;
        max-width: calc(100% - 45px) !important;
        margin-right: 0 !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] li {
        font-size: 20px !important;
        line-height: 1.7 !important;
    }

    [data-testid="stChatInput"] {
        width: 95% !important;
        max-width: 95% !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.html("""
<div class="app-header">

    <h1>🤖 Skin Cancer RAG Assistant</h1>

    <p>
        Evidence-grounded guidance for skin cancer prevention, risk factors, detection, and treatment.
    </p>

</div>
""")


# ============================================================
# SESSION MEMORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISCLAIMER
# ============================================================

def show_disclaimer():

    st.html("""
    <div class="disclaimer">

        ⚠️ <b>Medical Disclaimer:</b>

        This AI assistant provides general educational
        information and is not a substitute for professional
        medical advice, diagnosis, or treatment.

        Always consult a qualified healthcare professional
        for medical concerns.

    </div>
    """)


# ============================================================
# SOURCES
# ============================================================

def show_sources(citations):

    if not citations:
        return

    html = """
    <div class="sources-box">

        <div class="source-title">
            📚 Sources
        </div>
    """

    for index, citation in enumerate(citations, start=1):

        html += f"""
        <div class="source-card">

            <div class="source-title">
                {index}. {citation["document"]}
            </div>

            <div class="source-info">

                <b>Section:</b>
                {citation["section"]}

                &nbsp;&nbsp; | &nbsp;&nbsp;

                <b>Page:</b>
                {citation["page"]}

                <br>

                <b>Chunk:</b>

                <span class="source-id">
                    {citation["chunk_id"]}
                </span>

            </div>

        </div>
        """

    html += """
    </div>
    """

    st.html(html)


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    avatar = (
        "🤖"
        if message["role"] == "assistant"
        else "👤"
    )

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):

        if message["role"] == "user":
            st.markdown(
                '<span class="user-message-marker"></span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                message["content"]
            )
        else:
            st.markdown(
                '<span class="assistant-message-marker"></span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                message["content"]
            )


        # ====================================================
        # ASSISTANT DETAILS
        # ====================================================

        if (
            message["role"] == "assistant"
            and "details" in message
        ):

            details = message["details"]


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = details["confidence"]

            if confidence == "High":

                st.success(
                    "🟢 Confidence: High"
                )

            elif confidence == "Medium":

                st.warning(
                    "🟡 Confidence: Medium"
                )

            else:

                st.error(
                    "🔴 Confidence: Insufficient"
                )


            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            citations = details.get(
                "citations",
                []
            )

            show_sources(citations)


            # ------------------------------------------------
            # DISCLAIMER
            # ------------------------------------------------

            show_disclaimer()


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask a question about skin cancer..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if query:

    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    # Previous conversation only
    chat_history = st.session_state.messages.copy()

# Save current user message
    st.session_state.messages.append(
    {
        "role": "user",
        "content": query
    }
)


    # ========================================================
    # SHOW USER MESSAGE
    # ========================================================

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.markdown(
            '<span class="user-message-marker"></span>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div class=\"user-message-text\">{html.escape(query)}</div>",
            unsafe_allow_html=True,
        )


    # ========================================================
    # ASSISTANT
    # ========================================================

    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        st.markdown(
            '<span class="assistant-message-marker"></span>',
            unsafe_allow_html=True,
        )

        loading = st.empty()


        # ----------------------------------------------------
        # LOADING
        # ----------------------------------------------------

        loading.html("""
        <div class="loading-wrapper">

            <div class="robot">

                <div class="robot-antenna"></div>

            </div>

            <div class="loading-text">
                Analyzing the evidence...
            </div>

        </div>
        """)


        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        answer, results, confidence, citations = (
            generate_answer(
                query=query,
                chat_history=chat_history
            )
        )


        loading.empty()


        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        st.markdown(answer)


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        if confidence == "High":

            st.success(
                "🟢 Confidence: High"
            )

        elif confidence == "Medium":

            st.warning(
                "🟡 Confidence: Medium"
            )

        else:

            st.error(
                "🔴 Confidence: Insufficient"
            )


        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        show_sources(citations)


        # ----------------------------------------------------
        # DISCLAIMER
        # ----------------------------------------------------

        show_disclaimer()


    # ========================================================
    # SAVE ASSISTANT MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",

            "content": answer,

            "details": {

                "confidence": confidence,

                "citations": citations
            }
        }
    )