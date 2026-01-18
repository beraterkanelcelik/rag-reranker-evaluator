from __future__ import annotations

import streamlit as st

from frontend.api.client import APIClient
from frontend.components.dataset_section import render_dataset_section
from frontend.components.embedding_section import render_embedding_section
from frontend.components.evaluation_section import render_evaluation_section
from frontend.components.header import render_header
from frontend.components.results_section import render_results_section
from frontend.components.system_section import render_system_section


st.set_page_config(page_title="RAG + Reranker Evaluation", layout="wide", menu_items={})

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:wght@400;600;700&family=Space+Grotesk:wght@400;500;600&display=swap');

:root {
  --ink: #1f1c18;
  --ink-muted: #4d5b5f;
  --accent: #1b6f62;
  --accent-strong: #135348;
  --surface: #ffffff;
  --surface-alt: #f4f1ea;
  --border: rgba(31, 28, 24, 0.12);
  --shadow: 0 18px 40px rgba(20, 36, 34, 0.1);
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: "Space Grotesk", sans-serif;
  color: var(--ink);
  background: radial-gradient(circle at 20% 12%, #f5efe3 0%, #f9f7f2 45%, #e6efec 100%);
  background-attachment: fixed;
}

h1, h2, h3, h4, h5 {
  font-family: "Newsreader", serif;
  color: var(--ink);
}

p, span, label, div[data-testid="stMarkdownContainer"], div[data-testid="stCaption"], div[data-baseweb="input"] > label, div[data-testid="stSelectbox"] > label, div[data-testid="stNumberInput"] > label, div[data-baseweb="input"] > div > div, div[data-baseweb="select"] > div > div, div[data-testid="stTextInput"] > div > div, div[data-testid="stTextInput"] *, div[data-testid="stNumberInput"] *, div[data-testid="stSelectbox"] * {
  color: var(--ink) !important;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.block-container {
  padding-top: 2rem;
  padding-bottom: 4rem;
  animation: fadeInUp 0.6s ease;
}

section[data-testid="stSidebar"] {
  background: rgba(255, 255, 255, 0.85);
}

  to {
    opacity: 1;
    transform: translateY(0);
  }
}

* [data-testid="stHeader"] {
  display: none !important;
  height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  position: absolute !important;
  top: -100px !important;
  visibility: hidden !important;
}

* [data-testid="stToolbar"] {
  display: none !important;
  height: 0 !important;
  visibility: hidden !important;
}

.stApp > header {
  visibility: hidden;
}

[data-testid="stStatusWidget"] {
  display: none !important;
}

[data-testid="stFooter"] {
  display: none !important;
}

[data-testid="stAppViewContainer"] > .main {
  background: transparent;
}

[data-testid="stExpander"] {
  background: rgba(255, 255, 255, 0.95);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: var(--shadow);
  padding: 0.2rem 0.55rem 0.5rem 0.55rem;
}

[data-testid="stExpander"] > details > summary {
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--ink);
}

[data-testid="metric-container"] {
  background: var(--surface);
  border-radius: 12px;
  border: 1px solid var(--border);
  padding: 0.6rem 0.8rem;
  box-shadow: 0 10px 18px rgba(17, 35, 33, 0.06);
}

.stButton > button {
  background: linear-gradient(135deg, var(--accent), #2f8f81) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 0.45rem 1rem !important;
  font-weight: 600 !important;
}

.stButton > button:hover {
  background: linear-gradient(135deg, var(--accent-strong), #267066) !important;
}

.stButton > button:focus {
  box-shadow: 0 0 0 3px rgba(27, 111, 98, 0.2);
}

[data-baseweb="input"] > div {
  border-radius: 10px;
}

[data-testid="stCaption"] {
  color: var(--ink-muted);
}

[data-testid="stToolbar"] {
  display: none !important;
}


div[data-testid="stExpander"] > details > summary {
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--ink);
}

div[data-testid="metric-container"] {
  background: var(--surface);
  border-radius: 12px;
  border: 1px solid var(--border);
  padding: 0.6rem 0.8rem;
  box-shadow: 0 10px 18px rgba(17, 35, 33, 0.06);
}

.stButton > button {
  background: linear-gradient(135deg, var(--accent), #1a8a80);
  color: #ffffff;
  border: none;
  border-radius: 10px;
  padding: 0.45rem 1rem;
  font-weight: 600;
}

.stButton > button:hover {
  background: linear-gradient(135deg, var(--accent-strong), #16736b);
}

.stButton > button:focus {
  box-shadow: 0 0 0 3px rgba(15, 111, 106, 0.2);
}

div[data-baseweb="input"] > div {
  border-radius: 10px;
}

div[data-testid="stCaption"] {
  color: var(--ink-muted);
}

div[data-testid="stMetric"] * {
  color: var(--ink) !important;
}

input, select, textarea {
  color: var(--ink) !important;
  background: white !important;
}

div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, div[data-testid="stTextInput"] > div, div[data-testid="stNumberInput"] > div, div[data-testid="stSelectbox"] > div {
  background: white !important;
}

div[data-baseweb="input"] > div > div, div[data-baseweb="select"] > div > div, div[data-testid="stTextInput"] > div > div, div[data-testid="stNumberInput"] > div > div, div[data-testid="stSelectbox"] > div > div {
  color: var(--ink) !important;
  background: white !important;
}

div[data-testid="stNumberInput"] button {
  background: var(--accent) !important;
  color: white !important;
}

div[data-testid="stCheckbox"] input[type="checkbox"] {
  background: white !important;
}

div[data-testid="stCheckbox"] label {
  color: var(--ink) !important;
}

/* Code blocks light background */
[data-testid="stCodeBlock"] {
  background: #f9f9f9 !important;
  border: 1px solid var(--border) !important;
}

[data-testid="stCodeBlock"] pre {
  color: #1f1c18 !important;
}

/* Markdown fenced code blocks */
[data-testid="stMarkdownContainer"] pre,
[data-testid="stMarkdownContainer"] pre code {
  background: #f9f9f9 !important;
  color: #1f1c18 !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px;
  padding: 0.75rem 0.9rem;
}

[data-testid="stMarkdownContainer"] code:not(pre code) {
  background: #f9f9f9 !important;
  color: #1f1c18 !important;
}

/* Text areas for JSON display */
[data-testid="stTextArea"] textarea,
[data-testid="stTextArea"] textarea:disabled {
  color: #1f1c18 !important;
  background: #f9f9f9 !important;
}

/* Streamlit JSON tree */
[data-testid="stJson"],
[data-testid="stJson"] > div,
[data-testid="stJson"] > div > div {
  background: #f9f9f9 !important;
  color: #1f1c18 !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px;
  padding: 0.75rem 0.9rem;
  box-shadow: 0 8px 18px rgba(17, 35, 33, 0.05);
}

[data-testid="stJson"] * {
  background: transparent !important;
  color: #1f1c18 !important;
}

[data-testid="stJson"] pre,
[data-testid="stJson"] code {
  background: transparent !important;
  color: #1f1c18 !important;
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    const header = document.querySelector('[data-testid="stHeader"]');
    if (header) header.style.display = 'none';
    const toolbar = document.querySelector('[data-testid="stToolbar"]');
    if (toolbar) toolbar.style.display = 'none';
});
</script>
""", unsafe_allow_html=True)

if "health_status" in st.session_state:
    st.session_state.pop("health_status")
if "memory_status" in st.session_state:
    st.session_state.pop("memory_status")

client = APIClient()

render_header(client)

render_dataset_section(client)
render_embedding_section(client)
render_evaluation_section(client)
render_results_section(client)
render_system_section(client)
