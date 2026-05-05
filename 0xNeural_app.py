import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables once at the entry point
load_dotenv()

st.set_page_config(
    page_title="0xNeural",
    page_icon="🧠",
    layout="wide"
)

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables once at the entry point
load_dotenv()

st.set_page_config(
    page_title="0xNeural | Engineering Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Design System (CSS) ---
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #FFFFFF;
    }
    
    .stMarkdown {
        color: #C1C2C5;
    }
    
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 3rem;
        border-radius: 1rem;
        border: 1px solid #334155;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .hero-title {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #00FF41, #00A3FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Project Cards */
    .project-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid #334155;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .project-card:hover {
        transform: translateY(-5px);
        border-color: #00FF41;
    }
    
    .project-tag {
        background: #00FF41;
        color: #000000;
        padding: 0.2rem 0.6rem;
        border-radius: 0.4rem;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 1rem;
        display: inline-block;
    }
    
    /* Sidebar Fixes */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🧠 0xNeural</h1>
        <p style="font-size: 1.2rem; color: #94A3B8;">Advanced Web3 Engineering Suite & Neural Architectures</p>
        <p style="color: #64748B;">Pushing the boundaries of blockchain analysis and generative AI through custom-built engines.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Architectural Roadmap ---
st.markdown("### 🏗️ Our Architectural Roadmap")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="project-card">
            <span class="project-tag">Core Engine</span>
            <h3>Fraud Sentinel</h3>
            <p>A real-time Mempool monitoring engine powered by a <b>pure-Python Autograd MLP</b> built from absolute scratch.</p>
            <hr style="border-color: #334155;">
            <p style="font-size: 0.9rem; color: #94A3B8;">Calculus-driven fraud detection that flags malicious transactions before they finalize.</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="project-card">
            <span class="project-tag">Ingestion Layer</span>
            <h3>BPE Tokenizer</h3>
            <p>Custom <b>Byte-Pair Encoding</b> algorithm trained on 10M+ characters of Solidity source code.</p>
            <hr style="border-color: #334155;">
            <p style="font-size: 0.9rem; color: #94A3B8;">Achieves >2X efficiency in context preservation compared to standard LLM tokenizers.</p>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="project-card">
            <span class="project-tag">The Brain</span>
            <h3>NanoCopilot</h3>
            <p>Generative <b>Transformer</b> with 5.96M parameters implementing Multi-Head Self-Attention.</p>
            <hr style="border-color: #334155;">
            <p style="font-size: 0.9rem; color: #94A3B8;">An LLM trained to hallucinate and reason through complex Web3 smart contract logic.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.info("👈 Use the sidebar to explore each project in depth.")
