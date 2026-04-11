import streamlit as st

st.set_page_config(
    page_title="0xNeural",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Welcome to 0xNeural: The Web3 Engineering Suite")

st.markdown("""
    Welcome to 0xNeural, your comprehensive suite for advanced Web3 engineering. 
    This application showcases a collection of cutting-edge projects, each built 
    from scratch to push the boundaries of blockchain analysis, AI, and data processing.
    
    Navigate through the sidebar to explore the different projects.
""")

st.markdown("---")

st.header("Our Architectural Roadmap")

st.subheader("1. The Pure-Python Autograd Engine & MLP (The Fraud Sentinel)")
st.markdown("""
    *   **The Architecture:** A Multilayer Perceptron neural network and its underlying calculus engine (backpropagation/autograd) built from absolute scratch without PyTorch or TensorFlow.
    *   **The Training:** Trained on a dataset of numerical features to classify behavior.
    *   **The Web3 Application:** A real-time Mempool Sentinel. It reads 8 numerical features from pending Ethereum transactions, applies Min-Min scaling, and assigns a live "Risk Score" to predict and flag fraudulent wallets before a transaction is finalized on-chain.
""")

st.subheader("2. The Domain-Specific BPE Tokenizer (The Ingestion Layer)")
st.markdown("""
    *   **The Architecture:** A custom Byte-Pair Encoding (BPE) text-compression algorithm, identical to the foundational layer used by GPT-4 and LLaMA, built in pure Python.
    *   **The Training:** Connected to the Kaggle API to ingest and train on nearly 10 million characters of raw, verified Ethereum Smart Contracts and DeFi audit reports.
    *   **The Web3 Application:** The highly-optimized ingestion funnel for an AI Smart Contract Auditor. By pushing the vocabulary hyperparameter to 300 custom merges, it achieved a **2.91X compression ratio** on complex DeFi staking contracts, vastly outperforming standard models at reading Web3 syntax.
""")

st.subheader("3. The Transformer Architecture (The \"Brain\") — *Up Next*")
st.markdown("""
    *   **The Architecture:** The Self-Attention Mechanism. This is the exact mathematical breakthrough from the famous *Attention Is All You Need* paper.
    *   **The Goal:** Building the engine that actually reads the compressed integers from Project 2. While the Tokenizer shrinks the data, the Transformer allows the AI to look at a variable on line 150 and understand how it connects to a `require` statement on line 12.
    *   **The Web3 Application:** The core reasoning engine for the AI Smart Contract Auditor.
""")

st.markdown("---")
st.info("Select a project from the sidebar to get started!")
