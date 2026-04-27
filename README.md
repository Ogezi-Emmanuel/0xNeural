# 🧠 0xNeural: The Web3 Engineering Suite

Welcome to **0xNeural**, a comprehensive suite for advanced Web3 engineering. This project showcases a collection of cutting-edge tools and applications, each built from scratch to push the boundaries of blockchain analysis, AI, and data processing.

## 🏗️ Architectural Roadmap

### 1. The Pure-Python Autograd Engine & MLP (The Fraud Sentinel)
*   **The Architecture:** A Multilayer Perceptron (MLP) neural network and its underlying calculus engine (backpropagation/autograd) built from absolute scratch without PyTorch or TensorFlow.
*   **The Training:** Trained on a dataset of numerical features to classify wallet behavior.
*   **The Web3 Application:** A real-time **Mempool Sentinel**. It reads 8 numerical features from pending Ethereum transactions, applies Min-Max scaling, and assigns a live "Risk Score" to predict and flag fraudulent wallets before a transaction is finalized on-chain.
*   **High-Performance Execution Layer:** 
    *   **Asynchronous Architecture:** Rebuilt with `asyncio` and `aiohttp` for non-blocking network I/O.
    *   **Total Parallelism:** Concurrent execution of semantic data ingestion (Smart Contracts) and behavioral data fetching (Alchemy RPC).
    *   **Rate-Limit Resiliency (Safe Mode):** Integrated async Semaphores and throttled execution (25 transactions/block) to strictly respect Etherscan and Alchemy free-tier limits, preventing silent throttling or server disconnections.
    *   **Responsive Auto-Refresh:** Implements a non-blocking `st.rerun` pattern for real-time monitoring without UI freezes, with configurable intervals up to 5 minutes.

### 2. The Domain-Specific BPE Tokenizer (The Ingestion Layer API)
*   **The Architecture:** A custom Byte-Pair Encoding (BPE) text-compression algorithm, identical to the foundational layer used by GPT-4 and LLaMA, built in pure Python.
*   **Source Code:** [Ogezi-Emmanuel/0xneural-Tokenizer](https://github.com/Ogezi-Emmanuel/0xneural-Tokenizer)
*   **Deployment:** Deployed as a high-performance **FastAPI service** on Render: [zeroxneural-tokenizer.onrender.com](https://zeroxneural-tokenizer.onrender.com/).
*   **The Training:** Connected to the Kaggle API to ingest and train on nearly 10 million characters of raw, verified Ethereum Smart Contracts and DeFi audit reports.
*   **The Web3 Application:** Serves as the primary ingestion funnel for the Fraud Sentinel. By pushing the vocabulary hyperparameter to **2000 custom merges**, it achieves over **2X the efficiency** of standard models, providing superior context preservation for Web3 syntax.

### 3. The Transformer Architecture (The "Brain") — *Up Next*
*   **The Architecture:** The Self-Attention Mechanism. This is the exact mathematical breakthrough from the famous *Attention Is All You Need* paper.
*   **The Goal:** Building the engine that actually reads the compressed integers from Project 2. While the Tokenizer shrinks the data, the Transformer allows the AI to look at a variable on line 150 and understand how it connects to a `require` statement on line 12.
*   **The Web3 Application:** The core reasoning engine for the AI Smart Contract Auditor.

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   `pip` (Python package installer)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd 0xNeural
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your Environment Variables:**
    Create a `.env` file in the root directory. **Note:** Do not commit this file to version control.
    ```
    ALCHEMY_URL="https://eth-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_API_KEY"
    ```
    *The suite is designed to map the `ALCHEMY_URL` for both RPC calls and contract source code ingestion.*

4.  **Model Weights:**
    Ensure `fraud_model_weights.json` is in the root directory for the Fraud Sentinel to function.

### Running the Application

Start the Streamlit dashboard:
```bash
streamlit run 0xNeural_app.py
```

## 📂 Project Structure

*   `0xNeural_app.py`: Main entry point. Handles global environment loading and home page rendering.
*   `pages/`: 
    *   `Fraud_Sentinel.py`: The high-performance, asynchronous Mempool Sentinel.
*   `nn_model.py`: Pure-Python Autograd Engine and MLP implementation.
*   `fraud_model_weights.json`: Pre-trained neural network weights.
*   `requirements.txt`: Project dependencies.
*   `.env`: Local secrets (Alchemy URL).

## 🛠️ Technical Highlights (Fraud Sentinel)

- **AsyncWeb3:** Fully non-blocking Ethereum provider integration.
- **Concurrent `asyncio.gather`:** Processes concurrent RPC calls and semantic ingestion simultaneously.
- **Traffic Throttling:** Multi-stage semaphores (Etherscan: 5 req/s, Alchemy: 5 req/s) for maximum stability on free-tier providers.
- **Network Resilience:** Implements 10-second request timeouts and robust retry logic to handle server disconnections and transient network errors.
- **Robust Connection Layer:** Permanent handshake logic with fallbacks for Environment Variables, `.env` files, and Streamlit Secrets.
- **API-Driven Ingestion:** Seamless integration with the [0xNeural Tokenizer API](https://zeroxneural-tokenizer.onrender.com/) ([GitHub](https://github.com/Ogezi-Emmanuel/0xneural-Tokenizer)) for real-time contract analysis.
- **Non-Blocking Auto-Refresh:** Linear execution architecture using `st.rerun` for smooth UI performance.
