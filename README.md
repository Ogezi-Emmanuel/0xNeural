# 🧠 0xNeural: The Web3 Engineering Suite

Welcome to **0xNeural**, a comprehensive suite for advanced Web3 engineering. This project showcases a collection of cutting-edge tools and applications, each built from scratch to push the boundaries of blockchain analysis, AI, and data processing.

## 🎨 Design System: Dark Cyber-Industrial
The 0xNeural suite features a cohesive, high-performance UI designed for professional engineers:
- **Glassmorphism:** Frosted-glass components with blurred backdrops for a modern, deep aesthetic.
- **Cyber-Industrial Palette:** A dark #0E1117 background contrasted with high-visibility accents (Emerald Green for safety, Electric Blue for logic, and Crimson for risk).
- **Typography:** Powered by **JetBrains Mono** for technical data and **Inter** for structural readability.
- **Real-time Feedback:** Integrated animations, terminal-style outputs, and dynamic status indicators.

## 🏗️ Architectural Roadmap

### 1. The Pure-Python Autograd Engine & MLP (The Fraud Sentinel)
*   **The Architecture:** A Multilayer Perceptron (MLP) neural network and its underlying calculus engine (backpropagation/autograd) built from absolute scratch without PyTorch or TensorFlow.
*   **The Training:** Trained on a dataset of numerical features to classify wallet behavior.
*   **The Web3 Application:** A real-time **Mempool Sentinel**. It reads 8 numerical features from pending Ethereum transactions, applies Min-Max scaling, and assigns a live "Risk Score" to predict and flag fraudulent wallets before a transaction is finalized on-chain.
*   **High-Performance Execution Layer:** 
    *   **Asynchronous Architecture:** Rebuilt with `asyncio` and `aiohttp` for non-blocking network I/O.
    *   **Total Parallelism:** Concurrent execution of semantic data ingestion (Smart Contracts) and behavioral data fetching (Alchemy RPC).
    *   **Rate-Limit Resiliency (Safe Mode):** Integrated async Semaphores and throttled execution (25 transactions/block) to strictly respect Etherscan and Alchemy free-tier limits.
    *   **Responsive Auto-Refresh:** Implements a non-blocking `st.rerun` pattern for real-time monitoring without UI freezes.

### 2. The Domain-Specific BPE Tokenizer (The Ingestion Layer API)
*   **The Architecture:** A custom Byte-Pair Encoding (BPE) text-compression algorithm, identical to the foundational layer used by GPT-4 and LLaMA, built in pure Python.
*   **Deployment:** Deployed as a high-performance **FastAPI service** on Render: [zeroxneural-tokenizer.onrender.com](https://zeroxneural-tokenizer.onrender.com/).
*   **The Web3 Application:** Serves as the primary ingestion funnel for the Fraud Sentinel. By pushing the vocabulary hyperparameter to **2000 custom merges**, it achieves over **2X the efficiency** of standard models, providing superior context preservation for Web3 syntax.

### 3. The Transformer Architecture (The "Brain")
*   **The Architecture:** The Self-Attention Mechanism. This is the exact mathematical breakthrough from the famous *Attention Is All You Need* paper.
*   **The Execution:** A custom-built Generative Large Language Model (LLM) with 5.96M parameters, implementing Masked Multi-Head Self-Attention.
*   **The Web3 Application:** **NanoCopilot**. By analyzing every variable and relationship in a smart contract simultaneously, this "Brain" can hallucinate syntactically correct Solidity code. It features a **RAM-cached weight injection system** for near-instant inference and a **terminal-inspired UI** for observing neural output.

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

4.  **Model Weights:**
    Ensure `fraud_model_weights.json` and `nano_copilot_5.96M_weights.pth` are in the root directory.

### Running the Application

Start the Streamlit dashboard:
```bash
streamlit run 0xNeural_app.py
```

## 📂 Project Structure

*   `0xNeural_app.py`: Main entry point & global design system host.
*   `pages/`: 
    *   `Fraud_Sentinel.py`: High-performance, asynchronous Mempool Sentinel.
    *   `NanoCopilot.py`: Generative Transformer "Brain" with terminal UI.
*   `nn_model.py`: Pure-Python Autograd Engine and MLP implementation.
*   `web3_tokenizer_2000_merges.json`: BPE merge rules for Web3 syntax.
*   `fraud_model_weights.json`: Pre-trained weights for the Fraud Sentinel.
*   `nano_copilot_5.96M_weights.pth`: Pre-trained weights for NanoCopilot.
*   `requirements.txt`: Project dependencies.
*   `.env`: Local secrets (Alchemy URL).

## 🛠️ Technical Highlights

- **AsyncWeb3:** Fully non-blocking Ethereum provider integration.
- **Concurrent `asyncio.gather`:** Processes concurrent RPC calls and semantic ingestion simultaneously.
- **RAM-Based Model Caching:** Uses `@st.cache_resource` to keep heavy neural weights in server-side memory.
- **Traffic Throttling:** Multi-stage semaphores (Etherscan: 5 req/s, Alchemy: 5 req/s) for maximum stability.
- **Domain-Specific Tokenizer:** Seamless integration with the [0xNeural Tokenizer API](https://zeroxneural-tokenizer.onrender.com/) for real-time contract analysis.
