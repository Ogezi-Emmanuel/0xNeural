# 🧠 0xNeural: The Web3 Engineering Suite

Welcome to **0xNeural**, a comprehensive suite for advanced Web3 engineering. This project showcases a collection of cutting-edge tools and applications, each built from scratch to push the boundaries of blockchain analysis, AI, and data processing.

## Architectural Roadmap

### 1. The Pure-Python Autograd Engine & MLP (The Fraud Sentinel)
*   **The Architecture:** A Multilayer Perceptron neural network and its underlying calculus engine (backpropagation/autograd) built from absolute scratch without PyTorch or TensorFlow.
*   **The Training:** Trained on a dataset of numerical features to classify behavior.
*   **The Web3 Application:** A real-time Mempool Sentinel. It reads 8 numerical features from pending Ethereum transactions, applies Min-Max scaling, and assigns a live "Risk Score" to predict and flag fraudulent wallets before a transaction is finalized on-chain.

### 2. The Domain-Specific BPE Tokenizer (The Ingestion Layer)
*   **The Architecture:** A custom Byte-Pair Encoding (BPE) text-compression algorithm, identical to the foundational layer used by GPT-4 and LLaMA, built in pure Python.
*   **The Training:** Connected to the Kaggle API to ingest and train on nearly 10 million characters of raw, verified Ethereum Smart Contracts and DeFi audit reports.
*   **The Web3 Application:** The highly-optimized ingestion funnel for an AI Smart Contract Auditor. By pushing the vocabulary hyperparameter to 300 custom merges, it achieved a **2.91X compression ratio** on complex DeFi staking contracts, vastly outperforming standard models at reading Web3 syntax.

### 3. The Transformer Architecture (The "Brain") — *Up Next*
*   **The Architecture:** The Self-Attention Mechanism. This is the exact mathematical breakthrough from the famous *Attention Is All You Need* paper.
*   **The Goal:** Building the engine that actually reads the compressed integers from Project 2. While the Tokenizer shrinks the data, the Transformer allows the AI to look at a variable on line 150 and understand how it connects to a `require` statement on line 12.
*   **The Web3 Application:** The core reasoning engine for the AI Smart Contract Auditor.

## Getting Started

Follow these instructions to set up and run the 0xNeural application locally.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd 0xNeural
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    source venv/bin/activate # On macOS/Linux
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Alchemy API Key:**
    Create a `.env` file in the root directory of the project (`0xNeural/`) and add your Alchemy API URL:
    ```
    ALCHEMY_URL="https://eth-mainnet.g.alchemy.com/v2/YOUR_ALCHEMY_API_KEY"
    ```
    Replace `YOUR_ALCHEMY_API_KEY` with your actual Alchemy API key.

5.  **Ensure Model Weights are Present:**
    Make sure you have the `fraud_model_weights.json` file in the root directory. This file contains the pre-trained weights for the Fraud Sentinel's neural network.

### Running the Application

To start the Streamlit application, navigate to the root directory of the project in your terminal and run:

```bash
streamlit run 0xNeural_app.py
```

This will open the 0xNeural home page in your web browser. You can then navigate to the "Fraud Sentinel" page (or other future projects) using the sidebar.

## Project Structure

*   `0xNeural_app.py`: The main Streamlit application entry point and home page.
*   `pages/`: Directory containing individual Streamlit application pages.
    *   `Fraud_Sentinel.py`: The Web3 Mempool Sentinel application.
*   `nn_model.py`: Contains the pure-Python implementation of the Autograd Engine and MLP.
*   `fraud_model_weights.json`: Pre-trained weights for the Fraud Sentinel model.
*   `requirements.txt`: Lists all Python dependencies.
*   `.env`: (Not committed to Git) Stores sensitive environment variables like API keys.
