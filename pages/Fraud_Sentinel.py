import os
import json
import math
import collections
import time
import logging
import random
import asyncio
import aiohttp
import pandas as pd
import plotly.express as px
import streamlit as st
from web3 import Web3, AsyncWeb3
from web3.providers import AsyncHTTPProvider
from dotenv import load_dotenv
from nn_model import MLP, Value

# Standard load_dotenv() - robust for local dev and production
load_dotenv()

st.title("Web3 Mempool Sentinel")
st.write("Live Fraud Detection Engine Powered by Pure Python")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
KNOWN_SCAM_ADDRESSES = [
    "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef", # Example scam address 1
    "0x1234567890123456789012345678901234567890"  # Example scam address 2
]
HISTORY_BLOCK_WINDOW = 100 # Number of recent blocks to scan for historical transfers
ETHERSCAN_RATE_LIMIT = 5 # Requests per second
ALCHEMY_RATE_LIMIT = 15 # Requests per second for Alchemy free tier
# --- End Configuration ---

# --- PERMANENT CONNECTION LAYER ---
# This block handles Alchemy URL resolution from .env or System Environment
alchemy_url = os.getenv("ALCHEMY_URL")

# Fallback: Check if we are in a Streamlit Cloud environment or similar
if not alchemy_url:
    # Try to see if it's in streamlit secrets as a last resort
    if "ALCHEMY_URL" in st.secrets:
        alchemy_url = st.secrets["ALCHEMY_URL"]

if not alchemy_url:
    st.error("🚨 CRITICAL ERROR: ALCHEMY_URL not found.")
    st.info("Please ensure your ALCHEMY_URL is set in your .env file or environment variables.")
    st.stop()

# Initialize providers
etherscan_api_key = alchemy_url # Mapping as requested
w3_async = AsyncWeb3(AsyncHTTPProvider(alchemy_url))
w3 = Web3(Web3.HTTPProvider(alchemy_url))

# Final Connection Validation (Synchronous)
try:
    if not w3.is_connected():
        st.error("🚨 CONNECTION FAILED: Alchemy provider is not responding.")
        st.stop()
    st.success("✅ Connected to Ethereum Mainnet")
except Exception as e:
    st.error(f"🚨 PROVIDER ERROR: {str(e)}")
    st.stop()
# --- END CONNECTION LAYER ---

# Semaphores for rate limiting
etherscan_semaphore = asyncio.Semaphore(ETHERSCAN_RATE_LIMIT)
alchemy_semaphore = asyncio.Semaphore(ALCHEMY_RATE_LIMIT)

# Initialize session state variables
if 'live_data' not in st.session_state:
    st.session_state['live_data'] = []

# --- Rate Limit Handling Configuration ---
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 60 # seconds

async def make_async_alchemy_request(session, method, params):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    }
    for i in range(MAX_RETRIES):
        try:
            async with session.post(alchemy_url, json=payload) as response:
                if response.status == 429:
                    raise aiohttp.ClientResponseError(response.request_info, response.history, status=429)
                
                result = await response.json()
                if result and 'error' in result and 'code' in result['error'] and result['error']['code'] == 429:
                    raise aiohttp.ClientResponseError(response.request_info, response.history, status=429)
                
                if response.status == 400:
                    if method == "alchemy_getTokenBalances":
                        logging.warning(f"400 Client Error for {method}. Likely too many tokens. Returning None.")
                        return None
                    else:
                        logging.error(f"400 Client Error for {method}: {await response.text()}")
                        return None
                
                return result
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                delay = min(MAX_RETRY_DELAY, INITIAL_RETRY_DELAY * (2**i) + random.uniform(0, 1))
                logging.warning(f"Rate limit hit ({method}). Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
            else:
                logging.error(f"HTTP Error {e.status} for {method}: {e.message}")
                return None
        except Exception as e:
            logging.error(f"Unexpected error in {method}: {e}")
            return None
    return None

async def process_smart_contract(session, receiver_address):
    # This function ONLY handles Etherscan and the Tokenizer
    code = await w3_async.eth.get_code(receiver_address)
    if code == b'':
        return None, [], "N/A"

    etherscan_url = f"https://api.etherscan.io/v2/api?chainid=1&module=contract&action=getsourcecode&address={receiver_address}&apikey={etherscan_api_key or ''}"
    
    try:
        async with etherscan_semaphore:
            async with session.get(etherscan_url) as es_resp:
                es_data = await es_resp.json()
                if es_data['status'] == '1' and es_data['result'][0]['SourceCode']:
                    contract_source_code = es_data['result'][0]['SourceCode']
                    tokenizer_payload = {"source_code": contract_source_code}
                    
                    async with session.post("https://zeroxneural-tokenizer.onrender.com/api/v1/encode", json=tokenizer_payload, timeout=60) as tok_resp:
                        if tok_resp.status == 200:
                            token_data = await tok_resp.json()
                            compressed_tokens = token_data.get("tokens", [])
                            raw_len = len(contract_source_code)
                            token_len = len(compressed_tokens)
                            compression_ratio = f"{raw_len / token_len:.2f}X" if token_len > 0 else "0.0X"
                            return contract_source_code, compressed_tokens, compression_ratio
    except Exception as e:
        logging.warning(f"Ingestion failed for {receiver_address}: {e}")
    
    return None, [], "N/A"

async def process_single_transaction(session, tx, from_block_hex, alchemy_semaphore):
    sender_address = tx['from']
    receiver_address = tx.get('to')
    
    tasks = []
    
    # Task 0: Smart Contract Ingestion (Only if it has a receiver)
    if receiver_address and Web3.is_address(receiver_address):
        tasks.append(process_smart_contract(session, receiver_address))
    else:
        async def dummy_contract_return(): return None, [], "N/A"
        tasks.append(dummy_contract_return())

    # Task 1-4: Alchemy Historical Data (Protected by a new semaphore)
    async def fetch_alchemy_with_throttle(method, params):
        async with alchemy_semaphore:
            return await make_async_alchemy_request(session, method, params)

    tasks.extend([
        fetch_alchemy_with_throttle("alchemy_getAssetTransfers", {
            "fromBlock": from_block_hex, "toBlock": "latest", "fromAddress": sender_address, "category": ["external", "erc20"]
        }),
        fetch_alchemy_with_throttle("alchemy_getAssetTransfers", {
            "fromBlock": from_block_hex, "toBlock": "latest", "toAddress": sender_address, "category": ["external", "erc20"]
        }),
        fetch_alchemy_with_throttle("alchemy_getTokenBalances", {"owner": sender_address}),
        fetch_alchemy_with_throttle("eth_getBalance", [sender_address, "latest"])
    ])
    
    # Execute EVERYTHING simultaneously
    results = await asyncio.gather(*tasks)
    
    contract_source_code, compressed_tokens, compression_ratio = results[0]
    history_from = results[1]
    history_to = results[2]
    token_balances_resp = results[3]
    ether_balance_wei = results[4]
    
    historical_transfers = []
    if history_from and 'result' in history_from: historical_transfers.extend(history_from['result']['transfers'])
    if history_to and 'result' in history_to: historical_transfers.extend(history_to['result']['transfers'])
    
    token_balances = []
    if token_balances_resp and 'result' in token_balances_resp:
        token_balances = token_balances_resp['result']['tokenBalances']

    # Handle ether_balance_wei which could be a JSON-RPC dict result or a hex string
    if isinstance(ether_balance_wei, dict) and 'result' in ether_balance_wei:
        ether_balance_wei = ether_balance_wei['result']
    
    # Convert hex string to int if necessary
    if isinstance(ether_balance_wei, str) and ether_balance_wei.startswith('0x'):
        ether_balance_wei = int(ether_balance_wei, 16)
    elif ether_balance_wei is None:
        ether_balance_wei = 0

    ether_balance = float(Web3.from_wei(ether_balance_wei, 'ether'))

    # 3. Feature Calculation
    history = {'sent_to': collections.defaultdict(int), 'received_from': collections.defaultdict(int), 'received_values': []}
    for h_tx in historical_transfers:
        if h_tx['category'] == 'external':
            val = h_tx.get('value')
            v_eth = 0.0
            if isinstance(val, str): v_eth = float(Web3.from_wei(int(val, 16), 'ether'))
            elif isinstance(val, (int, float)): v_eth = float(val)
            
            if h_tx['from'] == sender_address: history['sent_to'][h_tx['to']] += 1
            elif h_tx['to'] == sender_address:
                history['received_from'][h_tx['from']] += 1
                history['received_values'].append(v_eth)

    total_received = sum(history['received_values'])
    unique_recv = len(history['received_from'])
    unique_sent = len(history['sent_to'])
    tx_freq = len(historical_transfers)
    min_val = min(history['received_values']) if history['received_values'] else 0.0
    max_val = max(history['received_values']) if history['received_values'] else 0.0

    features = [ether_balance, tx_freq, float(Web3.from_wei(tx['value'], 'ether')), total_received, unique_recv, unique_sent, min_val, max_val]
    
    return {
        "address": sender_address,
        "features": features,
        "receiver_address": receiver_address,
        "contract_source_code": contract_source_code,
        "compressed_tokens": compressed_tokens,
        "compression_ratio": compression_ratio
    }

async def get_live_features_async():
    latest_block = await w3_async.eth.get_block('latest', full_transactions=True)
    if not latest_block or not latest_block.transactions:
        st.session_state['live_data'] = []
        return

    from_block_hex = hex(max(0, latest_block.number - HISTORY_BLOCK_WINDOW))
    transactions = latest_block.transactions[:50]
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_single_transaction(session, tx, from_block_hex, alchemy_semaphore) for tx in transactions]
        results = await asyncio.gather(*tasks)
        st.session_state['live_data'] = [r for r in results if r is not None]

def get_live_features():
    asyncio.run(get_live_features_async())

# Sidebar for filtering
st.sidebar.header("Filter Results")
risk_level_filter = st.sidebar.selectbox(
    "Filter by Risk Level",
    ("All", "High-Risk 🚨🚨🚨", "Medium-Risk ⚠️⚠️", "Low-Risk 🟡", "Normal ✅")
)

st.sidebar.markdown("---")
st.sidebar.header("Auto-Refresh Settings")
refresh_interval = st.sidebar.slider(
    "Refresh Interval (seconds)",
    min_value=5,
    max_value=60,
    value=10,
    step=5
)
auto_refresh_enabled = st.sidebar.checkbox("Enable Auto-Refresh", value=False)


def normalize_features(raw_features):
    # These must match the exact min and max from your training CSV
    feature_mins = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    feature_maxs = [1000.0, 10000.0, 1000.0, 50000.0, 100.0, 100.0, 50.0, 5000.0]
    
    normalized = []
    for i in range(8):
        val = (raw_features[i] - feature_mins[i]) / (feature_maxs[i] - feature_mins[i])
        # Prevent division by zero and cap values
        val = max(0.0, min(1.0, float(val)))
        normalized.append(Value(val))
        
    return normalized

# 1. Initialize the exact architecture
model = MLP(8, [16, 8, 1])

# 2. Load the trained weights
try:
    with open('fraud_model_weights.json', 'r') as f:
        trained_weights = json.load(f)
except FileNotFoundError:
    st.error("Error: 'fraud_model_weights.json' not found. Please ensure the model weights file is in the correct directory.")
    logging.error("Error: 'fraud_model_weights.json' not found. Please ensure the model weights file is in the correct directory.")
    st.stop() # Stop the Streamlit app
except json.JSONDecodeError:
    st.error("Error: 'fraud_model_weights.json' is corrupted or not a valid JSON file.")
    logging.error("Error: 'fraud_model_weights.json' is corrupted or not a valid JSON file.")
    st.stop() # Stop the Streamlit app
except Exception as e:
    st.error(f"An unexpected error occurred while loading model weights: {e}")
    logging.error(f"An unexpected error occurred while loading model weights: {e}")
    st.stop()

# 3. Inject the weights into the model
try:
    model_parameters = model.parameters()
    if len(trained_weights) != len(model_parameters):
        st.error(f"Error: Mismatch between loaded weights ({len(trained_weights)}) and model parameters ({len(model_parameters)}). Model architecture might have changed.")
        logging.error(f"Error: Mismatch between loaded weights ({len(trained_weights)}) and model parameters ({len(model_parameters)}). Model architecture might have changed.")
        st.stop()
    for param, trained_weight in zip(model_parameters, trained_weights):
        param.data = trained_weight
except Exception as e:
    st.error(f"An error occurred while injecting model weights: {e}")
    logging.error(f"An error occurred while injecting model weights: {e}")
    st.stop()

def predict_fraud(normalized_features):
    # Pass the array through your pure-Python forward pass
    prediction = model(normalized_features)
    
    # Multi-level risk classification based on prediction score
    if prediction.data > 1.5:
        return "High-Risk 🚨🚨🚨"
    elif prediction.data > 0.5:
        return "Medium-Risk ⚠️⚠️"
    elif prediction.data > 0:
        return "Low-Risk 🟡"
    else:
        return "Normal ✅"

def highlight_risk(row):
    if row["Classification"] == "High-Risk 🚨🚨🚨":
        return ['background-color: #FFEBEE'] * len(row) # Very light red
    elif row["Classification"] == "Medium-Risk ⚠️⚠️":
        return ['background-color: #FFF3E0'] * len(row) # Very light orange
    elif row["Classification"] == "Low-Risk 🟡":
        return ['background-color: #FFFDE7'] * len(row) # Very light yellow
    elif row["Classification"] == "Normal ✅":
        return ['background-color: #E8F5E9'] * len(row) # Very light green
    return [''] * len(row)

def display_results(live_transactions):
    results = []
    for tx in live_transactions:
        scaled_input = normalize_features(tx['features'])
        prediction = model(scaled_input) # Get the raw prediction
        classification = predict_fraud(scaled_input)
        
        results.append({
            "Wallet Address": tx['address'],
            "Classification": classification,
            "Contract Interacted": tx['receiver_address'] if tx['contract_source_code'] else "None",
            "Tokens": len(tx['compressed_tokens']) if tx['compressed_tokens'] else 0,
            "Compression": tx['compression_ratio'],
            "Ether Balance": tx['features'][0],
            "Tx Frequency": tx['features'][1],
            "Live Ether Sent": tx['features'][2],
            "Total Received": tx['features'][3],
            "Unique Received From": tx['features'][4],
            "Unique Sent To": tx['features'][5],
            "Min Value Received": tx['features'][6],
            "Max Value Received": tx['features'][7]
        })
        
    # Calculate summary statistics
    risk_counts = collections.defaultdict(int)
    for res in results:
        risk_counts[res["Classification"]] += 1

    st.subheader("Risk Level Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("High-Risk", risk_counts["High-Risk 🚨🚨🚨"])
    with col2:
        st.metric("Medium-Risk", risk_counts["Medium-Risk ⚠️⚠️"])
    with col3:
        st.metric("Low-Risk", risk_counts["Low-Risk 🟡"])
    with col4:
        st.metric("Normal", risk_counts["Normal ✅"])

    # Create a DataFrame for the chart
    risk_df = pd.DataFrame(risk_counts.items(), columns=["Risk Level", "Count"])
    risk_df["Risk Level"] = pd.Categorical(risk_df["Risk Level"], ["High-Risk 🚨🚨🚨", "Medium-Risk ⚠️⚠️", "Low-Risk 🟡", "Normal ✅"])
    risk_df = risk_df.sort_values("Risk Level")

    st.subheader("Risk Distribution")
    fig = px.bar(risk_df, x="Risk Level", y="Count", color="Risk Level",
                 color_discrete_map={
                     "High-Risk 🚨🚨🚨": "red",
                     "Medium-Risk ⚠️⚠️": "orange",
                     "Low-Risk 🟡": "yellow",
                     "Normal ✅": "green"
                 },
                 title="Distribution of Wallet Risk Levels")
    fig.update_layout(autosize=True) # Make the plot responsive
    st.plotly_chart(fig)
        
    results_df = pd.DataFrame(results)

    # Apply filter based on sidebar selection
    if risk_level_filter != "All":
        filtered_results = [
            res for res in results if res["Classification"] == risk_level_filter
        ]
        if filtered_results:
            filtered_results_df = pd.DataFrame(filtered_results)
            st.dataframe(filtered_results_df.style.apply(highlight_risk, axis=1), width='stretch')
        else:
            st.info(f"No transactions found with risk level: {risk_level_filter}")
    else:
        st.dataframe(results_df.style.apply(highlight_risk, axis=1), width='stretch')

if auto_refresh_enabled:
    st.write(f"Auto-refresh enabled. Updating every {refresh_interval} seconds.")
    placeholder = st.empty()
    while auto_refresh_enabled:
        with placeholder.container():
            with st.spinner("Fetching live transactions..."):
                get_live_features() # This now populates st.session_state['live_data']
                display_results(st.session_state.get('live_data', []))
        time.sleep(refresh_interval)
else:
    if st.button("Scan Live Mempool"):
        with st.spinner("Fetching live transactions..."):
            get_live_features() # This now populates st.session_state['live_data']
            display_results(st.session_state.get('live_data', []))

def verify_model_loading(model_instance):
    """
    Verifies that the model has loaded parameters and they are Value objects.
    """
    if not hasattr(model_instance, 'parameters') or not callable(model_instance.parameters):
        st.error("Model instance does not have a callable 'parameters' method.")
        logging.error("Model instance does not have a callable 'parameters' method.")
        return False
    
    params = model_instance.parameters()
    if not params:
        st.error("Model loaded but has no parameters.")
        logging.error("Model loaded but has no parameters.")
        return False
    
    for p in params:
        if not isinstance(p, Value):
            st.error(f"Model parameter is not a Value object: {type(p)}")
            logging.error(f"Model parameter is not a Value object: {type(p)}")
            return False
    
    st.success("Model parameters loaded and verified as Value objects.")
    logging.info("Model parameters loaded and verified as Value objects.")
    return True

# Call verification function after model loading
verify_model_loading(model)
