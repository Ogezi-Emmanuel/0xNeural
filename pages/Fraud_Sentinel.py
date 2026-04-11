import os
from dotenv import load_dotenv
load_dotenv()

import math
import json
import streamlit as st
from web3 import Web3
from nn_model import MLP, Value
import collections
import time
import pandas as pd
import plotly.express as px
import requests
import random
import logging

st.title("Web3 Mempool Sentinel")
st.write("Live Fraud Detection Engine Powered by Pure Python")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
KNOWN_SCAM_ADDRESSES = [
    "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef", # Example scam address 1
    "0x1234567890123456789012345678901234567890"  # Example scam address 2
    # In a real application, this list would be loaded from a database or external API
]
HISTORY_BLOCK_WINDOW = 100 # Number of recent blocks to scan for historical transfers
# --- End Configuration ---

# Web3 Connection
alchemy_url = os.getenv("ALCHEMY_URL")
w3 = Web3(Web3.HTTPProvider(alchemy_url))

# Initialize session state variables
if 'live_data' not in st.session_state:
    st.session_state['live_data'] = []

# --- Rate Limit Handling Configuration ---
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 60 # seconds

def make_alchemy_request_with_retries(method, params):
    for i in range(MAX_RETRIES):
        try:
            response = w3.provider.make_request(method, params)
            # Check for Alchemy-specific rate limit error in the response body
            if response and 'error' in response and 'code' in response['error'] and response['error']['code'] == 429:
                raise requests.exceptions.RequestException("Alchemy rate limit exceeded (code 429)")
            return response
        except requests.exceptions.RequestException as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                delay = min(MAX_RETRY_DELAY, INITIAL_RETRY_DELAY * (2**i) + random.uniform(0, 1))
                st.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds... (Attempt {i+1}/{MAX_RETRIES})")
                logging.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds... (Attempt {i+1}/{MAX_RETRIES})")
                time.sleep(delay)
            elif "400 Client Error" in str(e):
                if method == "alchemy_getTokenBalances":
                    logging.warning(f"400 Client Error: Bad Request for method {method} with params {params}. This usually means too many token balances for the address. Returning None.")
                    return None # Do not re-raise, allow outer function to handle gracefully
                else:
                    logging.error(f"400 Client Error: Bad Request for method {method} with params {params}. Error: {e}")
                    raise # Re-raise the error after logging
            else:
                logging.error(f"An unhandled requests exception occurred for method {method} with params {params}. Error: {e}")
                raise # Re-raise other request exceptions
        except Exception as e:
            st.error(f"An unexpected error occurred during Alchemy request: {e}")
            logging.error(f"An unexpected error occurred during Alchemy request for method {method} with params {params}. Error: {e}")
            raise
    st.error(f"Failed to make Alchemy request after {MAX_RETRIES} retries.")
    logging.error(f"Failed to make Alchemy request after {MAX_RETRIES} retries for method {method} with params {params}.")
    return None

MAX_PAGES_PER_ADDRESS = 5 # Limit to prevent excessive fetching for very active addresses

def get_paginated_asset_transfers(from_block_hex, address, category, is_from_address):
    all_transfers = []
    page_key = None
    for page_num in range(MAX_PAGES_PER_ADDRESS):
        params = {
            "fromBlock": from_block_hex,
            "toBlock": "latest",
            "category": category
        }
        if is_from_address:
            params["fromAddress"] = address
        else:
            params["toAddress"] = address
        
        if page_key:
            params["pageKey"] = page_key

        response = make_alchemy_request_with_retries("alchemy_getAssetTransfers", params)
        
        if response and 'result' in response and 'transfers' in response['result']:
            all_transfers.extend(response['result']['transfers'])
            page_key = response['result'].get('pageKey')
            if not page_key: # No more pages
                break
        else:
            logging.warning(f"No transfers or invalid response for {address} (page {page_num + 1}).")
            break
    
    if page_key:
        logging.warning(f"Reached MAX_PAGES_PER_ADDRESS ({MAX_PAGES_PER_ADDRESS}) for {address}. Some transfers might be truncated.")
    
    return all_transfers

def get_live_features():
    # Fetch the latest block for live transactions
    latest_block = w3.eth.get_block('latest', full_transactions=True)
    if not latest_block or not latest_block.transactions:
        st.warning("No transactions found in the latest block.")
        logging.warning("No transactions found in the latest block.")
        st.session_state['live_data'] = [] # Ensure it's an empty list
        return # Exit the function
    
    latest_block_number = latest_block.number
    from_block_number = max(0, latest_block_number - HISTORY_BLOCK_WINDOW)
    from_block_hex = hex(from_block_number)

    transactions = latest_block.transactions[:50] # Grab the first 50
    if not transactions:
        st.warning("No transactions to process after filtering.")
        logging.warning("No transactions to process after filtering.")
        st.session_state['live_data'] = [] # Ensure it's an empty list
        return # Exit the function
    
    live_data = []
    for tx in transactions:
        sender_address = tx['from']
        if not w3.is_address(sender_address):
            st.warning(f"Invalid sender address encountered: {sender_address}. Skipping transaction.")
            logging.warning(f"Invalid sender address encountered: {sender_address}. Skipping transaction.")
            continue
        
        # --- Fetch comprehensive transaction history using Alchemy API ---
        # This is a placeholder for actual Alchemy API call and rate limit handling.
        # In a real application, you would use Alchemy's SDK or a more robust
        # request mechanism with proper error handling and pagination.
        try:
            # Using alchemy_getAssetTransfers as getTransactionHistoryByAddress is beta and might require specific setup
            # This will fetch transfers for the sender_address as 'from' or 'to'
            historical_transfers_from = get_paginated_asset_transfers(from_block_hex, sender_address, ["external", "erc20"], True)
            historical_transfers_to = get_paginated_asset_transfers(from_block_hex, sender_address, ["external", "erc20"], False)
            historical_transfers = historical_transfers_from + historical_transfers_to
            
            # Fetch token balances
            token_balances_response = make_alchemy_request_with_retries(
                "alchemy_getTokenBalances",
                {
                    "owner": sender_address
                }
            )
            # time.sleep(ALCHEMY_API_DELAY) # Basic rate limit handling - removed, handled by retry mechanism
            if not token_balances_response or 'result' not in token_balances_response or 'tokenBalances' not in token_balances_response['result']:
                logging.warning(f"Invalid or empty token_balances_response for {sender_address}. Skipping.")
                token_balances = []
            else:
                token_balances = token_balances_response['result']['tokenBalances']

            # Fetch native Ether balance
            ether_balance_wei = w3.eth.get_balance(sender_address)
            ether_balance = float(w3.from_wei(ether_balance_wei, 'ether'))
            logging.info(f"Successfully fetched data for {sender_address}.")
            
        except Exception as e:
            st.warning(f"Could not fetch full transaction history or token balances for {sender_address} using Alchemy API: {e}. Using limited data.")
            logging.warning(f"Could not fetch full transaction history or token balances for {sender_address} using Alchemy API: {e}. Using limited data.")
            historical_transfers = [] # Fallback to empty history if API call fails
            token_balances = [] # Fallback to empty balances if API call fails
            ether_balance = 0.0 # Fallback to 0.0 for Ether balance

        # Dictionary to store historical data for the sender
        history = {
            'sent_to': collections.defaultdict(int),
            'received_from': collections.defaultdict(int),
            'received_values': []
        }

        for h_tx in historical_transfers:
            # For simplicity, we'll only consider 'external' transfers for value and addresses
            # More complex logic would be needed for ERC20/ERC721/ERC1155 transfers
            if h_tx['category'] == 'external':
                value_raw = h_tx.get('value')
                value_eth = 0.0 # Initialize value_eth

                if value_raw is None:
                    value_eth = 0.0
                elif isinstance(value_raw, str):
                    try:
                        value_int = int(value_raw, 16)
                        value_eth = float(w3.from_wei(value_int, 'ether'))
                    except ValueError:
                        logging.warning(f"Invalid hex string for transaction value: {value_raw}. Defaulting to 0.")
                        value_eth = 0.0
                elif isinstance(value_raw, int):
                    value_eth = float(w3.from_wei(value_raw, 'ether'))
                elif isinstance(value_raw, float):
                    value_eth = value_raw # Already in Ether units
                else:
                    logging.warning(f"Unexpected type for transaction value: {type(value_raw)}. Value: {value_raw}. Defaulting to 0.")
                    value_eth = 0.0
                
                if h_tx['from'] == sender_address:
                    history['sent_to'][h_tx['to']] += 1
                elif h_tx['to'] == sender_address:
                    history['received_from'][h_tx['from']] += 1
                    history['received_values'].append(value_eth)

        # Calculate dynamic features
        total_received = sum(history['received_values'])
        unique_received_from = len(history['received_from'])
        unique_sent_to = len(history['sent_to'])
        transaction_frequency = len(historical_transfers) # Total number of historical transfers
        
        min_value_received = min(history['received_values']) if history['received_values'] else 0.0
        max_value_received = max(history['received_values']) if history['received_values'] else 0.0
        avg_value_received = (sum(history['received_values']) / len(history['received_values'])) if history['received_values'] else 0.0

        # Check for interaction with known scam addresses
        interacted_with_scam = 0.0
        for h_tx in historical_transfers:
            if h_tx['from'] in KNOWN_SCAM_ADDRESSES or h_tx['to'] in KNOWN_SCAM_ADDRESSES:
                interacted_with_scam = 1.0
                break
        
        # Calculate unique tokens held
        unique_tokens_held = len(token_balances)

        features = [
            ether_balance, # Native Ether balance
            transaction_frequency, 
            float(w3.from_wei(tx['value'], 'ether')), # Live Ether sent
            total_received,
            unique_received_from,
            unique_sent_to,
            min_value_received,
            max_value_received
        ]
        
        # Validate the length of the features list
        if len(features) != 8:
            st.warning(f"Feature list for {sender_address} has unexpected length: {len(features)}. Expected 8. Skipping transaction.")
            logging.warning(f"Feature list for {sender_address} has unexpected length: {len(features)}. Expected 8. Skipping transaction.")
            continue
            
        live_data.append({"address": sender_address, "features": features})
        
    # Store results in Streamlit session_state instead of returning
    st.session_state['live_data'] = live_data

if w3.is_connected():
    st.success("Connected to Ethereum Mainnet")
    logging.info("Connected to Ethereum Mainnet.")
else:
    st.error("Connection Failed")
    logging.error("Connection to Ethereum Mainnet Failed.")
    st.stop() # Stop the app if connection fails

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

if auto_refresh_enabled:
    st.write(f"Auto-refresh enabled. Updating every {refresh_interval} seconds.")
    placeholder = st.empty()
    while auto_refresh_enabled:
        with placeholder.container():
            with st.spinner("Fetching live transactions..."):
                get_live_features() # This now populates st.session_state['live_data']
                live_transactions = st.session_state.get('live_data', [])
                
                results = []
                for tx in live_transactions:
                    scaled_input = normalize_features(tx['features'])
                    prediction = model(scaled_input) # Get the raw prediction
                    classification = predict_fraud(scaled_input)
                    
                    results.append({
                        "Wallet Address": tx['address'],
                        "Classification": classification,
                        "Ether Balance": tx['features'][0], # New: Ether Balance
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
                st.dataframe(results_df.style.apply(highlight_risk, axis=1), width='stretch')

                # Apply filter based on sidebar selection
                if risk_level_filter != "All":
                    filtered_results = [
                        res for res in results if res["Classification"] == risk_level_filter
                    ]
                    filtered_results_df = pd.DataFrame(filtered_results)
                    st.dataframe(filtered_results_df.style.apply(highlight_risk, axis=1), width='stretch')
                else:
                    st.dataframe(results_df.style.apply(highlight_risk, axis=1), width='stretch')
        time.sleep(refresh_interval)
else:
    if st.button("Scan Live Mempool"):
        with st.spinner("Fetching live transactions..."):
            get_live_features() # This now populates st.session_state['live_data']
            live_transactions = st.session_state.get('live_data', [])
            
            results = []
            for tx in live_transactions:
                scaled_input = normalize_features(tx['features'])
                prediction = model(scaled_input) # Get the raw prediction
                classification = predict_fraud(scaled_input)
                
                results.append({
                    "Wallet Address": tx['address'],
                    "Classification": classification,
                    "Ether Balance": tx['features'][0], # New: Ether Balance
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
            st.dataframe(results_df.style.apply(highlight_risk, axis=1), width='stretch')

            # Apply filter based on sidebar selection
            if risk_level_filter != "All":
                filtered_results = [
                    res for res in results if res["Classification"] == risk_level_filter
                ]
                filtered_results_df = pd.DataFrame(filtered_results)
                st.dataframe(filtered_results_df.style.apply(highlight_risk, axis=1), width='stretch')
            else:
                st.dataframe(results_df.style.apply(highlight_risk, axis=1), width='stretch')

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

