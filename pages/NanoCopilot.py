import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import os

# --- Configuration ---
BLOCK_SIZE = 256 # context length
N_EMBD = 256     # embedding dimension (Matches weights)
N_HEAD = 8       # number of heads (Matches weights)
N_LAYER = 6      # number of layers (Matches weights)
DROPOUT = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VOCAB_SIZE = 2256 # vocab size (Matches weights)

st.set_page_config(
    page_title="0xNeural | NanoCopilot",
    page_icon="🤖",
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
    
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid #334155;
        margin-bottom: 2rem;
        text-align: left;
    }
    
    .hero-title {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #00A3FF, #00FF41);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Terminal UI */
    .terminal-header {
        background: #1E293B;
        padding: 8px 15px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        border: 1px solid #334155;
        border-bottom: none;
        display: flex;
        gap: 6px;
    }
    
    .terminal-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }

    /* Sidebar Fixes */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }

    /* Model Architecture Info */
    .arch-card {
        background: rgba(30, 41, 59, 0.4);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #334155;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🤖 NanoCopilot</h1>
        <p style="font-size: 1.1rem; color: #94A3B8;">Generative Transformer model trained to hallucinate syntactically correct Solidity code.</p>
        <div style="display: flex; gap: 10px; margin-top: 10px;">
            <span style="background: #1E293B; color: #00A3FF; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; border: 1px solid #00A3FF;">5.96M PARAMETERS</span>
            <span style="background: #1E293B; color: #00FF41; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; border: 1px solid #00FF41;">SOLIDITY SYNTAX</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Transformer Architecture Classes ---

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention running in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ A complete Transformer Block: Communication (Attention) followed by Computation (FeedForward) """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class NanoCopilot(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=10):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Tokenizer Logic (Replicated from Notebook) ---

class Web3Tokenizer:
    def __init__(self, merges, vocab):
        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        # Convert text to raw bytes
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            # Find the pair that can be merged first (based on the merge rules)
            stats = {}
            for pair in zip(tokens, tokens[1:]):
                stats[pair] = stats.get(pair, 0) + 1
            
            # We want to find the pair with the lowest index in our merges (the first one that happened)
            pair_to_merge = None
            for pair in stats:
                if pair in self.merges:
                    if pair_to_merge is None or self.merges[pair] < self.merges[pair_to_merge]:
                        pair_to_merge = pair
            
            if pair_to_merge is None:
                break # no more merges possible
                
            idx = self.merges[pair_to_merge]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair_to_merge[0] and tokens[i+1] == pair_to_merge[1]:
                    new_tokens.append(idx)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, ids):
        # Map IDs back to raw bytes and decode to string
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")

@st.cache_resource
def load_tokenizer_resource():
    import json
    # Use relative pathing to ensure portability
    root_dir = os.path.dirname(os.path.dirname(__file__))
    tokenizer_path = os.path.join(root_dir, 'web3_tokenizer_2000_merges.json')
    
    if not os.path.exists(tokenizer_path):
        # Fallback check in current directory
        tokenizer_path = os.path.join(os.path.dirname(__file__), 'web3_tokenizer_2000_merges.json')
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Critical: 'web3_tokenizer_2000_merges.json' not found at {tokenizer_path}")
    
    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)

        # Correctly reconstruct merges and vocab from the JSON structure
        merges = {tuple(map(int, k.split('|'))): v for k, v in tokenizer_data['merges'].items()}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        
        return Web3Tokenizer(merges, vocab)
    except Exception as e:
        raise RuntimeError(f"Error parsing tokenizer: {e}")

# --- Model Loading ---

def load_model_resource():
    model = NanoCopilot()
    # Use relative pathing for weights as well
    root_dir = os.path.dirname(os.path.dirname(__file__))
    weights_path = os.path.join(root_dir, 'nano_copilot_5.96M_weights.pth')
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Critical: 'nano_copilot_5.96M_weights.pth' not found at {weights_path}")

    try:
        state_dict = torch.load(weights_path, map_location=torch.device(DEVICE))
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading weights: {e}")

# --- Caching Layer (Session Persistence) ---
@st.cache_resource(ttl=3600) # Added TTL to help with cache rotation
def get_cached_brain(version="1.0.1"): # Incremented version to force cache invalidation
    try:
        tokenizer = load_tokenizer_resource()
        model = load_model_resource()
        return {"model": model, "tokenizer": tokenizer, "success": True, "version": version}
    except Exception as e:
        return {"error": str(e), "success": False}

# Access the cached resources
brain = get_cached_brain()

if not brain["success"]:
    st.error(f"🧠 Brain Load Error: {brain['error']}")
    st.stop()

model = brain["model"]
tokenizer = brain["tokenizer"]

# Double-check that the tokenizer has the encode method (safety for stale caches)
if not hasattr(tokenizer, 'encode'):
    st.warning("🔄 Stale cache detected. Refreshing tokenizer resources...")
    st.cache_resource.clear()
    st.rerun()

# --- UI Interface ---
st.markdown("### 🖥️ Neural Output Terminal")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("#### 🎲 Random Hallucination")
    st.write("Let the model generate code from a blank state.")
    gen_random = st.button("Generate Random Code")

with col_right:
    st.markdown("#### 🧠 Seeded Generation")
    seed_text = st.text_area("Start the contract:", value="contract FlashLoan {", height=100)
    gen_seeded = st.button("Generate from Seed")

if gen_random or gen_seeded:
    st.markdown("""
        <div class="terminal-header">
            <div class="terminal-dot" style="background: #FF5F56;"></div>
            <div class="terminal-dot" style="background: #FFBD2E;"></div>
            <div class="terminal-dot" style="background: #27C93F;"></div>
            <span style="color: #94A3B8; font-size: 0.8rem; margin-left: 10px; font-family: 'JetBrains Mono';">nanocopilot --generate-solidity</span>
        </div>
        """, unsafe_allow_html=True)
    
    placeholder = st.empty()
    
    if gen_random:
        context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        full_text = ""
        with st.spinner("Decoding from latent space..."):
            for i in range(16):
                generated_indices = model.generate(context, max_new_tokens=32)
                new_ids = generated_indices[0, -32:].tolist()
                new_text = tokenizer.decode(new_ids)
                full_text += new_text
                placeholder.code(full_text, language='solidity')
                context = generated_indices
                time.sleep(0.05)
    
    elif gen_seeded:
        with st.spinner("Encoding seed and projecting..."):
            input_ids = tokenizer.encode(seed_text)
            context = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
            full_text = seed_text
            
            # Show initial seed
            placeholder.code(full_text, language='solidity')
            
            for i in range(12): # Generate ~384 more tokens
                generated_indices = model.generate(context, max_new_tokens=32)
                new_ids = generated_indices[0, -32:].tolist()
                new_text = tokenizer.decode(new_ids)
                full_text += new_text
                placeholder.code(full_text, language='solidity')
                context = generated_indices
                time.sleep(0.05)

# Sidebar Status Indicators
st.sidebar.markdown("### 🧠 Brain Status")
st.sidebar.success("✅ Weights Loaded")
st.sidebar.info(f"**Params:** 5.96M\n\n**Context:** {BLOCK_SIZE} tokens\n\n**Device:** {DEVICE}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏗️ Architecture Specs")
st.sidebar.markdown(f"""
<div class="arch-card">
    <b>Transformer Block</b><br>
    - Layers: {N_LAYER}<br>
    - Heads: {N_HEAD}<br>
    - Embd: {N_EMBD}<br>
    - Dropout: {DROPOUT}
</div>
<div class="arch-card">
    <b>Tokenizer (BPE)</b><br>
    - Merges: 2000<br>
    - Vocab: {VOCAB_SIZE}
</div>
""", unsafe_allow_html=True)
