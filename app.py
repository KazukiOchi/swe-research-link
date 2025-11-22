import streamlit as st
import pandas as pd
import numpy as np
import torch
import math
import time
import os

# ==========================================
# 1. SWE Core Logic (Backend)
# ==========================================

def laplacian_apply(x, edges_u, edges_v, weights, inv_sqrt_deg, deg):
    if x.dim() == 1:
        inv = inv_sqrt_deg.to(x.dtype)
        deg_t = deg.to(x.dtype)
    else:
        inv = inv_sqrt_deg.to(x.dtype).unsqueeze(1)
        deg_t = deg.to(x.dtype).unsqueeze(1)
    
    x1 = x * inv
    y_tmp = torch.zeros_like(x1)
    w = weights.to(x.dtype)

    if x.dim() == 1:
        y_tmp.index_add_(0, edges_u, w * x1[edges_v])
        y_tmp.index_add_(0, edges_v, w * x1[edges_u])
    else:
        y_tmp.index_add_(0, edges_u, w.view(-1, 1) * x1[edges_v])
        y_tmp.index_add_(0, edges_v, w.view(-1, 1) * x1[edges_u])

    y = deg_t * x1 - y_tmp
    y = y * inv
    return y

def swe_propagate(psi0, tau, edges_u, edges_v, weights, inv_sqrt_deg, deg, taylor_order=12):
    out = psi0.clone()
    term = psi0.clone()
    for k in range(1, taylor_order + 1):
        term = laplacian_apply(term, edges_u, edges_v, weights, inv_sqrt_deg, deg)
        coeff = (1j * tau) ** k / math.factorial(k)
        out = out + coeff * term
    return out

def compute_synergy(query_nodes, num_nodes, tau, edges_u, edges_v, weights, inv_sqrt_deg, deg, taylor_order, device="cpu"):
    """
    シナジー計算 (確率ベース)
    Q=1: 確率 |ψ|^2 をそのまま返す
    Q>1: 干渉項 P_coherent - P_incoherent を返す
    """
    device = torch.device(device)
    query_nodes = list(set(query_nodes))
    Q = len(query_nodes)
    if Q == 0: return None

    # 初期状態
    psi0_multi = torch.zeros((num_nodes, Q), dtype=torch.cfloat, device=device)
    for col, nid in enumerate(query_nodes):
        if 0 <= nid < num_nodes:
            psi0_multi[nid, col] = 1.0 + 0j

    # 1. 時間発展 (Propagate)
    psi_multi = swe_propagate(psi0_multi, tau, edges_u, edges_v, weights, inv_sqrt_deg, deg, taylor_order)

    # 2. 個別の確率密度 |ψ_k|^2
    probs_each = psi_multi.abs().pow(2).float()

    # --- Case A: Single Query (単独検索) ---
    if Q == 1:
        # そのまま確率を返す
        return probs_each.squeeze()

    # --- Case B: Multi Query (シナジー探索) ---
    # Coherent sum (合成波): ψ_all = (1/√Q) Σ ψ_k
    psi_all = (1.0 / math.sqrt(Q)) * psi_multi.sum(dim=1)
    prob_all = psi_all.abs().pow(2).float() # |ψ_all|^2

    # Incoherent sum (非干渉和): mean(|ψ_k|^2)
    # エネルギー保存の観点から、Coherent側(1/√Q係数)と比較するには平均をとるのが適切
    probs_mean = probs_each.mean(dim=1)

    # Synergy = (干渉した結果の確率) - (単独の確率の平均)
    # 正であれば「強め合い」、負であれば「弱め合い」
    return prob_all - probs_mean

# ==========================================
# 2. Data Loading
# ==========================================

@st.cache_resource
def load_data(device_name="cpu"):
    device = torch.device(device_name)
    
    titles = {}
    title_path = 'arxiv_titles.csv'
    if os.path.exists(title_path):
        df = pd.read_csv(title_path)
        titles = dict(zip(df['node_id'], df['title']))
    
    try:
        from ogb.nodeproppred import NodePropPredDataset
        dataset = NodePropPredDataset(name='ogbn-arxiv')
        graph, _ = dataset[0]
    except ImportError:
        st.error("OGB library not installed.")
        return None, None

    num_nodes = graph['num_nodes']
    edge_index = graph['edge_index']

    src, dst = edge_index[0], edge_index[1]
    u_list = np.concatenate([src, dst])
    v_list = np.concatenate([dst, src])
    edges = np.stack([u_list, v_list], axis=1)
    edges = np.unique(edges, axis=0)
    mask = edges[:, 0] != edges[:, 1]
    edges = edges[mask]

    edges_u = torch.tensor(edges[:, 0], dtype=torch.long, device=device)
    edges_v = torch.tensor(edges[:, 1], dtype=torch.long, device=device)
    
    num_edges = edges_u.shape[0]
    weights = torch.ones(num_edges, dtype=torch.float32, device=device)
    weights *= 0.5
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    deg.index_add_(0, edges_u, weights)
    deg.index_add_(0, edges_v, weights)
    inv_sqrt_deg = 1.0 / torch.sqrt(deg + 1e-8)

    graph_data = (num_nodes, edges_u, edges_v, weights, inv_sqrt_deg, deg)
    return graph_data, titles

# ==========================================
# 3. Frontend UI
# ==========================================

st.set_page_config(
    page_title="Research↔Link",
    page_icon="🧬",
    layout="centered"
)

st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        text-align: left;
    }
    .result-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 10px;
        border-left: 5px solid #4e8cff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .result-title {
        font-size: 16px;
        font-weight: bold;
        color: #333;
        margin-bottom: 4px;
    }
    .result-meta {
        font-size: 12px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Research↔Link")
st.caption("Discover the missing link between research topics in Computer Science.")

DEVICE_OPT = "cuda" if torch.cuda.is_available() else "cpu"
TAU_FIXED = 1.57
TOP_K_FIXED = 10

with st.spinner("Initializing Knowledge Graph..."):
    graph_data, titles = load_data(DEVICE_OPT)

if not graph_data:
    st.stop()

num_nodes, edges_u, edges_v, weights, inv_sqrt_deg, deg = graph_data

if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

# --- 1. 検索エリア ---
st.subheader("1. Add Ingredients")
search_col, add_col = st.columns([3, 1])
with search_col:
    search_query = st.text_input("Search papers by keyword", placeholder="e.g. attention mechanism", label_visibility="collapsed")

if search_query:
    hits = []
    search_query_lower = search_query.lower().split()
    for nid, t in titles.items():
        t_low = str(t).lower()
        if all(q in t_low for q in search_query_lower):
            hits.append((nid, t))
#            if len(hits) >= 30: break
    
    if hits:
        st.markdown(f"Found {len(hits)} papers:")
        with st.container(height=250):
            for nid, t in hits:
                if st.button(f"➕ {t}", key=f"add_{nid}"):
                    if nid not in st.session_state.cart:
                        st.session_state.cart.append(nid)
                        st.toast(f"Added: {t[:20]}...", icon="✅")
    else:
        st.info("No papers found.")

st.markdown("---")

# --- 2. カートエリア ---
st.subheader("2. Your Mix")

if len(st.session_state.cart) == 0:
    st.info("No papers selected yet. Search and add papers above.")
else:
    for i, nid in enumerate(st.session_state.cart):
        t = titles.get(nid, "Unknown Title")
        col_txt, col_del = st.columns([5, 1])
        with col_txt:
            st.markdown(f"📄 **{t}**")
        with col_del:
            if st.button("🗑️", key=f"del_{i}"):
                st.session_state.cart.pop(i)
                st.rerun()
    
    st.markdown("")
    # ボタンのラベルを状況によって変える
    btn_label = "🧪 Mix & Discover Synergy" if len(st.session_state.cart) > 1 else "🔎 Search Related Papers"
    if st.button(btn_label, type="primary"):
        st.session_state.run_analysis = True

# --- 3. 解析結果 ---
if st.session_state.run_analysis and st.session_state.cart:
    st.markdown("---")
    st.subheader("3. Discovery Results")
    
    query_nodes = st.session_state.cart
    
    with st.spinner("Calculating probabilities & synergy..."):
        start_t = time.time()
        # 計算実行
        delta = compute_synergy(query_nodes, num_nodes, TAU_FIXED, edges_u, edges_v, weights, inv_sqrt_deg, deg, 12, DEVICE_OPT)
        elapsed = time.time() - start_t
    
    delta_np = delta.detach().cpu().numpy()
    query_set = set(query_nodes)

    # ソートして上位抽出
    candidates = [(i, delta_np[i]) for i in range(num_nodes) if i not in query_set]
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    st.success(f"Found {TOP_K_FIXED} results in {elapsed:.2f} sec!")

    for i, (nid, score) in enumerate(candidates[:TOP_K_FIXED]):
        t = titles.get(nid, "(No Title)")
        search_url = f"https://scholar.google.com/scholar?q={t.replace(' ', '+')}"
        
        st.markdown(f"""
        <div class="result-card">
            <div class="result-title">{i+1}. <a href="{search_url}" target="_blank" style="text-decoration:none; color:#333;">{t}</a></div>
            <div class="result-meta">Paper ID: {nid} | <a href="{search_url}" target="_blank">Search on Google Scholar ↗</a></div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Clear Results"):
        st.session_state.run_analysis = False
        st.rerun()
