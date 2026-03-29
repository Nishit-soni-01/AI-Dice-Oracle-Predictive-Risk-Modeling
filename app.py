import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
import random
import requests
from streamlit_lottie import st_lottie
import warnings
import os

# --- 1. SETTINGS & WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(page_title="AI Dice Oracle", page_icon="🔮", layout="wide")

# --- 2. SESSION STATE (HISTORY) ---
# This keeps track of your rolls even when the app reruns
if 'roll_history' not in st.session_state:
    st.session_state.roll_history = []

# --- 3. CUSTOM GLASSMORPHISM CSS ---
st.markdown("""
    <style>
    .main { background: #0F172A; color: #F8FAFC; }
    .stButton>button {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        color: white; border-radius: 12px; border: none;
        padding: 1rem; font-weight: 700; transition: 0.3s;
        width: 100%;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(124, 58, 237, 0.5); }
    [data-testid="stMetricValue"] { color: #A78BFA; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. ASSET LOADERS ---
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200: return None
        return r.json()
    except: return None

@st.cache_resource
def load_model():
    # Model trained using GaussianNB logic
    return pickle.load(open('dice_model.pkl', 'rb'))

lottie_url = "https://lottie.host/86644f8d-676b-4395-9788-f542a2012a6d/YI6YI6p5X0.json"
lottie_magic = load_lottie(lottie_url)

# --- 5. SIDEBAR (INSTRUCTIONS & HISTORY) ---
with st.sidebar:
    st.header("🎮 Game Controls")
    with st.expander("📖 How to Play", expanded=True):
        st.write("""
        1. **Set Target:** Choose a sum between 2-12.
        2. **Adjust Luck:** Tell the AI your 'Luck' factor (1-10).
        3. **Oracle Prediction:** The **Gaussian Naive Bayes** model predicts your win probability.
        4. **Roll:** Click to see if destiny matches the data!
        """)
    
    st.write("---")
    st.subheader("📜 Roll History")
    if st.session_state.roll_history:
        for entry in reversed(st.session_state.roll_history[-5:]):
            st.write(entry)
    else:
        st.caption("No rolls yet. Start your destiny!")

# --- 6. APP LOGIC ---
try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'dice_model.pkl' missing!")
    st.stop()

st.title("🔮 The AI Oracle: Predictive Dice")
st.markdown("### Bridging Gaussian Naive Bayes with Interactive Gaming")
st.write("---")

tab1, tab2 = st.tabs(["🎲 Game Environment", "📊 Model Diagnostics"])

with tab1:
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.subheader("Input Parameters")
        target = st.select_slider("Select Target Sum:", options=list(range(2, 13)), value=7)
        luck = st.slider("Luck Coefficient:", 1.0, 10.0, 5.0)
        
        # DataFrame ensures valid feature names for the model
        input_data = pd.DataFrame([[target, luck]], columns=['target', 'luck'])
        prob = model.predict_proba(input_data)[0][1]
        
        st.metric("Oracle's Win Probability", f"{prob*100:.2f}%")
        st.progress(prob)
    
    with c2:
        if lottie_magic:
            st_lottie(lottie_magic, height=300, key="oracle")
        else:
            st.markdown("<h1 style='text-align: center; font-size: 150px;'>🔮</h1>", unsafe_allow_html=True)

    if st.button("EXECUTE PROBABILITY ROLL"):
        placeholder = st.empty()
        for _ in range(12):
            d1, d2 = random.randint(1, 6), random.randint(1, 6)
            placeholder.markdown(f"<h1 style='text-align: center; font-size: 80px;'>🎲 {d1} &nbsp; 🎲 {d2}</h1>", unsafe_allow_html=True)
            time.sleep(0.08)
        
        f1, f2 = random.randint(1, 6), random.randint(1, 6)
        res = f1 + f2
        placeholder.markdown(f"<h1 style='text-align: center; font-size: 100px; color: #C084FC;'>🎲 {f1} &nbsp; 🎲 {f2}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center;'>Total Result: {res}</h2>", unsafe_allow_html=True)
        
        if res == target:
            st.success(f"🏆 DESTINY ACHIEVED: Rolled {res}!")
            st.balloons()
            st.session_state.roll_history.append(f"✅ Target {target}: Rolled {res} (WIN)")
        else:
            st.error(f"💀 THE ORACLE WAS CORRECT: Rolled {res}.")
            st.session_state.roll_history.append(f"❌ Target {target}: Rolled {res} (LOSS)")

with tab2:
    st.header("📊 Model Performance & Validation")
    img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
    with img_col2:
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="Confusion Matrix: Predicted vs Actual", use_column_width=True)
        else:
            st.error("⚠️ 'confusion_matrix.png' not found.")

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Algorithm", "GaussianNB")
    col_b.metric("Accuracy", "0.9225") # Consistent with notebook implementation
    col_c.metric("F1-Score", "0.96 ")