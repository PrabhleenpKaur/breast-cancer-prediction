"""
app.py  –  Breast Cancer Detection Web Application
----------------------------------------------------
Run:  streamlit run app/app.py
"""

import streamlit as st
import numpy as np
import os
import sys
import io
import time
from PIL import Image, ImageEnhance
import tempfile

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BreastScan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0f1117;
    color: #e8eaf0;
}

/* ── Top banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a1f35 0%, #0d1b2a 50%, #1a1035 100%);
    border: 1px solid #2a3050;
    border-radius: 12px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 60% 40%, rgba(99,102,241,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #f0f4ff;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1rem;
    color: #8892b0;
    margin-top: 0.4rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.4);
    color: #f87171;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* ── Upload card ── */
.upload-zone {
    background: #141824;
    border: 2px dashed #2d3550;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}

/* ── Result cards ── */
.result-card {
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.result-benign {
    background: linear-gradient(135deg, #0d2a1f, #0a2018);
    border: 1px solid #1a5c36;
}
.result-malignant {
    background: linear-gradient(135deg, #2a0d0d, #200a0a);
    border: 1px solid #5c1a1a;
}
.result-label {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
}
.result-benign .result-label  { color: #4ade80; }
.result-malignant .result-label { color: #f87171; }
.result-confidence {
    font-size: 0.9rem;
    color: #8892b0;
    margin-top: 0.3rem;
}

/* ── Metric chips ── */
.metric-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.metric-chip {
    background: #1e2535;
    border: 1px solid #2d3550;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.85rem;
    color: #c8d0e0;
}
.metric-chip span {
    color: #818cf8;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Progress bar colour ── */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2535;
}
section[data-testid="stSidebar"] * { color: #c8d0e0; }

/* ── Disclaimer box ── */
.disclaimer {
    background: #1a1a2e;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    font-size: 0.82rem;
    color: #8892b0;
    margin-top: 1.5rem;
}

/* ── Steps ── */
.step-item {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    margin-bottom: 0.9rem;
}
.step-num {
    background: #6366f1;
    color: white;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    font-size: 0.72rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.step-text { font-size: 0.88rem; color: #a8b2c8; line-height: 1.4; }

hr { border-color: #1e2535 !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor(model_path: str):
    """Load model once and cache across sessions."""
    try:
        from utils.inference import BreastCancerPredictor
        return BreastCancerPredictor(model_path), None
    except Exception as e:
        return None, str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()

    model_path = st.text_input(
        "Model path",
        value="model/best_model.keras",
        help="Path to your trained .keras or .h5 model file",
    )
    threshold = st.slider(
        "Decision threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Scores above this → Malignant",
    )

    st.divider()
    st.markdown("### 📋 How it works")
    for num, txt in [
        ("1", "Upload a mammogram or histopathology image"),
        ("2", "The CNN backbone (EfficientNetB0) extracts features"),
        ("3", "The classifier head outputs a probability"),
        ("4", "Threshold applied → Benign / Malignant label"),
    ]:
        st.markdown(
            f'<div class="step-item">'
            f'<div class="step-num">{num}</div>'
            f'<div class="step-text">{txt}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Model:** EfficientNetB0 + custom head")
    st.markdown("**Dataset:** BreastMNIST / BreaKHis")
    st.markdown("**Input:** 224 × 224 RGB")


# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">⚠ Educational Use Only</div>
  <p class="hero-title">🩺 BreastScan AI</p>
  <p class="hero-subtitle">
    Deep learning–powered breast cancer detection from mammogram &amp; histopathology images
  </p>
</div>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
predictor, load_error = load_predictor(model_path)

if load_error:
    st.warning(
        f"⚠️ Model not found at **{model_path}**. "
        "Run `python train.py` first, or adjust the path in the sidebar.\n\n"
        f"Error: `{load_error}`",
        icon="🔧",
    )
    st.info(
        "**Demo mode active** – predictions are simulated for UI demonstration.",
        icon="ℹ️",
    )
    demo_mode = True
else:
    st.success(f"✅ Model loaded from `{model_path}`", icon="🤖")
    demo_mode = False


# ── Main columns ─────────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("#### 📤 Upload Image")
    uploaded = st.file_uploader(
        "Choose a mammogram or histopathology image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        label_visibility="collapsed",
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)

        # Image info chips
        w, h = img.size
        st.markdown(
            f'<div class="metric-row">'
            f'<div class="metric-chip">Size <span>{w}×{h}</span></div>'
            f'<div class="metric-chip">Format <span>{uploaded.type.split("/")[1].upper()}</span></div>'
            f'<div class="metric-chip">File <span>{uploaded.size / 1024:.1f} KB</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )


with col_result:
    st.markdown("#### 🔬 Analysis Result")

    if not uploaded:
        st.markdown("""
        <div style="
            background:#141824; border:2px dashed #2d3550;
            border-radius:12px; padding:3rem 2rem;
            text-align:center; color:#4a5568;
        ">
            <div style="font-size:3rem; margin-bottom:0.8rem">🔬</div>
            <p style="font-size:1rem; margin:0">
                Upload an image to begin analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        analyze_btn = st.button(
            "▶  Run Analysis",
            use_container_width=True,
            type="primary",
        )

        if analyze_btn:
            with st.spinner("Analysing image …"):
            # Create a cross-platform temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp_file:
                    tmp_file.write(uploaded.getbuffer())
                    tmp_path = tmp_file.name

                time.sleep(0.6)  # brief pause for UX

                if demo_mode:
                    # Simulate a prediction for demo purposes
                    import random
                    raw = random.uniform(0.05, 0.95)
                    result = {
                        "label":       "Malignant" if raw >= threshold else "Benign",
                        "confidence":  round((raw if raw >= threshold else 1 - raw) * 100, 2),
                        "probability": round(raw, 4),
                        "raw_score":   raw,
                    }
                else:
                    result = predictor.predict(tmp_path, threshold=threshold)
                
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            # ── Display result ────────────────────────────────────────────
            is_malignant = result["label"] == "Malignant"
            card_cls     = "result-malignant" if is_malignant else "result-benign"
            icon         = "⚠️" if is_malignant else "✅"

            st.markdown(
                f'<div class="result-card {card_cls}">'
                f'  <p class="result-label">{icon} {result["label"]}</p>'
                f'  <p class="result-confidence">Confidence: {result["confidence"]}%</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Confidence bar
            prob = result["probability"]
            st.markdown(
                f"**Malignancy probability:** `{prob:.4f}`"
            )
            st.progress(float(prob))

            # Metric chips
            st.markdown(
                f'<div class="metric-row">'
                f'<div class="metric-chip">Raw score <span>{result["raw_score"]:.4f}</span></div>'
                f'<div class="metric-chip">Threshold <span>{threshold}</span></div>'
                f'<div class="metric-chip">Decision <span>{result["label"]}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Interpretation note
            st.divider()
            if is_malignant:
                st.error(
                    "The model detected features **consistent with malignancy**. "
                    "This is a screening aid only — please consult a qualified radiologist.",
                    icon="⚠️",
                )
            else:
                st.success(
                    "The model detected features **consistent with a benign lesion**. "
                    "Regular screening is still recommended.",
                    icon="✅",
                )


# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  <strong>⚠️ Disclaimer:</strong>
  This is an <strong>educational research project</strong> and is <em>not</em> intended for
  clinical diagnosis or medical use. Predictions should <strong>never</strong> replace
  evaluation by a qualified medical professional. Always consult a licensed physician
  for any health-related decisions.
</div>
""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
cols = st.columns(3)
cols[0].markdown("🩺 **BreastScan AI** v1.0")
cols[1].markdown("🤖 EfficientNetB0 · TensorFlow 2.x")
cols[2].markdown("📚 BreastMNIST / BreaKHis Dataset")
