import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pymongo import MongoClient
import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import base64
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors as rl_colors
import tempfile
import io

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Oil Spill Detection System",
    page_icon="https://img.icons8.com/fluency/48/oil-industry.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #07111f;
    color: #d1d9e6;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2e 0%, #07111f 100%);
    border-right: 1px solid #1a2d45;
}
section[data-testid="stSidebar"] * { color: #c9d6e3 !important; }

/* ── Main content area ── */
.main .block-container { padding: 2rem 2.5rem 3rem; }

/* ── Card ── */
.card {
    background: linear-gradient(135deg, #0d1b2e 0%, #101e30 100%);
    border: 1px solid #1e304a;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 22px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.45);
}

/* ── Section heading ── */
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #4ea8de;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e304a;
}

/* ── Page title ── */
.page-title {
    font-size: 2rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.5px;
}
.page-subtitle {
    font-size: 0.9rem;
    color: #7b8ea6;
    margin-top: 4px;
}

/* ── Metric box ── */
.metric-box {
    background: linear-gradient(135deg, #0f2036 0%, #0d1b2e 100%);
    border: 1px solid #1e304a;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
}
.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4ea8de;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1;
}
.metric-unit {
    font-size: 0.75rem;
    color: #7b8ea6;
    margin-top: 4px;
}

/* ── Status banner ── */
.status-critical {
    background: linear-gradient(90deg, #3b1010, #1e0808);
    border: 1px solid #7a1f1f;
    border-left: 4px solid #e53e3e;
    border-radius: 10px;
    padding: 14px 20px;
    color: #feb2b2;
    font-weight: 500;
}
.status-warning {
    background: linear-gradient(90deg, #352808, #1e1208);
    border: 1px solid #7a4f1f;
    border-left: 4px solid #dd6b20;
    border-radius: 10px;
    padding: 14px 20px;
    color: #fbd38d;
    font-weight: 500;
}
.status-info {
    background: linear-gradient(90deg, #0a2540, #07111f);
    border: 1px solid #1a4a7a;
    border-left: 4px solid #3182ce;
    border-radius: 10px;
    padding: 14px 20px;
    color: #90cdf4;
    font-weight: 500;
}
.status-ok {
    background: linear-gradient(90deg, #0a2b1e, #07111f);
    border: 1px solid #1a5c3a;
    border-left: 4px solid #38a169;
    border-radius: 10px;
    padding: 14px 20px;
    color: #9ae6b4;
    font-weight: 500;
}

/* ── Insight rows ── */
.insight-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid #131f2e;
}
.insight-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4ea8de;
    width: 120px;
    flex-shrink: 0;
    padding-top: 2px;
}
.insight-value {
    font-size: 0.88rem;
    color: #c9d6e3;
}

/* ── Image captions ── */
.img-caption {
    text-align: center;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4ea8de;
    margin-top: 8px;
}

/* ── History record ── */
.history-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid #131f2e;
    font-size: 0.85rem;
}
.history-name { color: #c9d6e3; font-weight: 500; }
.history-pct { color: #4ea8de; font-weight: 700; }

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #0d1b2e;
    border: 2px dashed #1e304a;
    border-radius: 14px;
    padding: 16px;
}

/* ── Buttons ── */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #1a6fb5, #1259a0) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(135deg, #2080cc, #1a6fb5) !important;
    box-shadow: 0 4px 14px rgba(30,111,181,0.45) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0d1b2e !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    color: #c9d6e3 !important;
}

/* ── Pyplot / chart backgrounds ── */
.stpyplot { background: transparent !important; }

/* ── Divider ── */
hr { border-top: 1px solid #1a2d45 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  MATPLOTLIB DARK STYLE HELPER
# ─────────────────────────────────────────
DARK_BG   = "#0d1b2e"
DARK_AX   = "#0f2036"
GRID_CLR  = "#1a2d45"
TEXT_CLR  = "#7b8ea6"
ACCENT    = "#4ea8de"
ACCENT2   = "#1a6fb5"

def apply_dark_style(fig, ax_list=None):
    fig.patch.set_facecolor(DARK_BG)
    if ax_list is None:
        ax_list = fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor(DARK_AX)
        ax.tick_params(colors=TEXT_CLR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)
        ax.title.set_color(TEXT_CLR)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.grid(True, color=GRID_CLR, linewidth=0.6, linestyle="--", alpha=0.6)

# ─────────────────────────────────────────
#  DATABASE (MongoDB — optional)
# ─────────────────────────────────────────
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    client.server_info()
    collection = client["oil_spill_db"]["oil_spill_results"]
except Exception:
    collection = None

# ─────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────
IMG_SIZE = 256

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return (2. * K.sum(y_true_f * y_pred_f) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def weighted_loss(y_true, y_pred):
    return tf.reduce_mean(
        tf.keras.backend.binary_crossentropy(y_true, y_pred)
    ) + (1 - dice_coef(y_true, y_pred))

@st.cache_resource
def load_model_cached():
    return load_model(
        "oil_spill_unet_model_v2.h5",
        custom_objects={
            "weighted_bce_dice_loss": weighted_loss,
            "dice_coef": dice_coef,
        },
        compile=False,
    )

model = load_model_cached()

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:10px 0 24px;'>
        <div style='font-size:1.3rem;font-weight:700;color:#e6edf3;'>Oil Spill Detection</div>
        <div style='font-size:0.78rem;color:#4ea8de;margin-top:2px;letter-spacing:0.06em;'>DEEP LEARNING SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**About**")
    st.markdown(
        "<span style='font-size:0.83rem;color:#7b8ea6;'>"
        "Semantic segmentation model (U-Net) trained on SAR satellite imagery "
        "to detect and delineate oil spill regions in marine environments."
        "</span>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**Pipeline**")
    steps = [
        ("1", "Upload SAR Image"),
        ("2", "Preprocessing"),
        ("3", "U-Net Inference"),
        ("4", "Probability Map"),
        ("5", "Binary Mask"),
        ("6", "Metrics + Report"),
    ]
    for num, label in steps:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;padding:5px 0;'>"
            f"<div style='width:22px;height:22px;border-radius:50%;background:#1a6fb5;"
            f"display:flex;align-items:center;justify-content:center;"
            f"font-size:0.7rem;font-weight:700;color:#fff;flex-shrink:0;'>{num}</div>"
            f"<span style='font-size:0.82rem;color:#c9d6e3;'>{label}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Threshold**")
    threshold = st.slider(
        "Segmentation threshold", 0.1, 0.9, 0.5, 0.05,
        help="Pixel probability above this value is classified as oil."
    )

    st.markdown("---")
    db_status = "Connected" if collection is not None else "Unavailable"
    db_color  = "#38a169" if collection is not None else "#e53e3e"
    st.markdown(
        f"<div style='font-size:0.78rem;color:#7b8ea6;'>"
        f"Database: <span style='color:{db_color};font-weight:600;'>{db_status}</span></div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────
#  PAGE HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class='card' style='margin-bottom:28px;'>
    <div class='page-title'>AI-Based Oil Spill Detection System</div>
    <div class='page-subtitle'>
        Upload a Synthetic Aperture Radar (SAR) image to analyze and detect oil spill regions
        using a deep learning U-Net segmentation model.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  PROJECT INTRODUCTION
# ─────────────────────────────────────────
with st.expander("About This System — Project Overview & Pipeline", expanded=False):

    st.markdown("""
<p style='font-size:0.88rem;color:#c9d6e3;line-height:1.75;margin-bottom:14px;'>
This system is a deep learning-based environmental monitoring tool designed to automatically
identify and measure oil spill contamination in
<strong style='color:#4ea8de;'>Synthetic Aperture Radar (SAR)</strong> satellite imagery.
SAR sensors penetrate cloud cover and operate day or night, providing consistent surveillance
data even in adverse weather conditions.
</p>
<div class='section-title'>What is SAR Imagery?</div>
<p style='font-size:0.85rem;color:#9cb4cc;line-height:1.72;margin-bottom:14px;'>
SAR images are radar-based satellite images where the ocean surface appears in varying shades
of grey. Clean water has high radar backscatter and appears bright, while an oil layer dampens
surface waves and creates a darker signature. This contrast is the physical basis for detecting
oil spills from space.
</p>
<div class='section-title'>Model Architecture — U-Net</div>
<p style='font-size:0.85rem;color:#9cb4cc;line-height:1.72;margin-bottom:14px;'>
The core of this system is a
<strong style='color:#4ea8de;'>U-Net convolutional neural network</strong>, originally developed
for biomedical image segmentation and widely adopted for remote sensing. It uses an
encoder-decoder structure with skip connections that preserve fine spatial detail.
The model outputs a per-pixel probability map indicating the likelihood of oil presence, which
is then thresholded to produce a binary segmentation mask.
</p>
<div class='section-title'>Detection Pipeline</div>
""", unsafe_allow_html=True)

    _steps = [
        ("Step 1 — Input",
         "A grayscale or single-band SAR image is uploaded. "
         "The image is read and decoded into a standard array format for processing."),
        ("Step 2 — Preprocessing",
         "The image is resized to 256 x 256 pixels and pixel values are normalised "
         "to the [0, 1] range to match the model’s training distribution."),
        ("Step 3 — U-Net Inference",
         "The normalised image is passed through the trained U-Net model, which "
         "produces a per-pixel probability map with values between 0 and 1."),
        ("Step 4 — Probability Map",
         "Each pixel receives a confidence score. Values near 1.0 indicate likely "
         "oil; values near 0.0 indicate clean ocean or land background."),
        ("Step 5 — Segmentation Mask",
         "A binary mask is created by applying a configurable threshold (default 0.5). "
         "Pixels above the threshold are classified as oil."),
        ("Step 6 — Analysis & Report",
         "Coverage metrics, spill boundaries, confidence scores, and historical trend "
         "data are computed and visualised. A detailed PDF report can be exported."),
    ]

    _c1, _c2, _c3 = st.columns(3)
    _cols = [_c1, _c2, _c3]
    for _i, (_title, _body) in enumerate(_steps):
        with _cols[_i % 3]:
            st.markdown(
                f"<div style='background:#0f2036;border:1px solid #1e304a;"
                f"border-radius:12px;padding:16px;margin-bottom:12px;'>"
                f"<div style='font-size:0.7rem;font-weight:700;color:#4ea8de;"
                f"letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;'>"
                f"{_title}</div>"
                f"<div style='font-size:0.82rem;color:#9cb4cc;line-height:1.6;'>"
                f"{_body}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("""
<div class='section-title' style='margin-top:4px;'>Loss Function — Weighted BCE + Dice</div>
<p style='font-size:0.85rem;color:#9cb4cc;line-height:1.72;'>
The model was trained using a combined loss:
<strong style='color:#4ea8de;'>Binary Cross-Entropy (BCE)</strong> for pixel-level
classification accuracy, and <strong style='color:#4ea8de;'>Dice Loss</strong> to maximise
the overlap between predicted and actual oil regions. This combination handles the class
imbalance problem, since oil pixels are far fewer than background pixels in most SAR images.
</p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  FILE UPLOAD
# ─────────────────────────────────────────
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Input Image</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a SAR image (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload a grayscale or RGB SAR satellite image for analysis.",
)
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  INFERENCE + DISPLAY
# ─────────────────────────────────────────
if uploaded_file is not None:

    # ── Preprocessing ──────────────────────
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_gray   = cv2.imdecode(file_bytes, 0)

    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_norm    = img_resized / 255.0
    pred        = model.predict(np.expand_dims(img_norm, axis=(0, -1)))[0].squeeze()

    mask          = pred > threshold
    oil_pixels    = int(np.sum(mask))
    total_pixels  = IMG_SIZE * IMG_SIZE
    oil_percentage = (oil_pixels / total_pixels) * 100
    confidence     = float(np.mean(pred[mask]) * 100) if oil_pixels > 0 else 0.0
    mean_prob      = float(np.mean(pred))
    max_prob       = float(np.max(pred))

    img_color = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay   = img_color.copy()
    overlay[mask] = [0, 60, 220]
    blended   = cv2.addWeighted(overlay, 0.55, img_color, 0.45, 0)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = img_color.copy()
    cv2.drawContours(boundary, contours, -1, (0, 210, 120), 2)

    num_regions = max(0, cv2.connectedComponents(mask.astype(np.uint8))[0] - 1)

    # ── Save to DB ──────────────────────────
    if collection is not None:
        _, buffer = cv2.imencode('.png', blended)
        collection.insert_one({
            "filename":       uploaded_file.name,
            "oil_percentage": float(oil_percentage),
            "confidence":     confidence,
            "image":          base64.b64encode(buffer).decode(),
            "timestamp":      datetime.datetime.utcnow(),
        })

    # ── Section: Image Grid ─────────────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Segmentation Output</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(img_gray, use_container_width=True, clamp=True)
        st.markdown("<div class='img-caption'>Original SAR</div>", unsafe_allow_html=True)
    with c2:
        st.image((mask.astype(np.uint8) * 255), use_container_width=True, clamp=True)
        st.markdown("<div class='img-caption'>Binary Mask</div>", unsafe_allow_html=True)
    with c3:
        st.image(blended, channels="BGR", use_container_width=True, clamp=True)
        st.markdown("<div class='img-caption'>Overlay</div>", unsafe_allow_html=True)
    with c4:
        st.image(boundary, channels="BGR", use_container_width=True, clamp=True)
        st.markdown("<div class='img-caption'>Boundaries</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section: Detection Explanation ──────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Detection Explanation</div>", unsafe_allow_html=True)

    # Build dynamic explanation text
    if oil_percentage > 5:
        severity_label = "HIGH-SEVERITY SPILL"
        severity_color = "#e53e3e"
        severity_desc = (
            f"The model has identified a <strong style='color:#e53e3e;'>significant oil spill</strong> covering "
            f"<strong>{oil_percentage:.2f}%</strong> of the analysed image area. "
            f"This corresponds to approximately <strong>{oil_pixels:,} pixels</strong>, indicating a large spatial footprint. "
            f"At this level, the contamination poses a serious and immediate threat to marine ecosystems. "
            f"Rapid containment measures, emergency response coordination, and regulatory notification are strongly advised."
        )
    elif oil_percentage > 1:
        severity_label = "MODERATE SPILL DETECTED"
        severity_color = "#dd6b20"
        severity_desc = (
            f"A <strong style='color:#dd6b20;'>moderate oil spill</strong> has been detected, covering "
            f"<strong>{oil_percentage:.2f}%</strong> of the image (<strong>{oil_pixels:,} pixels</strong>). "
            f"The spill has a noticeable spatial spread that may impact coastal or open-water ecosystems. "
            f"Active environmental monitoring, spill trajectory modelling, and preparedness for containment operations are recommended."
        )
    elif oil_percentage > 0.1:
        severity_label = "MINOR TRACES DETECTED"
        severity_color = "#3182ce"
        severity_desc = (
            f"The model detected <strong style='color:#3182ce;'>minor oil traces</strong> within the image, "
            f"covering <strong>{oil_percentage:.2f}%</strong> of the scene (<strong>{oil_pixels:,} pixels</strong>). "
            f"This could indicate an early-stage spill, surface sheen from natural seeps, or a remnant of a previously contained event. "
            f"Continued observation and follow-up imagery over the same region are advised to determine whether the spill is growing."
        )
    else:
        severity_label = "NO SPILL DETECTED"
        severity_color = "#38a169"
        severity_desc = (
            f"The model found <strong style='color:#38a169;'>no significant oil spill signature</strong> in this image. "
            f"Oil-classified pixels account for only <strong>{oil_percentage:.2f}%</strong> of the image — well below the alerting threshold. "
            f"The analysed area appears to be clean ocean or land with no detectable hydrocarbon contamination."
        )

    # Fragmentation assessment
    if num_regions == 0:
        frag_text = "No discrete spill regions were identified in the segmentation mask."
    elif num_regions == 1:
        frag_text = (
            "The spill appears as a single continuous region, suggesting a concentrated source "
            "with limited dispersion — consistent with a recent or contained spill event."
        )
    elif num_regions <= 4:
        frag_text = (
            f"{num_regions} discrete spill patches were identified. This fragmented pattern suggests the "
            "spill may be breaking apart due to wind, currents, or wave action, or could indicate "
            "multiple isolated seep or discharge points."
        )
    else:
        frag_text = (
            f"{num_regions} separate regions were detected. A highly fragmented pattern of this nature "
            "typically reflects advanced weathering of the spill, dispersal by ocean currents, "
            "or a distributed leakage source rather than a single point of origin."
        )

    # Confidence interpretation
    if confidence > 80:
        conf_text = (
            f"The model assigns a mean per-pixel confidence of <strong>{confidence:.1f}%</strong> across the "
            "detected oil regions, indicating high decisiveness in its predictions. "
            "The probability map shows sharp, well-defined boundaries between oil and background."
        )
    elif confidence > 40:
        conf_text = (
            f"The model shows moderate confidence (<strong>{confidence:.1f}%</strong>) across detected pixels. "
            "Some ambiguity exists at spill boundaries — this may reflect thin sheens, "
            "emulsified oil, or look-alike targets such as biogenic films or low-wind zones."
        )
    else:
        conf_text = (
            f"Confidence across detected pixels is relatively low (<strong>{confidence:.1f}%</strong>). "
            "Consider adjusting the segmentation threshold or validating with a higher-resolution image."
        )

    # Render the explanation
    st.markdown(
        f"""
        <div style='border-left:3px solid {severity_color};padding-left:18px;margin-bottom:18px;'>
            <div style='font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
                        text-transform:uppercase;color:{severity_color};margin-bottom:6px;'>{severity_label}</div>
            <p style='font-size:0.88rem;color:#c9d6e3;line-height:1.78;margin:0;'>{severity_desc}</p>
        </div>

        <div style='border-left:3px solid #4ea8de;padding-left:18px;margin-bottom:18px;'>
            <div style='font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
                        text-transform:uppercase;color:#4ea8de;margin-bottom:6px;'>Spatial Structure</div>
            <p style='font-size:0.88rem;color:#c9d6e3;line-height:1.78;margin:0;'>{frag_text}</p>
        </div>

        <div style='border-left:3px solid #805ad5;padding-left:18px;'>
            <div style='font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
                        text-transform:uppercase;color:#805ad5;margin-bottom:6px;'>Model Confidence</div>
            <p style='font-size:0.88rem;color:#c9d6e3;line-height:1.78;margin:0;'>{conf_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section: Detection Status ───────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Detection Status</div>", unsafe_allow_html=True)
    if oil_percentage > 5:
        st.markdown(
            "<div class='status-critical'>"
            "CRITICAL — High-severity spill detected. Immediate containment response is advised."
            "</div>", unsafe_allow_html=True)
    elif oil_percentage > 1:
        st.markdown(
            "<div class='status-warning'>"
            "MODERATE — Noticeable oil spill detected. Increased monitoring and assessment recommended."
            "</div>", unsafe_allow_html=True)
    elif oil_percentage > 0.1:
        st.markdown(
            "<div class='status-info'>"
            "MINOR — Small oil traces detected. Continued observation is advised."
            "</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='status-ok'>"
            "CLEAR — No significant oil spill detected in the analyzed image."
            "</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section: Key Metrics ────────────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Key Metrics</div>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        (m1, "Oil Coverage",   f"{oil_percentage:.2f}", "%"),
        (m2, "Oil Pixels",     f"{oil_pixels:,}",        "px"),
        (m3, "Confidence",     f"{confidence:.1f}",      "%"),
        (m4, "Mean Prob.",     f"{mean_prob:.3f}",       ""),
        (m5, "Spill Regions",  f"{num_regions}",         ""),
    ]
    for col, label, value, unit in metrics:
        with col:
            st.markdown(
                f"<div class='metric-box'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value'>{value}</div>"
                f"<div class='metric-unit'>{unit}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section: Charts (3-column) ──────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Visual Analytics</div>", unsafe_allow_html=True)
    ch1, ch2, ch3 = st.columns(3)

    # Chart 1 — Probability Heatmap
    with ch1:
        st.markdown("<p style='font-size:0.78rem;font-weight:600;color:#4ea8de;"
                    "text-align:center;text-transform:uppercase;letter-spacing:0.07em;"
                    "margin-bottom:10px;'>Probability Heatmap</p>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(4, 3.4))
        apply_dark_style(fig1, [ax1])
        im = ax1.imshow(pred, cmap="inferno", vmin=0, vmax=1)
        ax1.axis("off")
        cbar = fig1.colorbar(im, ax=ax1, fraction=0.040, pad=0.03)
        cbar.ax.tick_params(colors=TEXT_CLR, labelsize=7)
        cbar.outline.set_edgecolor(GRID_CLR)
        fig1.tight_layout(pad=0.5)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    # Chart 2 — Probability Distribution Histogram
    with ch2:
        st.markdown("<p style='font-size:0.78rem;font-weight:600;color:#4ea8de;"
                    "text-align:center;text-transform:uppercase;letter-spacing:0.07em;"
                    "margin-bottom:10px;'>Probability Distribution</p>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(4, 3.4))
        apply_dark_style(fig2, [ax2])
        vals = pred.flatten()
        n, bins, patches = ax2.hist(vals, bins=50, color=ACCENT2, edgecolor="none", alpha=0.85)
        # Color bars by value
        norm = mcolors.Normalize(vmin=bins[0], vmax=bins[-1])
        cmap = plt.cm.get_cmap("inferno")
        for patch, left_edge in zip(patches, bins[:-1]):
            patch.set_facecolor(cmap(norm(left_edge + (bins[1]-bins[0])/2)))
        ax2.axvline(threshold, color="#e53e3e", linewidth=1.5, linestyle="--", label=f"Threshold ({threshold})")
        ax2.set_xlabel("Probability", color=TEXT_CLR, fontsize=8)
        ax2.set_ylabel("Pixel Count", color=TEXT_CLR, fontsize=8)
        leg = ax2.legend(fontsize=7, facecolor=DARK_AX, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
        fig2.tight_layout(pad=0.5)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # Chart 3 — Oil vs Background Donut
    with ch3:
        st.markdown("<p style='font-size:0.78rem;font-weight:600;color:#4ea8de;"
                    "text-align:center;text-transform:uppercase;letter-spacing:0.07em;"
                    "margin-bottom:10px;'>Coverage Breakdown</p>", unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(4, 3.4))
        apply_dark_style(fig3, [ax3])
        bg_pix = total_pixels - oil_pixels
        sizes  = [oil_pixels, bg_pix]
        clrs   = ["#e53e3e", "#1a6fb5"]
        wedge_props = dict(width=0.52, edgecolor=DARK_BG, linewidth=2)
        wedges, texts, autotexts = ax3.pie(
            sizes, labels=["Oil", "Background"],
            colors=clrs, autopct="%1.1f%%",
            startangle=90, wedgeprops=wedge_props,
            textprops=dict(color=TEXT_CLR, fontsize=8),
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_color("#e6edf3")
            at.set_fontsize(8)
            at.set_fontweight("600")
        ax3.set(aspect="equal")
        fig3.tight_layout(pad=0.5)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section: Analytical Insights ────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Analytical Insights</div>", unsafe_allow_html=True)

    # Severity
    if oil_percentage > 5:
        severity_text = "High — Large spatial coverage; represents a major environmental hazard."
    elif oil_percentage > 1:
        severity_text = "Moderate — Noticeable spread; environmental impact likely. Monitor closely."
    elif oil_percentage > 0.1:
        severity_text = "Low — Minor traces present; early-stage or dispersed spill."
    else:
        severity_text = "None — No significant spill signature detected."

    # Density
    density = oil_pixels / total_pixels
    if density > 0.2:
        density_text = "Dense and concentrated — spill is well-defined with a compact footprint."
    elif density > 0.05:
        density_text = "Moderately dispersed — spill has partially fragmented boundaries."
    else:
        density_text = "Highly dispersed or trace-level — pixels are scattered across the image."

    # Confidence
    if confidence > 80:
        conf_text = "High — model predictions are highly decisive above the threshold."
    elif confidence > 50:
        conf_text = "Moderate — some uncertainty at spill boundaries."
    else:
        conf_text = "Low — marginal detections; consider re-evaluating the threshold."

    rows = [
        ("Severity",        severity_text),
        ("Spatial Density", density_text),
        ("Model Confidence",conf_text),
        ("Spill Regions",   f"{num_regions} discrete region(s) identified via connected-component analysis."),
        ("Max Probability", f"{max_prob:.4f} — highest per-pixel confidence score produced by the model."),
        ("Mean Probability",f"{mean_prob:.4f} — average confidence across all pixels in the image."),
    ]
    for label, value in rows:
        st.markdown(
            f"<div class='insight-row'>"
            f"<div class='insight-label'>{label}</div>"
            f"<div class='insight-value'>{value}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section: Downloads ───────────────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Export Results</div>", unsafe_allow_html=True)
    dl1, dl2, dl3 = st.columns(3)

    # Download overlay image
    with dl1:
        _, enc = cv2.imencode(".png", blended)
        st.download_button(
            label="Download Overlay Image",
            data=enc.tobytes(),
            file_name=f"oilspill_overlay_{uploaded_file.name}",
            mime="image/png",
            use_container_width=True,
        )

    # Download mask
    with dl2:
        _, menc = cv2.imencode(".png", mask.astype(np.uint8) * 255)
        st.download_button(
            label="Download Binary Mask",
            data=menc.tobytes(),
            file_name=f"oilspill_mask_{uploaded_file.name}",
            mime="image/png",
            use_container_width=True,
        )

    # PDF Report
    with dl3:
        if st.button("Generate PDF Report", use_container_width=True):
            with st.spinner("Building report..."):
                pdf_path  = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
                img_path  = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                cv2.imwrite(img_path, blended)

                doc    = SimpleDocTemplate(pdf_path, pagesize=letter)
                styles = getSampleStyleSheet()
                title_style = ParagraphStyle(
                    "CustomTitle",
                    parent=styles["Title"],
                    fontSize=18, spaceAfter=12,
                    textColor=rl_colors.HexColor("#1a6fb5"),
                )
                body_style = ParagraphStyle(
                    "CustomBody",
                    parent=styles["Normal"],
                    fontSize=11, spaceAfter=6,
                    textColor=rl_colors.HexColor("#333333"),
                )
                elements = [
                    Paragraph("Oil Spill Detection Report", title_style),
                    Spacer(1, 10),
                    Paragraph(f"File: {uploaded_file.name}", body_style),
                    Paragraph(f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", body_style),
                    Spacer(1, 12),
                    Paragraph(f"Oil Coverage: {oil_percentage:.2f}%", body_style),
                    Paragraph(f"Oil Pixels: {oil_pixels:,}", body_style),
                    Paragraph(f"Model Confidence: {confidence:.1f}%", body_style),
                    Paragraph(f"Mean Probability: {mean_prob:.4f}", body_style),
                    Paragraph(f"Max Probability: {max_prob:.4f}", body_style),
                    Paragraph(f"Spill Regions: {num_regions}", body_style),
                    Paragraph(f"Threshold Used: {threshold}", body_style),
                    Spacer(1, 16),
                    Paragraph("Segmentation Overlay:", body_style),
                    RLImage(img_path, width=320, height=320),
                ]
                doc.build(elements)

                with open(pdf_path, "rb") as pf:
                    st.download_button(
                        "Download PDF Report",
                        pf,
                        f"oilspill_report_{uploaded_file.name}.pdf",
                        use_container_width=True,
                    )
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  SECTION: PREDICTION HISTORY
# ─────────────────────────────────────────
if collection is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Prediction History</div>", unsafe_allow_html=True)
    with st.expander("View last 5 records", expanded=False):
        records = list(collection.find().sort("timestamp", -1).limit(5))
        if records:
            for rec in records:
                ts = rec.get("timestamp", "")
                ts_str = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
                st.markdown(
                    f"<div class='history-row'>"
                    f"<span class='history-name'>{rec.get('filename','N/A')}</span>"
                    f"<span style='color:#7b8ea6;font-size:0.8rem;'>{ts_str}</span>"
                    f"<span class='history-pct'>{rec.get('oil_percentage',0):.2f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if "image" in rec:
                    st.image(base64.b64decode(rec["image"]), width=220)
        else:
            st.markdown("<span style='font-size:0.85rem;color:#7b8ea6;'>No records found.</span>",
                        unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Trend Chart ─────────────────────────
    recs = list(collection.find().sort("timestamp", 1))
    if len(recs) > 1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Historical Trend — Oil Coverage (%)</div>",
                    unsafe_allow_html=True)
        df = pd.DataFrame({
            "Time": [r["timestamp"] for r in recs],
            "Oil Coverage (%)": [r["oil_percentage"] for r in recs],
        })
        fig_t, ax_t = plt.subplots(figsize=(10, 3))
        apply_dark_style(fig_t, [ax_t])
        ax_t.plot(df["Time"], df["Oil Coverage (%)"],
                  color=ACCENT, linewidth=2, marker="o", markersize=5,
                  markerfacecolor="#e53e3e", markeredgecolor=DARK_BG, markeredgewidth=1.5)
        ax_t.fill_between(df["Time"], df["Oil Coverage (%)"], alpha=0.12, color=ACCENT)
        ax_t.set_xlabel("Timestamp", color=TEXT_CLR, fontsize=8)
        ax_t.set_ylabel("Oil Coverage (%)", color=TEXT_CLR, fontsize=8)
        fig_t.autofmt_xdate(rotation=30)
        fig_t.tight_layout(pad=0.6)
        st.pyplot(fig_t, use_container_width=True)
        plt.close(fig_t)
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:40px 0 10px;color:#2d4055;font-size:0.78rem;'>
    Oil Spill Detection System &nbsp;|&nbsp; U-Net Semantic Segmentation &nbsp;|&nbsp; SAR Imagery Analysis
</div>
""", unsafe_allow_html=True)