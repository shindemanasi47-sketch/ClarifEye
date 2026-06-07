import os
import streamlit as st
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="ClarifEye · DR Grading",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

:root {
    --bg:        #f4f6f5;
    --surface:   #ffffff;
    --surface2:  #eef1f0;
    --border:    #dce3e1;
    --teal:      #2a7f6f;
    --teal-lt:   #d6eeea;
    --teal-md:   #4aab96;
    --slate:     #3d5450;
    --muted:     #7a9490;
    --text:      #1c2e2b;
    --text-soft: #4f6662;
    --grade0:    #2e7d5e;
    --grade1:    #6a9e3f;
    --grade2:    #b07d2a;
    --grade3:    #c0522a;
    --grade4:    #a83232;
    --r0:        #e8f5ef;
    --r1:        #f0f6e8;
    --r2:        #fdf3e3;
    --r3:        #fdf0e8;
    --r4:        #fdeaea;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
    padding-top: 2rem;
}
section[data-testid="stSidebar"] * { color: var(--text-soft) !important; }
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: var(--teal) !important; font-weight: 600; }

.main .block-container { background: var(--bg); padding-top: 2rem; }

.wordmark {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: var(--teal);
    letter-spacing: -0.5px;
    line-height: 1;
}
.wordmark-sub {
    font-size: 0.95rem;
    color: var(--muted);
    font-weight: 400;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 4px;
}
.header-bar {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}
.header-tag {
    background: var(--teal-lt);
    color: var(--teal);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
}

.metric-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
.metric-card {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
}
.metric-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: var(--teal);
    line-height: 1;
}
.metric-lbl { font-size: 0.78rem; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.06em; }

.upload-zone {
    background: var(--surface);
    border: 1.5px dashed var(--border);
    border-radius: 12px;
    padding: 2.5rem;
    text-align: center;
    color: var(--muted);
}
.upload-zone h4 { color: var(--teal); font-size: 1.05rem; margin: 0 0 6px; }

.result-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.result-box {
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin: 0.5rem 0 1.2rem;
    border-left: 4px solid;
}
.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    line-height: 1.1;
    margin: 0 0 6px;
}
.result-advice { font-size: 0.88rem; line-height: 1.6; color: var(--text-soft); margin-top: 8px; }

.section-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin: 1.6rem 0 1rem;
}

.image-label {
    font-size: 0.82rem;
    color: var(--muted);
    text-align: center;
    margin-top: 6px;
}

.grade-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.87rem;
}
.grade-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }

.disclaimer {
    background: #fffbea;
    border: 1px solid #e8d98a;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #6b5e20;
    margin-top: 1rem;
    line-height: 1.5;
}

.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2.5rem 0;
}

.footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.77rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
    letter-spacing: 0.04em;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────
CLASS_NAMES    = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
GRADE_COLORS   = ['#2e7d5e', '#6a9e3f', '#b07d2a', '#c0522a', '#a83232']
GRADE_BG       = ['#e8f5ef', '#f0f6e8', '#fdf3e3', '#fdf0e8', '#fdeaea']
SEVERITY_LABEL = ['None', 'Low', 'Moderate', 'High', 'Critical']
ADVICE = [
    "No signs of diabetic retinopathy were detected. Maintain routine annual ophthalmic examinations and glycaemic control.",
    "Early signs of mild NPDR present. A follow-up ophthalmic review is recommended within 12 months.",
    "Moderate non-proliferative changes identified. Ophthalmologist consultation advised within 6 months.",
    "Severe NPDR detected. Urgent ophthalmologist referral is recommended within 1 month.",
    "Proliferative diabetic retinopathy identified. Immediate specialist evaluation is required."
]

# ─── Model ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import gdown
    import os
    if not os.path.exists('best_dr_finetuned.keras'):
        gdown.download(
            'https://drive.google.com/uc?id=1AHb4caKZbNXBOg9u0xROjJvU1pVgqJk-',
            'best_dr_finetuned.keras',
            quiet=False
        )
    return tf.keras.models.load_model('best_dr_finetuned.keras')

def predict(img: Image.Image, model):
    arr = np.array(img.resize((128, 128)).convert('RGB')) / 255.0
    probs = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
    return probs

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ClarifEye")
    st.markdown("### DR Grading System")
    st.markdown("---")

    st.markdown("**Model Specifications**")
    specs = {
        "Architecture": "MobileNetV2",
        "Input Size": "128 × 128 px",
        "Classes": "5",
        "Kappa Score": "0.71",
        "Accuracy": "68%",
        "Fine-tuned Layers": "30"
    }
    for k, v in specs.items():
        st.markdown(f"<div style='display:flex;justify-content:space-between;"
                    f"padding:5px 0;border-bottom:1px solid #eee;font-size:0.85rem'>"
                    f"<span>{k}</span><strong style='color:#2a7f6f'>{v}</strong></div>",
                    unsafe_allow_html=True)

    st.markdown("<div class='section-title' style='margin-top:1.5rem'>DR Grade Reference</div>",
                unsafe_allow_html=True)
    grades_info = [
        ("#2e7d5e", "Grade 0", "No DR"),
        ("#6a9e3f", "Grade 1", "Mild NPDR"),
        ("#b07d2a", "Grade 2", "Moderate NPDR"),
        ("#c0522a", "Grade 3", "Severe NPDR"),
        ("#a83232", "Grade 4", "Proliferative"),
    ]
    for color, grade, name in grades_info:
        st.markdown(
            f"<div class='grade-row'>"
            f"<div class='grade-dot' style='background:{color}'></div>"
            f"<span style='color:#7a9490;font-size:0.8rem'>{grade}</span>"
            f"<span style='margin-left:auto;font-size:0.85rem;color:#1c2e2b'>{name}</span>"
            f"</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='disclaimer'>
    <strong>Research Use Only.</strong> This tool is not a substitute for clinical diagnosis.
    Always consult a qualified ophthalmologist for medical decisions.
    </div>
    """, unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <div>
        <div class="wordmark">ClarifEye</div>
        <div class="wordmark-sub">Diabetic Retinopathy Grading System</div>
    </div>
    <div class="header-tag">Research Preview</div>
</div>
""", unsafe_allow_html=True)

# ─── Metric Row ───────────────────────────────────────────────
st.markdown("""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-val">68%</div>
        <div class="metric-lbl">Classification Accuracy</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">0.71</div>
        <div class="metric-lbl">Quadratic Weighted Kappa</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">5</div>
        <div class="metric-lbl">Severity Grades</div>
    </div>
    <div class="metric-card">
        <div class="metric-val">MNV2</div>
        <div class="metric-lbl">Base Architecture</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Upload ───────────────────────────────────────────────────
st.markdown("<div class='section-title'>Analysis</div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload one or more retinal fundus images (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# ─── Results ──────────────────────────────────────────────────
if uploaded_files:
    model = load_model()

    for i, uploaded_file in enumerate(uploaded_files):
        img   = Image.open(uploaded_file)
        probs = predict(img, model)
        idx   = int(np.argmax(probs))
        color = GRADE_COLORS[idx]
        bg    = GRADE_BG[idx]

        if i > 0:
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        st.markdown(f"<div class='section-title'>Image {i+1} — {uploaded_file.name}</div>",
                    unsafe_allow_html=True)

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.image(img, use_container_width=True)
            st.markdown(f"<div class='image-label'>{uploaded_file.name}</div>",
                        unsafe_allow_html=True)

        with right:
            st.markdown("<div class='section-title'>Diagnosis</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-box" style="background:{bg};border-color:{color}">
                <div class="result-pill" style="background:{color}20;color:{color}">
                    Severity · {SEVERITY_LABEL[idx]}
                </div>
                <div class="result-label" style="color:{color}">{CLASS_NAMES[idx]}</div>
                <div class="result-advice">{ADVICE[idx]}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='section-title'>Confidence by Grade</div>",
                        unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                x=probs * 100,
                y=CLASS_NAMES,
                orientation='h',
                marker=dict(color=GRADE_COLORS, line=dict(width=0)),
                text=[f"{p*100:.1f}%" for p in probs],
                textposition='outside',
                cliponaxis=False
            ))
            fig.update_layout(
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font=dict(color='#4f6662', family='DM Sans', size=12),
                xaxis=dict(range=[0, 115], showgrid=False, zeroline=False,
                           ticksuffix='%', tickfont=dict(size=11)),
                yaxis=dict(showgrid=False, tickfont=dict(size=12)),
                margin=dict(l=0, r=40, t=10, b=10),
                height=230
            )
            st.plotly_chart(fig, use_container_width=True, key=f"bar_{i}")

        # ─── Donut ────────────────────────────────────────────
        st.markdown("<div class='section-title'>Probability Distribution</div>",
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            fig2 = go.Figure(go.Pie(
                labels=CLASS_NAMES,
                values=probs,
                hole=0.6,
                marker=dict(colors=GRADE_COLORS, line=dict(color='#ffffff', width=2)),
                textinfo='label+percent',
                hovertemplate='%{label}: %{percent}<extra></extra>',
                textfont=dict(family='DM Sans', size=12)
            ))
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#4f6662', family='DM Sans'),
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
                height=320,
                annotations=[dict(
                    text=f"<b>{CLASS_NAMES[idx]}</b>",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=15, color=color, family='DM Serif Display')
                )]
            )
            st.plotly_chart(fig2, use_container_width=True, key=f"donut_{i}")

else:
    st.markdown("""
    <div class="upload-zone">
        <h4>Upload one or more retinal fundus photographs to begin grading</h4>
        <p style="font-size:0.85rem;margin:0">Supported formats: PNG · JPG · JPEG · Multiple files allowed</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title' style='margin-top:2rem'>Model Comparison</div>",
                unsafe_allow_html=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name='Accuracy (%)', x=['Basic CNN', 'MobileNetV2'], y=[39, 68],
        marker_color=['#b0c4bf', '#2a7f6f'],
        text=['39%', '68%'], textposition='outside'
    ))
    fig3.add_trace(go.Bar(
        name='Kappa × 100', x=['Basic CNN', 'MobileNetV2'], y=[29, 71],
        marker_color=['#c8d8d4', '#4aab96'],
        text=['0.29', '0.71'], textposition='outside'
    ))
    fig3.update_layout(
        barmode='group',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(color='#4f6662', family='DM Sans', size=12),
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=1.1),
        yaxis=dict(showgrid=True, gridcolor='#eef1f0', range=[0, 85], zeroline=False),
        xaxis=dict(showgrid=False),
        margin=dict(t=30, b=20),
        height=340
    )
    st.plotly_chart(fig3, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    ClarifEye · Diabetic Retinopathy Grading System &nbsp;·&nbsp;
    Built with Streamlit &amp; TensorFlow &nbsp;·&nbsp; Research Use Only
</div>
""", unsafe_allow_html=True)
