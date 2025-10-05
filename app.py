import streamlit as st
from image_tab import image_tab
from video_tab import video_tab
from map_tab import map_tab
from ultralytics import YOLO

# Konfigurasi halaman
st.set_page_config(
    page_title="AI Damage Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling dan header
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }

    .stTitle {
        color: #1f77b4;
        font-size: 3rem !important;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 class="stTitle">🔍 AI Road Damage Detection System</h1>
    <p style="font-size: 1.2rem; color: #666; margin-top: -1rem;">
        Deteksi kerusakan jalan otomatis menggunakan YOLO11 dengan pemetaan GPS
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar konfigurasi
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <h3>⚙️ Pengaturan</h3>
        <p>Konfigurasi model dan parameter deteksi</p>
    </div>
    """, unsafe_allow_html=True)

    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    st.markdown("---")
    st.markdown("""
    ###  Info Aplikasi
    - **Model**: YOLO11
    - **Format Support**: 
      -  Gambar: JPG, PNG, JPEG
      -  Video: MP4, MOV, AVI
    - **GPS**: Auto-extract dari EXIF
    - **Status**: Ready
    """)

# Session state inisialisasi
if 'location_data' not in st.session_state:
    st.session_state.location_data = []

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("model/best.pt")  

model = load_model()
if model is None:
    st.stop()

# Menu Utama
tab1, tab2, tab3 = st.tabs([" Upload Gambar", " Upload Video", 
                            " Peta Lokasi"])
with tab1:
    image_tab(model, confidence_threshold)
with tab2:
    video_tab(model, confidence_threshold)
with tab3:
    map_tab()
