import streamlit as st
import base64

def load_base64_file(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

def homepage():
    # Load images and video
    logo_data = load_base64_file("static/images/logo.png")
    about_img_data = load_base64_file("static/images/about.jpg")
    background_img_data = load_base64_file("static/images/background.jpg")
    video_data = load_base64_file("static/video/background.mp4")

    st.set_page_config(page_title="Voice‑Viz RAG", layout="wide")

    # HTML & CSS
    st.markdown(f"""
        <style>
            html, body, .stApp {{
                margin: 0;
                padding: 0;
                overflow-x: hidden;
                background-image: url("data:image/jpg;base64,{background_img_data}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
        }}
            .nav-bar {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                z-index: 1000;
                background-color: white;
                padding: 1.2rem 2rem;
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
            }}

        .nav-left img {{
                height: 60px;
            }}
            .nav-right a {{
                margin-left: 30px;
                text-decoration: none;
                color: #00264d;
                font-weight: 600;
                font-size: 17px;
            }}
            .nav-right a:hover {{
                color: #0073e6;
            }}

        .hero-section {{
            position: relative;
            height: 100vh;
            width: 100%;
            overflow: hidden;
        }}

        .hero-video {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            z-index: 1;
        }}

        .hero-overlay {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                color: white;
                z-index: 2;
                text-align: center;
                padding: 0 20px;
                background-color: rgba(0, 0, 0, 0.4);
            }}

        .hero-overlay h1 {{
                font-size: 3rem;
                font-weight: bold;
                text-shadow: 2px 2px 6px #000000;
            }}
            .hero-overlay p {{
                font-size: 1.5rem;
                margin-top: 1rem;
                text-shadow: 1px 1px 4px #000000;
            }}

        .start-btn-wrapper {{
                margin-top: 2rem;
                z-index: 3;
                text-align: center;
            }}
            .start-btn-wrapper button {{
                padding: 0.8rem 2.2rem;
                background-color: white;
                color: #0074D9;
                border: none;
                border-radius: 8px;
                font-size: 1.2rem;
                font-weight: bold;
                cursor: pointer;
                transition: 0.3s;
            }}

        .start-btn-wrapper button:hover {{
                background-color: #0074D9;
                color: white;
        }}
        .split-section {{
                display: flex;
                flex-wrap: wrap;
                background-color: rgba(255,255,255,0.95);
                border-radius: 12px;
                margin: 2rem auto;
                padding: 2rem;
                width: 90%;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
                align-items: center;
                justify-content: space-between;
            }}
            .split-section img {{
                max-width: 45%;
                height: 450px;
                object-fit: cover;
                border-radius: 10px;
                flex: 1;
            }}
            .split-content {{
                flex: 1;
                padding: 1.5rem;
                color: #00264d;
                font-size: 1.05rem;
            }}
            .split-content h2 {{
                font-size: 1.7rem;
                margin-bottom: 1rem;
                color: #00264d;
            }}

        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        header {{ visibility: hidden; }}
    </style>

    <!-- NAV BAR -->
    <div class="nav-bar">
        <div class="nav-left">
            <img src="data:image/png;base64,{logo_data}" alt="Logo">
        </div>
        <div class="nav-right">
            <a href="#about">About</a>
        </div>
    </div>

    <!-- HERO SECTION -->
    <div class="hero-section">
        <video class="hero-video" autoplay muted loop>
            <source src="data:video/mp4;base64,{video_data}" type="video/mp4">
        </video>
        <div class="hero-overlay">
            <h1>Welcome to Swetcha AI</h1>
            <p>Your smart voice-based PDF assistant to explore, ask, and understand any document.</p>
            <div class="start-btn-wrapper">
                <form action="" method="post">
                </form>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)


    # Start Generating Button
    with st.container():
        st.markdown('<div class="start-btn-wrapper">', unsafe_allow_html=True)
        if st.button("🚀 Start Generating"):
            st.session_state.start_app = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # About Section
    st.markdown('<hr id="about">', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="split-section">
            <img src="data:image/jpg;base64,{about_img_data}" alt="About Voice‑Viz">
            <div class="split-content">
                <h2>\U0001F4AC About This App</h2>
                <p><b>Voice‑Viz RAG</b> is a multilingual, voice-enabled helpdesk powered by Generative AI and Retrieval-Augmented Generation. Upload manuals, ask questions, and receive clear answers in your language — via voice or text.</p>
                <ul style="list-style: none; padding-left: 0;">
                    <li>\u2705 Multilingual PDF understanding</li>
                    <li>\u2705 Voice & text-based Q&A</li>
                    <li>\u2705 Real-time AI video avatars</li>
                    <li>\u2705 Audio response with waveform</li>
                    <li>\u2705 SQLite-powered interaction history</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)
