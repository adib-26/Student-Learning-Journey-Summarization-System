# --- UI/Animation Injection (place after st.set_page_config and before st.title) ---
import streamlit as st
import streamlit.components.v1 as components


def inject_ui_animations():
    st.markdown("""
    <style>

    /* ===============================
       WORKING INITIAL PAGE GLOW
    =============================== */

    @keyframes pageGlow {
        0% { 
            background: radial-gradient(circle at 50% 50%, rgba(0,200,255,0.4), transparent 50%);
            opacity: 0;
        }
        50% { 
            background: radial-gradient(circle at 50% 50%, rgba(0,200,255,0.35), transparent 60%);
            opacity: 1;
        }
        100% { 
            background: radial-gradient(circle at 50% 50%, rgba(0,200,255,0.15), transparent 70%);
            opacity: 1;
        }
    }

    .stApp {
        position: relative;
        background: #0a0e1a;
        color: #e6eef8;
        min-height: 100vh;
        animation: pageGlow 2s ease forwards;
    }

    /* ===============================
       CLEAN GLASS CONTAINER
    =============================== */

    .main .block-container {
        background: rgba(255,255,255,0.02);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        animation: containerFade 0.8s ease forwards;
    }

    @keyframes containerFade {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ===============================
       IMPROVED METRICS
    =============================== */

    .stMetric {
        border-radius: 12px;
        padding: 18px;
        background: linear-gradient(135deg, rgba(0,150,255,0.08), rgba(80,40,200,0.08));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }

    .stMetric:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,150,255,0.25);
        border-color: rgba(0,200,255,0.3);
    }

    /* ===============================
       CHART ANIMATIONS
    =============================== */

    .stPlotlyChart, .stAltairChart, .stVegaLiteChart {
        opacity: 0;
        transform: translateY(30px);
        animation: chartSlide 0.8s ease forwards;
        animation-delay: 0.2s;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
    }

    @keyframes chartSlide {
        to { opacity: 1; transform: translateY(0); }
    }

    .stPlotlyChart:hover, .stAltairChart:hover, .stVegaLiteChart:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(0,150,255,0.2);
    }

    </style>
    """, unsafe_allow_html=True)


# ---------- PREMIUM SUMMARY WITH ADVANCED ANIMATIONS ----------

def animated_summary(summary_text):
    components.html(f"""
    <style>
    body {{
        margin: 0;
        padding: 0;
        background: transparent;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}

    #summary-wrapper {{
        margin-top: 30px;
        padding: 10px;
    }}

    #summary-box {{
        position: relative;
        overflow: hidden;

        padding: 32px;
        border-radius: 24px;

        background:
            linear-gradient(
                145deg,
                rgba(17, 24, 39, 0.96),
                rgba(10, 15, 30, 0.96)
            );

        border: 1px solid rgba(255,255,255,0.08);

        box-shadow:
            0 10px 40px rgba(0,0,0,0.45),
            inset 0 1px 0 rgba(255,255,255,0.04);

        backdrop-filter: blur(18px);

        opacity: 0;
        transform: translateY(20px);
        animation: fadeUp 0.8s ease forwards;
    }}

    @keyframes fadeUp {{
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    /* Soft glow */
    #summary-box::before {{
        content: "";
        position: absolute;
        width: 300px;
        height: 300px;

        top: -120px;
        right: -120px;

        background: radial-gradient(
            circle,
            rgba(0, 200, 255, 0.18),
            transparent 70%
        );

        filter: blur(40px);
        pointer-events: none;
    }}

    /* Top accent line */
    .top-accent {{
        position: absolute;
        top: 0;
        left: 0;

        width: 100%;
        height: 3px;

        background: linear-gradient(
            90deg,
            #00d4ff,
            #7c4dff,
            #00d4ff
        );

        background-size: 200% auto;

        animation: shimmer 6s linear infinite;
    }}

    @keyframes shimmer {{
        0% {{
            background-position: 0% center;
        }}
        100% {{
            background-position: 200% center;
        }}
    }}

    .summary-header {{
        display: flex;
        align-items: center;
        gap: 12px;

        margin-bottom: 22px;
        position: relative;
        z-index: 2;
    }}

    .summary-icon {{
        width: 42px;
        height: 42px;

        border-radius: 12px;

        display: flex;
        align-items: center;
        justify-content: center;

        background:
            linear-gradient(
                135deg,
                rgba(0,212,255,0.18),
                rgba(124,77,255,0.18)
            );

        border: 1px solid rgba(255,255,255,0.08);

        font-size: 20px;

        box-shadow:
            0 4px 15px rgba(0,212,255,0.15);
    }}

    .summary-title {{
        color: white;
        font-size: 1.2rem;
        font-weight: 650;
        letter-spacing: 0.3px;
    }}

    #summary-content {{
        position: relative;
        z-index: 2;

        color: rgba(230,236,245,0.92);

        font-size: 1rem;
        line-height: 1.8;

        letter-spacing: 0.2px;
    }}

    /* subtle hover */
    #summary-box:hover {{
        transform: translateY(-2px);
        transition: all 0.3s ease;
        box-shadow:
            0 16px 50px rgba(0,0,0,0.5),
            0 0 25px rgba(0,212,255,0.08);
    }}

    </style>

    <div id="summary-wrapper">
        <div id="summary-box">

            <div class="top-accent"></div>

            <div class="summary-header">
                <div class="summary-icon">🔖</div>
                <div class="summary-title">
                    Analysis Summary
                </div>
            </div>

            <div id="summary-content">
                {summary_text}
            </div>

        </div>
    </div>

    <script>
    setTimeout(() => {{
        document.getElementById("summary-wrapper")
        .scrollIntoView({{
            behavior: "smooth",
            block: "center"
        }});
    }}, 300);
    </script>

    """, height=350)
    # --- End UI/Animation Injection ---