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


# ---------- PROFESSIONAL SUMMARY WITH REVEAL ANIMATION ----------

def animated_summary(summary_text):
    components.html(f"""
    <style>
    body {{
        margin: 0;
        padding: 0;
        background: transparent;
    }}

    #summary-wrapper {{
        margin-top: 40px;
        perspective: 1000px;
    }}

    #summary-box {{
        padding: 36px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(15,30,60,0.95), rgba(10,20,45,0.95));
        border: 1px solid rgba(0,200,255,0.3);
        box-shadow: 
            0 20px 60px rgba(0,0,0,0.6),
            inset 0 1px 0 rgba(255,255,255,0.1);
        backdrop-filter: blur(16px);
        position: relative;
        overflow: hidden;
        transform: scale(0.9);
        opacity: 0;
        animation: boxReveal 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
    }}

    @keyframes boxReveal {{
        0% {{
            transform: scale(0.9) rotateX(10deg);
            opacity: 0;
        }}
        100% {{
            transform: scale(1) rotateX(0deg);
            opacity: 1;
        }}
    }}

    /* LEFT CURTAIN */
    #summary-box::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 50%;
        height: 100%;
        background: linear-gradient(90deg, #0a0e1a 0%, transparent 100%);
        z-index: 10;
        animation: curtainLeft 1.2s ease-out forwards 0.3s;
    }}

    @keyframes curtainLeft {{
        0% {{ transform: translateX(0); }}
        100% {{ transform: translateX(-100%); }}
    }}

    /* RIGHT CURTAIN */
    #summary-box::after {{
        content: "";
        position: absolute;
        top: 0;
        right: 0;
        width: 50%;
        height: 100%;
        background: linear-gradient(-90deg, #0a0e1a 0%, transparent 100%);
        z-index: 10;
        animation: curtainRight 1.2s ease-out forwards 0.3s;
    }}

    @keyframes curtainRight {{
        0% {{ transform: translateX(0); }}
        100% {{ transform: translateX(100%); }}
    }}

    /* SCANNING LIGHT - APPEARS AFTER CURTAINS */
    .scan-light {{
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent 0%,
            rgba(0,200,255,0.1) 45%,
            rgba(0,200,255,0.3) 50%,
            rgba(0,200,255,0.1) 55%,
            transparent 100%
        );
        z-index: 5;
        animation: scanSweep 2s ease-in-out 1.5s forwards;
    }}

    @keyframes scanSweep {{
        0% {{ left: -100%; }}
        100% {{ left: 100%; }}
    }}

    /* TITLE SECTION */
    .summary-header {{
        position: relative;
        z-index: 2;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 2px solid rgba(0,200,255,0.3);
        opacity: 0;
        transform: translateY(-20px);
        animation: headerSlide 0.6s ease forwards 1.6s;
    }}

    @keyframes headerSlide {{
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    .summary-title {{
        font-size: 1.4em;
        font-weight: 600;
        color: #00d9ff;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 0;
        text-shadow: 0 0 20px rgba(0,217,255,0.5);
    }}

    /* CONTENT */
    #summary-content {{
        position: relative;
        z-index: 2;
        line-height: 1.9;
        color: #e6eef8;
        font-size: 1.05em;
        opacity: 0;
        transform: translateX(-30px);
        animation: contentSlide 0.8s ease forwards 1.8s;
    }}

    @keyframes contentSlide {{
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}

    /* GRADIENT WAVE BACKGROUND */
    .wave-bg {{
        position: absolute;
        bottom: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at center, 
            rgba(0,200,255,0.08) 0%,
            rgba(100,50,255,0.05) 50%,
            transparent 70%
        );
        opacity: 0;
        animation: waveReveal 1.5s ease forwards 2s, waveMove 8s ease-in-out infinite 3.5s;
        z-index: 0;
    }}

    @keyframes waveReveal {{
        to {{ opacity: 1; }}
    }}

    @keyframes waveMove {{
        0%, 100% {{ 
            transform: translate(0%, 0%) scale(1); 
        }}
        33% {{ 
            transform: translate(5%, -5%) scale(1.05); 
        }}
        66% {{ 
            transform: translate(-5%, 5%) scale(0.95); 
        }}
    }}

    /* CORNER ACCENTS */
    .corner-accent {{
        position: absolute;
        width: 40px;
        height: 40px;
        border: 2px solid rgba(0,200,255,0.5);
        z-index: 3;
        opacity: 0;
        animation: cornerFade 0.5s ease forwards 2.2s;
    }}

    @keyframes cornerFade {{
        to {{ opacity: 1; }}
    }}

    .corner-tl {{
        top: 10px;
        left: 10px;
        border-right: none;
        border-bottom: none;
        border-radius: 4px 0 0 0;
    }}

    .corner-br {{
        bottom: 10px;
        right: 10px;
        border-left: none;
        border-top: none;
        border-radius: 0 0 4px 0;
    }}
    </style>

    <div id="summary-wrapper">
        <div id="summary-box">
            <div class="scan-light"></div>
            <div class="wave-bg"></div>
            <div class="corner-accent corner-tl"></div>
            <div class="corner-accent corner-br"></div>

            <div class="summary-header">
                <h3 class="summary-title">📊 Analysis Summary</h3>
            </div>

            <div id="summary-content">
                {summary_text}
            </div>
        </div>
    </div>

    <script>
    setTimeout(() => {{
        document.getElementById("summary-wrapper").scrollIntoView({{ 
            behavior: "smooth", 
            block: "center" 
        }});
    }}, 400);
    </script>
    """, height=380)

# --- End UI/Animation Injection ---