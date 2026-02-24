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
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    #summary-wrapper {{
        margin-top: 40px;
        perspective: 1200px;
    }}

    #summary-box {{
        padding: 40px;
        border-radius: 20px;
        background: linear-gradient(135deg, 
            rgba(15,30,60,0.98) 0%, 
            rgba(10,20,45,0.98) 50%,
            rgba(20,25,50,0.98) 100%);
        border: 1px solid rgba(0,200,255,0.4);
        box-shadow: 
            0 25px 70px rgba(0,0,0,0.7),
            0 10px 30px rgba(0,200,255,0.1),
            inset 0 1px 0 rgba(255,255,255,0.15);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
        transform: scale(0.92) rotateX(15deg);
        opacity: 0;
        animation: boxReveal 1s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
    }}

    @keyframes boxReveal {{
        0% {{
            transform: scale(0.92) rotateX(15deg);
            opacity: 0;
        }}
        100% {{
            transform: scale(1) rotateX(0deg);
            opacity: 1;
        }}
    }}

    /* ANIMATED GRID BACKGROUND */
    .grid-bg {{
        position: absolute;
        inset: 0;
        background-image: 
            linear-gradient(rgba(0,200,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,200,255,0.03) 1px, transparent 1px);
        background-size: 30px 30px;
        opacity: 0;
        animation: gridFade 1s ease forwards 0.5s;
        z-index: 0;
    }}

    @keyframes gridFade {{
        to {{ opacity: 1; }}
    }}

    /* LEFT CURTAIN WITH GRADIENT */
    #summary-box::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 50%;
        height: 100%;
        background: linear-gradient(90deg, 
            #0a0e1a 0%, 
            rgba(10,14,26,0.95) 70%,
            transparent 100%);
        z-index: 10;
        animation: curtainLeft 1.4s cubic-bezier(0.65, 0, 0.35, 1) forwards 0.4s;
        box-shadow: 10px 0 30px rgba(0,0,0,0.5);
    }}

    @keyframes curtainLeft {{
        0% {{ transform: translateX(0); }}
        100% {{ transform: translateX(-100%); }}
    }}

    /* RIGHT CURTAIN WITH GRADIENT */
    #summary-box::after {{
        content: "";
        position: absolute;
        top: 0;
        right: 0;
        width: 50%;
        height: 100%;
        background: linear-gradient(-90deg, 
            #0a0e1a 0%, 
            rgba(10,14,26,0.95) 70%,
            transparent 100%);
        z-index: 10;
        animation: curtainRight 1.4s cubic-bezier(0.65, 0, 0.35, 1) forwards 0.4s;
        box-shadow: -10px 0 30px rgba(0,0,0,0.5);
    }}

    @keyframes curtainRight {{
        0% {{ transform: translateX(0); }}
        100% {{ transform: translateX(100%); }}
    }}

    /* DOUBLE SCANNING LIGHT */
    .scan-light {{
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent 0%,
            rgba(0,200,255,0.08) 40%,
            rgba(0,200,255,0.25) 50%,
            rgba(0,200,255,0.08) 60%,
            transparent 100%
        );
        z-index: 5;
        animation: scanSweep 2.2s ease-in-out 1.8s forwards;
        filter: blur(2px);
    }}

    @keyframes scanSweep {{
        0% {{ left: -100%; }}
        100% {{ left: 100%; }}
    }}

    .scan-light-2 {{
        position: absolute;
        top: 0;
        left: -100%;
        width: 50%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent 0%,
            rgba(100,50,255,0.15) 50%,
            transparent 100%
        );
        z-index: 5;
        animation: scanSweep2 2.5s ease-in-out 2.2s forwards;
        filter: blur(3px);
    }}

    @keyframes scanSweep2 {{
        0% {{ left: -100%; }}
        100% {{ left: 110%; }}
    }}

    /* HOLOGRAPHIC PARTICLES */
    .particle {{
        position: absolute;
        width: 2px;
        height: 2px;
        background: #00d9ff;
        border-radius: 50%;
        box-shadow: 0 0 8px #00d9ff, 0 0 15px #00d9ff;
        opacity: 0;
        animation: particleFloat 4s ease-in-out infinite;
        z-index: 1;
    }}

    .particle:nth-child(1) {{ left: 10%; animation-delay: 2.5s; }}
    .particle:nth-child(2) {{ left: 25%; animation-delay: 2.8s; }}
    .particle:nth-child(3) {{ left: 40%; animation-delay: 3.1s; }}
    .particle:nth-child(4) {{ left: 55%; animation-delay: 3.4s; }}
    .particle:nth-child(5) {{ left: 70%; animation-delay: 3.7s; }}
    .particle:nth-child(6) {{ left: 85%; animation-delay: 4s; }}

    @keyframes particleFloat {{
        0% {{ 
            bottom: -10px; 
            opacity: 0; 
            transform: scale(0) translateX(0);
        }}
        10% {{ opacity: 0.8; transform: scale(1) translateX(5px); }}
        50% {{ transform: translateX(-10px); }}
        90% {{ opacity: 0.8; transform: scale(0.8) translateX(5px); }}
        100% {{ 
            bottom: 110%; 
            opacity: 0; 
            transform: scale(0) translateX(0);
        }}
    }}

    /* TITLE SECTION WITH GLOW */
    .summary-header {{
        position: relative;
        z-index: 2;
        margin-bottom: 24px;
        padding-bottom: 18px;
        border-bottom: 2px solid transparent;
        border-image: linear-gradient(90deg, 
            transparent, 
            rgba(0,200,255,0.6), 
            rgba(100,50,255,0.6), 
            transparent
        ) 1;
        opacity: 0;
        transform: translateY(-25px);
        animation: headerSlide 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) forwards 2s;
    }}

    @keyframes headerSlide {{
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    .summary-title {{
        font-size: 1.5em;
        font-weight: 700;
        background: linear-gradient(90deg, #00d9ff, #64c8ff, #00d9ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin: 0;
        animation: titleShimmer 3s ease-in-out infinite;
        filter: drop-shadow(0 0 20px rgba(0,217,255,0.6));
    }}

    @keyframes titleShimmer {{
        0%, 100% {{ background-position: 0% center; }}
        50% {{ background-position: 100% center; }}
    }}

    /* CONTENT WITH STAGGER */
    #summary-content {{
        position: relative;
        z-index: 2;
        line-height: 2;
        color: #e6eef8;
        font-size: 1.08em;
        opacity: 0;
        transform: translateX(-40px);
        animation: contentSlide 1s cubic-bezier(0.34, 1.56, 0.64, 1) forwards 2.3s;
    }}

    @keyframes contentSlide {{
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}

    /* DYNAMIC WAVE BACKGROUND */
    .wave-bg {{
        position: absolute;
        bottom: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: 
            radial-gradient(circle at 30% 50%, rgba(0,200,255,0.12) 0%, transparent 50%),
            radial-gradient(circle at 70% 50%, rgba(100,50,255,0.08) 0%, transparent 50%);
        opacity: 0;
        animation: waveReveal 2s ease forwards 2.5s, waveMove 10s ease-in-out infinite 4.5s;
        z-index: 0;
    }}

    @keyframes waveReveal {{
        to {{ opacity: 1; }}
    }}

    @keyframes waveMove {{
        0%, 100% {{ 
            transform: translate(0%, 0%) rotate(0deg); 
        }}
        25% {{ 
            transform: translate(5%, -5%) rotate(2deg); 
        }}
        50% {{ 
            transform: translate(-3%, 3%) rotate(-2deg); 
        }}
        75% {{ 
            transform: translate(3%, -3%) rotate(1deg); 
        }}
    }}

    /* ENHANCED CORNER ACCENTS */
    .corner-accent {{
        position: absolute;
        width: 50px;
        height: 50px;
        z-index: 3;
        opacity: 0;
        animation: cornerReveal 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) forwards 2.8s;
    }}

    @keyframes cornerReveal {{
        0% {{ opacity: 0; transform: scale(0); }}
        100% {{ opacity: 1; transform: scale(1); }}
    }}

    .corner-tl {{
        top: 15px;
        left: 15px;
        border-top: 3px solid rgba(0,200,255,0.7);
        border-left: 3px solid rgba(0,200,255,0.7);
        border-radius: 6px 0 0 0;
        box-shadow: 
            -5px -5px 20px rgba(0,200,255,0.3),
            inset 2px 2px 5px rgba(0,200,255,0.2);
    }}

    .corner-br {{
        bottom: 15px;
        right: 15px;
        border-bottom: 3px solid rgba(100,50,255,0.7);
        border-right: 3px solid rgba(100,50,255,0.7);
        border-radius: 0 0 6px 0;
        box-shadow: 
            5px 5px 20px rgba(100,50,255,0.3),
            inset -2px -2px 5px rgba(100,50,255,0.2);
    }}

    /* ENERGY PULSE BORDER */
    .energy-border {{
        position: absolute;
        inset: -3px;
        border-radius: 20px;
        background: linear-gradient(90deg,
            transparent,
            rgba(0,200,255,0.5),
            transparent,
            rgba(100,50,255,0.5),
            transparent
        );
        background-size: 200% 100%;
        opacity: 0;
        animation: borderFade 1s ease forwards 3.2s, borderPulse 4s linear infinite 4.2s;
        z-index: -1;
        filter: blur(6px);
    }}

    @keyframes borderFade {{
        to {{ opacity: 1; }}
    }}

    @keyframes borderPulse {{
        0% {{ background-position: 0% 0; }}
        100% {{ background-position: 200% 0; }}
    }}

    /* AMBIENT GLOW */
    .ambient-glow {{
        position: absolute;
        inset: -80px;
        border-radius: 50%;
        background: radial-gradient(circle, 
            rgba(0,200,255,0.15) 0%,
            rgba(100,50,255,0.1) 40%,
            transparent 70%
        );
        filter: blur(50px);
        opacity: 0;
        animation: glowPulse 6s ease-in-out infinite 3.5s;
        z-index: -2;
    }}

    @keyframes glowPulse {{
        0%, 100% {{ 
            opacity: 0.4; 
            transform: scale(1); 
        }}
        50% {{ 
            opacity: 0.7; 
            transform: scale(1.15); 
        }}
    }}
    </style>

    <div id="summary-wrapper">
        <div id="summary-box">
            <div class="grid-bg"></div>
            <div class="scan-light"></div>
            <div class="scan-light-2"></div>
            <div class="wave-bg"></div>
            <div class="energy-border"></div>
            <div class="ambient-glow"></div>

            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>

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
    }}, 500);
    </script>
    """, height=420)

# --- End UI/Animation Injection ---