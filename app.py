import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# IMPORT PROJECT BACKEND
# ============================================================

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.append(str(SRC_DIR))

from integration.road_safety_analyzer import analyze_video


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Road Intelligence",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# HTML HELPER
# ============================================================

def html(content):
    st.html(content)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analyzed_file" not in st.session_state:
    st.session_state.analyzed_file = None


# ============================================================
# DESIGN SYSTEM
# ============================================================

html("""
<style>

:root {
    --bg: #070b0e;
    --panel: #0c1216;
    --line: rgba(255,255,255,0.075);
    --text: #edf2f3;
    --muted: #718087;
    --safe: #7ea596;
    --medium: #ddb16b;
    --danger: #c76868;
}

.stApp {
    background:
        radial-gradient(
            circle at 80% 5%,
            rgba(72,105,110,0.12),
            transparent 25%
        ),
        linear-gradient(
            180deg,
            #080d10,
            #06090b
        );

    color: var(--text);
}

.block-container {
    max-width: 1450px;
    padding-top: 2.5rem;
    padding-bottom: 2rem;
}

header {
    background: transparent !important;
}

#MainMenu, footer {
    visibility: hidden;
}


/* HEADER */

.brand {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding-bottom: 16px;
    margin-bottom: 20px;

    border-bottom: 1px solid var(--line);
}

.brand-kicker {
    font-size: 9px;
    letter-spacing: 0.28em;
    color: #65757c;
    margin-bottom: 6px;
}

.brand-title {
    font-size: 28px;
    font-weight: 600;
    letter-spacing: -0.035em;
}

.brand-title span {
    font-weight: 300;
    color: #91a8a1;
}

.active {
    display: flex;
    align-items: center;
    gap: 9px;

    border: 1px solid rgba(126,165,150,0.22);
    border-radius: 999px;

    padding: 8px 12px;

    font-size: 9px;
    letter-spacing: 0.17em;

    color: #9db2aa;
}

.active-dot {
    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #86ae9f;

    box-shadow:
        0 0 14px rgba(134,174,159,0.65);
}


/* LABELS */

.section-label {
    font-size: 9px;
    letter-spacing: 0.22em;
    color: #66767d;
    margin-bottom: 9px;
}


/* PANELS */

.panel {
    border: 1px solid var(--line);

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.035),
            rgba(255,255,255,0.012)
        );

    border-radius: 15px;
    padding: 20px;
}


/* UPLOAD */

.upload-intro {
    text-align: center;
    padding: 35px 15px 15px;
}

.upload-title {
    font-size: 20px;
    font-weight: 400;
    color: #dce4e6;
}

.upload-subtitle {
    margin-top: 7px;
    font-size: 11px;
    color: #68777d;
}


/* RISK */

.risk-panel {
    min-height: 285px;

    display: flex;
    flex-direction: column;
    justify-content: center;

    position: relative;
    overflow: hidden;
}

.risk-caption {
    font-size: 11px;
    color: #77868c;
}

.risk-number {
    margin-top: 12px;

    font-size: 72px;
    line-height: 0.9;

    font-weight: 250;
    letter-spacing: -0.07em;
}

.risk-number span {
    font-size: 19px;
    color: #78858a;
}

.risk-state {
    margin-top: 12px;

    font-size: 14px;
    letter-spacing: 0.22em;
}

.risk-track {
    margin-top: 28px;

    position: relative;

    height: 4px;

    border-radius: 999px;

    background:
        linear-gradient(
            90deg,
            var(--safe) 0%,
            var(--safe) 35%,
            var(--medium) 35%,
            var(--medium) 65%,
            var(--danger) 65%,
            var(--danger) 100%
        );
}

.risk-marker {
    position: absolute;

    top: 50%;

    transform: translate(-50%, -50%);

    width: 12px;
    height: 12px;

    border-radius: 50%;

    background: #f0d3a3;

    box-shadow:
        0 0 0 5px rgba(240,211,163,0.10),
        0 0 18px rgba(240,211,163,0.4);
}

.risk-labels {
    display: flex;
    justify-content: space-between;

    margin-top: 8px;

    font-size: 8px;
    letter-spacing: 0.14em;

    color: #596970;
}


/* SCENE */

.pulse {
    display: grid;
    grid-template-columns: repeat(4, 1fr);

    border: 1px solid var(--line);
    border-radius: 14px;

    overflow: hidden;

    background: var(--line);
    gap: 1px;
}

.pulse-cell {
    background: #0b1114;
    padding: 18px 16px;
}

.pulse-value {
    font-size: 25px;
    font-weight: 300;
    color: #e4eaec;
}

.pulse-label {
    margin-top: 4px;

    font-size: 8px;
    letter-spacing: 0.13em;

    color: #64747a;
}


/* REASONS */

.reason {
    display: flex;
    align-items: flex-start;
    gap: 12px;

    padding: 12px 0;

    border-bottom:
        1px solid rgba(255,255,255,0.055);
}

.reason:last-child {
    border-bottom: none;
}

.reason-mark {
    width: 9px;
    height: 9px;

    margin-top: 5px;

    border:
        1px solid rgba(221,177,107,0.65);

    transform: rotate(45deg);

    flex-shrink: 0;
}

.reason-text {
    font-size: 12px;
    line-height: 1.45;

    color: #9da9ad;
}


/* PIPELINE */

.pipeline {
    margin-top: 22px;

    padding: 20px 5px;

    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
}

.pipeline-grid {
    display: grid;

    grid-template-columns:
        1fr 80px
        1fr 80px
        1fr 80px
        1fr;

    align-items: center;
}

.stage {
    text-align: center;
}

.stage-dot {
    width: 8px;
    height: 8px;

    margin: auto;
    margin-bottom: 8px;

    border-radius: 50%;

    background: #82a69a;

    box-shadow:
        0 0 12px rgba(130,166,154,0.35);
}

.stage-name {
    font-size: 8px;
    letter-spacing: 0.18em;
    color: #64747a;
}

.stage-model {
    margin-top: 5px;
    font-size: 12px;
    color: #bac5c8;
}

.connector {
    height: 1px;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(130,166,154,0.45),
            transparent
        );
}


/* PERFORMANCE */

.performance {
    display: grid;
    grid-template-columns: repeat(4, 1fr);

    margin-top: 18px;
    gap: 20px;
}

.performance-value {
    font-size: 15px;
    color: #d9e0e2;
}

.performance-label {
    margin-top: 3px;

    font-size: 8px;
    letter-spacing: 0.13em;

    color: #5e6d73;
}


/* VIDEO */

video {
    width: 100% !important;

    border-radius: 14px !important;

    border:
        1px solid rgba(255,255,255,0.08);

    background: #030506;
}


/* BUTTON */

.stButton > button {
    width: 100%;

    border-radius: 10px;

    border:
        1px solid rgba(126,165,150,0.30);

    background:
        rgba(126,165,150,0.10);

    color: #cbd8d4;

    font-size: 12px;
    letter-spacing: 0.08em;

    padding: 0.7rem;
}

.stButton > button:hover {
    border-color:
        rgba(126,165,150,0.65);

    color: white;
}


/* NOTE */

.note {
    margin-top: 18px;

    font-size: 8px;
    letter-spacing: 0.10em;
    line-height: 1.6;

    color: #4e5d63;
}

</style>
""")


# ============================================================
# HEADER
# ============================================================

html("""
<div class="brand">

    <div>
        <div class="brand-kicker">
            TEMPORAL ROAD-SAFETY PERCEPTION
        </div>

        <div class="brand-title">
            ROAD <span>INTELLIGENCE</span>
        </div>
    </div>

    <div class="active">
        <div class="active-dot"></div>
        AI SAFETY SYSTEM
    </div>

</div>
""")


# ============================================================
# UPLOAD
# ============================================================

html("""
<div class="upload-intro">

    <div class="upload-title">
        Analyze a road scene
    </div>

    <div class="upload-subtitle">
        Upload a dashcam or road video to begin
        spatial and temporal safety analysis.
    </div>

</div>
""")


uploaded_file = st.file_uploader(
    "Road video",
    type=[
        "mp4",
        "avi",
        "mov"
    ],
    label_visibility="collapsed"
)


# ============================================================
# HANDLE NEW FILE
# ============================================================

if uploaded_file is not None:

    current_file_identity = (
        uploaded_file.name,
        uploaded_file.size
    )


    if (
        st.session_state.analyzed_file
        != current_file_identity
    ):

        st.session_state.analysis_result = None
        st.session_state.analyzed_file = None


    analyze_clicked = st.button(
        "ANALYZE SCENE"
    )


    if analyze_clicked:

        # ----------------------------------------------------
        # SAVE UPLOAD
        # ----------------------------------------------------

        temp_dir = Path(
            "outputs/dashboard_runtime"
        )

        temp_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        input_path = (
            temp_dir
            / "uploaded_video.mp4"
        )


        output_path = (
            temp_dir
            / "analyzed_video.mp4"
        )


        with open(
            input_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        progress_bar = st.progress(
            0,
            text="Initializing road intelligence..."
        )


        def update_progress(value):

            progress_bar.progress(
                min(
                    int(value * 100),
                    100
                ),
                text=(
                    f"Analyzing road scene... "
                    f"{int(value * 100)}%"
                )
            )


        # ----------------------------------------------------
        # RUN AI
        # ----------------------------------------------------

        try:

            with st.spinner(
                "YOLO + U-Net + LSTM analysis in progress..."
            ):

                result = analyze_video(
                    input_path=input_path,
                    output_path=output_path,
                    progress_callback=update_progress
                )


            progress_bar.progress(
                100,
                text="Analysis complete."
            )


            st.session_state.analysis_result = (
                result
            )

            st.session_state.analyzed_file = (
                current_file_identity
            )


        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )


# ============================================================
# RESULTS
# ============================================================

result = st.session_state.analysis_result


if result is not None:

    st.divider()


    risk_percent = float(
        result["risk_percent"]
    )

    risk_level = result[
        "risk_level"
    ]

    scene = result[
        "scene"
    ]

    reasons = result[
        "reasons"
    ]


    # --------------------------------------------------------
    # RISK COLORS
    # --------------------------------------------------------

    if risk_level == "LOW":

        risk_color = "#7ea596"

    elif risk_level == "MEDIUM":

        risk_color = "#ddb16b"

    else:

        risk_color = "#c76868"


    # --------------------------------------------------------
    # MAIN RESULTS
    # --------------------------------------------------------

    video_col, risk_col = st.columns(
        [2.35, 1],
        gap="large"
    )


    with video_col:

        html("""
        <div class="section-label">
            ANALYZED ROAD SCENE
        </div>
        """)


        output_video_path = Path(
            "outputs/dashboard_runtime/analyzed_video_web.mp4"
        )


        if output_video_path.exists():

            with open(output_video_path, "rb") as video_file:
                video_bytes = video_file.read()

            st.video(
            video_bytes,
            format="video/mp4"
        )


    with risk_col:

        html(
            f"""
            <div class="section-label">
                TEMPORAL STATE
            </div>

            <div class="panel risk-panel">

                <div class="risk-caption">
                    Current inferred risk
                </div>

                <div
                    class="risk-number"
                    style="color:{risk_color};"
                >
                    {risk_percent:.1f}
                    <span>%</span>
                </div>

                <div
                    class="risk-state"
                    style="color:{risk_color};"
                >
                    {risk_level}
                </div>

                <div class="risk-track">

                    <div
                        class="risk-marker"
                        style="
                            left:
                            {max(0, min(risk_percent, 100))}%;
                        "
                    ></div>

                </div>

                <div class="risk-labels">
                    <span>LOW</span>
                    <span>MEDIUM</span>
                    <span>HIGH</span>
                </div>

            </div>
            """
        )


    # ========================================================
    # RISK TRAJECTORY
    # ========================================================

    risk_history = result[
        "risk_history"
    ]


    if risk_history:

        html("""
        <br>
        <div class="section-label">
            TEMPORAL RISK TRAJECTORY
        </div>
        """)


        risk_df = pd.DataFrame(
            risk_history
        )


        risk_df["risk_percent"] = (
            risk_df["risk"]
            * 100
        )


        chart_data = (
            risk_df[
                [
                    "time",
                    "risk_percent"
                ]
            ]
            .set_index(
                "time"
            )
        )


        st.line_chart(
            chart_data,
            height=210,
            y_label="Risk %",
            x_label="Video time (seconds)"
        )


    # ========================================================
    # SCENE + WHY NOW
    # ========================================================

    scene_col, reason_col = st.columns(
        [1.35, 1],
        gap="large"
    )


    with scene_col:

        html(
            f"""
            <div class="section-label">
                SCENE PULSE
            </div>

            <div class="pulse">

                <div class="pulse-cell">
                    <div class="pulse-value">
                        {scene["vehicles"]:02d}
                    </div>

                    <div class="pulse-label">
                        VEHICLES
                    </div>
                </div>


                <div class="pulse-cell">
                    <div class="pulse-value">
                        {scene["pedestrians"]:02d}
                    </div>

                    <div class="pulse-label">
                        PEDESTRIANS
                    </div>
                </div>


                <div class="pulse-cell">
                    <div class="pulse-value">
                        {scene["on_road_objects"]:02d}
                    </div>

                    <div class="pulse-label">
                        ROAD-ACTIVE
                    </div>
                </div>


                <div class="pulse-cell">
                    <div class="pulse-value">
                        {scene["heavy_vehicles"]:02d}
                    </div>

                    <div class="pulse-label">
                        HEAVY VEHICLES
                    </div>
                </div>

            </div>
            """
        )


    with reason_col:

        html("""
        <div class="section-label">
            WHY NOW?
        </div>

        <div class="panel">
        """)


        for reason in reasons:

            html(
                f"""
                <div class="reason">

                    <div class="reason-mark"></div>

                    <div class="reason-text">
                        {reason}
                    </div>

                </div>
                """
            )


        html("""
        </div>
        """)


    # ========================================================
    # PIPELINE
    # ========================================================

    html(
        f"""
        <div class="pipeline">

            <div class="section-label">
                INTELLIGENCE PATHWAY
            </div>

            <div class="pipeline-grid">

                <div class="stage">
                    <div class="stage-dot"></div>
                    <div class="stage-name">
                        PERCEPTION
                    </div>
                    <div class="stage-model">
                        YOLO11n
                    </div>
                </div>

                <div class="connector"></div>


                <div class="stage">
                    <div class="stage-dot"></div>
                    <div class="stage-name">
                        CONTEXT
                    </div>
                    <div class="stage-model">
                        U-Net
                    </div>
                </div>

                <div class="connector"></div>


                <div class="stage">
                    <div class="stage-dot"></div>
                    <div class="stage-name">
                        TEMPORAL
                    </div>
                    <div class="stage-model">
                        LSTM
                    </div>
                </div>

                <div class="connector"></div>


                <div class="stage">
                    <div class="stage-dot"></div>
                    <div class="stage-name">
                        DECISION
                    </div>

                    <div
                        class="stage-model"
                        style="color:{risk_color};"
                    >
                        {risk_level}
                    </div>
                </div>

            </div>

        </div>
        """
    )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    html(
        f"""
        <div class="performance">

            <div>
                <div class="performance-value">
                    {result["fps"]:.2f} FPS
                </div>

                <div class="performance-label">
                    END-TO-END THROUGHPUT
                </div>
            </div>


            <div>
                <div class="performance-value">
                    {result["latency_ms"]:.2f} ms
                </div>

                <div class="performance-label">
                    AVERAGE LATENCY
                </div>
            </div>


            <div>
                <div class="performance-value">
                    15 × 14
                </div>

                <div class="performance-label">
                    TEMPORAL OBSERVATION
                </div>
            </div>


            <div>
                <div class="performance-value">
                    {result["device"].upper()}
                </div>

                <div class="performance-label">
                    COMPUTE BACKEND
                </div>
            </div>

        </div>
        """
    )


    html("""
    <div class="note">

        RISK STATE IS AN APPLICATION-LEVEL INTERPRETATION
        OF THE BINARY TEMPORAL MODEL OUTPUT. CONTEXTUAL
        OBSERVATIONS ARE DERIVED FROM OBJECT DETECTION
        AND DRIVABLE-AREA SEGMENTATION.

    </div>
    """)


else:

    # ========================================================
    # IDLE STATE
    # ========================================================

    html("""
    <br><br>

    <div class="panel" style="
        text-align:center;
        padding:50px 20px;
    ">

        <div style="
            font-size:11px;
            letter-spacing:0.18em;
            color:#6b7b81;
        ">
            SYSTEM READY
        </div>

        <div style="
            margin-top:12px;
            font-size:17px;
            color:#b7c3c6;
        ">
            Waiting for road-scene input
        </div>

        <div style="
            margin-top:7px;
            font-size:11px;
            color:#59686e;
        ">
            Upload a video and select ANALYZE SCENE
            to begin perception and temporal reasoning.
        </div>

    </div>
    """)