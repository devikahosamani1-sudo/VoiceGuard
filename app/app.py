import streamlit as st
import librosa
import numpy as np
import joblib
import os
import tempfile


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VoiceGuard AI",
    page_icon="🎙️",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

.upload-title {
    font-size: 22px;
    font-weight: 600;
}

.footer {
    text-align: center;
    font-size: 14px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "model/voiceguard_model.pkl"

try:

    model = joblib.load(MODEL_PATH)

except Exception as e:

    st.error(f"❌ Model load error: {e}")
    st.stop()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=16000
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    features = np.mean(
        mfcc,
        axis=1
    )

    return features.reshape(1, -1)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎙️ VoiceGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Real Voice & Fake Voice Detection'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# PROJECT INTRODUCTION
# ============================================================

st.info(
    "🛡️ VoiceGuard AI analyzes audio characteristics "
    "using Machine Learning to classify a voice as "
    "REAL or AI-GENERATED."
)


# ============================================================
# HOW IT WORKS
# ============================================================

with st.expander("ℹ️ How VoiceGuard AI Works"):

    st.write("""
    **1️⃣ Upload Audio**  
    Upload a WAV audio file.

    **2️⃣ Feature Extraction**  
    The system extracts MFCC features from the audio.

    **3️⃣ Machine Learning Analysis**  
    The trained ML model analyzes the extracted features.

    **4️⃣ Voice Classification**  
    The system predicts REAL VOICE or FAKE / AI VOICE.

    **5️⃣ Confidence Score**  
    A confidence percentage is displayed with the result.
    """)


st.divider()


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="upload-title">📤 Upload Your Audio</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a WAV file to check whether the voice is real "
    "or AI-generated."
)

uploaded_file = st.file_uploader(
    "Choose a WAV audio file",
    type=["wav"]
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    st.audio(
        uploaded_file,
        format="audio/wav"
    )

    st.success(
        "✅ Audio uploaded successfully!"
    )


    # --------------------------------------------------------
    # Temporary File
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        temp_path = temp_file.name


    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    with st.spinner(
        "🔍 Analyzing audio... Please wait"
    ):

        try:

            features = extract_features(
                temp_path
            )

            prediction = model.predict(
                features
            )[0]

            probabilities = model.predict_proba(
                features
            )[0]

            confidence = (
                max(probabilities) * 100
            )


            # =================================================
            # RESULT
            # =================================================

            st.divider()

            st.subheader(
                "🔍 Detection Result"
            )


            if prediction == 0:

                st.success(
                    "✅ REAL VOICE"
                )

                st.write(
                    "The uploaded audio is classified "
                    "as a real human voice."
                )

            else:

                st.error(
                    "⚠️ FAKE / AI VOICE"
                )

                st.write(
                    "The uploaded audio is classified "
                    "as an AI-generated or synthetic voice."
                )


            # =================================================
            # CONFIDENCE
            # =================================================

            st.metric(
                label="🎯 Detection Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(
                int(confidence)
            )


        except Exception as e:

            st.error(
                f"❌ Error while analyzing audio: {e}"
            )


        finally:

            if os.path.exists(temp_path):

                os.remove(temp_path)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer">'
    '🎙️ VoiceGuard AI | Machine Learning Voice Detection'
    '</div>',
    unsafe_allow_html=True
)