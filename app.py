import streamlit as st
import numpy as np
import joblib
import cv2
import tempfile
import os

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Autism Screening Tool",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Autism Screening Prediction")
st.caption("Research demo — Bharati Vidyapeeth's College of Engineering, New Delhi | Not a medical diagnosis tool.")

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
@st.cache_resource
def load_questionnaire_models():
    rf     = joblib.load('rf_model (1).pkl')
    scaler = joblib.load('scaler (1).pkl')
    return rf, scaler

@st.cache_resource
def load_video_model():
    """Load the CNN video model trained on SSBD dataset."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('video_model.h5')
        le    = joblib.load('video_label_encoder.pkl')
        return model, le
    except Exception as e:
        return None, None

rf, scaler = load_questionnaire_models()

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2 = st.tabs(["📝 Questionnaire Screening", "🎥 Video Behavior Analysis"])


# ══════════════════════════════════════════
# TAB 1 — QUESTIONNAIRE (unchanged logic)
# ══════════════════════════════════════════
with tab1:
    st.header("Answer the following 10 questions:")
    st.info("Based on the AQ-10 autism screening questionnaire.")

    def yesno(val):
        return 1 if val == "Yes" else 0

    q1  = st.radio("1. I often notice small sounds when others do not.", ('No', 'Yes'))
    q2  = st.radio("2. I usually concentrate more on the whole picture, rather than the small details.", ('No', 'Yes'))
    q3  = st.radio("3. I find it easy to do more than one thing at once.", ('No', 'Yes'))
    q4  = st.radio("4. If there is an interruption, I can switch back to what I was doing very quickly.", ('No', 'Yes'))
    q5  = st.radio("5. I find it easy to 'read between the lines' when someone is talking to me.", ('No', 'Yes'))
    q6  = st.radio("6. I know how to tell if someone listening to me is getting bored.", ('No', 'Yes'))
    q7  = st.radio("7. When I'm reading a story, I find it difficult to work out the characters' intentions.", ('No', 'Yes'))
    q8  = st.radio("8. I like to collect information about categories of things.", ('No', 'Yes'))
    q9  = st.radio("9. I find it easy to work out what someone is thinking or feeling just by looking at their face.", ('No', 'Yes'))
    q10 = st.radio("10. I find it difficult to make new friends.", ('No', 'Yes'))

    a_scores = [yesno(q) for q in [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]]

    st.markdown("---")
    st.subheader("Demographics")

    age    = st.number_input("Age", min_value=5.0, max_value=100.0, value=30.0)
    gender = st.radio("Gender", ["Male", "Female"])

    country_options = {
        "India": 52, "United States": 1, "Canada": 30,
        "UK": 51, "Australia": 12, "Other": 0
    }
    country_label  = st.selectbox("Country of residence", list(country_options.keys()))
    contry_of_res  = country_options[country_label]

    relation_options = {
        "Self": 1, "Parent": 2, "Relative": 3,
        "Health professional": 4, "Other": 0
    }
    relation_label = st.selectbox("Relation (who is answering?)", list(relation_options.keys()))
    relation       = relation_options[relation_label]

    ethnicity_options = {
        "Unknown": 0, "White-European": 1, "Middle Eastern": 2, "Pasifika": 3,
        "Black": 4, "Others": 5, "Hispanic": 6, "Asian": 7,
        "Turkish": 8, "South Asian": 9, "Latino": 10
    }
    ethnicity_label = st.selectbox("Ethnicity", list(ethnicity_options.keys()))
    ethnicity       = ethnicity_options[ethnicity_label]

    jaundice = st.radio("Born with Jaundice?", ["No", "Yes"])
    austim   = st.radio("Family history of autism?", ["No", "Yes"])

    # Feature engineering
    total_A_score   = sum(a_scores)
    mean_A_score    = np.mean(a_scores)
    std_A_score     = np.std(a_scores)
    A_score_high_flag = 1 if total_A_score >= 6 else 0

    answers = (
        a_scores
        + [age]
        + [1 if gender == "Male" else 0]
        + [int(ethnicity)]
        + [1 if jaundice == "Yes" else 0]
        + [1 if austim == "Yes" else 0]
        + [contry_of_res]
        + [relation]
        + [total_A_score, mean_A_score, std_A_score, A_score_high_flag]
    )

    if st.button("🔍 Predict ASD Risk", use_container_width=True):
        data        = np.array([answers])
        data_scaled = scaler.transform(data)
        pred        = rf.predict(data_scaled)[0]
        prob        = rf.predict_proba(data_scaled)[0]

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ **Result: ASD Traits Detected**  \nConfidence: {prob[1]*100:.1f}%")
        else:
            st.success(f"✅ **Result: No ASD Traits Detected**  \nConfidence: {prob[0]*100:.1f}%")

        st.metric("Total AQ Score", total_A_score, help="Score ≥ 6 is associated with ASD traits")
        st.caption("⚕️ This is not a medical diagnosis. Please consult a healthcare professional.")


# ══════════════════════════════════════════
# TAB 2 — VIDEO ANALYSIS (NEW)
# ══════════════════════════════════════════
with tab2:
    st.header("Video-Based Behavior Analysis")
    st.markdown("""
    Upload a short video of the individual. The CNN model trained on the 
    **SSBD dataset** detects ASD-related self-stimulatory (stimming) behaviors:
    
    | Behavior | Description |
    |---|---|
    | 🙌 Arm Flapping | Repetitive flapping of arms/hands |
    | 🤕 Head Banging | Repetitive head-banging motion |
    | 🌀 Spinning | Body spinning in circles |
    """)

    st.info("💡 **Note**: This video model is trained to detect stimming behaviors associated with ASD. Presence of these behaviors is one indicator, not a definitive diagnosis.")

    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Keep the video under 2 minutes for best performance."
    )

    if uploaded_video is not None:
        # Show the uploaded video
        st.video(uploaded_video)
        uploaded_video.seek(0)  # Reset after st.video reads it

        video_model, le = load_video_model()

        if video_model is None:
            st.warning("""
            ⚠️ **Video model not found.**  
            Please train it first using the notebook and place `video_model.h5` 
            and `video_label_encoder.pkl` in the same directory as this app.
            """)
        else:
            if st.button("🎬 Analyze Video", use_container_width=True):
                with st.spinner("Extracting frames and analyzing behavior..."):

                    # Save video to temp file
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_video.read())
                    tfile.close()

                    # ── Frame extraction ──
                    cap         = cv2.VideoCapture(tfile.name)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps          = cap.get(cv2.CAP_PROP_FPS)
                    duration_sec = total_frames / fps if fps > 0 else 0

                    n_samples = min(30, max(10, total_frames // 10))
                    indices   = np.linspace(0, total_frames - 1, n_samples, dtype=int)
                    frames    = []

                    for idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.resize(frame, (224, 224))
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame)
                    cap.release()
                    os.unlink(tfile.name)

                    if len(frames) == 0:
                        st.error("Could not read frames from the video. Please try a different file.")
                    else:
                        # ── Predict on each frame, average probabilities ──
                        frames_array = np.array(frames, dtype=np.float32) / 255.0
                        preds_all    = video_model.predict(frames_array, verbose=0)  # (n_frames, n_classes)
                        avg_probs    = preds_all.mean(axis=0)
                        pred_idx     = np.argmax(avg_probs)
                        pred_class   = le.inverse_transform([pred_idx])[0]
                        confidence   = avg_probs[pred_idx] * 100

                        st.markdown("---")
                        st.subheader("📊 Analysis Results")

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Detected Behavior", pred_class.replace('_', ' ').title())
                        col2.metric("Confidence", f"{confidence:.1f}%")
                        col3.metric("Frames Analyzed", len(frames))

                        st.markdown("**Behavior Probability Distribution:**")
                        for cls, prob in sorted(zip(le.classes_, avg_probs), key=lambda x: -x[1]):
                            label = cls.replace('_', ' ').title()
                            st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")

                        # ── ASD Risk interpretation ──
                        st.markdown("---")
                        stimming_behaviors = ['arm_flapping', 'headbanging', 'spinning']
                        if pred_class.lower() in stimming_behaviors and confidence > 60:
                            st.warning(f"""
                            ⚠️ **Stimming behavior detected: {pred_class.replace('_', ' ').title()}**  
                            This behavior is commonly associated with ASD.  
                            We recommend combining this result with the questionnaire screening.
                            """)
                        else:
                            st.success("""
                            ✅ **No significant stimming behavior detected.**  
                            No prominent ASD-related motor behavior identified in this video.
                            """)

                        st.caption("⚕️ This is a research demonstration. Always consult a qualified medical professional for diagnosis.")

    # ── How to get the SSBD dataset ──
    with st.expander("📥 How to obtain the SSBD video dataset & train the model"):
        st.markdown("""
        **SSBD — Self-Stimulatory Behaviour Dataset** is the standard public dataset for this task.
        
        **Step 1 — Get the dataset:**
        - Visit: https://rolandgoecke.net/research/datasets/ssbd/
        - Download `url-list.pdf` → use the provided script or `yt-dlp` to download videos
        - OR use the curated version: https://github.com/Samwei1/autism-related-behavior
        
        **Step 2 — Folder structure expected:**
        ```
        ssbd_videos/
          arm_flapping/  video1.mp4, video2.mp4 ...
          headbanging/   video1.mp4 ...
          spinning/      video1.mp4 ...
        ```
        
        **Step 3 — Run the Video Training section in the notebook (Section 6)**  
        It will save `video_model.h5` and `video_label_encoder.pkl` — place these alongside `app.py`.
        """)
