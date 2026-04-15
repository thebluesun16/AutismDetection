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
    # Try to load the SMOTE-improved model first, else fall back to original
    try:
        rf = joblib.load('rf_model_smote.pkl')
        st.toast("Using SMOTE-improved Random Forest model ✅")
    except:
        rf = joblib.load('rf_model (1).pkl')
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

@st.cache_resource
def load_cnnlstm_model():
    """Load the CNN-LSTM temporal model (new — Future Scope)."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('cnnlstm_model.h5')
        le    = joblib.load('cnnlstm_label_encoder.pkl')
        return model, le
    except:
        return None, None

rf, scaler = load_questionnaire_models()


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def extract_frames_for_lstm(video_path, seq_len=10, img_size=224):
    """
    Extract exactly seq_len uniformly spaced frames from a video.
    Used for the CNN-LSTM model that needs a fixed-length sequence.
    Returns shape: (1, seq_len, 224, 224, 3) or None
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 5:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, seq_len, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()

    if len(frames) < seq_len:
        return None

    arr = np.array(frames[:seq_len], dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]    # add batch dimension → (1, seq_len, 224, 224, 3)


def analyse_face_landmarks_from_frames(frames_rgb):
    """
    Runs MediaPipe face mesh on a list of RGB frames.
    Returns average Eye Aspect Ratio (EAR) and an eye contact assessment.
    
    MediaPipe eye landmark indices:
      Left eye  : 362, 385, 387, 263, 373, 380
      Right eye : 33,  160, 158, 133, 153, 144
    """
    try:
        import mediapipe as mp
    except ImportError:
        return None   # MediaPipe not installed

    LEFT_EYE  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33,  160, 158, 133, 153, 144]

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        min_detection_confidence=0.5
    )
    ear_list = []

    for frame in frames_rgb:
        results = face_mesh.process(frame)
        if results.multi_face_landmarks:
            lm   = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            def ear(eye_pts):
                pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in eye_pts]
                v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
                v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
                hz = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
                return (v1 + v2) / (2.0 * hz + 1e-6)

            avg_ear = (ear(LEFT_EYE) + ear(RIGHT_EYE)) / 2.0
            ear_list.append(avg_ear)

    face_mesh.close()

    if not ear_list:
        return {'avg_ear': 0.0, 'faces_found': 0, 'assessment': 'No face detected'}

    avg = float(np.mean(ear_list))
    return {
        'avg_ear'    : round(avg, 4),
        'faces_found': len(ear_list),
        'assessment' : 'Normal eye contact' if avg >= 0.20 else 'Possible eye contact avoidance'
    }


def dsm5_score(a_scores, age, jaundice, family_history):
    """
    Approximates DSM-5 clinical criteria using AQ-10 answers.
    Domain A: Social communication (Q2, Q5, Q6, Q9)
    Domain B: Restricted/repetitive behavior (Q1, Q7, Q8)
    Returns a score out of 10 and a risk label.
    """
    domain_a = a_scores[1] + a_scores[4] + a_scores[5] + a_scores[8]
    domain_b = a_scores[0] + a_scores[6] + a_scores[7]
    bonus    = (0.5 if jaundice == 1 else 0) + (1.0 if family_history == 1 else 0)
    total    = min(domain_a + domain_b + bonus, 10)
    if total >= 6:
        label = "🔴 High DSM-5 Alignment"
    elif total >= 4:
        label = "🟡 Moderate DSM-5 Alignment"
    else:
        label = "🟢 Low DSM-5 Alignment"
    return domain_a, domain_b, round(total, 1), label


# ─────────────────────────────────────────
# TABS  (now 3 tabs)
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📝 Questionnaire Screening",
    "🎥 Video Behavior Analysis",
    "📷 Real-Time Webcam Detection"   # NEW — Future Scope
])


# ══════════════════════════════════════════
# TAB 1 — QUESTIONNAIRE  (same as before + DSM-5 score)
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

    total_A_score    = sum(a_scores)
    mean_A_score     = np.mean(a_scores)
    std_A_score      = np.std(a_scores)
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

        # ── 🆕 DSM-5 Clinical Validation Score ──
        st.markdown("---")
        st.subheader("🏥 DSM-5 Clinical Alignment Score")
        st.caption("This maps your answers to real DSM-5 diagnostic criteria (research approximation only).")

        j_val = 1 if jaundice == "Yes" else 0
        f_val = 1 if austim   == "Yes" else 0
        dom_a, dom_b, dsm_total, dsm_label = dsm5_score(a_scores, age, j_val, f_val)

        col1, col2, col3 = st.columns(3)
        col1.metric("Domain A\n(Social Communication)", f"{dom_a}/4")
        col2.metric("Domain B\n(Repetitive Behavior)", f"{dom_b}/3")
        col3.metric("DSM-5 Score", f"{dsm_total}/10")
        st.info(dsm_label)

        st.caption("⚕️ This is not a medical diagnosis. Please consult a healthcare professional.")


# ══════════════════════════════════════════
# TAB 2 — VIDEO ANALYSIS (same as before + CNN-LSTM + MediaPipe)
# ══════════════════════════════════════════
with tab2:
    st.header("Video-Based Behavior Analysis")
    st.markdown("""
    Upload a short video of the individual. Two models analyse the video:
    
    | Model | What it does |
    |---|---|
    | **MobileNetV2** (original) | Classifies each frame separately |
    | **CNN-LSTM** 🆕 | Captures motion patterns across a sequence of frames |
    
    Detected behaviors: 🙌 Arm Flapping | 🤕 Head Banging | 🌀 Spinning
    """)

    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Keep the video under 2 minutes for best performance."
    )

    if uploaded_video is not None:
        st.video(uploaded_video)
        uploaded_video.seek(0)

        video_model, le         = load_video_model()
        cnnlstm_model, le_lstm  = load_cnnlstm_model()

        if video_model is None and cnnlstm_model is None:
            st.warning("""
            ⚠️ **Video models not found.**  
            Please train them using the notebook (Sections 6 & 10) and place  
            `video_model.h5`, `cnnlstm_model.h5` and their label encoders alongside this app.
            """)
        else:
            model_choice = st.radio(
                "Choose model for analysis:",
                ["MobileNetV2 (frame-by-frame)", "CNN-LSTM (temporal — recommended 🆕)"]
            )

            if st.button("🎬 Analyze Video", use_container_width=True):
                with st.spinner("Extracting frames and analysing behavior..."):

                    # Save video to a temp file
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_video.read())
                    tfile.close()

                    # ─── MobileNetV2 (original frame-by-frame) ───
                    if "MobileNetV2" in model_choice and video_model is not None:
                        cap          = cv2.VideoCapture(tfile.name)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        n_samples    = min(30, max(10, total_frames // 10))
                        indices      = np.linspace(0, total_frames - 1, n_samples, dtype=int)
                        frames       = []
                        for idx in indices:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                            ret, frame = cap.read()
                            if ret:
                                frame = cv2.resize(frame, (224, 224))
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frames.append(frame)
                        cap.release()

                        if frames:
                            frames_array = np.array(frames, dtype=np.float32) / 255.0
                            preds_all    = video_model.predict(frames_array, verbose=0)
                            avg_probs    = preds_all.mean(axis=0)
                            pred_idx     = np.argmax(avg_probs)
                            pred_class   = le.inverse_transform([pred_idx])[0]
                            confidence   = avg_probs[pred_idx] * 100
                            frames_used  = len(frames)
                            class_labels = le.classes_

                    # ─── CNN-LSTM (temporal) ───
                    elif "CNN-LSTM" in model_choice and cnnlstm_model is not None:
                        seq = extract_frames_for_lstm(tfile.name, seq_len=10)
                        if seq is not None:
                            preds_seq  = cnnlstm_model.predict(seq, verbose=0)   # (1, n_classes)
                            avg_probs  = preds_seq[0]
                            pred_idx   = np.argmax(avg_probs)
                            pred_class = le_lstm.inverse_transform([pred_idx])[0]
                            confidence = avg_probs[pred_idx] * 100
                            frames_used = 10
                            class_labels = le_lstm.classes_
                        else:
                            st.error("Could not read frames for CNN-LSTM. Try a different video.")
                            os.unlink(tfile.name)
                            st.stop()
                    else:
                        st.warning("Selected model not available. Please train it from the notebook.")
                        os.unlink(tfile.name)
                        st.stop()

                    # ─── Results ───
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Detected Behavior", pred_class.replace('_', ' ').title())
                    col2.metric("Confidence", f"{confidence:.1f}%")
                    col3.metric("Frames Analyzed", frames_used)

                    st.markdown("**Behavior Probability Distribution:**")
                    for cls, prob in sorted(zip(class_labels, avg_probs), key=lambda x: -x[1]):
                        label = cls.replace('_', ' ').title()
                        st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")

                    # ─── 🆕 MediaPipe Facial Analysis ───
                    st.markdown("---")
                    st.subheader("👁️ Facial Landmark Analysis (MediaPipe)")
                    cap2   = cv2.VideoCapture(tfile.name)
                    total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
                    idxs2  = np.linspace(0, total2 - 1, 20, dtype=int)
                    rgb_frames = []
                    for idx in idxs2:
                        cap2.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                        ret, frame = cap2.read()
                        if ret:
                            rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cap2.release()

                    mp_result = analyse_face_landmarks_from_frames(rgb_frames)

                    if mp_result is None:
                        st.info("ℹ️ MediaPipe not installed. Run `pip install mediapipe` to enable facial analysis.")
                    else:
                        mcol1, mcol2 = st.columns(2)
                        mcol1.metric("Avg Eye Aspect Ratio (EAR)", mp_result['avg_ear'],
                                     help="Higher = eyes more open. Below 0.20 may indicate avoidance.")
                        mcol2.metric("Frames with Face Detected", mp_result['faces_found'])
                        if mp_result['avg_ear'] < 0.20 and mp_result['faces_found'] > 0:
                            st.warning(f"⚠️ **{mp_result['assessment']}**  \n"
                                       "EAR below 0.20 — eyes appear less open, which may indicate "
                                       "gaze avoidance. This is one indicator assessed in autism screening.")
                        elif mp_result['faces_found'] > 0:
                            st.success(f"✅ **{mp_result['assessment']}**  \n"
                                       f"EAR = {mp_result['avg_ear']} — eye openness appears normal.")
                        else:
                            st.info("No face detected in video frames.")

                    # ─── ASD Risk interpretation ───
                    st.markdown("---")
                    stimming = ['arm_flapping', 'headbanging', 'spinning']
                    if pred_class.lower() in stimming and confidence > 60:
                        st.warning(f"⚠️ **Stimming behavior detected: {pred_class.replace('_', ' ').title()}**  \n"
                                   "This behavior is commonly associated with ASD.  \n"
                                   "Combine this with the questionnaire screening for a fuller picture.")
                    else:
                        st.success("✅ **No significant stimming behavior detected.**  \n"
                                   "No prominent ASD-related motor behavior identified in this video.")

                    os.unlink(tfile.name)
                    st.caption("⚕️ Research demonstration only. Consult a qualified medical professional for diagnosis.")

    with st.expander("📥 How to obtain the SSBD video dataset & train the models"):
        st.markdown("""
        **SSBD — Self-Stimulatory Behaviour Dataset**
        
        **Step 1 — Get the dataset:**  
        Visit https://rolandgoecke.net/research/datasets/ssbd/ and download videos.
        
        **Step 2 — Folder structure:**
        ```
        ssbd_videos/
          arm_flapping/  video1.mp4 ...
          headbanging/   video1.mp4 ...
          spinning/      video1.mp4 ...
        ```
        
        **Step 3 — Run the notebook:**  
        - Section 6 → saves `video_model.h5`  
        - Section 10 → saves `cnnlstm_model.h5`  
        Place all `.h5` and `.pkl` files next to this `app.py`.
        """)


# ══════════════════════════════════════════
# TAB 3 — 🆕 REAL-TIME WEBCAM DETECTION  (Future Scope)
# ══════════════════════════════════════════
with tab3:
    st.header("📷 Real-Time Webcam Detection")
    st.info("""
    **Future Scope Feature** — Capture a photo from your webcam and analyse it instantly.
    
    This uses your device camera to:
    - Detect **facial landmarks** (eye contact, expression) via MediaPipe
    - Run **ASD behavior classification** on the captured frame
    - Show **DSM-5 social domain score** based on facial features
    
    > 💡 For a full video, use the **Video Behavior Analysis** tab.
    """)

    # st.camera_input() is Streamlit's built-in webcam capture widget
    # It opens the device camera and lets the user take a photo
    webcam_image = st.camera_input("📸 Take a photo for real-time analysis")

    if webcam_image is not None:
        # Read the captured image into a numpy array
        file_bytes = np.asarray(bytearray(webcam_image.read()), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        st.image(img_rgb, caption="Captured frame", use_column_width=True)
        st.markdown("---")

        with st.spinner("Analysing the captured image..."):

            # ── MediaPipe facial analysis on single frame ──
            mp_result = analyse_face_landmarks_from_frames([img_rgb])

            st.subheader("👁️ Facial Landmark Analysis")
            if mp_result is None:
                st.warning("MediaPipe not installed. Run `pip install mediapipe` in your terminal.")
            elif mp_result['faces_found'] == 0:
                st.warning("⚠️ No face detected in this image. Please ensure the face is clearly visible.")
            else:
                col1, col2 = st.columns(2)
                col1.metric("Eye Aspect Ratio (EAR)", mp_result['avg_ear'],
                            help="Above 0.20 = normal. Below 0.20 = possible gaze avoidance.")
                col2.metric("EAR Interpretation", "Normal" if mp_result['avg_ear'] >= 0.20 else "Low")

                if mp_result['avg_ear'] < 0.20:
                    st.warning(f"⚠️ **{mp_result['assessment']}**  \n"
                               "Low EAR detected — this may indicate eye contact avoidance, "
                               "which is one of the social communication signs assessed in autism screening.")
                else:
                    st.success(f"✅ **{mp_result['assessment']}**  \n"
                               "Eye openness appears within normal range.")

            # ── Classify the captured frame using video model ──
            st.subheader("🎯 Behavior Classification (Single Frame)")
            video_model_cam, le_cam = load_video_model()

            if video_model_cam is None:
                st.info("ℹ️ Video model not loaded. Train it from Section 6 of the notebook.")
            else:
                frame_resized = cv2.resize(img_rgb, (224, 224))
                frame_norm    = np.array([frame_resized], dtype=np.float32) / 255.0
                pred_probs    = video_model_cam.predict(frame_norm, verbose=0)[0]
                pred_idx      = np.argmax(pred_probs)
                pred_class    = le_cam.inverse_transform([pred_idx])[0]
                confidence    = pred_probs[pred_idx] * 100

                st.metric("Detected Behavior", pred_class.replace('_', ' ').title())
                st.metric("Confidence", f"{confidence:.1f}%")

                st.markdown("**All class probabilities:**")
                for cls, prob in sorted(zip(le_cam.classes_, pred_probs), key=lambda x: -x[1]):
                    st.progress(float(prob), text=f"{cls.replace('_',' ').title()}: {prob*100:.1f}%")

                st.caption("⚠️ Single-frame classification is less accurate than full video analysis.")

        st.markdown("---")
        st.caption("⚕️ Real-time webcam analysis is a research demonstration. Always consult a qualified medical professional.")

    st.markdown("""
    ---
    ### 📌 How to get better results with the webcam:
    - Make sure the face is **well-lit** and clearly visible
    - Look directly at the camera for eye contact analysis
    - For **stimming behavior detection**, use the Video tab with a full clip instead
    """)
