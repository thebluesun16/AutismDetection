import streamlit as st
import numpy as np
import joblib
import tempfile
import os

# ──────────────────────────────────────────────────────────
# SAFE IMPORTS
# ──────────────────────────────────────────────────────────
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ──────────────────────────────────────────────────────────
# KERAS COMPATIBILITY PATCH
# ──────────────────────────────────────────────────────────
if TF_AVAILABLE:
    from tensorflow.keras.layers import Dense as _OrigDense
    from tensorflow.keras.layers import Conv2D as _OrigConv2D
    from tensorflow.keras.layers import LSTM as _OrigLSTM

    class _CompatDense(_OrigDense):
        def __init__(self, *args, quantization_config=None, **kwargs):
            super().__init__(*args, **kwargs)

    class _CompatConv2D(_OrigConv2D):
        def __init__(self, *args, quantization_config=None, **kwargs):
            super().__init__(*args, **kwargs)

    class _CompatLSTM(_OrigLSTM):
        def __init__(self, *args, quantization_config=None, **kwargs):
            super().__init__(*args, **kwargs)

    COMPAT_OBJECTS = {
        'Dense':  _CompatDense,
        'Conv2D': _CompatConv2D,
        'LSTM':   _CompatLSTM,
    }
else:
    COMPAT_OBJECTS = {}

# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Autism Screening Tool", page_icon="🧠", layout="centered")
st.title("🧠 Autism Screening Prediction")
st.caption("Final Review Project | Bharati Vidyapeeth's College of Engineering")

# ──────────────────────────────────────────────────────────
# MODEL LOADING (CACHED)
# ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _path(filename):
    return os.path.join(BASE_DIR, filename)

@st.cache_resource
def load_all_models():
    models = {}

    # --- Questionnaire models ---
    rf_path     = _path('rf_model_smote.pkl')
    scaler_path = _path('scaler.pkl')

    if os.path.exists(rf_path) and os.path.exists(scaler_path):
        try:
            models['rf']     = joblib.load(rf_path)
            models['scaler'] = joblib.load(scaler_path)
        except Exception as e:
            st.warning(f"⚠️ Could not load questionnaire model: {e}")
    else:
        st.warning("⚠️ rf_model_smote.pkl or scaler.pkl not found in repo root.")

    # --- Video model (MobileNetV2 frame-by-frame) ---
    if TF_AVAILABLE:
        video_keras = _path('video_model.keras')
        video_h5    = _path('video_model.h5')
        video_le    = _path('video_label_encoder.pkl')

        video_path = video_keras if os.path.exists(video_keras) else video_h5

        if os.path.exists(video_path) and os.path.exists(video_le):
            try:
                models['video']    = tf.keras.models.load_model(
                                         video_path,
                                         compile=False,
                                         custom_objects=COMPAT_OBJECTS
                                     )
                models['video_le'] = joblib.load(video_le)
            except Exception as e:
                st.warning(f"⚠️ Could not load video model: {e}")
        else:
            st.warning("⚠️ video_model (.keras or .h5) or video_label_encoder.pkl not found.")

        # --- CNN-LSTM model (temporal patterns) ---
        cnnlstm_h5 = _path('cnnlstm_model.h5')
        cnnlstm_le = _path('cnnlstm_label_encoder.pkl')

        if os.path.exists(cnnlstm_h5) and os.path.exists(cnnlstm_le):
            try:
                models['cnnlstm']    = tf.keras.models.load_model(
                                           cnnlstm_h5,
                                           compile=False,
                                           custom_objects=COMPAT_OBJECTS
                                       )
                models['cnnlstm_le'] = joblib.load(cnnlstm_le)
            except Exception as e:
                st.warning(f"⚠️ Could not load CNN-LSTM model: {e}")
        else:
            st.info("ℹ️ cnnlstm_model.h5 not found — temporal analysis will be skipped.")

    return models

models = load_all_models()

# ──────────────────────────────────────────────────────────
# DSM-5 SCORING FUNCTION
# ──────────────────────────────────────────────────────────
def dsm5_clinical_score(a_scores, age, jaundice, family_history):
    domain_a = a_scores[1] + a_scores[4] + a_scores[5] + a_scores[8]
    domain_b = a_scores[0] + a_scores[6] + a_scores[7]

    risk_bonus = 0
    if jaundice       == 1: risk_bonus += 0.5
    if family_history == 1: risk_bonus += 1.0
    if age <= 10           : risk_bonus += 0.5

    raw   = domain_a + domain_b + risk_bonus
    score = round(min(raw, 10), 1)

    if score >= 6:   cat = "🔴 High DSM-5 Alignment (Likely ASD traits)"
    elif score >= 4: cat = "🟡 Moderate DSM-5 Alignment (Possible ASD traits)"
    else:            cat = "🟢 Low DSM-5 Alignment (Unlikely ASD traits)"

    return domain_a, domain_b, score, cat

# ──────────────────────────────────────────────────────────
# EAR + MAR ANALYSIS  (Eye & Mouth Aspect Ratios)
# ──────────────────────────────────────────────────────────
def compute_ear(landmarks, eye_indices):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def compute_mar(landmarks, mouth_indices):
    """
    Mouth Aspect Ratio using 4 key mouth landmarks.
    High MAR → mouth open; low MAR → mouth closed.
    Unusual or frozen MAR can indicate atypical expression patterns.
    """
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in mouth_indices])
    vert  = np.linalg.norm(pts[0] - pts[1])   # top-to-bottom
    horiz = np.linalg.norm(pts[2] - pts[3])   # left-to-right
    return vert / (horiz + 1e-6)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [13,  14,  78,  308]   # top-lip, bottom-lip, left-corner, right-corner

def get_ear_mar_analysis(video_path):
    if not CV2_AVAILABLE or not MP_AVAILABLE:
        return None, None, None, "OpenCV / MediaPipe not available in this environment."

    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh    = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        cap          = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return None, None, None, "Could not read video frames — check file format."

        ear_values = []
        mar_values = []
        sample_idx = np.linspace(0, total_frames - 1, min(30, total_frames), dtype=int)

        for idx in sample_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm  = results.multi_face_landmarks[0].landmark
                ear = (compute_ear(lm, LEFT_EYE) + compute_ear(lm, RIGHT_EYE)) / 2.0
                mar = compute_mar(lm, MOUTH)
                ear_values.append(ear)
                mar_values.append(mar)

        cap.release()
        face_mesh.close()

        if not ear_values:
            return None, None, None, "No face detected in the video. Ensure the face is clearly visible."

        avg_ear    = float(np.mean(ear_values))
        avg_mar    = float(np.mean(mar_values))
        blink_rate = sum(1 for e in ear_values if e < 0.20) / len(ear_values) * 100
        status     = "Normal Eye Contact" if avg_ear > 0.22 else "Reduced Eye Contact / Gaze Avoidance"
        return avg_ear, avg_mar, blink_rate, status

    except Exception as e:
        return None, None, None, f"Analysis error: {e}"

# ──────────────────────────────────────────────────────────
# VIDEO MODEL PREDICTION  (MobileNetV2 — frame-by-frame)
# ──────────────────────────────────────────────────────────
def predict_video_model(video_path, model, le):
    try:
        cap          = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_preds  = []

        sample_idx = np.linspace(0, total_frames - 1, min(20, total_frames), dtype=int)

        for idx in sample_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            img = cv2.resize(frame, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img, verbose=0)
            frame_preds.append(pred[0])

        cap.release()

        if not frame_preds:
            return None, None

        avg_pred   = np.mean(frame_preds, axis=0)
        class_idx  = int(np.argmax(avg_pred))
        label      = le.inverse_transform([class_idx])[0]
        confidence = float(avg_pred[class_idx]) * 100
        return label, confidence

    except Exception as e:
        return None, None

# ──────────────────────────────────────────────────────────
# CNN-LSTM MODEL PREDICTION  (temporal sequence)
# ──────────────────────────────────────────────────────────
def predict_cnnlstm_model(video_path, model, le):
    """
    CNN-LSTM expects a sequence of frames as input: (1, n_frames, H, W, 3).
    Matches the training approach from Section 10 of the notebook.
    """
    try:
        cap          = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return None, None

        # Determine expected sequence length from model input shape
        # Model input: (None, timesteps, H, W, C)
        try:
            n_frames = model.input_shape[1]   # timesteps dimension
            img_size = model.input_shape[2]   # spatial H
        except Exception:
            n_frames = 20
            img_size = 224

        sample_idx = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        frames = []

        for idx in sample_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                # Pad with a blank frame if read fails
                frames.append(np.zeros((img_size, img_size, 3), dtype=np.float32))
                continue

            frame = cv2.resize(frame, (img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)

        cap.release()

        if len(frames) < 2:
            return None, None

        # Pad or trim to exactly n_frames
        while len(frames) < n_frames:
            frames.append(np.zeros((img_size, img_size, 3), dtype=np.float32))
        frames = frames[:n_frames]

        seq        = np.expand_dims(np.array(frames), axis=0)   # (1, n_frames, H, W, 3)
        pred       = model.predict(seq, verbose=0)[0]
        class_idx  = int(np.argmax(pred))
        label      = le.inverse_transform([class_idx])[0]
        confidence = float(pred[class_idx]) * 100
        return label, confidence

    except Exception as e:
        return None, None

# ──────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📝 Questionnaire", "🎥 Video Analysis"])

# ══════════════════════════════════════════════════════════
# TAB 1 — QUESTIONNAIRE
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("AQ-10 Autism Screening Questionnaire")
    st.info("Answer each question honestly. This tool does **not** replace a clinical diagnosis.")
    st.markdown("**Answer 0 (Disagree / Rarely) or 1 (Agree / Often) for each question:**")

    questions = [
        "Q1. I notice small sounds when others do not.",
        "Q2. I find it hard to see the 'big picture' when looking at details.",
        "Q3. I find it easy to do more than one thing at once.",
        "Q4. If there is an interruption, I can switch back easily.",
        "Q5. I find it hard to read between the lines.",
        "Q6. I know how to tell if someone is bored.",
        "Q7. When reading a story, I find it hard to figure out characters' intentions.",
        "Q8. I like to collect information about categories of things.",
        "Q9. I find it easy to work out what someone is thinking by looking at them.",
        "Q10. I find social situations easy.",
    ]

    a_scores = []
    cols = st.columns(2)
    for i, q in enumerate(questions):
        with cols[i % 2]:
            val = st.selectbox(q, options=[0, 1], key=f"aq_{i}")
            a_scores.append(val)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        age    = st.number_input("Age", min_value=2, max_value=100, value=25)
    with col_b:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    col_c, col_d = st.columns(2)
    with col_c:
        jaundice       = st.selectbox("Born with Jaundice?", ["No", "Yes"])
    with col_d:
        family_history = st.selectbox("Family history of Autism?", ["No", "Yes"])

    jaundice_val       = 1 if jaundice == "Yes" else 0
    family_history_val = 1 if family_history == "Yes" else 0
    gender_val         = 1 if gender == "Male" else 0

    if st.button("🔍 Predict ASD Risk", use_container_width=True):
        aq_total = sum(a_scores)

        dom_a, dom_b, dsm5_score, dsm5_cat = dsm5_clinical_score(
            a_scores, age, jaundice_val, family_history_val
        )

        st.markdown("---")
        st.subheader("📊 Results")

        # --- FIX: Show Domain B score too (was missing from original) ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AQ-10 Total Score",    f"{aq_total} / 10")
        col2.metric("DSM-5 Approx. Score",  f"{dsm5_score} / 10")
        col3.metric("Domain A (Social)",    f"{dom_a} / 4")
        col4.metric("Domain B (Behaviour)", f"{dom_b} / 3")

        st.markdown(f"**DSM-5 Category:** {dsm5_cat}")

        if 'rf' in models and 'scaler' in models:
            try:
                # --- Feature engineering to match notebook training ---
                a_arr         = np.array(a_scores)
                mean_a        = float(a_arr.mean())
                std_a         = float(a_arr.std())
                high_flag     = int(aq_total >= 6)

                features = np.array(
                    a_scores + [age, jaundice_val, gender_val, family_history_val,
                                aq_total, mean_a, std_a, high_flag]
                ).reshape(1, -1)

                expected = models['scaler'].n_features_in_
                if features.shape[1] < expected:
                    pad      = np.zeros((1, expected - features.shape[1]))
                    features = np.hstack([features, pad])
                elif features.shape[1] > expected:
                    features = features[:, :expected]

                features_scaled = models['scaler'].transform(features)
                pred            = models['rf'].predict(features_scaled)[0]
                proba           = models['rf'].predict_proba(features_scaled)[0]
                confidence      = round(float(max(proba)) * 100, 1)

                if pred == 1:
                    st.error(f"🔴 **Model Prediction: High ASD Risk** ({confidence}% confidence)")
                else:
                    st.success(f"🟢 **Model Prediction: Low ASD Risk** ({confidence}% confidence)")

            except Exception as e:
                st.warning(f"Model prediction failed: {e}")
                if aq_total >= 6:
                    st.error("🔴 **Rule-based: High ASD Risk** (AQ-10 ≥ 6)")
                else:
                    st.success("🟢 **Rule-based: Low ASD Risk** (AQ-10 < 6)")
        else:
            if aq_total >= 6:
                st.error("🔴 **Screening Result: High ASD Risk** (AQ-10 ≥ 6)")
            else:
                st.success("🟢 **Screening Result: Low ASD Risk** (AQ-10 < 6)")

        st.caption("⚠️ This is a research screening tool — NOT a clinical diagnosis.")

# ══════════════════════════════════════════════════════════
# TAB 2 — VIDEO ANALYSIS
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("Behavioral Video Analysis")
    st.info("Upload a short video (5–30 sec) of the person in a social/play setting.")

    uploaded_file = st.file_uploader("Upload Video (.mp4 / .mov / .avi)", type=['mp4', 'mov', 'avi'])

    if uploaded_file:
        st.video(uploaded_file)

        if st.button("▶️ Run AI Analysis", use_container_width=True):
            with st.spinner("Analysing video frames... please wait."):

                suffix     = os.path.splitext(uploaded_file.name)[-1] or ".mp4"
                tfile      = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tfile.write(uploaded_file.getbuffer())
                tfile.flush()
                video_path = tfile.name

                # --- EAR + MAR analysis ---
                avg_ear, avg_mar, blink_rate, ear_status = get_ear_mar_analysis(video_path)

                st.markdown("---")
                st.subheader("📊 Analysis Results")

                # ── Eye & Mouth metrics ──
                if avg_ear is not None:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Avg EAR Score",  f"{avg_ear:.3f}")
                    col2.metric("Avg MAR Score",  f"{avg_mar:.3f}")
                    col3.metric("Blink Rate",     f"{blink_rate:.1f}%")
                    col4.metric("Gaze Status",    ear_status)

                    if avg_ear > 0.25:
                        st.success("✅ Eye contact appears **normal**.")
                    elif avg_ear > 0.18:
                        st.warning("Reduce")
                    else:
                        st.error("🔴 **Significant gaze avoidance** detected.")

                    # MAR interpretation
                    if avg_mar < 0.05:
                        st.info("👄 Mouth mostly closed — limited facial expression detected.")
                    elif avg_mar > 0.30:
                        st.warning("👄 Mouth frequently open — possible atypical expression pattern.")
                    else:
                        st.success("👄 Mouth expression appears within typical range.")

                else:
                    st.warning(f"EAR/MAR Analysis: {ear_status}")

                st.divider()

                # ── MobileNetV2 frame-by-frame prediction ──
                if 'video' in models and 'video_le' in models and CV2_AVAILABLE:
                    with st.spinner("Running MobileNetV2 frame analysis..."):
                        label, confidence = predict_video_model(
                            video_path, models['video'], models['video_le']
                        )
                    if label is not None:
                        st.markdown(f"**🖼️ MobileNetV2 Prediction:** `{label}` — {confidence:.1f}% confidence")
                    else:
                        st.warning("MobileNetV2 model could not process the video frames.")
                elif 'video' not in models:
                    st.info("ℹ️ MobileNetV2 model not loaded — showing EAR/MAR analysis only.")

                # ── CNN-LSTM temporal prediction ──
                if 'cnnlstm' in models and 'cnnlstm_le' in models and CV2_AVAILABLE:
                    with st.spinner("Running CNN-LSTM temporal analysis..."):
                        cnnlstm_label, cnnlstm_conf = predict_cnnlstm_model(
                            video_path, models['cnnlstm'], models['cnnlstm_le']
                        )
                    if cnnlstm_label is not None:
                        st.markdown(
                            f"**🎬 CNN-LSTM Temporal Prediction:** `{cnnlstm_label}` "
                            f"— {cnnlstm_conf:.1f}% confidence"
                        )
                        st.caption(
                            "CNN-LSTM analyses motion patterns across the full video sequence "
                            "(arm flapping, headbanging, spinning)."
                        )
                    else:
                        st.warning("CNN-LSTM model could not process the video sequence.")

                try:
                    os.unlink(video_path)
                except Exception:
                    pass

        st.caption("⚠️ This tool is for research purposes only and does not constitute a medical diagnosis.")
