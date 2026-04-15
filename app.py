import streamlit as st
import numpy as np
import joblib
import tempfile
import os
import cv2
import mediapipe as mp
import tensorflow as tf

# ─────────────────────────────────────────
# PATH HELPER
# ─────────────────────────────────────────
# Using getcwd() is more reliable on Streamlit Cloud for finding uploaded .h5/.pkl files
BASE_DIR = os.getcwd()

def _model_path(filename):
    """Return the absolute path for a model file."""
    return os.path.join(BASE_DIR, filename)

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
    rf = joblib.load(_model_path('rf_model_smote.pkl'))
    scaler = joblib.load(_model_path('scaler.pkl'))
    return rf, scaler

@st.cache_resource
def load_video_model():
    """Load the CNN video model."""
    h5_path = _model_path('video_model.h5')
    if os.path.exists(h5_path):
        try:
            model = tf.keras.models.load_model(h5_path, compile=False)
            le = joblib.load(_model_path('video_label_encoder.pkl'))
            return model, le
        except Exception as e:
            st.error(f"Error loading video_model: {e}")
    return None, None

@st.cache_resource
def load_cnnlstm_model():
    """Load the CNN-LSTM temporal model."""
    h5_path = _model_path('cnnlstm_model.h5')
    if os.path.exists(h5_path):
        try:
            model = tf.keras.models.load_model(h5_path, compile=False)
            le = joblib.load(_model_path('cnnlstm_label_encoder.pkl'))
            return model, le
        except Exception as e:
            st.error(f"Error loading cnnlstm_model: {e}")
    return None, None

# Initializing global models
try:
    rf, scaler = load_questionnaire_models()
except Exception as e:
    st.error("Questionnaire models (rf_model_smote.pkl/scaler.pkl) not found in root directory.")

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def extract_frames_for_lstm(video_path, seq_len=10, img_size=224):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 5:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, seq_len, dtype=int)
    frames = []
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
    return arr[np.newaxis, ...] 


def analyse_face_landmarks_from_frames(frames_rgb):
    LEFT_EYE  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33,  160, 158, 133, 153, 144]

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        min_detection_confidence=0.5
    )
    ear_list = []

    for frame in frames_rgb:
        results = face_mesh.process(frame)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
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
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📝 Questionnaire Screening",
    "🎥 Video Behavior Analysis",
    "📷 Real-Time Webcam Detection"
])


# ══════════════════════════════════════════
# TAB 1 — QUESTIONNAIRE
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

    country_options = {"India": 52, "United States": 1, "Canada": 30, "UK": 51, "Australia": 12, "Other": 0}
    country_label  = st.selectbox("Country of residence", list(country_options.keys()))
    contry_of_res  = country_options[country_label]

    relation_options = {"Self": 1, "Parent": 2, "Relative": 3, "Health professional": 4, "Other": 0}
    relation_label = st.selectbox("Relation (who is answering?)", list(relation_options.keys()))
    relation       = relation_options[relation_label]

    ethnicity_options = {"Unknown": 0, "White-European": 1, "Middle Eastern": 2, "Pasifika": 3, "Black": 4, "Others": 5, "Hispanic": 6, "Asian": 7, "Turkish": 8, "South Asian": 9, "Latino": 10}
    ethnicity_label = st.selectbox("Ethnicity", list(ethnicity_options.keys()))
    ethnicity       = ethnicity_options[ethnicity_label]

    jaundice = st.radio("Born with Jaundice?", ["No", "Yes"])
    austim   = st.radio("Family history of autism?", ["No", "Yes"])

    if st.button("🔍 Predict ASD Risk", use_container_width=True):
        total_A_score = sum(a_scores)
        mean_A_score  = np.mean(a_scores)
        std_A_score   = np.std(a_scores)
        A_score_high_flag = 1 if total_A_score >= 6 else 0

        answers = a_scores + [age] + [1 if gender == "Male" else 0] + [int(ethnicity)] + \
                  [1 if jaundice == "Yes" else 0] + [1 if austim == "Yes" else 0] + \
                  [contry_of_res] + [relation] + \
                  [total_A_score, mean_A_score, std_A_score, A_score_high_flag]

        data = np.array([answers])
        data_scaled = scaler.transform(data)
        pred = rf.predict(data_scaled)[0]
        prob = rf.predict_proba(data_scaled)[0]

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ **Result: ASD Traits Detected** \nConfidence: {prob[1]*100:.1f}%")
        else:
            st.success(f"✅ **Result: No ASD Traits Detected** \nConfidence: {prob[0]*100:.1f}%")

        st.metric("Total AQ Score", total_A_score)
        
        st.markdown("---")
        st.subheader("🏥 DSM-5 Clinical Alignment Score")
        j_val = 1 if jaundice == "Yes" else 0
        f_val = 1 if austim == "Yes" else 0
        dom_a, dom_b, dsm_total, dsm_label = dsm5_score(a_scores, age, j_val, f_val)

        col1, col2, col3 = st.columns(3)
        col1.metric("Domain A", f"{dom_a}/4")
        col2.metric("Domain B", f"{dom_b}/3")
        col3.metric("DSM-5 Score", f"{dsm_total}/10")
        st.info(dsm_label)


# ══════════════════════════════════════════
# TAB 2 — VIDEO ANALYSIS
# ══════════════════════════════════════════
with tab2:
    st.header("Video-Based Behavior Analysis")
    uploaded_video = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov', 'mkv'])

    if uploaded_video is not None:
        st.video(uploaded_video)
        video_model, le = load_video_model()
        cnnlstm_model, le_lstm = load_cnnlstm_model()

        if video_model is None and cnnlstm_model is None:
            st.warning("⚠️ Video models (.h5) or label encoders (.pkl) not found in directory.")
        else:
            model_choice = st.radio("Choose model:", ["MobileNetV2 (frame-by-frame)", "CNN-LSTM (temporal)"])

            if st.button("🎬 Analyze Video", use_container_width=True):
                with st.spinner("Analyzing..."):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_video.read())
                    tfile.close()

                    if "MobileNetV2" in model_choice and video_model is not None:
                        cap = cv2.VideoCapture(tfile.name)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        indices = np.linspace(0, total_frames - 1, 20, dtype=int)
                        frames = []
                        for idx in indices:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                            ret, frame = cap.read()
                            if ret:
                                frame = cv2.resize(frame, (224, 224))
                                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        cap.release()

                        if frames:
                            frames_array = np.array(frames, dtype=np.float32) / 255.0
                            preds_all = video_model.predict(frames_array, verbose=0)
                            avg_probs = preds_all.mean(axis=0)
                            pred_idx = np.argmax(avg_probs)
                            pred_class = le.inverse_transform([pred_idx])[0]
                            confidence = avg_probs[pred_idx] * 100
                            
                            st.subheader("📊 Analysis Results")
                            st.metric("Detected Behavior", pred_class.title())
                            st.metric("Confidence", f"{confidence:.1f}%")

                    elif "CNN-LSTM" in model_choice and cnnlstm_model is not None:
                        seq = extract_frames_for_lstm(tfile.name, seq_len=10)
                        if seq is not None:
                            preds_seq = cnnlstm_model.predict(seq, verbose=0)
                            avg_probs = preds_seq[0]
                            pred_idx = np.argmax(avg_probs)
                            pred_class = le_lstm.inverse_transform([pred_idx])[0]
                            st.metric("Detected Behavior", pred_class.title())
                            st.metric("Confidence", f"{avg_probs[pred_idx]*100:.1f}%")

                    # MediaPipe
                    cap2 = cv2.VideoCapture(tfile.name)
                    ret, frame_test = cap2.read()
                    if ret:
                        res = analyse_face_landmarks_from_frames([cv2.cvtColor(frame_test, cv2.COLOR_BGR2RGB)])
                        st.metric("Avg Eye Aspect Ratio", res['avg_ear'])
                        st.info(res['assessment'])
                    cap2.release()
                    os.unlink(tfile.name)


# ══════════════════════════════════════════
# TAB 3 — REAL-TIME WEBCAM DETECTION
# ══════════════════════════════════════════
with tab3:
    st.header("📷 Real-Time Webcam Detection")
    webcam_image = st.camera_input("Take a photo")

    if webcam_image is not None:
        file_bytes = np.asarray(bytearray(webcam_image.read()), dtype=np.uint8)
        img_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        
        with st.spinner("Analysing..."):
            mp_result = analyse_face_landmarks_from_frames([img_rgb])
            st.subheader("👁️ Facial Analysis")
            if mp_result['faces_found'] > 0:
                st.metric("Eye Aspect Ratio (EAR)", mp_result['avg_ear'])
                st.success(mp_result['assessment'])
            else:
                st.warning("No face detected.")

            video_model_cam, le_cam = load_video_model()
            if video_model_cam:
                frame_resized = cv2.resize(img_rgb, (224, 224))
                frame_norm = np.array([frame_resized], dtype=np.float32) / 255.0
                pred_probs = video_model_cam.predict(frame_norm, verbose=0)[0]
                pred_idx = np.argmax(pred_probs)
                st.metric("Behavior", le_cam.inverse_transform([pred_idx])[0].title())
