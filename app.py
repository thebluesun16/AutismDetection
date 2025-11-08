import streamlit as st
import numpy as np
import joblib

# Load model and scaler
rf = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Autism Screening Prediction (Demo)")

st.header("Answer the following:")

# Real AQ-10 Adult Screening Questions (True=Yes, False=No)
q1 = st.radio("1. I often notice small sounds when others do not.", ('No', 'Yes'))
q2 = st.radio("2. I usually concentrate more on the whole picture, rather than the small details.", ('No', 'Yes'))
q3 = st.radio("3. I find it easy to do more than one thing at once.", ('No', 'Yes'))
q4 = st.radio("4. If there is an interruption, I can switch back to what I was doing very quickly.", ('No', 'Yes'))
q5 = st.radio("5. I find it easy to 'read between the lines' when someone is talking to me.", ('No', 'Yes'))
q6 = st.radio("6. I know how to tell if someone listening to me is getting bored.", ('No', 'Yes'))
q7 = st.radio("7. When I’m reading a story, I find it difficult to work out the characters’ intentions.", ('No', 'Yes'))
q8 = st.radio("8. I like to collect information about categories of things (e.g., types of car, types of bird, types of train, types of plant etc).", ('No', 'Yes'))
q9 = st.radio("9. I find it easy to work out what someone is thinking or feeling just by looking at their face.", ('No', 'Yes'))
q10 = st.radio("10. I find it difficult to make new friends.", ('No', 'Yes'))

age = st.number_input("Age", min_value=5, max_value=100, value=30)
gender = st.radio("Gender", ["Male", "Female"])

# Data Processing: Encode answers and categorical features as required by your model
def yesno(val): return 1 if val == 'Yes' else 0

answers = [
    yesno(q1), yesno(q2), yesno(q3), yesno(q4), yesno(q5),
    yesno(q6), yesno(q7), yesno(q8), yesno(q9), yesno(q10),
    age,
    1 if gender == "Male" else 0
]
data = np.array([answers])
data_scaled = scaler.transform(data)

if st.button("Predict"):
    pred = rf.predict(data_scaled)[0]
    st.write("Prediction (1=ASD, 0=No ASD):", int(pred))
    st.info("This is not medical advice—just a demonstration AI model.")
