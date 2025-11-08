import streamlit as st
import numpy as np
import joblib

# Load model and scaler
rf = joblib.load('rf_model (1).pkl')
scaler = joblib.load('scaler (1).pkl')

st.title("Autism Screening Prediction (Demo)")
st.header("Answer the following:")

def yesno(val):
    return 1 if val == "Yes" else 0

# Collect A1 to A10 scores
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

# User-friendly dropdowns for country and relation
country_options = {
    "India": 52,
    "United States": 1,
    "Canada": 30,
    "UK": 51,
    "Australia": 12,
    "Other": 0
}
country_label = st.selectbox("Country of residence", list(country_options.keys()))
contry_of_res = country_options[country_label]

relation_options = {
    "Self": 1,
    "Parent": 2,
    "Relative": 3,
    "Health professional": 4,
    "Other": 0
}
relation_label = st.selectbox("Relation (who is answering?)", list(relation_options.keys()))
relation = relation_options[relation_label]

# Other fields
age = st.number_input("Age", min_value=5, max_value=100, value=30)
gender = st.radio("Gender", ["Male", "Female"])
ethnicity = st.number_input("Ethnicity (enter code)", min_value=0, max_value=20, value=0)
jaundice = st.radio("Jaundice (Yes/No)", ["No", "Yes"])
austim = st.radio("Family history of autism (austim)", ["No", "Yes"])

# Define a_scores list properly
a_scores = [yesno(q) for q in [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]]

# Engineered features
total_A_score = sum(a_scores)
mean_A_score = np.mean(a_scores)
std_A_score = np.std(a_scores)
A_score_high_flag = 1 if total_A_score >= 6 else 0

# Build feature vector in exact order used in training
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

data = np.array([answers])
data_scaled = scaler.transform(data)

if st.button("Predict"):
    pred = rf.predict(data_scaled)[0]
    st.write("Prediction (1=ASD, 0=No ASD):", int(pred))
    st.info("This is not medical advice—just a demonstration AI model.")
