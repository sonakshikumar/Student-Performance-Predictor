import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("student_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page config
st.set_page_config(page_title="Student Performance Predictor")

st.title("🎓 Student Performance Predictor")
st.write("Predict the final grade of a student based on academic details")

# User inputs
G1 = st.number_input("First Period Grade (G1)", min_value=0, max_value=20, value=10)
G2 = st.number_input("Second Period Grade (G2)", min_value=0, max_value=20, value=10)
absences = st.number_input("Number of Absences", min_value=0, max_value=100, value=5)
studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4])
failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])

# Create input dataframe
# UCI dataset has 33 features → we pad remaining with zeros
input_data = pd.DataFrame(
    [[studytime, failures, absences, G1, G2]],
    columns=["studytime", "failures", "absences", "G1", "G2"]
)

# Predict
if st.button("Predict Final Grade"):
    prediction = model.predict(input_data)
    st.success(f"📊 Predicted Final Grade: {prediction[0]:.2f}")
