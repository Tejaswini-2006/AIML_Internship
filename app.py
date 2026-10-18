
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("student_performance_random_forest.pkl")

model = load_model()

st.title("🎓 Student Performance Predictor")
st.write(
    "Enter student details to estimate the final mathematics "
    "grade (G3) out of 20."
)

with st.form("student_form"):
    st.subheader("Student Information")

    school = st.selectbox("School", ["GP", "MS"])
    sex = st.selectbox("Sex", ["F", "M"])
    age = st.slider("Age", 15, 22, 17)
    address = st.selectbox("Address", ["U", "R"])
    famsize = st.selectbox("Family Size", ["GT3", "LE3"])
    Pstatus = st.selectbox("Parents' Cohabitation", ["T", "A"])

    Medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4])
    Fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4])

    jobs = ["teacher", "health", "services", "at_home", "other"]
    Mjob = st.selectbox("Mother's Job", jobs)
    Fjob = st.selectbox("Father's Job", jobs)

    reason = st.selectbox(
        "School Choice Reason",
        ["home", "reputation", "course", "other"]
    )
    guardian = st.selectbox(
        "Guardian", ["mother", "father", "other"]
    )

    traveltime = st.selectbox("Travel Time", [1, 2, 3, 4])
    studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4])
    failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])

    st.subheader("Support and Activities")

    schoolsup = st.selectbox("School Support", ["yes", "no"])
    famsup = st.selectbox("Family Support", ["yes", "no"])
    paid = st.selectbox("Extra Paid Classes", ["yes", "no"])
    activities = st.selectbox("Extracurricular Activities", ["yes", "no"])
    nursery = st.selectbox("Attended Nursery School", ["yes", "no"])
    higher = st.selectbox("Wants Higher Education", ["yes", "no"])
    internet = st.selectbox("Internet Access", ["yes", "no"])
    romantic = st.selectbox("In a Romantic Relationship", ["yes", "no"])

    st.subheader("Lifestyle and Attendance")

    famrel = st.slider("Family Relationship (1–5)", 1, 5, 4)
    freetime = st.slider("Free Time (1–5)", 1, 5, 3)
    goout = st.slider("Going Out (1–5)", 1, 5, 3)
    Dalc = st.slider("Weekday Alcohol Consumption (1–5)", 1, 5, 1)
    Walc = st.slider("Weekend Alcohol Consumption (1–5)", 1, 5, 1)
    health = st.slider("Health (1–5)", 1, 5, 3)
    absences = st.number_input(
        "School Absences", min_value=0, max_value=100, value=0
    )

    submitted = st.form_submit_button("Predict Final Grade")

if submitted:
    student = pd.DataFrame([{
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences
    }])

    try:
        prediction = float(model.predict(student)[0])
        prediction = max(0, min(20, prediction))

        st.success("Prediction completed!")
        st.metric("Estimated Final Grade", f"{prediction:.2f} / 20")
        st.caption(
            "This is a machine-learning estimate, not a guaranteed "
            "grade or an official student assessment."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
