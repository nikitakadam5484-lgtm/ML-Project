
import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "diabetes_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))
# ---------------- BACKGROUND ---------------- #

st.markdown("""
<div style="
background: linear-gradient(90deg,#009688,#4db6ac);
padding:20px;
border-radius:15px;
text-align:center;
margin-bottom:20px;
">

<h1 style="color:white;">
🩺 Diabetes Prediction System
</h1>

<p style="color:white;font-size:18px;">
Predict Diabetes Risk Using Machine Learning
</p>

</div>
""", unsafe_allow_html=True)

# ---------------- BANNER ---------------- #

st.image(
    "https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=1200",
    use_container_width=True
)

# ---------------- TITLE ---------------- #

st.markdown(
    '<p class="title">🩺 Diabetes Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict Diabetes Risk using Machine Learning</p>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("🏥 About Project")

st.sidebar.info("""
This application predicts whether a patient
is likely to have diabetes using a trained
Random Forest Machine Learning model.

Dataset:
PIMA Indian Diabetes Dataset
""")

# ---------------- INPUT FORM ---------------- #

st.subheader("📋 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=300,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=1000,
        value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=100.0,
        value=25.0
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=5.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=25
    )

# ---------------- PREDICTION ---------------- #

with st.expander("🔄 Project Workflow"):

    st.markdown("""
    ### System Flow

    Dataset Collection

    ⬇

    Data Cleaning

    ⬇

    Feature Scaling

    ⬇

    Random Forest Model

    ⬇

    Diabetes Prediction

    ⬇

    Medical Report Generation
    """)

    with st.expander("📚 About Diabetes"):

     st.markdown("""
    **Diabetes** is a chronic disease that occurs when the body cannot properly regulate blood sugar levels.

    ### Common Symptoms
    - Frequent urination
    - Increased thirst
    - Fatigue
    - Blurred vision
    - Unexplained weight loss

    ### Risk Factors
    - Obesity
    - Family history
    - Physical inactivity
    - High blood pressure
    - Poor diet

    ### Prevention
    - Healthy diet
    - Regular exercise
    - Weight management
    - Routine medical checkups
    """)

if st.button("🔍 Predict Diabetes", use_container_width=True):

    input_data = pd.DataFrame({
        "Pregnancies":[pregnancies],
        "Glucose":[glucose],
        "BloodPressure":[blood_pressure],
        "SkinThickness":[skin_thickness],
        "Insulin":[insulin],
        "BMI":[bmi],
        "DPF":[dpf],
        "Age":[age]
    })

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]

    probability = model.predict_proba(scaled_data)

    diabetes_prob = probability[0][1] * 100

    st.markdown("## 📈 Prediction Result")

    if prediction == 1:

        st.markdown(
            f"""
            <div class="result-bad">
            ⚠ HIGH RISK OF DIABETES
            <br><br>
            Risk Score: {diabetes_prob:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        st.warning("""
    🩺 Health Recommendations

    • Consult a healthcare professional.
    • Monitor blood sugar regularly.
    • Reduce sugar and processed foods.
    • Exercise at least 30 minutes daily.
    • Maintain a healthy weight.
    • Drink sufficient water.
    • Get regular health checkups.
    """)

    else:

        st.markdown(
            f"""
            <div class="result-good">
            ✅ LOW RISK OF DIABETES
            <br><br>
            Confidence: {100-diabetes_prob:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.success("""
    🌿 Healthy Lifestyle Suggestions

    • Continue balanced nutrition.
    • Exercise regularly.
    • Maintain healthy BMI.
    • Limit sugary beverages.
    • Get routine health screenings.
    • Stay hydrated.
    • Prioritize quality sleep.
    """)

    # Probability Cards

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Diabetes Probability",
            f"{diabetes_prob:.2f}%"
        )

    with c2:
        st.metric(
            "No Diabetes Probability",
            f"{100-diabetes_prob:.2f}%"
        )

    st.markdown("---")

    # Gauge Meter

    st.subheader("📊 Diabetes Risk Meter")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=diabetes_prob,
        title={'text': "Risk Percentage"},
        gauge={
            'axis': {'range': [0,100]},
            'steps': [
                {'range': [0,30], 'color': 'lightgreen'},
                {'range': [30,70], 'color': 'yellow'},
                {'range': [70,100], 'color': 'salmon'}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Patient Summary

    st.subheader("👤 Patient Summary")

    c1, c2 = st.columns(2)

    with c1:
        st.info(f"Age : {age}")
        st.info(f"BMI : {bmi}")

    with c2:
        st.info(f"Glucose : {glucose}")
        st.info(f"Blood Pressure : {blood_pressure}")

    st.markdown("---")

    # PDF REPORT

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", size=14)

    pdf.cell(
        200,
        10,
        txt="Diabetes Prediction Report",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Age : {age}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Glucose : {glucose}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"BMI : {bmi}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Diabetes Risk : {diabetes_prob:.2f}%",
        ln=True
    )

    pdf.output("report.pdf")

    with open("report.pdf", "rb") as file:

        st.download_button(
            "📄 Download Medical Report",
            file,
            file_name="Diabetes_Report.pdf"
        )

        st.markdown("---")

st.markdown("""
<div style='text-align:center; padding:15px;'>

<h4>🩺 Diabetes Prediction System</h4>

<p>
Machine Learning Based Healthcare Application
</p>

<p style='color:gray;'>
Developed by Nikita Kadam, Poorva Jangam, Saee Jambhale | B.Tech Computer Science & Engineering
</p>

<p style='color:gray;'>
© 2026 All Rights Reserved
</p>

</div>
""", unsafe_allow_html=True)
