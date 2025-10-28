import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="🏥 Medical Cost & Insurance Approval Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (dull, professional theme)
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-container {
        background: #f7fafc;
        padding: 2rem;
        border-radius: 1rem;
        border-left: 5px solid #4a5568;
        margin: 1rem 0;
    }
    .cost-metric, .approved-status, .not-approved-status {
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .approved-status {
        background: #a3d9b1;
        color: #1b4332;
    }
    .not-approved-status {
        background: #f8d7da;
        color: #721c24;
    }
    .info-box {
        background: #f1f3f4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #6c757d;
        margin: 1rem 0;
        color: #000000 !important;
    }
    .stButton>button {
        background: #4a5568;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        width: 100%;
    }
    .stButton>button:hover {
        background: #2d3748;
    }
    .patient-profile-header,
    h3.patient-profile-header {
        color: #ffffff !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Helper function to get model paths
def get_model_path(filename):
    current_dir = Path(__file__).parent
    model_path = current_dir.parent / "models" / filename
    return str(model_path)


# Load models and preprocessor
@st.cache_resource
def load_models():
    try:
        regression_model = joblib.load(get_model_path("regression_model.pkl"))
        classification_model = joblib.load(get_model_path("classification_model.pkl"))
        scaler = joblib.load(get_model_path("scaler.pkl"))
        feature_columns = joblib.load(get_model_path("feature_columns.pkl"))
        return regression_model, classification_model, scaler, feature_columns
    except FileNotFoundError as e:
        st.error("❌ Model files not found. Please ensure the models are trained and saved first.")
        st.error(f"Missing file: {e}")
        return None, None, None, None


# Preprocessing function
def preprocess_input(age, sex, bmi, children, smoker, region, feature_columns, scaler):
    input_data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df, columns=['sex', 'smoker', 'region'], drop_first=False)

    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[feature_columns]
    input_scaled = scaler.transform(input_encoded)
    return input_scaled


# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">🏥 Medical Cost & Insurance Approval Prediction</h1>', unsafe_allow_html=True)

    regression_model, classification_model, scaler, feature_columns = load_models()
    if regression_model is None:
        st.stop()

    # Sidebar input
    st.sidebar.header("📋 Patient Information")
    st.sidebar.markdown("*Please fill in the patient details below:*")

    age = st.sidebar.slider("👶 Age", 18, 100, 30, 1)
    bmi = st.sidebar.slider("⚖️ BMI (Body Mass Index)", 10.0, 50.0, 25.0, 0.1)
    children = st.sidebar.slider("👶 Number of Children", 0, 5, 0, 1)
    sex = st.sidebar.selectbox("⚥ Sex", ["female", "male"])
    smoker = st.sidebar.selectbox("🚬 Smoker Status", ["no", "yes"])
    region = st.sidebar.selectbox("🗺️ Region", ["southwest", "southeast", "northwest", "northeast"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 BMI Categories")
    st.sidebar.markdown("""
    - **Underweight:** BMI < 18.5  
    - **Normal:** BMI 18.5 - 24.9  
    - **Overweight:** BMI 25.0 - 29.9  
    - **Obese:** BMI ≥ 30.0
    """)

    if bmi < 18.5:
        bmi_status = "Underweight"
    elif bmi < 25:
        bmi_status = "Normal"
    elif bmi < 30:
        bmi_status = "Overweight"
    else:
        bmi_status = "Obese"

    st.sidebar.markdown(f"**Current Status:** {bmi_status}")

    # Quick prediction preview
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔮 Quick Preview")
    try:
        quick_input_scaled = preprocess_input(age, sex, bmi, children, smoker, region, feature_columns, scaler)
        quick_predicted_cost = regression_model.predict(quick_input_scaled)[0]
        quick_predicted_approval = classification_model.predict(quick_input_scaled)[0]

        st.sidebar.metric("💰 Estimated Cost", f"₹{quick_predicted_cost:,.0f}")
        approval_status = "✅ Likely Approved" if quick_predicted_approval == 1 else "❌ May be Rejected"
        st.sidebar.markdown(f"**🎭 Status:** {approval_status}")

    except Exception as e:
        st.sidebar.error(f"Preview unavailable: {str(e)[:50]}...")

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<h3 class="patient-profile-header">👤 Patient Profile Summary</h3>', unsafe_allow_html=True)
        profile_info = f"""
        <div class="info-box">
        <h4>📋 Patient Details:</h4>
        <ul>
        <li><strong>Age:</strong> {age} years old</li>
        <li><strong>Sex:</strong> {sex.title()}</li>
        <li><strong>BMI:</strong> {bmi} ({bmi_status})</li>
        <li><strong>Children:</strong> {children}</li>
        <li><strong>Smoker:</strong> {'Yes' if smoker == 'yes' else 'No'}</li>
        <li><strong>Region:</strong> {region.title()}</li>
        </ul>
        </div>
        """
        st.markdown(profile_info, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🚨 Risk Factors")
        risk_factors = []
        if smoker == "yes":
            risk_factors.append("🚬 Tobacco use")
        if bmi >= 30:
            risk_factors.append("⚖️ Obesity")
        if age >= 50:
            risk_factors.append("👴 Advanced age")

        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.markdown("✅ No major risk factors identified")

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🔮 Generate Prediction", key="predict_button"):
            try:
                input_scaled = preprocess_input(age, sex, bmi, children, smoker, region, feature_columns, scaler)
                predicted_cost = regression_model.predict(input_scaled)[0]
                predicted_approval = classification_model.predict(input_scaled)[0]
                approval_probability = classification_model.predict_proba(input_scaled)[0]

                st.markdown("## 🎯 Prediction Results")

                # Cost display
                st.markdown(f"""
                    <div style="background:#d1e7dd;color:#1b4332;
                    padding:1.5rem;border-radius:1rem;text-align:center;">
                    <h1>₹{predicted_cost:,.2f}</h1>
                    <p>Estimated Annual Premium</p></div>
                """, unsafe_allow_html=True)

                cost_per_month = predicted_cost / 12
                st.metric("Monthly Premium", f"₹{cost_per_month:,.2f}")

                # Approval Status
                st.markdown("---")
                st.markdown("### 🎭 Approval Status")

                if predicted_approval == 1:
                    approval_html = """
                    <div class="approved-status">
                        <h2>✅ APPROVED</h2>
                        <p>Insurance application likely to be approved</p>
                    </div>
                    """
                    confidence = approval_probability[1]
                else:
                    approval_html = """
                    <div class="not-approved-status">
                        <h2>❌ NOT APPROVED</h2>
                        <p>Insurance application may be rejected</p>
                    </div>
                    """
                    confidence = approval_probability[0]

                st.markdown(approval_html, unsafe_allow_html=True)
                st.progress(confidence)
                st.markdown(f"**Confidence:** {confidence:.1%}")

            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")

    # About Section
    st.markdown("---")
    st.markdown("## ℹ️ About This Application")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🎯 Purpose
        - Predict insurance costs  
        - Assess approval likelihood  
        - Provide insights into cost drivers
        """)

    with col2:
        st.markdown("""
        ### 🧠 Models Used
        - Linear Regression (cost prediction)  
        - Random Forest (approval classification)  
        - StandardScaler for normalization
        """)

    with col3:
        st.markdown("""
        ### ⚙️ Features
        - Real-time prediction  
        - Risk factor analysis  
        - Confidence scoring
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#777;">
        <p><strong>🏥 Medical Cost & Insurance Approval Prediction System</strong></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()