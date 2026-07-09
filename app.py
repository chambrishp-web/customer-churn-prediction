import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# Load Machine Learning Model
# ==========================================
model = joblib.load("customer_churn_model.pkl")
feature_names = pd.read_csv("feature_names.csv").iloc[:, 0].tolist()


def prepare_features_for_model(df):
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]

    if all(col in df.columns for col in feature_names):
        return df[feature_names].copy()

    required_raw_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]

    missing_columns = [col for col in required_raw_columns if col not in df.columns]
    if missing_columns:
        st.error(
            "Unsupported upload. Please upload a CSV with the telecom customer columns such as: "
            + ", ".join(required_raw_columns)
        )
        st.stop()

    encoded = {}

    encoded["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(float)
    encoded["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(float)
    encoded["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0).astype(float)
    encoded["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0).astype(float)

    encoded["gender_Male"] = df["gender"].astype(str).str.strip().eq("Male").astype(int)
    encoded["Partner_Yes"] = df["Partner"].astype(str).str.strip().eq("Yes").astype(int)
    encoded["Dependents_Yes"] = df["Dependents"].astype(str).str.strip().eq("Yes").astype(int)
    encoded["PhoneService_Yes"] = df["PhoneService"].astype(str).str.strip().eq("Yes").astype(int)

    encoded["MultipleLines_No phone service"] = df["MultipleLines"].astype(str).str.strip().eq("No phone service").astype(int)
    encoded["MultipleLines_Yes"] = df["MultipleLines"].astype(str).str.strip().eq("Yes").astype(int)

    encoded["InternetService_Fiber optic"] = df["InternetService"].astype(str).str.strip().eq("Fiber optic").astype(int)
    encoded["InternetService_No"] = df["InternetService"].astype(str).str.strip().eq("No").astype(int)

    encoded["OnlineSecurity_No internet service"] = df["OnlineSecurity"].astype(str).str.strip().eq("No internet service").astype(int)
    encoded["OnlineSecurity_Yes"] = df["OnlineSecurity"].astype(str).str.strip().eq("Yes").astype(int)

    encoded["OnlineBackup_No internet service"] = df["OnlineBackup"].astype(str).str.strip().eq("No internet service").astype(int)
    encoded["OnlineBackup_Yes"] = df["OnlineBackup"].astype(str).str.strip().eq("Yes").astype(int)

    encoded["DeviceProtection_No internet service"] = df["DeviceProtection"].astype(str).str.strip().eq("No internet service").astype(int)
    encoded["DeviceProtection_Yes"] = df["DeviceProtection"].astype(str).str.strip().eq("Yes").astype(int)

    encoded["TechSupport_No internet service"] = df["TechSupport"].astype(str).str.strip().eq("No internet service").astype(int)
    encoded["TechSupport_Yes"] = df["TechSupport"].astype(str).str.strip().eq("Yes").astype(int)

    encoded["StreamingTV_No internet service"] = df["StreamingTV"].astype(str).str.strip().eq("No internet service").astype(int)
    encoded["StreamingTV_Yes"] = df["StreamingTV"].astype(str).str.strip().eq("Yes").astype(int)

    encoded["StreamingMovies_No internet service"] = df["StreamingMovies"].astype(str).str.strip().eq("No internet service").astype(int)
    encoded["StreamingMovies_Yes"] = df["StreamingMovies"].astype(str).str.strip().eq("Yes").astype(int)

    encoded["Contract_One year"] = df["Contract"].astype(str).str.strip().eq("One year").astype(int)
    encoded["Contract_Two year"] = df["Contract"].astype(str).str.strip().eq("Two year").astype(int)

    encoded["PaperlessBilling_Yes"] = df["PaperlessBilling"].astype(str).str.strip().eq("Yes").astype(int)

    encoded["PaymentMethod_Credit card (automatic)"] = df["PaymentMethod"].astype(str).str.strip().eq("Credit card (automatic)").astype(int)
    encoded["PaymentMethod_Electronic check"] = df["PaymentMethod"].astype(str).str.strip().eq("Electronic check").astype(int)
    encoded["PaymentMethod_Mailed check"] = df["PaymentMethod"].astype(str).str.strip().eq("Mailed check").astype(int)

    prepared = pd.DataFrame(encoded)
    return prepared.reindex(columns=feature_names, fill_value=0)

# ==========================================
# Logo
# ==========================================
if os.path.exists("logo.png"):
    st.image("logo.png", width=170)

# ==========================================
# Title
# ==========================================
st.title("📊 Customer Churn Prediction System")
st.caption("Predict whether a telecom customer is likely to leave the company.")

st.divider()

# ==========================================
# Sidebar
# ==========================================
st.sidebar.title("📊 Analytics Dashboard")
st.sidebar.success("Version 1.0")

st.sidebar.info("Built using")

st.sidebar.write("✔ Python")

st.sidebar.write("✔ Streamlit")

st.sidebar.write("✔ Pandas")

st.sidebar.write("✔ Scikit-Learn")

st.sidebar.write("✔ Random Forest")
st.sidebar.success("Customer Churn Prediction")

st.sidebar.write("### 👨 Developer")
st.sidebar.info("Chambrish Prabhu")

st.sidebar.write("### 🤖 Model")
st.sidebar.success("Random Forest Classifier")

st.sidebar.write("### 🎯 Accuracy")
st.sidebar.success("82.11 %")

st.sidebar.write("### 📌 Purpose")

st.sidebar.write("""
Predict customers who are likely to churn.

This helps companies retain valuable customers by providing offers before they leave.
""")

# ==========================================
# Customer Information
# ==========================================
st.header("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )

    MonthlyCharges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    TotalCharges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=850.0
    )

with col2:

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    phone = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple = st.selectbox(
        "Multiple Lines",
        ["No",
         "Yes",
         "No phone service"]
    )

st.divider()

# ==========================================
# Internet Services
# ==========================================
st.header("🌐 Internet Services")

internet = st.selectbox(
    "Internet Service",
    ["DSL",
     "Fiber optic",
     "No"]
)

def internet_option(title):

    option = st.selectbox(
        title,
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    yes = 1 if option == "Yes" else 0
    no_service = 1 if option == "No internet service" else 0

    return yes, no_service


OnlineSecurity_Yes, OnlineSecurity_No = internet_option("Online Security")

OnlineBackup_Yes, OnlineBackup_No = internet_option("Online Backup")

DeviceProtection_Yes, DeviceProtection_No = internet_option("Device Protection")

TechSupport_Yes, TechSupport_No = internet_option("Tech Support")

StreamingTV_Yes, StreamingTV_No = internet_option("Streaming TV")

StreamingMovies_Yes, StreamingMovies_No = internet_option("Streaming Movies")

st.divider()

# ==========================================
# Contract Details
# ==========================================
st.header("📄 Contract Details")

contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

paperless = st.selectbox(
    "Paperless Billing",
    [
        "No",
        "Yes"
    ]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

st.divider()

# ==========================================
# Predict Button
# ==========================================
predict = st.button(
    "🔍 Predict Customer Churn",
    use_container_width=True
)
# ==========================================
# Prediction
# ==========================================

if predict:

    # Encode Inputs
    gender_Male = 1 if gender == "Male" else 0
    SeniorCitizen = 1 if senior == "Yes" else 0
    Partner_Yes = 1 if partner == "Yes" else 0
    Dependents_Yes = 1 if dependents == "Yes" else 0
    PhoneService_Yes = 1 if phone == "Yes" else 0

    MultipleLines_Yes = 1 if multiple == "Yes" else 0
    MultipleLines_No_phone_service = 1 if multiple == "No phone service" else 0

    InternetService_Fiber_optic = 1 if internet == "Fiber optic" else 0
    InternetService_No = 1 if internet == "No" else 0

    Contract_One_year = 1 if contract == "One year" else 0
    Contract_Two_year = 1 if contract == "Two year" else 0

    PaperlessBilling_Yes = 1 if paperless == "Yes" else 0

    PaymentMethod_Credit_card_automatic = 1 if payment == "Credit card (automatic)" else 0
    PaymentMethod_Electronic_check = 1 if payment == "Electronic check" else 0
    PaymentMethod_Mailed_check = 1 if payment == "Mailed check" else 0

    # Create Input DataFrame
    input_data = pd.DataFrame([{

        "SeniorCitizen": SeniorCitizen,
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,

        "gender_Male": gender_Male,
        "Partner_Yes": Partner_Yes,
        "Dependents_Yes": Dependents_Yes,

        "PhoneService_Yes": PhoneService_Yes,

        "MultipleLines_No phone service": MultipleLines_No_phone_service,
        "MultipleLines_Yes": MultipleLines_Yes,

        "InternetService_Fiber optic": InternetService_Fiber_optic,
        "InternetService_No": InternetService_No,

        "OnlineSecurity_No internet service": OnlineSecurity_No,
        "OnlineSecurity_Yes": OnlineSecurity_Yes,

        "OnlineBackup_No internet service": OnlineBackup_No,
        "OnlineBackup_Yes": OnlineBackup_Yes,

        "DeviceProtection_No internet service": DeviceProtection_No,
        "DeviceProtection_Yes": DeviceProtection_Yes,

        "TechSupport_No internet service": TechSupport_No,
        "TechSupport_Yes": TechSupport_Yes,

        "StreamingTV_No internet service": StreamingTV_No,
        "StreamingTV_Yes": StreamingTV_Yes,

        "StreamingMovies_No internet service": StreamingMovies_No,
        "StreamingMovies_Yes": StreamingMovies_Yes,

        "Contract_One year": Contract_One_year,
        "Contract_Two year": Contract_Two_year,

        "PaperlessBilling_Yes": PaperlessBilling_Yes,

        "PaymentMethod_Credit card (automatic)": PaymentMethod_Credit_card_automatic,
        "PaymentMethod_Electronic check": PaymentMethod_Electronic_check,
        "PaymentMethod_Mailed check": PaymentMethod_Mailed_check

    }])

    # Model Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    stay_probability = probability[0] * 100
    churn_probability = probability[1] * 100

    st.divider()

    st.header("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN.")
    else:
        st.success("✅ Customer is likely to STAY.")

    # Dashboard Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Stay Probability",
            f"{stay_probability:.2f}%"
        )

    with col2:
        st.metric(
            "Churn Probability",
            f"{churn_probability:.2f}%"
        )

    with col3:

        if churn_probability < 30:
            risk = "🟢 LOW"

        elif churn_probability < 70:
            risk = "🟡 MEDIUM"

        else:
            risk = "🔴 HIGH"

        st.metric(
            "Risk Level",
            risk
        )

    st.progress(churn_probability / 100)

    st.divider()

    st.subheader("📈 Stay vs Churn Probability")

    chart = pd.DataFrame({
        "Probability": [
            stay_probability,
            churn_probability
        ]
    }, index=["Stay", "Churn"])

    st.bar_chart(chart)

    # ==========================
    # Probability Pie Chart
    # ==========================

    st.subheader("🥧 Churn Probability Distribution")

    fig, ax = plt.subplots(figsize=(5,5))

    ax.pie(
        [stay_probability, churn_probability],
        labels=["Stay", "Churn"],
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.05,0.05)
    )

    ax.axis("equal")

    st.pyplot(fig)

    # ==========================
    # Risk Meter
    # ==========================

    st.subheader("📊 Customer Risk Score")

    st.progress(churn_probability/100)

    st.write(f"Risk Score : {churn_probability:.2f}%")

    if churn_probability >= 70:
        st.error("🔴 High Risk Customer")

    elif churn_probability >= 40:
        st.warning("🟡 Medium Risk Customer")

    else:
        st.success("🟢 Low Risk Customer")

    # ==========================
    # Prediction Details
    # ==========================

    st.subheader("📋 Prediction Details")

    details = pd.DataFrame({
        "Parameter":[
            "Prediction",
            "Stay Probability",
            "Churn Probability",
            "Risk"
        ],
        "Value":[
            "Churn" if prediction else "Stay",
            f"{stay_probability:.2f}%",
            f"{churn_probability:.2f}%",
            risk
        ]
    })

    st.table(details)

    # ==========================================
    # Customer Summary
    # ==========================================

    st.divider()

    st.subheader("📋 Customer Summary")

    summary = pd.DataFrame({
        "Field": [
            "Gender",
            "Senior Citizen",
            "Tenure",
            "Monthly Charges",
            "Total Charges",
            "Partner",
            "Dependents",
            "Internet Service",
            "Contract",
            "Payment Method"
        ],
        "Value": [
            gender,
            senior,
            tenure,
            MonthlyCharges,
            TotalCharges,
            partner,
            dependents,
            internet,
            contract,
            payment
        ]
    })

    st.table(summary)

    # ==========================================
    # Feature Importance
    # ==========================================

    st.divider()

    st.subheader("⭐ Top 5 Important Features")

    importance = pd.DataFrame({
        "Feature": input_data.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    top5 = importance.head(5)

    st.bar_chart(
        top5.set_index("Feature")
    )

    # ==========================================
    # Prediction Explanation
    # ==========================================

    st.divider()

    st.subheader("🔍 Why this Prediction?")

    if prediction == 1:

        if contract == "Month-to-month":
            st.write("✅ Month-to-Month contract increases churn risk.")

        if MonthlyCharges > 70:
            st.write("✅ High monthly charges increase churn probability.")

        if tenure < 12:
            st.write("✅ Customer has low tenure.")

        if internet == "Fiber optic":
            st.write("✅ Fiber Optic users have higher churn in this dataset.")

        if paperless == "Yes":
            st.write("✅ Paperless billing slightly increases churn risk.")

    else:

        st.success("Customer characteristics indicate low churn risk.")

    # ==========================================
    # Recommended Actions
    # ==========================================

    st.divider()

    st.subheader("💡 Business Recommendations")

    if prediction == 1:

        st.success("✔ Offer loyalty discount")

        st.success("✔ Recommend One-Year or Two-Year Contract")

        st.success("✔ Improve customer support")

        st.success("✔ Contact customer before renewal")

        st.success("✔ Offer personalized retention package")

    else:

        st.info("✔ Continue providing excellent customer service.")

        st.info("✔ Recommend premium plans.")

        st.info("✔ Reward customer loyalty.")

    # ==========================
    # Download Report
    # ==========================

    st.divider()

    st.subheader("📥 Download Prediction Report")

    report = pd.DataFrame([{
        "Gender": gender,
        "Senior Citizen": senior,
        "Tenure": tenure,
        "Monthly Charges": MonthlyCharges,
        "Total Charges": TotalCharges,
        "Partner": partner,
        "Dependents": dependents,
        "Internet Service": internet,
        "Contract": contract,
        "Payment Method": payment,
        "Prediction": "Churn" if prediction == 1 else "Stay",
        "Churn Probability": f"{churn_probability:.2f}%"
    }])

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download CSV Report",
        data=csv,
        file_name="churn_prediction_report.csv",
        mime="text/csv"
    )
    # ==========================
    # Download PDF Report
    # ==========================

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate("prediction_report.pdf")

    story = []

    story.append(Paragraph("<b>Customer Churn Prediction Report</b>", styles["Title"]))

    story.append(Paragraph(f"Prediction : {'Churn' if prediction==1 else 'Stay'}", styles["Normal"]))

    story.append(Paragraph(f"Stay Probability : {stay_probability:.2f}%", styles["Normal"]))

    story.append(Paragraph(f"Churn Probability : {churn_probability:.2f}%", styles["Normal"]))

    story.append(Paragraph(f"Risk Level : {risk}", styles["Normal"]))

    doc.build(story)

    with open("prediction_report.pdf","rb") as pdf_file:

        st.download_button(
            "📄 Download PDF Report",
            pdf_file,
            file_name="prediction_report.pdf",
            mime="application/pdf"
        )

# ==========================
# Bulk Prediction (CSV Upload)
# ==========================

st.divider()

st.subheader("📂 Bulk Customer Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.write(df.columns.tolist())

    st.write("📊 Uploaded Data Preview")
    st.dataframe(df.head())

    feature_frame = prepare_features_for_model(df)

    predictions = model.predict(feature_frame)
    probs = model.predict_proba(feature_frame)

    df["Prediction"] = predictions
    df["Churn Probability (%)"] = (probs[:, 1] * 100).round(2)
    df["Prediction"] = df["Prediction"].map({0: "Stay", 1: "Churn"})

    st.write("📊 Prediction Results")
    st.dataframe(df)

    csv_bulk = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Bulk Predictions",
        data=csv_bulk,
        file_name="bulk_churn_predictions.csv",
        mime="text/csv"
    )

    st.info("🎉 Bulk prediction completed successfully.")
    st.divider()

st.markdown("""
---
### Customer Churn Prediction System

Machine Learning Powered Telecom Analytics

Developed by **Chambrish Prabhu**

© 2026 All Rights Reserved
""")
