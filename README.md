# Customer Churn Prediction

This project is a Streamlit-based machine learning application that predicts whether a telecom customer is likely to churn.

## Project Overview

```mermaid
flowchart LR
    A[User] --> B[Streamlit Web App]
    B --> C[Customer Input Form]
    C --> D[Feature Encoding]
    D --> E[Loaded Model]
    E --> F[Churn Prediction]
    F --> G[Prediction Result UI]

    H[feature_names.csv] --> D
    I[customer_churn_model.pkl] --> E
    J[WA_Fn-UseC_-Telco-Customer-Churn.csv] --> K[Training / Reference Dataset]
```

## Prediction Workflow

```mermaid
sequenceDiagram
    participant User
    participant StreamlitApp as Streamlit App
    participant Encoder as Feature Encoder
    participant Model as Random Forest Model

    User->>StreamlitApp: Enter customer details
    StreamlitApp->>Encoder: Prepare input features
    Encoder-->>StreamlitApp: Encoded feature vector
    StreamlitApp->>Model: Predict churn probability
    Model-->>StreamlitApp: Churn / Stay result
    StreamlitApp-->>User: Show prediction and probability
```

## Repository Structure

```mermaid
flowchart TB
    ROOT[Customer Churn Prediction]
    ROOT --> APP[app.py]
    ROOT --> REQ[requirements.txt]
    ROOT --> MODEL[customer_churn_model.pkl]
    ROOT --> FEATURE[feature_names.csv]
    ROOT --> DATA[WA_Fn-UseC_-Telco-Customer-Churn.csv]
    ROOT --> LOGO[logo.png]
    ROOT --> README[README.md]

    APP --> UI[Streamlit Interface]
    APP --> PREP[Feature Preparation Logic]
    APP --> PRED[Prediction Logic]
```

## Suggested GitHub README Section

You can paste the following snippet directly into your GitHub README:

```md
## Architecture Diagram

```mermaid
flowchart LR
    A[User] --> B[Streamlit Web App]
    B --> C[Customer Input Form]
    C --> D[Feature Encoding]
    D --> E[Random Forest Model]
    E --> F[Churn Prediction Result]
```
```
