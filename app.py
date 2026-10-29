"""Streamlit demo for the customer churn risk model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from churn_project.modeling import FEATURES

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "churn_pipeline.joblib"
METRICS_PATH = ROOT / "artifacts" / "model_metrics.json"

st.set_page_config(
    page_title="Customer Churn Risk Explorer",
    page_icon="📉",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return model, metrics


if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error("The trained model files are missing. Run `python scripts/train.py` first.")
    st.stop()

model, report = load_artifacts()
test = report["test_metrics"]
threshold = report["decision_threshold"]

st.title("📉 Customer Churn Risk Explorer")
st.caption(
    "An end-to-end machine learning project for exploring churn patterns and "
    "testing customer risk scores. Scores support prioritization; they do not "
    "predict an individual outcome with certainty."
)

overview, explore, predict, validation = st.tabs(
    ["Project overview", "Explore churn", "Try a prediction", "Model validation"]
)

with overview:
    st.subheader("Business question")
    st.write(
        "Which customers may be at higher risk of leaving, and how could a retention "
        "team prioritize a human review or outreach? The demo uses the public-style "
        "Telco customer churn dataset and compares logistic regression and random forest "
        "baselines with an XGBoost, CatBoost, and LightGBM stacking ensemble."
    )
    a, b, c, d = st.columns(4)
    a.metric("Customers in dataset", f"{report['dataset']['rows']:,}")
    b.metric("Churn rate", f"{report['dataset']['churn_rate']:.1%}")
    c.metric("Selected model", report["model_name"])
    d.metric("Held-out test F1", f"{test['f1']:.3f}")
    st.info(
        f"The score is flagged at a threshold of {threshold:.3f}. That threshold was "
        "chosen using training-only cross-validation, not the held-out test labels. "
        "A business should validate outreach capacity, costs, and outcomes before use."
    )
    st.markdown("**Workflow:** data validation → preprocessing pipeline → cross-validated model comparison → threshold selection → held-out evaluation → interactive demo.")

with explore:
    st.subheader("Churn patterns in the dataset")
    st.caption("These are descriptive associations in this dataset, not causal effects.")
    profile_cols = st.columns(2)
    for container, title, key in [
        (profile_cols[0], "Churn rate by contract", "Contract"),
        (profile_cols[1], "Churn rate by tenure", "TenureGroup"),
        (profile_cols[0], "Churn rate by internet service", "InternetService"),
        (profile_cols[1], "Churn rate by payment method", "PaymentMethod"),
    ]:
        rows = report["profiles"][key]
        chart = pd.DataFrame(rows).set_index("category")
        container.markdown(f"**{title}**")
        container.bar_chart(chart[["churn_rate"]].rename(columns={"churn_rate": "Churn rate"}))
        container.dataframe(
            chart.rename(columns={"churn_rate": "Churn rate", "customers": "Customers"}),
            use_container_width=True,
        )

with predict:
    st.subheader("Estimate a customer’s churn risk")
    st.write("Enter a profile, then submit it to receive a model score and a threshold-based review flag.")
    with st.form("customer_profile"):
        left, middle, right = st.columns(3)
        with left:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda value: "Yes" if value else "No")
            partner = st.selectbox("Has a partner", ["No", "Yes"])
            dependents = st.selectbox("Has dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = (
                st.selectbox("Multiple lines", ["No", "Yes"])
                if phone == "Yes"
                else "No phone service"
            )
        with middle:
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            if internet == "No":
                online_security = online_backup = device_protection = "No internet service"
                tech_support = streaming_tv = streaming_movies = "No internet service"
                st.caption("Internet add-ons are set to ‘No internet service’.")
            else:
                online_security = st.selectbox("Online security", ["No", "Yes"])
                online_backup = st.selectbox("Online backup", ["No", "Yes"])
                device_protection = st.selectbox("Device protection", ["No", "Yes"])
                tech_support = st.selectbox("Tech support", ["No", "Yes"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                streaming_movies = st.selectbox("Streaming movies", ["No", "Yes"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with right:
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=500.0, value=75.0, step=1.0)
            default_total = float(round(tenure * monthly, 2))
            total = st.number_input(
                "Total charges so far ($)", min_value=0.0, max_value=100000.0,
                value=default_total, step=10.0,
                help="The default is tenure × monthly charges; adjust it if you have a known customer total.",
            )
        submitted = st.form_submit_button("Estimate churn risk", type="primary", use_container_width=True)

    if submitted:
        customer = pd.DataFrame(
            [{
                "gender": gender, "SeniorCitizen": senior, "Partner": partner,
                "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
                "MultipleLines": multiple_lines, "InternetService": internet,
                "OnlineSecurity": online_security, "OnlineBackup": online_backup,
                "DeviceProtection": device_protection, "TechSupport": tech_support,
                "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
                "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment, "MonthlyCharges": monthly,
                "TotalCharges": total,
            }],
            columns=FEATURES,
        )
        probability = float(model.predict_proba(customer)[0, 1])
        score_col, status_col = st.columns([1, 2])
        score_col.metric("Estimated churn probability", f"{probability:.1%}")
        if probability >= threshold:
            status_col.warning(
                "**Flag for review:** this score is at or above the selected threshold. "
                "Consider checking the customer’s situation before choosing an action."
            )
        else:
            status_col.success(
                "**Below review threshold:** the model score is below the selected cutoff. "
                "This does not guarantee the customer will stay."
            )
        st.progress(min(max(probability, 0.0), 1.0))

with validation:
    st.subheader("Held-out test set")
    st.caption("The test split was held aside during model and threshold selection.")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Precision", f"{test['precision']:.3f}")
    m2.metric("Recall", f"{test['recall']:.3f}")
    m3.metric("F1", f"{test['f1']:.3f}")
    m4.metric("ROC AUC", f"{test['roc_auc']:.3f}")
    m5.metric("PR AUC", f"{test['pr_auc']:.3f}")
    tn, fp = test["confusion_matrix"][0]
    fn, tp = test["confusion_matrix"][1]
    st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
    st.dataframe(
        pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual: stayed", "Actual: churned"], columns=["Predicted: stayed", "Predicted: churned"]),
        use_container_width=True,
    )
    comparison = pd.DataFrame(report["cross_validation_comparison"]).set_index("name")
    st.markdown("**Model comparison on training-only out-of-fold predictions**")
    st.dataframe(comparison, use_container_width=True)
    if report["top_model_factors"]:
        st.markdown("**Global model factors**")
        st.caption("These describe model-level associations; they are not causal explanations for an individual prediction.")
        st.dataframe(pd.DataFrame(report["top_model_factors"]), use_container_width=True)
    st.caption(
        f"Threshold: {threshold:.3f} · test customers: {test['support']:,} · "
        f"split: {report['dataset']['split']} · generated: {report['generated_at_utc']}"
    )
\n