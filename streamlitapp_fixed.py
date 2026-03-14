import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Load Model & Transformer
model = joblib.load("income_model.pkl")
pt    = joblib.load("capital_net_transformer.pkl")

# Page Config
st.set_page_config(page_title="Income Prediction", layout="wide")

# Header
st.title("💰 Income Prediction App")
st.image("income5.png", use_container_width=True)
st.write("Predict whether a person's income is **>50K or ≤50K** based on demographic features.")

# Sidebar Inputs
st.sidebar.header("Enter Person Information")

age = st.sidebar.number_input("Age", 18, 100, 30)

workclass = st.sidebar.selectbox("Workclass", [
    "State-gov", "Self-emp-not-inc", "Private", "Federal-gov",
    "Local-gov", "Self-emp-inc", "Without-pay"
])

education_level = st.sidebar.selectbox("Education Level", [
    "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
    "Assoc-acdm", "7th-8th", "Doctorate", "Assoc-voc", "Prof-school",
    "5th-6th", "10th", "Preschool", "12th", "1st-4th"
])

education_num = st.sidebar.number_input("Education Num", 1, 16, 10)

marital_status = st.sidebar.selectbox("Marital Status", [
    "Never-married", "Married-civ-spouse", "Divorced",
    "Married-spouse-absent", "Separated", "Married-AF-spouse", "Widowed"
])
is_married = 1 if "Married" in marital_status else 0

occupation = st.sidebar.selectbox("Occupation", [
    "Adm-clerical", "Exec-managerial", "Handlers-cleaners",
    "Prof-specialty", "Other-service", "Sales", "Transport-moving",
    "Farming-fishing", "Machine-op-inspct", "Tech-support",
    "Craft-repair", "Protective-serv", "Armed-Forces", "Priv-house-serv"
])

relationship = st.sidebar.selectbox("Relationship", [
    "Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"
])

race = st.sidebar.selectbox("Race", [
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
])

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

hours_per_week = st.sidebar.number_input("Hours per week", 1, 100, 40)

native_country = st.sidebar.selectbox("Country", [
    "United-States", "Cuba", "Jamaica", "India", "Mexico", "Puerto-Rico",
    "Honduras", "England", "Canada", "Germany", "Iran", "Philippines",
    "Poland", "Columbia", "Cambodia", "Thailand", "Ecuador", "Laos",
    "Taiwan", "Haiti", "Portugal", "Dominican-Republic", "El-Salvador",
    "France", "Guatemala", "Italy", "China", "South", "Japan", "Yugoslavia",
    "Peru", "Outlying-US(Guam-USVI-etc)", "Scotland", "Trinadad&Tobago",
    "Greece", "Nicaragua", "Vietnam", "Hong", "Ireland", "Hungary", "Holand-Netherlands"
])

capital_net_raw = st.sidebar.number_input("Capital Net", -5000.0, 100000.0, 0.0)

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.35,
    step=0.05,
    help="Lower = catch more >50K cases (higher recall). Raise = more conservative."
)

predict_button = st.sidebar.button("Predict")

# Transform capital_net
capital_net_transformed = pt.transform([[capital_net_raw]])[0][0]

# Input DataFrame
input_data = pd.DataFrame({
    "age":             [age],
    "workclass":       [workclass],
    "education_level": [education_level],
    "education-num":   [education_num],
    "occupation":      [occupation],
    "relationship":    [relationship],
    "race":            [race],
    "sex":             [sex],
    "hours-per-week":  [hours_per_week],
    "native-country":  [native_country],
    "capital_net":     [capital_net_transformed],
    "is_married":      [is_married],
})

# Prediction
if predict_button:
    proba      = model.predict_proba(input_data)[0][1]
    prediction = 1 if proba >= threshold else 0

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"Income >50K 💰  (Probability: {proba:.2f} | Threshold: {threshold})")
    else:
        st.error(f"Income ≤50K  (Probability: {proba:.2f} | Threshold: {threshold})")

st.markdown("<br><br>", unsafe_allow_html=True)

# Layout for Charts
col1, col2 = st.columns(2)

# Feature Importance
with col1:
    st.subheader("Top 10 Feature Importance")

    preprocessor      = model.named_steps["preprocessing"]
    ohe               = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out())
    num_features      = ["age", "education-num", "hours-per-week", "capital_net", "is_married"]
    all_feature_names = cat_feature_names + num_features

    xgb_model   = model.named_steps["model"]
    importances = pd.Series(
        xgb_model.feature_importances_,
        index=all_feature_names
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 9))
    sns.barplot(x=importances.head(10), y=importances.head(10).index, ax=ax)
    ax.set_title("Top 10 Feature Importance")
    plt.tight_layout()
    st.pyplot(fig)

# Class Distribution
with col2:
    st.subheader("Class Distribution")
    df = pd.read_csv("census.csv")
    fig, ax = plt.subplots(figsize=(9, 9))
    sns.countplot(x="income", data=df, ax=ax)
    ax.set_title("Income Distribution")
    plt.tight_layout()
    st.pyplot(fig)
