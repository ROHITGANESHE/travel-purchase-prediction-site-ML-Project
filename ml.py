# ===================== IMPORTS ===================== #
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ===================== PAGE CONFIG ===================== #
st.set_page_config(
    page_title=" Tourism Analytics",
    page_icon="🧳",
    layout="wide"
)

st.title("🧳Tourism Analytics & Prediction Suite")
st.markdown("---")
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: #ffffff;
}
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-left: 6px solid #00f260;
    padding: 20px;
    border-radius: 15px;
}
[data-testid="stMetricValue"] {
    color: #00f260 !important;
}
.header-text {
    font-size: 42px;
    font-weight: 800;
    background: -webkit-linear-gradient(#0575E6, #00f260);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
[data-testid="stSidebar"] {
    background-color: #0b1116;
}
</style>
""", unsafe_allow_html=True)

# ===================== LOAD DATA ===================== #
@st.cache_data
def load_data():
    df = pd.read_csv("travelclean1.csv")
    if "CustomerID" in df.columns:
        df.drop(columns=["CustomerID"], inplace=True)

    # Ensure target is numeric
    df["ProdTaken"] = df["ProdTaken"].astype(int)
    return df

df = load_data()

# ===================== TARGET ===================== #
TARGET = "ProdTaken"
X = df.drop(columns=[TARGET])
y = df[TARGET]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# ===================== SIDEBAR ===================== #
with st.sidebar:
    st.header("⚙️ Configuration")

    model_choice = st.selectbox(
        "Select Model",
        ["Random Forest", "Logistic Regression", "SVM"]
    )

#     threshold = st.slider(
#         "Prediction Threshold",
#         0.1, 0.9, 0.4, 0.05
#     )

# ===================== TABS ===================== #
tab_eda, tab_model, tab_pred = st.tabs(
    ["📊 EDA Dashboard", "🛠️ Model Training", "🔮 Live Prediction"]
)

# ==================================================
# TAB 1: EDA
# ==================================================
num1 = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

with tab_eda:
    st.subheader("📌 Key Performance Indicators")

    total_customers = len(df)
    purchased = df["ProdTaken"].sum()
    not_purchased = total_customers - purchased
    conversion_rate = round((purchased / total_customers) * 100, 2)

    avg_followups = round(
        df.loc[df["ProdTaken"] == 1, "NumberOfFollowups"].mean(), 
    )
    avg_age = round(df["Age"].mean(),)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Customers", total_customers)
    c2.metric("Purchased", purchased)
    c3.metric("Not Purchased", not_purchased)
    c4.metric("Conversion Rate", f"{conversion_rate}%")
    c5.metric("Avg Follow-ups", avg_followups)
    c6.metric("Avg Age", avg_age)

    st.divider()
    st.subheader("📊 Data Exploration")

    analysis_type = st.selectbox(
        "Analysis Type",
        ["Univariate", "Bivariate"]
    )

    if analysis_type == "Univariate":
        var_type = st.radio("Variable Type", ["Numerical", "Categorical"])

        if var_type == "Numerical":
            col = st.selectbox("Select Column", num1)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.histogram(df, x=col), use_container_width=True)
            with c2:
                st.plotly_chart(px.box(df, y=col), use_container_width=True)

        else:
            col = st.selectbox("Select Column", cat_cols)
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, "Count"]

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.bar(counts, x=col, y="Count"), use_container_width=True)
            with c2:
                st.plotly_chart(px.pie(counts, names=col, values="Count"), use_container_width=True)

    else:
        relation = st.selectbox(
            "Relationship Type",
            ["Num vs Num", "Num vs Cat", "Cat vs Cat"]
        )

        if relation == "Num vs Num":
            x = st.selectbox("X", num1)
            y_ = st.selectbox("Y", num1)
            st.plotly_chart(px.scatter(df, x=x, y=y_), use_container_width=True)

        elif relation == "Num vs Cat":
            num = st.selectbox("Numerical", num1)
            cat = st.selectbox("Categorical", cat_cols)
            st.plotly_chart(px.box(df, x=cat, y=num1), use_container_width=True)

        else:
            x = st.selectbox("X Category", cat_cols)
            y_ = st.selectbox("Y Category", cat_cols)
            ct = pd.crosstab(df[x], df[y_])
            st.plotly_chart(px.imshow(ct, text_auto=True), use_container_width=True)

# ==================================================
# TAB 2: MODEL TRAINING
# ==================================================
with tab_model:
    st.subheader("🛠️ Model Training & Evaluation")

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        model = SVC(probability=True, class_weight="balanced")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    c2.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
    c3.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
    c4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2%}")

    cm = confusion_matrix(y_test, y_pred)
    st.plotly_chart(px.imshow(cm, text_auto=True), use_container_width=True)

# ==================================================
# TAB 3: LIVE PREDICTION
# ==================================================
with tab_pred:
    st.subheader("🔮 Live Customer Prediction")

    input_data = {}
    cols = st.columns(3)

    for i, col in enumerate(X.columns):
        with cols[i % 3]:
            if col in cat_cols:
                input_data[col] = st.selectbox(col, df[col].unique())
            else:
                input_data[col] = st.number_input(col, value=int(df[col].median()))

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prob = pipeline.predict_proba(input_df)[0][1]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            gauge={"axis": {"range": [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        if prob >= 0.5:
            st.success(f"✅ BUY (Confidence: {prob:.1%})")
        else:
            st.error(f"❌ NO BUY (Confidence: {prob:.1%})")
