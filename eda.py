import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ================= PAGE CONFIG ================= #
st.set_page_config(
    page_title="Tourism  Dashboard",
    layout="wide",
    page_icon="🧳"
)

# ================= THEME ================= #
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

# ================= LOAD DATA ================= #
# ✅ Use relative path (best practice)
file_path = "travelclean1.csv"

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error("❌ File not found. Please place travelclean1.csv in the same folder.")
    st.stop()

# ================= DATA TYPE FIX ================= #
# Convert categorical-like numerics to object
cat_like_cols = [
    'CityTier', 'ProdTaken', 'NumberOfPersonVisiting', 'OwnCar',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'NumberOfChildrenVisiting', 'Passport', 'PitchSatisfactionScore'
]

df[cat_like_cols] = df[cat_like_cols].astype(str)

df_selection = df.copy()

# ================= HEADER ================= #
st.markdown("<div class='header-text'>🧳 Tourism Market Insights Dashboard </div>", unsafe_allow_html=True)

# ================= KPI SECTION ================= #
st.subheader("📌 Key Performance Indicators")

total_customers = len(df_selection)

# ✅ Correct ProdTaken logic
purchased = (df_selection['ProdTaken'] == '1').sum()
not_purchased = total_customers - purchased

conversion_rate = round((purchased / total_customers) * 100, 2)

avg_followups = round(
    df_selection.loc[df_selection['ProdTaken'] == '1', 'NumberOfFollowups']
    .astype(float).mean(),
)

avg_age = round(df_selection['Age'].mean(),)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Customers", total_customers)
c2.metric("Purchased", purchased)
c3.metric("Not Purchased", not_purchased)
c4.metric("Conversion Rate", f"{conversion_rate}%")
c5.metric("Avg Follow-ups", avg_followups)
c6.metric("Avg Age", avg_age)

st.divider()
st.subheader("📊 Data Exploration")

# ================= COLUMN TYPES ================= #
num_cols = df_selection.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df_selection.select_dtypes(include=['object']).columns.tolist()

# ================= SIDEBAR ================= #
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["Univariate Analysis", "Bivariate Analysis"]
)

# ================= UNIVARIATE ================= #
if analysis_type == "Univariate Analysis":

    variable_type = st.sidebar.selectbox(
        "Select Variable Type",
        ["Numerical", "Categorical"]
    )

    # -------- NUMERICAL -------- #
    if variable_type == "Numerical":
        col = st.sidebar.selectbox("Select Numerical Column", num_cols)

        colA, colB = st.columns(2)

        with colA:
            fig, ax = plt.subplots(figsize=(6,4))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")

            sns.histplot(df_selection[col], kde=True, ax=ax,
                         color="#00f260", edgecolor="black", alpha=0.6)

            ax.tick_params(colors="white")
            ax.set_xlabel(col, color="white")
            ax.set_ylabel("Count", color="white")

            for spine in ax.spines.values():
                spine.set_color((1,1,1,0.3))

            st.pyplot(fig)

        with colB:
            fig, ax = plt.subplots(figsize=(6,3.6))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")

            sns.boxplot(x=df_selection[col], ax=ax, color="#0575E6")

            ax.tick_params(colors="white")
            ax.set_xlabel(col, color="white")

            for spine in ax.spines.values():
                spine.set_color((1,1,1,0.3))

            st.pyplot(fig)

    # -------- CATEGORICAL -------- #
    else:
        col = st.sidebar.selectbox("Select Categorical Column", cat_cols)

        cat_df = (
            df_selection[col]
            .value_counts()
            .rename_axis(col)
            .reset_index(name="count")
        )

        colA, colB = st.columns(2)

        with colA:
            fig = px.bar(
                cat_df, x=col, y="count",
                template="plotly_dark",
                color_discrete_sequence=["#00f260"]
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with colB:
            fig = px.pie(
                cat_df, names=col, values="count",
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Tealgrn
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

# ================= BIVARIATE ================= #
else:
    relation_type = st.sidebar.selectbox(
        "Select Relationship",
        ["Num vs Num", "Cat vs Cat", "Num vs Cat"]
    )

    colA, colB = st.columns(2)

    if relation_type == "Num vs Num":
        x = st.sidebar.selectbox("Select X", num_cols)
        y = st.sidebar.selectbox("Select Y", num_cols)

        with colA:
            fig = px.scatter(df_selection, x=x, y=y,
                             template="plotly_dark",
                             color_discrete_sequence=["#00f260"])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with colB:
            fig = px.density_heatmap(df_selection, x=x, y=y,
                                     template="plotly_dark",
                                     color_continuous_scale="Viridis")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    elif relation_type == "Cat vs Cat":
        x = st.sidebar.selectbox("Select X", cat_cols)
        y = st.sidebar.selectbox("Select Y", cat_cols)

        with colA:
            fig = px.histogram(df_selection, x=x, color=y,
                               barmode="group",
                               template="plotly_dark")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with colB:
            ctab = pd.crosstab(df_selection[x], df_selection[y])
            fig = px.imshow(ctab, text_auto=True, aspect="auto",
                             template="plotly_dark",
                             color_continuous_scale="Tealgrn")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    else:
        x = st.sidebar.selectbox("Select Numerical", num_cols)
        y = st.sidebar.selectbox("Select Category", cat_cols)

        with colA:
            fig = px.box(df_selection, x=y, y=x,
                         template="plotly_dark",
                         color_discrete_sequence=["#0575E6"])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with colB:
            mean_df = df_selection.groupby(y)[x].mean().reset_index()
            fig = px.bar(mean_df, x=y, y=x,
                         template="plotly_dark",
                         color_discrete_sequence=["#00f260"])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
