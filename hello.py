import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ---------------- UI SETTINGS ----------------
st.set_page_config(page_title="AI Dashboard", layout="wide")

# ---------------- LOGO ----------------
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("logo.png", width=150)

st.title("🧠 Customer Purchase Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.image("logo.png", width=120)
st.sidebar.title("⚙️ Controls")

show_data = st.sidebar.checkbox("Show Data", True)
show_cluster = st.sidebar.checkbox("Show Clustering", True)
show_prediction = st.sidebar.checkbox("Show Prediction", True)

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload CSV", type=["csv"])

def find_col(df, keywords):
    for col in df.columns:
        for k in keywords:
            if k in col.lower():
                return col
    return None

if file:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    st.success("✅ File uploaded successfully!")

    product_col = find_col(df, ["product", "item"])
    category_col = find_col(df, ["category", "type"])
    amount_col = find_col(df, ["amount", "price", "cost", "sales"])

    # ---------------- METRICS ----------------
    st.subheader("📌 Quick Insights")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", len(df))
    c2.metric("Products", df[product_col].nunique() if product_col else "N/A")
    c3.metric("Categories", df[category_col].nunique() if category_col else "N/A")

    # ---------------- CATEGORY (ANIMATED PIE) ----------------
    if category_col:
        st.subheader("📦 Category Distribution (Animated)")

        cat = df[category_col].value_counts().reset_index()
        cat.columns = ['Category', 'Count']

        fig2 = px.pie(cat, names='Category', values='Count')
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- CLUSTERING ----------------
    if show_cluster and amount_col:
        st.subheader("🤖 Customer Segmentation")

        X = df[[amount_col]].dropna()

        if len(X) > 2:
            kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
            df['Cluster'] = kmeans.fit_predict(X)

            fig3 = px.scatter(
                df,
                x=df.index,
                y=amount_col,
                color=df['Cluster'].astype(str),
                title="Customer Clusters"
            )

            st.plotly_chart(fig3, use_container_width=True)

    # ---------------- 🚀 ADVANCED PREDICTION (FIXED + ANIMATED) ----------------
    if show_prediction and amount_col:
        st.subheader("📈 AI Advanced Forecast (Animated)")

        df_clean = df.dropna(subset=[amount_col]).copy()
        df_clean['Index'] = range(len(df_clean))

        if len(df_clean) > 3:

            # ---------------- MODELS ----------------
            lin_model = LinearRegression()
            lin_model.fit(df_clean[['Index']], df_clean[amount_col])

            poly_model = make_pipeline(PolynomialFeatures(3), LinearRegression())
            poly_model.fit(df_clean[['Index']], df_clean[amount_col])

            # ---------------- FUTURE ----------------
            future_steps = 15
            future = pd.DataFrame({'Index': range(len(df_clean), len(df_clean)+future_steps)})

            lin_pred = lin_model.predict(future)
            poly_pred = poly_model.predict(future)

            # ---------------- SAFE DATA ----------------
            actual = pd.DataFrame({
                "Index": df_clean['Index'],
                "Value": df_clean[amount_col],
                "Type": "Actual"
            })

            lin_df = pd.DataFrame({
                "Index": future['Index'],
                "Value": lin_pred,
                "Type": "Linear"
            })

            poly_df = pd.DataFrame({
                "Index": future['Index'],
                "Value": poly_pred,
                "Type": "Polynomial"
            })

            full = pd.concat([actual, lin_df, poly_df])

            # ---------------- ANIMATED GRAPH (FIXED) ----------------
            fig = px.line(
                full,
                x="Index",
                y="Value",
                color="Type",
                markers=True,
                title="📊 Animated AI Sales Forecast"
            )

            fig.update_traces(line_shape="spline")  # smooth animation feel

            st.plotly_chart(fig, use_container_width=True)

            # ---------------- INSIGHTS ----------------
            st.subheader("🧠 AI Insights")

            st.info(f"📊 Average Sales: {df_clean[amount_col].mean():.2f}")
            st.info(f"📈 Max Sales: {df_clean[amount_col].max():.2f}")
            st.info(f"📉 Min Sales: {df_clean[amount_col].min():.2f}")

            trend = "📈 Increasing" if poly_pred[-1] > df_clean[amount_col].mean() else "📉 Stable/Down"
            st.success(f"Forecast Trend: {trend}")

        else:
            st.warning("⚠️ Not enough data for prediction")

else:
    st.info("📂 Please upload a CSV file to start analysis")
