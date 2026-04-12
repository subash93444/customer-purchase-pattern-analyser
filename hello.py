import streamlit as st
import pandas as pd
import plotly.express as px
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

# ---------------- AUTO COLUMN DETECTION FUNCTION ----------------
def find_col(df, keywords):
    for col in df.columns:
        for k in keywords:
            if k in col.lower():
                return col
    return None

if file:
    df = pd.read_csv(file)

    # ---------------- CLEAN COLUMN NAMES ----------------
    df.columns = df.columns.str.strip().str.lower()

    st.success("✅ File uploaded successfully!")

    # ---------------- AUTO DETECT COLUMNS ----------------
    product_col = find_col(df, ["product", "item"])
    category_col = find_col(df, ["category", "type"])
    amount_col = find_col(df, ["amount", "price", "cost", "sales"])

    # ---------------- METRICS ----------------
    st.subheader("📌 Quick Insights")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Rows", len(df))
    c2.metric("Products", df[product_col].nunique() if product_col else "N/A")
    c3.metric("Categories", df[category_col].nunique() if category_col else "N/A")

    # ---------------- DATA VIEW ----------------
    if show_data:
        with st.expander("🔍 View Dataset"):
            st.dataframe(df)

    # ---------------- CATEGORY ----------------
    if category_col:
        st.subheader("📦 Category Distribution")

        cat = df[category_col].value_counts().reset_index()
        cat.columns = ['Category', 'Count']

        fig2 = px.pie(cat, names='Category', values='Count')
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- CLUSTERING ----------------
    if show_cluster:
        st.subheader("🤖 Customer Segmentation")

        if amount_col:
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

    # ---------------- 🚀 ADVANCED PREDICTION ----------------
    if show_prediction:
        st.subheader("📈 AI Advanced Sales Forecast")

        if amount_col:
            df_clean = df.dropna(subset=[amount_col]).copy()
            df_clean['Index'] = range(len(df_clean))

            if len(df_clean) > 3:

                # ---------------- MOVING AVERAGE ----------------
                df_clean["Moving_Avg"] = df_clean[amount_col].rolling(window=3).mean()

                # ---------------- LINEAR MODEL ----------------
                lin_model = LinearRegression()
                lin_model.fit(df_clean[['Index']], df_clean[amount_col])

                # ---------------- POLYNOMIAL MODEL (AI CURVE) ----------------
                poly_model = make_pipeline(PolynomialFeatures(3), LinearRegression())
                poly_model.fit(df_clean[['Index']], df_clean[amount_col])

                # ---------------- FUTURE PREDICTION ----------------
                future_steps = 15
                future = pd.DataFrame({'Index': range(len(df_clean), len(df_clean) + future_steps)})

                lin_pred = lin_model.predict(future)
                poly_pred = poly_model.predict(future)

                # ---------------- DATA COMBINE ----------------
                full = pd.DataFrame({
                    "Index": list(df_clean['Index']) + list(future['Index']),
                    "Linear": list(df_clean[amount_col]) + list(lin_pred),
                    "Polynomial": list(df_clean[amount_col]) + list(poly_pred)
                })

                # ---------------- PLOT ----------------
                fig4 = px.line(
                    full,
                    x="Index",
                    y=["Linear", "Polynomial"],
                    title="AI Sales Forecast (Linear vs Polynomial)"
                )

                st.plotly_chart(fig4, use_container_width=True)

                # ---------------- AI INSIGHTS ----------------
                st.subheader("🧠 AI Insights")

                st.info(f"📊 Average Sales: {df_clean[amount_col].mean():.2f}")
                st.info(f"📈 Max Sales: {df_clean[amount_col].max():.2f}")
                st.info(f"📉 Min Sales: {df_clean[amount_col].min():.2f}")

                trend = "📈 Increasing" if poly_pred[-1] > df_clean[amount_col].mean() else "📉 Stable/Down"
                st.success(f"Forecast Trend: {trend}")

            else:
                st.warning("⚠️ Not enough data for advanced prediction")

        else:
            st.warning("⚠️ Amount column not found")

else:
    st.info("📂 Please upload a CSV file to start analysis")
