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
st.title("🧠 Smart AI Customer Dashboard")

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload ANY CSV File", type=["csv"])

# ---------------- SAFE FINDER ----------------
def find_col(df, keywords):
    for col in df.columns:
        for k in keywords:
            if k in col.lower():
                return col
    return None

if file:

    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower()

        st.success("✅ File uploaded successfully!")

        st.subheader("📊 Raw Data Preview")
        st.dataframe(df.head())

        # ---------------- AUTO DETECT ----------------
        product_col = find_col(df, ["product", "item", "name"])
        category_col = find_col(df, ["category", "type", "class"])
        amount_col = find_col(df, ["amount", "price", "cost", "sales", "value", "total", "revenue"])

        # ---------------- SAFE NUMERIC FIX (IMPORTANT) ----------------
        num_cols = df.select_dtypes(include=['number']).columns

        if len(num_cols) == 0:
            df["dummy_index"] = range(len(df))
            num_cols = ["dummy_index"]

        # ---------------- METRICS ----------------
        st.subheader("📌 Quick Insights")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Numeric Columns", len(num_cols))

        # ---------------- ALWAYS SHOW CHART ----------------
        st.subheader("📊 Data Overview Chart (Auto)")

        fig1 = px.histogram(df, x=num_cols[0])
        fig1.update_layout(bargap=0.2)

        st.plotly_chart(fig1, use_container_width=True)

        # ---------------- CATEGORY ----------------
        if category_col:
            st.subheader("📦 Category Distribution")

            cat = df[category_col].value_counts().reset_index()
            cat.columns = ['Category', 'Count']

            fig2 = px.pie(cat, names='Category', values='Count')
            st.plotly_chart(fig2, use_container_width=True)

        # ---------------- CLUSTERING ----------------
        st.subheader("🤖 Clustering (Auto AI)")

        X = df[[num_cols[0]]].dropna()

        if len(X) > 2:
            kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
            df["Cluster"] = kmeans.fit_predict(X)

            fig3 = px.scatter(
                df,
                x=df.index,
                y=num_cols[0],
                color=df["Cluster"].astype(str),
                title="AI Customer Clusters"
            )

            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("⚠️ Not enough data for clustering")

        # ---------------- PREDICTION ----------------
        st.subheader("📈 AI Forecast (Smart Prediction)")

        col = num_cols[0]

        df_clean = df.dropna(subset=[col]).copy()
        df_clean["Index"] = range(len(df_clean))

        if len(df_clean) > 3:

            # Linear model
            lin_model = LinearRegression()
            lin_model.fit(df_clean[['Index']], df_clean[col])

            # Polynomial model
            poly_model = make_pipeline(PolynomialFeatures(3), LinearRegression())
            poly_model.fit(df_clean[['Index']], df_clean[col])

            future = pd.DataFrame({'Index': range(len(df_clean), len(df_clean)+15)})

            lin_pred = lin_model.predict(future)
            poly_pred = poly_model.predict(future)

            full = pd.DataFrame({
                "Index": list(df_clean["Index"]) + list(future["Index"]),
                "Value": list(df_clean[col]) + list(poly_pred),
                "Type": ["Actual"]*len(df_clean) + ["Predicted"]*len(future)
            })

            fig4 = px.line(
                full,
                x="Index",
                y="Value",
                color="Type",
                markers=True,
                title="📊 AI Forecast (Animated View)"
            )

            fig4.update_traces(line_shape="spline")

            st.plotly_chart(fig4, use_container_width=True)

            # ---------------- INSIGHTS ----------------
            st.subheader("🧠 AI Insights")

            st.info(f"📊 Mean: {df_clean[col].mean():.2f}")
            st.info(f"📈 Max: {df_clean[col].max():.2f}")
            st.info(f"📉 Min: {df_clean[col].min():.2f}")

        else:
            st.warning("⚠️ Not enough data for prediction")

    except Exception as e:
        st.error("❌ Something went wrong but app is safe now")
        st.exception(e)

else:
    st.info("📂 Upload any CSV file to start analysis")
