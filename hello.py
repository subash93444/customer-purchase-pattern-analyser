import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ---------------- UI ----------------
st.set_page_config(page_title="AI Dashboard", layout="wide")
st.title("🧠 Smart AI Customer Dashboard")

file = st.file_uploader("Upload ANY CSV File", type=["csv"])

# ---------------- SAFE COLUMN FINDER ----------------
def find_col(df, keywords):
    for col in df.columns:
        for k in keywords:
            if k in col.lower():
                return col
    return None

# ---------------- MAIN ----------------
if file:
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower()

        st.success("✅ File Loaded Successfully")
        st.dataframe(df.head())

        # ---------------- NUMERIC SAFE ----------------
        num_cols = df.select_dtypes(include='number').columns.tolist()

        if len(num_cols) == 0:
            df["auto_index"] = range(len(df))
            num_cols = ["auto_index"]

        main_col = num_cols[0]

        # ---------------- INSIGHTS ----------------
        st.subheader("📌 Quick Insights")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Numeric Columns", len(num_cols))

        # ---------------- CHART 1 ----------------
        st.subheader("📊 Data Overview")
        fig1 = px.histogram(df, x=main_col)
        st.plotly_chart(fig1, use_container_width=True)

        # ---------------- CLUSTERING ----------------
        st.subheader("🤖 Clustering AI")

        if len(df) > 3:
            X = df[[main_col]].dropna()

            model = KMeans(n_clusters=3, n_init=10, random_state=42)
            df["Cluster"] = model.fit_predict(X)

            fig2 = px.scatter(
                df,
                x=df.index,
                y=main_col,
                color=df["Cluster"].astype(str)
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Not enough data for clustering")

        # ---------------- PREDICTION ----------------
        st.subheader("📈 AI Forecast")

        clean = df.dropna(subset=[main_col]).copy()
        clean["Index"] = range(len(clean))

        if len(clean) > 3:

            lin = LinearRegression()
            lin.fit(clean[['Index']], clean[main_col])

            poly = make_pipeline(PolynomialFeatures(3), LinearRegression())
            poly.fit(clean[['Index']], clean[main_col])

            future = pd.DataFrame({'Index': range(len(clean), len(clean)+15)})

            lin_pred = lin.predict(future)
            poly_pred = poly.predict(future)

            full = pd.DataFrame({
                "Index": list(clean["Index"]) + list(future["Index"]),
                "Linear": list(clean[main_col]) + list(lin_pred),
                "Polynomial": list(clean[main_col]) + list(poly_pred)
            })

            fig3 = px.line(full, x="Index", y=["Linear", "Polynomial"], markers=True)
            st.plotly_chart(fig3, use_container_width=True)

        else:
            st.warning("Not enough data for prediction")

    except Exception as e:
        st.error("❌ File error handled safely")
        st.exception(e)

else:
    st.info("📂 Upload CSV to start analysis")
