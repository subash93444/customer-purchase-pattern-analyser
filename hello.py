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

# ---------------- SAFE COLUMN FINDER ----------------
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

        # ---------------- SAFE METRICS ----------------
        st.subheader("📌 Quick Insights")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Numeric Columns", len(df.select_dtypes(include='number').columns))

        # ---------------- ALWAYS SHOW CHART 1 ----------------
        st.subheader("📊 Data Overview Chart")

        num_cols = df.select_dtypes(include='number').columns

        if len(num_cols) > 0:
            fig1 = px.bar(df.head(10), x=num_cols[0], title="Numeric Data Overview (Auto)")
        else:
            fig1 = px.bar(x=["No Numeric Data"], y=[1], title="No Numeric Column Found")

        st.plotly_chart(fig1, use_container_width=True)

        # ---------------- CATEGORY CHART ----------------
        if category_col:
            st.subheader("📦 Category Distribution")

            cat = df[category_col].value_counts().reset_index()
            cat.columns = ['Category', 'Count']

            fig2 = px.pie(cat, names='Category', values='Count')
            st.plotly_chart(fig2, use_container_width=True)

        # ---------------- CLUSTERING ----------------
        st.subheader("🤖 Clustering (Auto)")

        if len(num_cols) >= 1:
            X = df[[num_cols[0]]].dropna()

            if len(X) > 2:
                kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
                df['Cluster'] = kmeans.fit_predict(X)

                fig3 = px.scatter(
                    df,
                    x=df.index,
                    y=num_cols[0],
                    color=df['Cluster'].astype(str),
                    title="Customer Clustering"
                )

                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("Not enough data for clustering")

        # ---------------- ADVANCED PREDICTION ----------------
        st.subheader("📈 AI Forecast (Auto Prediction)")

        if len(num_cols) > 0:

            col = num_cols[0]

            df_clean = df.dropna(subset=[col]).copy()
            df_clean["Index"] = range(len(df_clean))

            if len(df_clean) > 3:

                # Linear
                lin_model = LinearRegression()
                lin_model.fit(df_clean[['Index']], df_clean[col])

                # Polynomial
                poly_model = make_pipeline(PolynomialFeatures(3), LinearRegression())
                poly_model.fit(df_clean[['Index']], df_clean[col])

                future = pd.DataFrame({'Index': range(len(df_clean), len(df_clean)+15)})

                lin_pred = lin_model.predict(future)
                poly_pred = poly_model.predict(future)

                full = pd.DataFrame({
                    "Index": list(df_clean['Index']) + list(future['Index']),
                    "Linear": list(df_clean[col]) + list(lin_pred),
                    "Polynomial": list(df_clean[col]) + list(poly_pred)
                })

                fig4 = px.line(
                    full,
                    x="Index",
                    y=["Linear", "Polynomial"],
                    markers=True,
                    title="📊 AI Forecast (Animated View)"
                )

                st.plotly_chart(fig4, use_container_width=True)

                # ---------------- INSIGHTS ----------------
                st.subheader("🧠 AI Insights")

                st.info(f"📊 Mean: {df_clean[col].mean():.2f}")
                st.info(f"📈 Max: {df_clean[col].max():.2f}")
                st.info(f"📉 Min: {df_clean[col].min():.2f}")

            else:
                st.warning("Not enough data for prediction")

        else:
            st.warning("No numeric column found for prediction")

    except Exception as e:
        st.error("❌ Error occurred but app is safe now (no crash)")
        st.exception(e)

else:
    st.info("📂 Upload any CSV file to start analysis")
