import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ---------------- UI ----------------
st.set_page_config(page_title="AI Dashboard", layout="wide")
st.title("🧠 Universal AI Dataset Dashboard")

file = st.file_uploader("Upload ANY CSV File", type=["csv"])

# ---------------- SAFE FINDER ----------------
def safe_numeric(df):
    nums = df.select_dtypes(include='number').columns.tolist()
    if len(nums) == 0:
        df["auto_index"] = range(len(df))
        nums = ["auto_index"]
    return nums

# ---------------- MAIN ----------------
if file:
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower()

        st.success("✅ File Loaded Successfully")
        st.dataframe(df.head())

        # ---------------- SAFE COLUMNS ----------------
        num_cols = safe_numeric(df)
        col = num_cols[0]

        # ---------------- QUICK INSIGHTS ----------------
        st.subheader("📌 Quick Insights")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Numeric Columns", len(num_cols))

        # ---------------- DATA OVERVIEW ----------------
        st.subheader("📊 Data Overview")

        fig1 = px.histogram(df, x=col)
        st.plotly_chart(fig1, use_container_width=True)

        # ---------------- CLUSTERING (SAFE) ----------------
        st.subheader("🤖 Auto Clustering")

        if len(df) > 3:
            X = df[[col]].dropna()

            model = KMeans(n_clusters=3, n_init=10, random_state=42)
            df["cluster"] = model.fit_predict(X)

            fig2 = px.scatter(
                df,
                x=df.index,
                y=col,
                color=df["cluster"].astype(str),
                title="AI Clusters"
            )

            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Not enough data for clustering")

        # ---------------- PREDICTION (SAFE AI) ----------------
        st.subheader("📈 AI Prediction")

        clean = df[[col]].dropna().copy()
        clean["index"] = range(len(clean))

        if len(clean) > 3:

            model = LinearRegression()
            model.fit(clean[["index"]], clean[col])

            future = pd.DataFrame({"index": range(len(clean), len(clean)+10)})
            pred = model.predict(future)

            full = pd.DataFrame({
                "index": list(clean["index"]) + list(future["index"]),
                "value": list(clean[col]) + list(pred),
                "type": ["Actual"] * len(clean) + ["Predicted"] * len(pred)
            })

            fig3 = px.line(
                full,
                x="index",
                y="value",
                color="type",
                markers=True
            )

            st.plotly_chart(fig3, use_container_width=True)

        else:
            st.warning("Not enough data for prediction")

    except Exception as e:
        st.error("⚠️ File processed safely but something went wrong")
        st.exception(e)

else:
    st.info("📂 Upload any CSV file to start analysis")
