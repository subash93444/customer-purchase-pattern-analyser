import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Dashboard", layout="wide")

st.title("🧠 Customer Purchase Dashboard")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    # 🔥 Clean column names
    df.columns = df.columns.str.strip().str.lower()

    st.subheader("Data Preview")
    st.dataframe(df)

    st.write("Columns:", df.columns)  # DEBUG

    # ---------------------------
    # 📊 Top Products
    # ---------------------------
    st.subheader("Top Products")

    if 'product' in df.columns:
        st.bar_chart(df['product'].value_counts().head(5))
    else:
        st.error("❌ 'Product' column not found")

    # ---------------------------
    # 📂 Category
    # ---------------------------
    st.subheader("Category Distribution")

    if 'category' in df.columns:
        st.bar_chart(df['category'].value_counts())
    else:
        st.error("❌ 'Category' column not found")

    # ---------------------------
    # 🤖 AI Clustering
    # ---------------------------
    st.subheader("Customer Segmentation (Clustering)")

    if 'amount' in df.columns:
        X = df[['amount']]

        kmeans = KMeans(n_clusters=3, n_init=10)
        df['cluster'] = kmeans.fit_predict(X)

        fig, ax = plt.subplots()
        ax.scatter(df.index, df['amount'], c=df['cluster'])
        ax.set_xlabel("Customer Index")
        ax.set_ylabel("Amount")
        ax.set_title("Cluster Graph")

        st.pyplot(fig)
    else:
        st.error("❌ 'Amount' column not found")

    # ---------------------------
    # 📈 Prediction
    # ---------------------------
    st.subheader("Sales Prediction (AI)")

    if 'amount' in df.columns:
        df['index'] = range(len(df))

        X = df[['index']]
        y = df['amount']

        model = LinearRegression()
        model.fit(X, y)

        future = pd.DataFrame({'index': range(len(df), len(df)+5)})
        predictions = model.predict(future)

        fig2, ax2 = plt.subplots()

        ax2.plot(df['index'], df['amount'], marker='o', label="Actual")
        ax2.plot(future['index'], predictions, marker='o', linestyle='--', label="Predicted")

        ax2.set_xlabel("Customer Index")
        ax2.set_ylabel("Amount")
        ax2.set_title("Sales Prediction Graph")
        ax2.legend()

        st.pyplot(fig2)