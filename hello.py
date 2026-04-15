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

    st.subheader("Data Preview")
    st.dataframe(df)

    # Top products
    st.subheader("Top Products")
    st.bar_chart(df['Product'].value_counts().head(5))

    # Category
    st.subheader("Category Distribution")
    st.bar_chart(df['Category'].value_counts())

    # ---------------------------
    # 🤖 AI Clustering
    # ---------------------------
    st.subheader("Customer Segmentation (Clustering)")

    if 'Amount' in df.columns:
        X = df[['Amount']]
        kmeans = KMeans(n_clusters=3)
        df['Cluster'] = kmeans.fit_predict(X)

        fig, ax = plt.subplots()
        ax.scatter(df.index, df['Amount'], c=df['Cluster'])
        ax.set_xlabel("Customer Index")
        ax.set_ylabel("Amount")
        ax.set_title("Cluster Graph")
        st.pyplot(fig)

    # ---------------------------
    # 📈 Prediction (NEW)
    # ---------------------------
    st.subheader("Sales Prediction (AI)")

    if 'Amount' in df.columns:
        df['Index'] = range(len(df))

        X = df[['Index']]
        y = df['Amount']

        model = LinearRegression()
        model.fit(X, y)

        future = pd.DataFrame({'Index': range(len(df), len(df)+5)})
        predictions = model.predict(future)

        # Combine original + prediction
        fig2, ax2 = plt.subplots()

        ax2.plot(df['Index'], df['Amount'], marker='o', label="Actual")
        ax2.plot(future['Index'], predictions, marker='o', linestyle='--', label="Predicted")

        ax2.set_xlabel("Customer Index")
        ax2.set_ylabel("Amount")
        ax2.set_title("Sales Prediction Graph")
        ax2.legend()

        st.pyplot(fig2)
