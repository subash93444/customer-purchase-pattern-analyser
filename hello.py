import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ---------------- UI SETTINGS ----------------
st.set_page_config(page_title="AI Dashboard", layout="wide")

# ---------------- LOGO ----------------
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image("logo.png", width=150)

# ---------------- TITLE (UPDATED) ----------------
st.title("📊 Product & Customer Insights System")
st.subheader("Data Analysis • Visualization • Insights")

# ---------------- SIDEBAR ----------------
st.sidebar.image("logo.png", width=120)
st.sidebar.title("⚙️ Controls")

show_data = st.sidebar.checkbox("Show Data", True)
show_cluster = st.sidebar.checkbox("Show Clustering", True)
show_prediction = st.sidebar.checkbox("Show Prediction", True)

# ---------------- UPLOAD ----------------
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    # ---------------- METRICS ----------------
    st.subheader("📌 Quick Insights")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", len(df))
    c2.metric("Products", df['Product'].nunique())
    c3.metric("Categories", df['Category'].nunique())

    # ---------------- DATA ----------------
    if show_data:
        with st.expander("🔍 View Dataset"):
            st.dataframe(df)

    # ---------------- TOP PRODUCTS ----------------
    st.subheader("📊 Top Products (Animated)")

    top_products = df['Product'].value_counts().reset_index()
    top_products.columns = ['Product', 'Count']

    fig1 = px.bar(
        top_products,
        x='Product',
        y='Count',
        text='Count',
        title="Top Products"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ---------------- CATEGORY ----------------
    st.subheader("📦 Category Distribution (Animated)")

    cat = df['Category'].value_counts().reset_index()
    cat.columns = ['Category', 'Count']

    fig2 = px.pie(
        cat,
        names='Category',
        values='Count',
        title="Category Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ---------------- CLUSTERING ----------------
    if show_cluster:
        st.subheader("🤖 Customer Segmentation (Animated)")

        if 'Amount' in df.columns:
            X = df[['Amount']]
            kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
            df['Cluster'] = kmeans.fit_predict(X)

            fig3 = px.scatter(
                df,
                x=df.index,
                y='Amount',
                color=df['Cluster'].astype(str),
                title="Customer Clusters"
            )

            st.plotly_chart(fig3, use_container_width=True)

    # ---------------- PREDICTION ----------------
    if show_prediction:
        st.subheader("📈 Sales Prediction (Animated)")

        if 'Amount' in df.columns:
            df['Index'] = range(len(df))

            model = LinearRegression()
            model.fit(df[['Index']], df['Amount'])

            future = pd.DataFrame({'Index': range(len(df), len(df)+5)})
            predictions = model.predict(future)

            full = pd.DataFrame({
                "Index": list(df['Index']) + list(future['Index']),
                "Amount": list(df['Amount']) + list(predictions),
                "Type": ["Actual"]*len(df) + ["Predicted"]*len(predictions)
            })

            fig4 = px.line(
                full,
                x="Index",
                y="Amount",
                color="Type",
                markers=True,
                title="Sales Prediction"
            )

            st.plotly_chart(fig4, use_container_width=True)
