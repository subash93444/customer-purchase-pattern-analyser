import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

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

# ---------------- UPLOAD ----------------
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    # ---------------- FILTERS (NEW ADDED) ----------------
    st.sidebar.subheader("🔍 Filters")

    if 'Category' in df.columns:
        category = st.sidebar.selectbox("Category", ["All"] + list(df['Category'].unique()))
        if category != "All":
            df = df[df['Category'] == category]

    if 'Product' in df.columns:
        product = st.sidebar.selectbox("Product", ["All"] + list(df['Product'].unique()))
        if product != "All":
            df = df[df['Product'] == product]

    st.divider()

    # ---------------- KPI METRICS ----------------
    st.subheader("📌 Business Metrics")

    total_customers = df['CustomerID'].nunique() if 'CustomerID' in df.columns else len(df)
    total_revenue = df['Amount'].sum() if 'Amount' in df.columns else 0
    transactions = len(df)
    avg_order_value = total_revenue / transactions if transactions > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total Customers", total_customers)
    c2.metric("💰 Total Revenue", f"₹{total_revenue}")
    c3.metric("🧾 Transactions", transactions)
    c4.metric("📊 Avg Order Value", f"₹{avg_order_value:.2f}")

    st.divider()

    # ---------------- DOWNLOAD BUTTON (NEW ADDED) ----------------
    st.subheader("⬇️ Download Data")

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name="customer_data.csv",
        mime="text/csv"
    )

    st.divider()

    # ---------------- RAW DATA ----------------
    st.subheader("📂 Raw Data")

    if show_data:
        st.dataframe(df)

    st.divider()

    # ---------------- CUSTOMER ANALYSIS ----------------
    st.subheader("👥 Customer Analysis")

    if 'CustomerID' in df.columns and 'Amount' in df.columns:
        customer_df = df.groupby('CustomerID')['Amount'].sum().reset_index()
        customer_df = customer_df.sort_values(by='Amount', ascending=False)

        fig = px.bar(
            customer_df.head(10),
            x='CustomerID',
            y='Amount',
            title="Top Customers by Spending",
            text='Amount'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------- PRODUCT RECOMMENDATION ----------------
    st.subheader("🛍️ Product Recommendation")

    if 'Product' in df.columns:
        popular_products = df['Product'].value_counts().head(5)

        for product, count in popular_products.items():
            st.write(f"🔥 {product} (Bought {count} times)")

    st.divider()

    # ---------------- TOP PRODUCTS ----------------
    st.subheader("📊 Top Products")

    if 'Product' in df.columns:
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

    st.divider()

    # ---------------- CATEGORY ----------------
    st.subheader("📦 Category Distribution")

    if 'Category' in df.columns:
        cat = df['Category'].value_counts().reset_index()
        cat.columns = ['Category', 'Count']

        fig2 = px.pie(
            cat,
            names='Category',
            values='Count',
            title="Category Distribution"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ---------------- CUSTOMER SPENDING OVERVIEW ----------------
    st.subheader("💰 Customer Spending Overview")

    if 'Amount' in df.columns:
        fig3 = px.histogram(
            df,
            x='Amount',
            nbins=20,
            title="Customer Spending Distribution"
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ---------------- CLUSTERING ----------------
    if show_cluster:
        st.subheader("🤖 Customer Segmentation")

        if 'Amount' in df.columns:
            X = df[['Amount']]
            kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
            df['Cluster'] = kmeans.fit_predict(X)

            fig4 = px.scatter(
                df,
                x=df.index,
                y='Amount',
                color=df['Cluster'].astype(str),
                title="Customer Clusters"
            )
            st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ---------------- PREDICTION ----------------
    if show_prediction:
        st.subheader("📈 Sales Prediction")

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

            fig5 = px.line(
                full,
                x="Index",
                y="Amount",
                color="Type",
                markers=True,
                title="Sales Prediction"
            )

            st.plotly_chart(fig5, use_container_width=True)
