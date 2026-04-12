import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import datetime as dt

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="AI Customer Dashboard", layout="wide")

# ---------------- DARK THEME ----------------
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .stApp {
        background-color: #0e1117;
    }
    h1, h2, h3, h4 {
        color: #00ADB5;
    }
    .stMetric {
        background-color: #1c1f26;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOGO + TITLE ----------------
col1, col2 = st.columns([1,5])

with col1:
    st.image("logo.png", width=90)

with col2:
    st.title("🚀 AI-Powered Customer Analytics Dashboard")

st.markdown("Analyze customer behavior using AI 🚀")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

show_cluster = st.sidebar.checkbox("Show Clustering", True)
show_prediction = st.sidebar.checkbox("Show Prediction", True)

search = st.sidebar.text_input("🔍 Search Customer ID")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:

    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)

    df = load_data(file).dropna()

    if search:
        df = df[df['CustomerID'].astype(str).str.contains(search)]

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        start = st.sidebar.date_input("Start Date", df['Date'].min())
        end = st.sidebar.date_input("End Date", df['Date'].max())

        df = df[(df['Date'] >= pd.to_datetime(start)) &
                (df['Date'] <= pd.to_datetime(end))]

    if 'Category' in df.columns:
        category = st.sidebar.selectbox("Category", ["All"] + list(df['Category'].unique()))
        if category != "All":
            df = df[df['Category'] == category]

    if 'Product' in df.columns:
        product = st.sidebar.selectbox("Product", ["All"] + list(df['Product'].unique()))
        if product != "All":
            df = df[df['Product'] == product]

    if df.empty:
        st.error("⚠️ No data available")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 ML", "📂 Data"])

    # ================= DASHBOARD =================
    with tab1:

        total_customers = df['CustomerID'].nunique()
        total_revenue = df['Amount'].sum()
        transactions = len(df)
        avg_order_value = total_revenue / transactions if transactions else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers", total_customers)
        c2.metric("Revenue", f"₹{total_revenue}")
        c3.metric("Transactions", transactions)
        c4.metric("Avg Order", f"₹{avg_order_value:.2f}")

        st.divider()

        st.subheader("Top Customers")
        customer_df = df.groupby('CustomerID')['Amount'].sum().reset_index()
        customer_df = customer_df.sort_values(by='Amount', ascending=False)

        st.plotly_chart(px.bar(customer_df.head(10), x='CustomerID', y='Amount'))

        st.subheader("Category Distribution")
        cat = df['Category'].value_counts().reset_index()
        cat.columns = ['Category', 'Count']
        st.plotly_chart(px.pie(cat, names='Category', values='Count'))

        if 'Date' in df.columns:
            trend = df.groupby('Date')['Amount'].sum().reset_index()
            st.subheader("Revenue Trend")
            st.plotly_chart(px.line(trend, x='Date', y='Amount'))

    # ================= ML =================
    with tab2:

        snapshot_date = df['Date'].max()

        rfm = df.groupby('CustomerID').agg({
            'Date': lambda x: (snapshot_date - x.max()).days,
            'CustomerID': 'count',
            'Amount': 'sum'
        })

        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        rfm = rfm.reset_index()

        st.dataframe(rfm.head())

        # SAFE CLUSTERING FIX ✅
        if show_cluster:
            if len(rfm) >= 3:
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                rfm['Cluster'] = kmeans.fit_predict(rfm[['Recency','Frequency','Monetary']])
                st.plotly_chart(px.scatter(rfm, x='Frequency', y='Monetary', color='Cluster'))
            else:
                st.warning("⚠️ Not enough data for clustering")

        # PREDICTION
        if show_prediction:
            df = df.sort_values('Date')
            df['Days'] = (df['Date'] - df['Date'].min()).dt.days

            model = LinearRegression()
            model.fit(df[['Days']], df['Amount'])

            future = pd.DataFrame({'Days': range(df['Days'].max()+1, df['Days'].max()+6)})
            preds = model.predict(future)

            st.write("Next 5 Days Prediction:", preds)

    # ================= DATA =================
    with tab3:

        st.dataframe(df)

        st.download_button("Download CSV", df.to_csv(index=False), "data.csv")
