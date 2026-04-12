import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import datetime as dt

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="AI Customer Dashboard", layout="wide")

# ---------------- ANIMATION CSS ----------------
st.markdown("""
<style>
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1f1f1f, #2b2b2b);
    padding: 15px;
    border-radius: 12px;
    color: white;
    transition: 0.3s;
}
[data-testid="stMetric"]:hover {
    transform: scale(1.05);
}

.stPlotlyChart {
    animation: fadeIn 1.2s ease-in-out;
}

@keyframes fadeIn {
    from {opacity:0; transform: translateY(20px);}
    to {opacity:1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGO + TITLE ----------------
col1, col2 = st.columns([1,5])

with col1:
    st.image("logo.png", width=90)

with col2:
    st.title("🚀 AI-Powered Customer Analytics Dashboard")

st.markdown("This dashboard analyzes customer purchase behavior using AI techniques like clustering, RFM analysis, and prediction.")

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

    df = load_data(file)
    df = df.dropna()

    if search:
        df = df[df['CustomerID'].astype(str).str.contains(search)]

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        start = st.sidebar.date_input("Start Date", df['Date'].min())
        end = st.sidebar.date_input("End Date", df['Date'].max())

        df = df[(df['Date'] >= pd.to_datetime(start)) &
                (df['Date'] <= pd.to_datetime(end))]

    if 'Category' in df.columns:
        category = st.sidebar.selectbox("Category", ["All"] + list(df['Category'].dropna().unique()))
        if category != "All":
            df = df[df['Category'] == category]

    if 'Product' in df.columns:
        product = st.sidebar.selectbox("Product", ["All"] + list(df['Product'].dropna().unique()))
        if product != "All":
            df = df[df['Product'] == product]

    if df.empty:
        st.error("⚠️ No data available after filtering")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 ML", "📂 Data"])

    # ================= DASHBOARD =================
    with tab1:

        st.subheader("📌 Business Metrics")

        total_customers = df['CustomerID'].nunique()
        total_revenue = df['Amount'].sum()
        transactions = len(df)
        avg_order_value = total_revenue / transactions if transactions > 0 else 0

        if 'Date' in df.columns:
            last_week = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=7)]
            prev_week = df[(df['Date'] < df['Date'].max() - pd.Timedelta(days=7)) &
                           (df['Date'] >= df['Date'].max() - pd.Timedelta(days=14))]
            growth = last_week['Amount'].sum() - prev_week['Amount'].sum()
        else:
            growth = 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Customers", total_customers)
        c2.metric("💰 Revenue", f"₹{total_revenue}", delta=f"{growth:.2f}")
        c3.metric("🧾 Transactions", transactions)
        c4.metric("📊 Avg Order", f"₹{avg_order_value:.2f}")

        st.divider()

        # ✅ AI INSIGHT CARDS
        st.subheader("🧠 AI Insights")

        colA, colB, colC = st.columns(3)

        with colA:
            if df['Amount'].mean() > df['Amount'].median():
                st.success("💡 High Spending Customers")
            else:
                st.warning("⚠️ Low Spending Customers")

        with colB:
            top_cat = df['Category'].value_counts().idxmax()
            st.info(f"🔥 Top Category\n{top_cat}")

        with colC:
            top_product = df['Product'].value_counts().idxmax()
            st.info(f"🏆 Top Product\n{top_product}")

        st.divider()

        customer_df = df.groupby('CustomerID')['Amount'].sum().reset_index()
        customer_df = customer_df.sort_values(by='Amount', ascending=False)

        st.subheader("🏆 Top Customers")

        fig_bar = px.bar(customer_df.head(10), x='CustomerID', y='Amount')
        fig_bar.update_layout(transition_duration=800)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("📦 Category Distribution")
        cat = df['Category'].value_counts().reset_index()
        cat.columns = ['Category', 'Count']

        fig_pie = px.pie(cat, names='Category', values='Count')
        fig_pie.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_pie)

        if 'Date' in df.columns:
            st.subheader("📈 Revenue Trend")
            trend = df.groupby('Date')['Amount'].sum().reset_index()

            fig_line = px.line(trend, x='Date', y='Amount')
            fig_line.update_layout(transition_duration=800)
            st.plotly_chart(fig_line)

    # ================= ML =================
    with tab2:

        st.subheader("📊 RFM Analysis")

        snapshot_date = df['Date'].max()

        rfm = df.groupby('CustomerID').agg({
            'Date': lambda x: (snapshot_date - x.max()).days,
            'CustomerID': 'count',
            'Amount': 'sum'
        })

        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        rfm = rfm.reset_index()

        st.dataframe(rfm.head())

        # ✅ SAFE KMEANS FIX
        if show_cluster:

            X = rfm[['Recency', 'Frequency', 'Monetary']].dropna()

            if len(X) < 2:
                st.warning("⚠️ Not enough data for clustering")
            else:
                n_clusters = min(3, len(X))

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                rfm['Cluster'] = kmeans.fit_predict(X)

                st.plotly_chart(px.scatter(rfm, x='Frequency', y='Monetary', color='Cluster'))

    # ================= DATA =================
    with tab3:

        st.subheader("📂 Raw Data")
        st.dataframe(df)
