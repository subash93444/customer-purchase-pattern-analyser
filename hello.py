import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import datetime as dt

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="AI Customer Dashboard", layout="wide")

st.title("🚀 AI-Powered Customer Analytics Dashboard")

st.markdown("This dashboard analyzes customer purchase behavior using AI techniques like clustering, RFM analysis, and prediction.")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

show_cluster = st.sidebar.checkbox("Show Clustering", True)
show_prediction = st.sidebar.checkbox("Show Prediction", True)

search = st.sidebar.text_input("🔍 Search Customer ID")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:

    # ---------------- LOAD + CACHE ----------------
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)

    df = load_data(file)

    df = df.dropna()

    # ---------------- SEARCH ----------------
    if search:
        df = df[df['CustomerID'].astype(str).str.contains(search)]

    # ---------------- DATE FILTER ----------------
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        start = st.sidebar.date_input("Start Date", df['Date'].min())
        end = st.sidebar.date_input("End Date", df['Date'].max())

        df = df[(df['Date'] >= pd.to_datetime(start)) &
                (df['Date'] <= pd.to_datetime(end))]

    # ---------------- CATEGORY FILTER ----------------
    if 'Category' in df.columns:
        category = st.sidebar.selectbox("Category", ["All"] + list(df['Category'].dropna().unique()))
        if category != "All":
            df = df[df['Category'] == category]

    # ---------------- PRODUCT FILTER ----------------
    if 'Product' in df.columns:
        product = st.sidebar.selectbox("Product", ["All"] + list(df['Product'].dropna().unique()))
        if product != "All":
            df = df[df['Product'] == product]

    # ---------------- ERROR ----------------
    if df.empty:
        st.error("⚠️ No data available after filtering")
        st.stop()

    # ---------------- TABS ----------------
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 ML", "📂 Data"])

    # =========================================================
    # ---------------- DASHBOARD ----------------
    # =========================================================
    with tab1:

        st.subheader("📌 Business Metrics")

        total_customers = df['CustomerID'].nunique()
        total_revenue = df['Amount'].sum()
        transactions = len(df)
        avg_order_value = total_revenue / transactions if transactions > 0 else 0

        # KPI Growth
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

        # ---------------- AI INSIGHTS ----------------
        st.subheader("🧠 AI Insights")

        if df['Amount'].mean() > df['Amount'].median():
            st.success("💡 Customers are high spenders")
        else:
            st.warning("⚠️ Customers are low spenders")

        top_cat = df['Category'].value_counts().idxmax()
        st.info(f"🔥 Most popular category: {top_cat}")

        top_product = df['Product'].value_counts().idxmax()
        st.info(f"🏆 Top Product: {top_product}")

        st.divider()

        # ---------------- TOP CUSTOMERS ----------------
        customer_df = df.groupby('CustomerID')['Amount'].sum().reset_index()
        customer_df = customer_df.sort_values(by='Amount', ascending=False)

        st.subheader("🏆 Top Customers")
        st.table(customer_df.head(5))

        fig = px.bar(customer_df.head(10), x='CustomerID', y='Amount')
        st.plotly_chart(fig, use_container_width=True)

        # ---------------- CATEGORY ----------------
        st.subheader("📦 Category Distribution")

        cat = df['Category'].value_counts().reset_index()
        cat.columns = ['Category', 'Count']

        fig2 = px.pie(cat, names='Category', values='Count')
        st.plotly_chart(fig2, use_container_width=True)

        # ---------------- REVENUE TREND ----------------
        if 'Date' in df.columns:
            st.subheader("📈 Revenue Trend")

            trend = df.groupby('Date')['Amount'].sum().reset_index()
            fig_trend = px.line(trend, x='Date', y='Amount')
            st.plotly_chart(fig_trend, use_container_width=True)

        # ---------------- HEATMAP ----------------
        st.subheader("🔥 Sales Heatmap")

        pivot = df.pivot_table(values='Amount',
                               index='Category',
                               columns='Product',
                               aggfunc='sum')

        fig_heat = px.imshow(pivot)
        st.plotly_chart(fig_heat, use_container_width=True)

    # =========================================================
    # ---------------- ML ----------------
    # =========================================================
    with tab2:

        # ---------------- RFM ANALYSIS ----------------
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

        # ---------------- CLUSTERING ----------------
        if show_cluster:

            X = rfm[['Recency', 'Frequency', 'Monetary']]

            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            rfm['Cluster'] = kmeans.fit_predict(X)

            def label(row):
                if row['Cluster'] == 0:
                    return "High Value 💰"
                elif row['Cluster'] == 1:
                    return "Regular 🙂"
                else:
                    return "Low Value ⚠️"

            rfm['Segment'] = rfm.apply(label, axis=1)

            fig4 = px.scatter(rfm,
                              x='Frequency',
                              y='Monetary',
                              color='Segment')
            st.plotly_chart(fig4, use_container_width=True)

        # ---------------- PREDICTION ----------------
        if show_prediction:

            st.subheader("📈 Sales Prediction")

            df = df.sort_values('Date')
            df['Days'] = (df['Date'] - df['Date'].min()).dt.days

            model = LinearRegression()
            model.fit(df[['Days']], df['Amount'])

            future_days = pd.DataFrame({
                'Days': range(df['Days'].max()+1, df['Days'].max()+6)
            })

            predictions = model.predict(future_days)

            future_dates = pd.date_range(df['Date'].max(), periods=6)[1:]

            full = pd.DataFrame({
                "Date": list(df['Date']) + list(future_dates),
                "Amount": list(df['Amount']) + list(predictions),
                "Type": ["Actual"]*len(df) + ["Predicted"]*len(predictions)
            })

            fig5 = px.line(full, x="Date", y="Amount", color="Type", markers=True)
            st.plotly_chart(fig5, use_container_width=True)

    # =========================================================
    # ---------------- DATA ----------------
    # =========================================================
    with tab3:

        st.subheader("📂 Raw Data")
        st.dataframe(df)

        st.download_button("⬇️ Download CSV",
                           df.to_csv(index=False),
                           file_name="customer_data.csv",
                           mime="text/csv")

        # Insights download
        st.download_button("📥 Download Insights",
                           f"Top Category: {top_cat}",
                           file_name="insights.txt")
