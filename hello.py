import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="AI Customer Dashboard", layout="wide")

st.title("🚀 AI-Powered Customer Analytics Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

show_cluster = st.sidebar.checkbox("Show Clustering", True)
show_prediction = st.sidebar.checkbox("Show Prediction", True)

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    # ---------------- CLEAN DATA ----------------
    df = df.dropna()

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

    # ---------------- ERROR HANDLING ----------------
    if df.empty:
        st.error("⚠️ No data available after filtering")
        st.stop()

    # ---------------- TABS ----------------
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 ML", "📂 Data"])

    # =========================================================
    # ---------------- DASHBOARD TAB ----------------
    # =========================================================
    with tab1:

        st.subheader("📌 Business Metrics")

        total_customers = df['CustomerID'].nunique() if 'CustomerID' in df.columns else len(df)
        total_revenue = df['Amount'].sum() if 'Amount' in df.columns else 0
        transactions = len(df)
        avg_order_value = total_revenue / transactions if transactions > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Customers", total_customers)
        c2.metric("💰 Revenue", f"₹{total_revenue}")
        c3.metric("🧾 Transactions", transactions)
        c4.metric("📊 Avg Order", f"₹{avg_order_value:.2f}")

        st.divider()

        # ---------------- AI INSIGHTS ----------------
        st.subheader("🧠 AI Insights")

        if 'Amount' in df.columns:
            if df['Amount'].mean() > df['Amount'].median():
                st.success("💡 Customers are high spenders")
            else:
                st.warning("⚠️ Customers are low spenders")

        if 'Category' in df.columns:
            top_cat = df['Category'].value_counts().idxmax()
            st.info(f"🔥 Most popular category: {top_cat}")

        if 'Product' in df.columns:
            top_product = df['Product'].value_counts().idxmax()
            st.info(f"🏆 Top Product: {top_product}")

        st.divider()

        # ---------------- CUSTOMER ANALYSIS ----------------
        if 'CustomerID' in df.columns and 'Amount' in df.columns:

            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            df = df.dropna(subset=['Amount', 'CustomerID'])

            customer_df = df.groupby('CustomerID')['Amount'].sum().reset_index()
            customer_df = customer_df.sort_values(by='Amount', ascending=False)

            st.subheader("🏆 Top Customers")
            st.table(customer_df.head(5))

            fig = px.bar(customer_df.head(10),
                         x='CustomerID',
                         y='Amount',
                         title="Top Customers")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ---------------- CATEGORY ----------------
        if 'Category' in df.columns:
            st.subheader("📦 Category Distribution")

            cat = df['Category'].value_counts().reset_index()
            cat.columns = ['Category', 'Count']

            fig2 = px.pie(cat,
                          names='Category',
                          values='Count')
            st.plotly_chart(fig2, use_container_width=True)

        # ---------------- REVENUE ----------------
        if 'Category' in df.columns and 'Amount' in df.columns:
            st.subheader("💰 Revenue by Category")

            cat_rev = df.groupby('Category')['Amount'].sum().reset_index()

            fig3 = px.bar(cat_rev,
                          x='Category',
                          y='Amount')
            st.plotly_chart(fig3, use_container_width=True)

        # ---------------- REVENUE TREND ----------------
        if 'Date' in df.columns and 'Amount' in df.columns:
            st.subheader("📈 Revenue Trend")

            trend = df.groupby('Date')['Amount'].sum().reset_index()

            fig_trend = px.line(trend,
                                x='Date',
                                y='Amount',
                                title="Revenue Over Time")
            st.plotly_chart(fig_trend, use_container_width=True)

    # =========================================================
    # ---------------- ML TAB ----------------
    # =========================================================
    with tab2:

        # ---------------- CLUSTERING ----------------
        if show_cluster and 'CustomerID' in df.columns and 'Amount' in df.columns:

            st.subheader("🤖 Customer Segmentation")

            customer_df = df.groupby('CustomerID').agg({
                'Amount': 'sum',
                'CustomerID': 'count'
            }).rename(columns={'CustomerID': 'Frequency'}).reset_index()

            X = customer_df[['Amount', 'Frequency']]

            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            customer_df['Cluster'] = kmeans.fit_predict(X)

            fig4 = px.scatter(customer_df,
                              x='Frequency',
                              y='Amount',
                              color=customer_df['Cluster'].astype(str),
                              title="Customer Segments")
            st.plotly_chart(fig4, use_container_width=True)

        # ---------------- PREDICTION (IMPROVED) ----------------
        if show_prediction and 'Amount' in df.columns and 'Date' in df.columns:

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

            fig5 = px.line(full,
                           x="Date",
                           y="Amount",
                           color="Type",
                           markers=True)
            st.plotly_chart(fig5, use_container_width=True)

    # =========================================================
    # ---------------- DATA TAB ----------------
    # =========================================================
    with tab3:

        st.subheader("📂 Raw Data")
        st.dataframe(df)

        st.subheader("⬇️ Download Data")

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            file_name="customer_data.csv",
            mime="text/csv"
        )
