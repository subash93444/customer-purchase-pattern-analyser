import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import datetime as dt

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="AI Customer Dashboard", layout="wide")

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

        st.subheader("🧠 AI Insights")

        if df['Amount'].mean() > df['Amount'].median():
            st.success("💡 Customers are high spenders")
        else:
            st.warning("⚠️ Customers are low spenders")

        top_cat = df['Category'].value_counts().idxmax()
        st.info(f"🔥 Most popular category: {top_cat}")

        top_product = df['Product'].value_counts().idxmax()
        st.info(f"🏆 Top Product: {top_product}")

        st.subheader("📢 Business Summary")
        st.write(f"""
        - Total Revenue: ₹{total_revenue}
        - Top Category: {top_cat}
        - Avg Order Value: ₹{avg_order_value:.2f}

        👉 Customers are {'high' if avg_order_value > 500 else 'low'} spenders  
        👉 Focus on {top_cat} category for growth  
        """)

        st.subheader("💡 Recommendations")

        if total_revenue > 10000:
            st.success("Increase stock for high-performing products")
        else:
            st.warning("Focus on marketing strategies to boost sales")

        if avg_order_value < 300:
            st.info("Offer combo deals to increase order value")

        st.divider()

        customer_df = df.groupby('CustomerID')['Amount'].sum().reset_index()
        customer_df = customer_df.sort_values(by='Amount', ascending=False)

        st.subheader("🏆 Top Customers")
        st.table(customer_df.head(5))

        st.plotly_chart(px.bar(customer_df.head(10), x='CustomerID', y='Amount'), use_container_width=True)

        st.subheader("📦 Category Distribution")
        cat = df['Category'].value_counts().reset_index()
        cat.columns = ['Category', 'Count']
        st.plotly_chart(px.pie(cat, names='Category', values='Count'))

        if 'Date' in df.columns:
            st.subheader("📈 Revenue Trend")
            trend = df.groupby('Date')['Amount'].sum().reset_index()
            st.plotly_chart(px.line(trend, x='Date', y='Amount'))

        st.subheader("🔥 Sales Heatmap")
        pivot = df.pivot_table(values='Amount', index='Category', columns='Product', aggfunc='sum')
        st.plotly_chart(px.imshow(pivot))

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

        st.subheader("💰 Customer Lifetime Value")
        clv = df.groupby('CustomerID')['Amount'].mean() * df.groupby('CustomerID')['CustomerID'].count()
        st.write(clv.head())

        if show_cluster:
            X = rfm[['Recency', 'Frequency', 'Monetary']]
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            rfm['Cluster'] = kmeans.fit_predict(X)

            rfm['Segment'] = rfm['Cluster'].map({0:"High Value 💰",1:"Regular 🙂",2:"Low Value ⚠️"})
            st.plotly_chart(px.scatter(rfm, x='Frequency', y='Monetary', color='Segment'))

            st.subheader("🧠 Segment Insights")
            st.write(rfm['Segment'].value_counts())

        if show_prediction:

            st.subheader("📈 Sales Prediction")

            df = df.sort_values('Date')
            df['Days'] = (df['Date'] - df['Date'].min()).dt.days

            model = LinearRegression()
            model.fit(df[['Days']], df['Amount'])

            future_days = pd.DataFrame({'Days': range(df['Days'].max()+1, df['Days'].max()+6)})
            predictions = model.predict(future_days)

            future_dates = pd.date_range(df['Date'].max(), periods=6)[1:]

            full = pd.DataFrame({
                "Date": list(df['Date']) + list(future_dates),
                "Amount": list(df['Amount']) + list(predictions),
                "Type": ["Actual"]*len(df) + ["Predicted"]*len(predictions)
            })

            st.plotly_chart(px.line(full, x="Date", y="Amount", color="Type"))

            score = r2_score(df['Amount'], model.predict(df[['Days']]))
            st.write(f"📊 Model Accuracy (R²): {score:.2f}")

            st.subheader("🎯 What-If Prediction")
            days = st.slider("Future Days", 1, 30, 5)
            pred = model.predict(pd.DataFrame({'Days': [df['Days'].max() + days]}))
            st.write(f"Predicted Revenue: ₹{pred[0]:.2f}")

    # ================= DATA =================
    with tab3:

        st.subheader("📂 Raw Data")
        st.dataframe(df)

        st.download_button("⬇️ Download CSV",
                           df.to_csv(index=False),
                           file_name="customer_data.csv")

        st.download_button("📥 Download Insights",
                           f"Top Category: {top_cat}",
                           file_name="insights.txt")
