import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sqlite3

# ---------------- DATABASE ----------------

def init_db():
    conn = sqlite3.connect('users.db')
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT,
        password TEXT
    )
    """)
    conn.close()

def register_user(username, password):
    conn = sqlite3.connect('users.db')
    conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    result = cursor.fetchone()
    conn.close()
    return result

init_db()

# ---------------- LOGIN SYSTEM ----------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if not st.session_state.logged_in:

    if choice == "Login":
        st.title("🔐 Login")

        user = st.text_input("Username")
        pwd = st.text_input("Password", type='password')

        if st.button("Login"):
            if login_user(user, pwd):
                st.session_state.logged_in = True
                st.success("Login Successful ✅")
                st.rerun()
            else:
                st.error("Invalid Credentials ❌")

    elif choice == "Register":
        st.title("📝 Register")

        new_user = st.text_input("Username")
        new_pwd = st.text_input("Password", type='password')

        if st.button("Register"):
            register_user(new_user, new_pwd)
            st.success("Account Created ✅")

    st.stop()

# ---------------- MAIN DASHBOARD ----------------

st.set_page_config(page_title="AI Customer Dashboard", layout="wide")

st.title("🚀 AI-Powered Customer Analytics Dashboard")

st.sidebar.title("⚙️ Controls")
show_cluster = st.sidebar.checkbox("Show Clustering", True)
show_prediction = st.sidebar.checkbox("Show Prediction", True)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

search = st.sidebar.text_input("🔍 Search Customer ID")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:

    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)

    df = load_data(file)
    df = df.dropna()

    # 🔥 REQUIRED COLUMNS CHECK
    if 'CustomerID' not in df.columns or 'Amount' not in df.columns:
        st.error("CSV must contain CustomerID and Amount columns")
        st.stop()

    if search:
        df = df[df['CustomerID'].astype(str).str.contains(search)]

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        start = st.sidebar.date_input("Start Date", df['Date'].min())
        end = st.sidebar.date_input("End Date", df['Date'].max())

        df = df[(df['Date'] >= pd.to_datetime(start)) &
                (df['Date'] <= pd.to_datetime(end))]

    if df.empty:
        st.error("⚠️ No data available after filtering")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 ML", "📂 Data"])

    # ---------------- DASHBOARD ----------------
    with tab1:

        total_customers = df['CustomerID'].nunique()
        total_revenue = df['Amount'].sum()
        transactions = len(df)
        avg_order_value = total_revenue / transactions if transactions > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Customers", total_customers)
        c2.metric("💰 Revenue", f"₹{total_revenue}")
        c3.metric("🧾 Transactions", transactions)
        c4.metric("📊 Avg Order", f"₹{avg_order_value:.2f}")

        st.subheader("🧠 AI Insights")

        if df['Amount'].mean() > df['Amount'].median():
            st.success("💡 High-value customers detected")
        else:
            st.warning("⚠️ Low spending behavior detected")

        st.subheader("🏆 Top Customers")
        customer_df = df.groupby('CustomerID')['Amount'].sum().reset_index()
        customer_df = customer_df.sort_values(by='Amount', ascending=False)
        st.dataframe(customer_df.head(5))

        st.plotly_chart(px.bar(customer_df.head(10), x='CustomerID', y='Amount'), use_container_width=True)

        if 'Date' in df.columns:
            st.subheader("📈 Revenue Trend")
            trend = df.groupby('Date')['Amount'].sum().reset_index()
            st.plotly_chart(px.line(trend, x='Date', y='Amount'))

    # ---------------- ML ----------------
    with tab2:

        if 'Date' in df.columns:

            snapshot_date = df['Date'].max()

            rfm = df.groupby('CustomerID').agg({
                'Date': lambda x: (snapshot_date - x.max()).days,
                'CustomerID': 'count',
                'Amount': 'sum'
            })

            rfm.columns = ['Recency', 'Frequency', 'Monetary']
            rfm = rfm.reset_index()

            st.subheader("📊 RFM Analysis")
            st.dataframe(rfm.head())

            if show_cluster:
                X = rfm[['Recency', 'Frequency', 'Monetary']]

                if len(X) > 2:
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    rfm['Cluster'] = kmeans.fit_predict(X)

                    st.plotly_chart(px.scatter(rfm, x='Frequency', y='Monetary', color='Cluster'))

        if show_prediction and 'Date' in df.columns:

            st.subheader("📈 Prediction")

            df = df.sort_values('Date')
            df['Days'] = (df['Date'] - df['Date'].min()).dt.days

            model = LinearRegression()
            model.fit(df[['Days']], df['Amount'])

            score = r2_score(df['Amount'], model.predict(df[['Days']]))
            st.write(f"Model Accuracy: {score:.2f}")

            if score < 0.5:
                st.warning("⚠️ Low accuracy model")

    # ---------------- DATA ----------------
    with tab3:
        st.dataframe(df)
