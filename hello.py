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

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload CSV", type=["csv"])

# ---------------- AUTO COLUMN DETECTION FUNCTION ----------------
def find_col(df, keywords):
    for col in df.columns:
        for k in keywords:
            if k in col.lower():
                return col
    return None

if file:
    df = pd.read_csv(file)

    # ---------------- CLEAN COLUMN NAMES ----------------
    df.columns = df.columns.str.strip().str.lower()

    st.success("✅ File uploaded successfully!")

    # ---------------- AUTO DETECT COLUMNS ----------------
    product_col = find_col(df, ["product", "item"])
    category_col = find_col(df, ["category", "type"])
    amount_col = find_col(df, ["amount", "price", "cost", "sales"])

    # ---------------- METRICS ----------------
    st.subheader("📌 Quick Insights")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Rows", len(df))
    c2.metric("Products", df[product_col].nunique() if product_col else "N/A")
    c3.metric("Categories", df[category_col].nunique() if category_col else "N/A")

    # ---------------- DATA VIEW ----------------
    if show_data:
        with st.expander("🔍 View Dataset"):
            st.dataframe(df)

    # ---------------- VALIDATION WARNING ----------------
    if not product_col:
        st.error("❌ Product column not found in dataset!")
    else:
        # ---------------- TOP PRODUCTS ----------------
        st.subheader("📊 Top Products")

        top_products = df[product_col].value_counts().reset_index()
        top_products.columns = ['Product', 'Count']

        fig1 = px.bar(top_products, x='Product', y='Count', text='Count')
        st.plotly_chart(fig1, use_container_width=True)

    # ---------------- CATEGORY ----------------
    if category_col:
        st.subheader("📦 Category Distribution")

        cat = df[category_col].value_counts().reset_index()
        cat.columns = ['Category', 'Count']

        fig2 = px.pie(cat, names='Category', values='Count')
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- CLUSTERING ----------------
    if show_cluster:
        st.subheader("🤖 Customer Segmentation")

        if amount_col:
            X = df[[amount_col]].dropna()

            if len(X) > 2:
                kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
                df['Cluster'] = kmeans.fit_predict(X)

                fig3 = px.scatter(
                    df,
                    x=df.index,
                    y=amount_col,
                    color=df['Cluster'].astype(str),
                    title="Customer Clusters"
                )

                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("⚠️ Amount column not found for clustering")

    # ---------------- PREDICTION ----------------
    if show_prediction:
        st.subheader("📈 Sales Prediction")

        if amount_col:
            df = df.dropna(subset=[amount_col])
            df['Index'] = range(len(df))

            model = LinearRegression()
            model.fit(df[['Index']], df[amount_col])

            future = pd.DataFrame({'Index': range(len(df), len(df)+5)})
            predictions = model.predict(future)

            full = pd.DataFrame({
                "Index": list(df['Index']) + list(future['Index']),
                "Amount": list(df[amount_col]) + list(predictions),
                "Type": ["Actual"]*len(df) + ["Predicted"]*len(predictions)
            })

            fig4 = px.line(full, x="Index", y="Amount", color="Type", markers=True)

            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("⚠️ Amount column not found for prediction")

else:
    st.info("📂 Please upload a CSV file to start analysis")
