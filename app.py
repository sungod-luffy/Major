import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import bcrypt
import hashlib
import os

# Initialize database
def init_db():
    if not os.path.exists('users.db'):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL
                    )''')
        conn.commit()
        conn.close()

init_db()

# Hash password
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Verify password
def verify_password(stored_hash, provided_password):
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_hash.encode('utf-8'))

# Register new user
def register_user(username, email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        hashed_pw = hash_password(password)
        c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                  (username, email, hashed_pw))
        conn.commit()
        conn.close()
        return True, "Registration successful!"
    except sqlite3.IntegrityError as e:
        conn.close()
        if "username" in str(e):
            return False, "Username already exists"
        else:
            return False, "Email already registered"
    except Exception as e:
        conn.close()
        return False, f"Error: {str(e)}"

# Login user
def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    
    if result is None:
        return False, "Username not found"
    
    if verify_password(result[0], password):
        return True, "Login successful!"
    else:
        return False, "Incorrect password"

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
if 'show_login' not in st.session_state:
    st.session_state.show_login = False

# Login/Register Page
def show_auth_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 Churn Prediction System")
        st.write("---")
        
        default_index = 0 if st.session_state.show_login else 1
        auth_choice = st.radio("Choose Action:", ["Login", "Register"], horizontal=True, index=default_index)
        st.session_state.show_login = False
        
        if auth_choice == "Register":
            st.subheader("Create New Account")
            with st.form("register_form"):
                new_username = st.text_input("Username")
                new_email = st.text_input("Email")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                
                if st.form_submit_button("Register"):
                    if not new_username or not new_email or not new_password:
                        st.error("Please fill in all fields")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        success, message = register_user(new_username, new_email, new_password)
                        if success:
                            st.success(message)
                            st.info("Redirecting to login...")
                            st.session_state.show_login = True
                            st.rerun()
                        else:
                            st.error(message)
        
        else:  # Login
            st.subheader("Login to Your Account")
            with st.form("login_form"):
                login_username = st.text_input("Username")
                login_password = st.text_input("Password", type="password")
                
                if st.form_submit_button("Login"):
                    if not login_username or not login_password:
                        st.error("Please enter username and password")
                    else:
                        success, message = login_user(login_username, login_password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.username = login_username
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

# Main app
if not st.session_state.logged_in:
    show_auth_page()
else:
    # Logout button in sidebar
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()
    
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    
    # Sidebar with project info and file upload
    st.sidebar.title('Churn Prediction System')
st.sidebar.info('Upload your customer data and explore churn risk using unsupervised learning and explainable AI.')
st.sidebar.markdown('---')
st.sidebar.header('Upload Data')
uploaded_file = st.sidebar.file_uploader('Upload your customer data (CSV)', type=['csv'])
st.sidebar.markdown('---')
st.sidebar.write('Developed with ❤️ using Streamlit')

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write('### Raw Data', data.head())
    st.write('#### Columns in your data:', list(data.columns))

    required_cols = ['Customer ID', 'InvoiceDate', 'Invoice', 'Quantity', 'Price']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.warning("Please upload a CSV with the required columns or update the code to match your data.")
    else:
        # User selects number of clusters
        n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=8, value=4, help='Choose how many customer segments to create')

        # Tabs for each analysis section
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            'RFM Analysis', 'RFM Distributions', 'Clustering', 'Churn Prediction', 'Explainable AI (SHAP)'])

        with tab1:
            st.write('## RFM Analysis')
            st.info('RFM (Recency, Frequency, Monetary) analysis segments customers based on how recently, how often, and how much they purchase. This helps identify valuable and at-risk customers.')
            data['TotalPrice'] = data['Quantity'] * data['Price']
            # Ensure InvoiceDate is datetime
            data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
            rfm = data.groupby('Customer ID').agg({
                'InvoiceDate': lambda x: (data['InvoiceDate'].max() - x.max()).days,
                'Invoice': 'nunique',
                'TotalPrice': 'sum'
            })
            rfm.columns = ['Recency', 'Frequency', 'Monetary']
            st.dataframe(rfm.head())
            st.metric('Total Customers', len(rfm))
            st.metric('Avg Recency', int(rfm['Recency'].mean()))
            st.metric('Avg Frequency', round(rfm['Frequency'].mean(), 2))
            st.metric('Avg Monetary', round(rfm['Monetary'].mean(), 2))

        with tab2:
            st.write('### RFM Feature Distributions')
            st.caption('These histograms show the distribution of Recency, Frequency, and Monetary values across all customers.')
            fig_rfm, axs = plt.subplots(1, 3, figsize=(15, 4))
            axs[0].hist(rfm['Recency'], bins=20, color='skyblue')
            axs[0].set_title('Recency')
            axs[1].hist(rfm['Frequency'], bins=20, color='lightgreen')
            axs[1].set_title('Frequency')
            axs[2].hist(rfm['Monetary'], bins=20, color='salmon')
            axs[2].set_title('Monetary')
            st.pyplot(fig_rfm)

        with tab3:
            st.write('## Hybrid Clustering')
            st.info('Clustering groups customers with similar RFM profiles. KMeans and Agglomerative clustering are used to find natural segments in your customer base.')
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(rfm_scaled)
            agg = AgglomerativeClustering(n_clusters=n_clusters)
            agg_labels = agg.fit_predict(rfm_scaled)
            rfm['KMeans_Cluster'] = kmeans_labels
            rfm['Agg_Cluster'] = agg_labels
            st.dataframe(rfm.head())
            st.write('### Cluster Scatterplot (Recency vs Monetary)')
            st.caption('Each point is a customer, colored by their cluster. This helps visualize how clusters separate based on Recency and Monetary value.')
            fig_scatter, ax_scatter = plt.subplots()
            scatter = ax_scatter.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['KMeans_Cluster'], cmap='tab10', alpha=0.7)
            legend1 = ax_scatter.legend(*scatter.legend_elements(), title="KMeans Cluster")
            ax_scatter.add_artist(legend1)
            ax_scatter.set_xlabel('Recency')
            ax_scatter.set_ylabel('Monetary')
            st.pyplot(fig_scatter)

        with tab4:
            st.write('## Churn Prediction (Cluster-based)')
            st.info('Customers in certain clusters (e.g., high Recency, low Frequency/Monetary) may be at higher risk of churn. This unsupervised approach uses clusters as a proxy for churn risk.')
            cluster_counts = rfm['KMeans_Cluster'].value_counts().sort_index()
            st.write('### Number of Customers per KMeans Cluster')
            st.caption('This bar chart shows how many customers are in each cluster.')
            fig_bar, ax_bar = plt.subplots()
            ax_bar.bar(cluster_counts.index.astype(str), cluster_counts.values, color='orchid')
            ax_bar.set_xlabel('KMeans Cluster')
            ax_bar.set_ylabel('Number of Customers')
            st.pyplot(fig_bar)
            st.write(rfm.groupby('KMeans_Cluster').mean())
            # Download button for cluster results
            csv = rfm.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button('Download Clustered Data as CSV', csv, 'clustered_customers.csv', 'text/csv')

        with tab5:
            st.write('## Explainable AI (SHAP)')
            st.info('SHAP (SHapley Additive exPlanations) explains which RFM features are most important for assigning customers to clusters, helping you understand the drivers of churn risk.')
            explainer = shap.KernelExplainer(kmeans.predict, rfm_scaled)
            shap_values = explainer.shap_values(rfm_scaled[:50])
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, rfm.iloc[:50, :3], show=False)
            st.pyplot(fig)

        st.write('---')
        st.write('This is a demo. For production, tune clustering and RFM logic to your data.')
else:
    st.info('Awaiting CSV file upload.')
