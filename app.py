import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
from datetime import datetime

# 1. Page Config
st.set_page_config(page_title="UHPC Strength Predictor", page_icon="🏗️", layout="wide")

# 2. Load Trained Model
@st.cache_resource
def load_model():
    # استبدلي اسم الملف باسم ملف الموديل عندك
    model_path = "model.pkl" 
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# 3. Database Setup (Persistent Storage)
def init_db():
    conn = sqlite3.connect("experiments.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            scms REAL,
            temp REAL,
            water REAL,
            cement_ratio REAL,
            stf_ratio REAL,
            sp REAL,
            predicted_strength REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(scms, temp, water, cement, stf, sp, strength):
    conn = sqlite3.connect("experiments.db")
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO predictions (timestamp, scms, temp, water, cement_ratio, stf_ratio, sp, predicted_strength)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (now, scms, temp, water, cement, stf, sp, strength))
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect("experiments.db")
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df

init_db()

# 4. GUI Design (English Only)
st.title("🏗️ UHPC Compressive Strength Prediction System")
st.write("Input the mixture parameters to predict residual compressive strength and store attempt history.")

st.sidebar.header("🛠️ Input Parameters")
scms = st.sidebar.number_input("SCMS Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
temp = st.sidebar.number_input("Temperature (°C)", min_value=20.0, max_value=1000.0, value=25.0, step=10.0)
water = st.sidebar.number_input("Water Ratio", min_value=0.0, max_value=1.0, value=0.08, step=0.005)
cement_ratio = st.sidebar.number_input("Cement Ratio", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
stf_ratio = st.sidebar.number_input("STF Ratio", min_value=0.0, max_value=0.2, value=0.02, step=0.001)
sp = st.sidebar.number_input("SP Dosage", min_value=0.0, max_value=0.1, value=0.01, step=0.001)

# Prediction Logic
if st.button("🚀 Predict & Log Experiment", type="primary"):
    if model is not None:
        # ترتيب المدخلات وفقاً لما يتوقعه الموديل
        input_data = pd.DataFrame([[scms, temp, water, cement_ratio, stf_ratio, sp]], 
                                  columns=['SCMS', 'temp', 'Water', 'Cement ratio', 'STF ratio', 'SP'])
        
        pred = model.predict(input_data)[0]
        pred_value = round(float(pred), 2)
        
        # حفظ النتيجة في قاعدة البيانات
        save_prediction(scms, temp, water, cement_ratio, stf_ratio, sp, pred_value)
        st.success(f"✅ Prediction Result: **{pred_value} MPa**")
    else:
        st.error("⚠️ Model file (`model.pkl`) not found! Please make sure it is uploaded.")

st.divider()

# 5. History and CSV Download
st.subheader("📊 Experiment History Log")
history_df = load_history()

if not history_df.empty:
    st.dataframe(history_df, use_container_width=True)
    
    csv = history_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download History Log as CSV",
        data=csv,
        file_name=f"UHPC_Predictions_Log_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
else:
    st.info("No attempts recorded yet. Run a prediction to start logging.")