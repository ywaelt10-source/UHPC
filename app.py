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
            cement_ratio REAL,
            sf_ratio REAL,
            fa_ratio REAL,
            water REAL,
            sand_ratio REAL,
            sp REAL,
            qp REAL,
            stf REAL,
            fiber_type REAL,
            fiber_ratio REAL,
            temp REAL,
            predicted_fcu REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(scms, cement, sf, fa, water, sand, sp, qp, stf, f_type, f_ratio, temp, fcu):
    conn = sqlite3.connect("experiments.db")
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO predictions (timestamp, cement_ratio, sf_ratio, fa_ratio, scms, water, sand_ratio, sp, qp, stf, fiber_type, fiber_ratio, temp, predicted_fcu)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (now, cement, sf, fa, scms, water, sand, sp, qp, stf, f_type, f_ratio, temp, fcu))
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect("experiments.db")
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df

init_db()

# 4. GUI Design
st.title("🏗️ UHPC Compressive Strength (Fcu) Prediction System")
st.write("Input the 12 mixture parameters to predict compressive strength ($F_{cu}$) accurately.")

st.sidebar.header("🛠️ Input Parameters")

# Sidebar inputs configured with high float precision (format="%.4f")
scms = st.sidebar.number_input("SCMS (mass fraction)", min_value=0.0, max_value=2.0, value=0.2000, step=0.0100, format="%.4f")
cement_ratio = st.sidebar.number_input("Cement ratio (ratio)", min_value=0.0, max_value=5.0, value=1.0000, step=0.0100, format="%.4f")
sf_ratio = st.sidebar.number_input("SF ratio (ratio)", min_value=0.0, max_value=2.0, value=0.2500, step=0.0100, format="%.4f")
fa_ratio = st.sidebar.number_input("FA ratio (ratio)", min_value=0.0, max_value=2.0, value=0.0000, step=0.0100, format="%.4f")
water = st.sidebar.number_input("Water (mass fraction)", min_value=0.0, max_value=2.0, value=0.1800, step=0.0010, format="%.4f")
sand_ratio = st.sidebar.number_input("Sand ratio (ratio)", min_value=0.0, max_value=5.0, value=1.1000, step=0.0100, format="%.4f")
sp = st.sidebar.number_input("SP (mass fraction)", min_value=0.0, max_value=1.0, value=0.0150, step=0.0001, format="%.4f")
qp = st.sidebar.number_input("QP (mass fraction)", min_value=0.0, max_value=2.0, value=0.0000, step=0.0100, format="%.4f")
stf = st.sidebar.number_input("STF (vol. fraction)", min_value=0.0, max_value=1.0, value=0.0200, step=0.0001, format="%.4f")
fiber_type = st.sidebar.number_input("additional fiber type (encoded)", min_value=0.0, max_value=10.0, value=0.0000, step=1.0000, format="%.1f")
fiber_ratio = st.sidebar.number_input("additional fiber ratio (vol. fraction)", min_value=0.0, max_value=1.0, value=0.0000, step=0.0001, format="%.4f")
temp = st.sidebar.number_input("temp (°C)", min_value=20.0, max_value=1000.0, value=20.0000, step=1.0000, format="%.2f")

# Prediction Logic
if st.button("🚀 Predict & Log Experiment", type="primary"):
    if model is not None:
        try:
            # Construct DataFrame with exact feature names and order required by the trained model
            input_data = pd.DataFrame([{
               'SCMS': scms,
                'Cement ratio': cement_ratio,
                'SF ratio': sf_ratio,
                'FA ratio': fa_ratio,
                'Water': water,
                'Sand ratio': sand_ratio,
                'SP': sp,
                'QP': qp,
                'STF': stf,
                'additional fiber type': fiber_type,
                'additional fiber ratio': fiber_ratio,
                'temp': temp
            }])
            
            # Unrounded full-precision prediction value
            pred = model.predict(input_data)[0]
            pred_value = float(pred)
            
            save_prediction(cement_ratio, sf_ratio, fa_ratio, scms, water, sand_ratio, sp, qp, stf, fiber_type, fiber_ratio, temp, pred_value)
            st.success(f"✅ Predicted Compressive Strength ($F_{{cu}}$): **{pred_value} MPa**")
            
        except Exception as e:
            st.error(f"⚠️ Prediction Error: {e}")
    else:
        st.error("⚠️ Model file (`model.pkl`) not found!")

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
        file_name=f"UHPC_Fcu_Predictions_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
else:
    st.info("No attempts recorded yet. Run a prediction to start logging.")
