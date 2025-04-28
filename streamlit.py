import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Load model dan scaler
model = tf.lite.Interpreter(model_path="model_heart_failure.tflite")
model.allocate_tensors()

scaler = joblib.load("scaler.pkl")

# Dapetin detail input/output
input_details = model.get_input_details()
output_details = model.get_output_details()

st.title("Prediksi Risiko Kematian Akibat Gagal Jantung ❤️")

# Form input user
age = st.number_input('Usia', min_value=1, max_value=120, value=50)
ejection_fraction = st.number_input('Ejection Fraction (%)', min_value=10, max_value=80, value=40)
serum_creatinine = st.number_input('Serum Creatinine (mg/dL)', min_value=0.1, max_value=10.0, value=1.0)
serum_sodium = st.number_input('Serum Sodium (mEq/L)', min_value=100, max_value=150, value=135)
time = st.number_input('Follow-up Time (hari)', min_value=1, max_value=400, value=100)
anaemia = st.selectbox('Anaemia', ['Tidak Ada', 'Ada'])
high_blood_pressure = st.selectbox('Tekanan Darah Tinggi', ['Tidak Ada', 'Ada'])
sex = st.selectbox('Jenis Kelamin', ['Wanita', 'Pria'])

#mapping
anaemia = 1 if anaemia == 'Ada' else 0
high_blood_pressure = 1 if high_blood_pressure == 'Ada' else 0
sex = 1 if sex == 'Pria' else 0

# Button prediksi
if st.button('Prediksi'):
    data = np.array([[age, ejection_fraction, serum_creatinine, serum_sodium, time]])
    data_scaled = scaler.transform(data)

    # Tambah kolom kategori (anaemia, high_blood_pressure, sex)
    final_data = np.hstack((data_scaled, [[anaemia, high_blood_pressure, sex]]))

    # Prediksi
    model.set_tensor(input_details[0]['index'], final_data.astype(np.float32))
    model.invoke()
    prediction = model.get_tensor(output_details[0]['index'])

    if prediction[0][0] > 0.5:
        st.error("⚠️ Pasien diprediksi MENINGGAL akibat gagal jantung.")
    else:
        st.success("✅ Pasien diprediksi TIDAK MENINGGAL akibat gagal jantung.")
