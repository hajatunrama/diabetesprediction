import streamlit as st
import numpy as np
import joblib

# =========================
# KONFIGURASI HALAMAN
# =========================

st.set_page_config(
    page_title="Sistem Prediksi Diabetes",
    page_icon="🩺",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load("diabetes_model.pkl")

# =========================
# HEADER
# =========================

st.title("🩺 Sistem Prediksi Diabetes")
st.markdown(
    """
    Aplikasi Machine Learning untuk memprediksi risiko diabetes
    menggunakan algoritma **K-Nearest Neighbors (KNN)**.
    """
)

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("📋 Informasi")

    st.info(
        """
        **Dataset**
        - Pima Indians Diabetes Dataset

        **Algoritma**
        - K-Nearest Neighbors (KNN)

        **Deployment**
        - Streamlit Cloud
        """
    )

    st.header("👨‍🎓 Identitas")

    st.write("Nama : Abdul Hajatun")
    st.write("NIM : 22.12.2588")
    st.write("Kelas : SI 05")

# =========================
# METRIC
# =========================

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Model", "KNN")

with m2:
    st.metric("Jumlah Fitur", "8")

with m3:
    st.metric("Status", "Aktif")

st.divider()

# =========================
# FORM INPUT
# =========================

st.subheader("📝 Input Data Pasien")

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=200,
        value=100
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=150,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=1000,
        value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

# =========================
# TOMBOL PREDIKSI
# =========================

if st.button("🔍 Prediksi Diabetes"):

    features = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    prediction = model.predict(features)[0]

    st.divider()

    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:

        st.error(
            "⚠️ Pasien memiliki indikasi Diabetes."
        )

    else:

        st.success(
            "✅ Pasien tidak terindikasi Diabetes."
        )

    st.subheader("📄 Data Input")

    st.dataframe({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "Blood Pressure": [blood_pressure],
        "Skin Thickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DPF": [dpf],
        "Age": [age]
    })

# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    "Sistem Prediksi Diabetes berbasis Machine Learning menggunakan algoritma KNN."
)
