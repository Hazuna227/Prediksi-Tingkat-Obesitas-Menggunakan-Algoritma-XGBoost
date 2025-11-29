# streamlit run app.py


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Dashboard Prediksi Obesitas", layout="wide")

# Sidebar Navigasi dengan ikon dan pemisah
st.sidebar.markdown("""
<div style='font-size:1.3rem;font-weight:bold;margin-bottom:10px;'>
    🧭 <span style='color:#1e3c72;'>Navigasi Dashboard</span>
</div>
<hr style='margin:0 0 10px 0;border:1px solid #e0e0e0;'>
""", unsafe_allow_html=True)
sidebar_option = st.sidebar.radio(
    "",
    (
        "Tampilkan Semua",
        "📁 Dataset Awal",
        "🧹 Data Setelah Cleansing",
        "🔎 Fitur & Target",
        "⚙️ Data Setelah Normalisasi",
        "📄 Classification Report",
        "🧩 Confusion Matrix",
        "🎯 Form Prediksi"
    ),
    index=0
)
st.sidebar.markdown("""
<hr style='margin:10px 0 0 0;border:1px solid #e0e0e0;'>
<div style='font-size:0.95rem;color:#888;margin-top:8px;'>
    <b>Tips:</b> Pilih menu untuk menampilkan bagian tertentu.<br>
    Pilih <b>Tampilkan Semua</b> untuk melihat seluruh proses sekaligus.
</div>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="background:linear-gradient(90deg,#1e3c72,#2a5298);padding:24px 10px 16px 10px;border-radius:12px;margin-bottom:10px;">
    <h1 style="color:white;margin-bottom:0;font-size:2.2rem;">📊 Dashboard Prediksi Obesitas <span style="font-size:1.2rem;">(XGBoost)</span></h1>
    <div style="color:#e0e0e0;font-size:1.05rem;margin-top:10px;">
        <b>Nama Kelompok:</b>
        <ul style='margin:6px 0 0 18px;padding:0;'>
            <li>Andika Putra Apriyatna</li>
            <li>Figo Firnanda</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# Load data yang diperlukan
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
df_clean = pd.read_csv("hasil_cleansing.csv")
df_norm = pd.read_csv("obesity_normalized.csv")
report_df = pd.read_csv("classification_report.csv", index_col=0)
cm = pd.read_csv("confusion_matrix.csv", index_col=0)
model = joblib.load("xgb_obesity_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")


# Fungsi untuk menampilkan setiap bagian
def tampil_dataset():
    st.markdown("<h3 style='color:#1e3c72;'>📁 1. Dataset Awal</h3>", unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True, height=220)

def tampil_cleansing():
    st.markdown("<h3 style='color:#1e3c72;'>🧹 2. Data Setelah Cleansing</h3>", unsafe_allow_html=True)
    st.dataframe(df_clean.head(), use_container_width=True, height=215, hide_index=True)
    st.metric("Jumlah Data Setelah Hapus Duplikat", len(df_clean))

def tampil_fitur_target():
    st.markdown("<h3 style='color:#1e3c72;'>🔎 3. Fitur & Target</h3>", unsafe_allow_html=True)
    selected_features = ["Weight", "Height", "FCVC", "FAVC", "family_history_with_overweight", "NCP", "Age", "MTRANS"]
    target = "NObeyesdad"
    fitur_html = "<ul style='margin-bottom:0;padding-left:18px;'>" + ''.join([f"<li>{f}</li>" for f in selected_features]) + "</ul>"
    kelas_html = "<ul style='margin-bottom:0;padding-left:18px;'>" + ''.join([f"<li>{k}</li>" for k in df_clean[target].unique()]) + "</ul>"

    # Tambahkan penjelasan arti fitur
    arti_fitur = {
        "Weight": "Berat badan (kg)",
        "Height": "Tinggi badan (m)",
        "FCVC": "Frekuensi konsumsi sayur (0-3)",
        "FAVC": "Sering mengonsumsi makanan tinggi kalori (Ya/Tidak)",
        "family_history_with_overweight": "Riwayat keluarga dengan kelebihan berat badan",
        "NCP": "Jumlah makan utama per hari",
        "Age": "Umur responden (tahun)",
        "MTRANS": "Transportasi utama (kategori)"
    }
    arti_html = "<ul style='margin-bottom:0;padding-left:18px;'>" + ''.join([f"<li><b>{k}</b>: {v}</li>" for k,v in arti_fitur.items()]) + "</ul>"

    st.markdown(f"""
    <div style='background:#f5f7fa;padding:14px 10px 8px 10px;border-radius:10px;min-height:100px;'>
        <div style='display:flex;gap:18px;'>
            <div style='min-width:120px;'>
                <b>Fitur yang Dipakai:</b>
                {fitur_html}
            </div>
            <div style='min-width:120px;'>
                <b>Kelas Target:</b>
                {kelas_html}
            </div>
        </div>
        <div style='margin-top:12px;'>
            <b>Arti Masing-masing Fitur:</b>
            {arti_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def tampil_normalisasi():
    st.markdown("<h3 style='color:#1e3c72;'>⚙️ 4. Data Setelah Normalisasi & Encoding</h3>", unsafe_allow_html=True)
    st.dataframe(df_norm.head(), use_container_width=True, height=220)

def tampil_classification_report():
    st.markdown("<h3 style='color:#1e3c72;'>📄 5. Classification Report</h3>", unsafe_allow_html=True)
    st.dataframe(report_df, use_container_width=True, height=390)

def tampil_confusion_matrix():
    st.markdown("<h3 style='color:#1e3c72;'>🧩 6. Confusion Matrix</h3>", unsafe_allow_html=True)
    # Ukuran default untuk tampilan berdampingan (all), lebih kecil jika hanya confusion matrix
    fig_size = (5,4)
    title_size = 13
    label_size = 11
    if sidebar_option == "🧩 Confusion Matrix":
        fig_size = (3.2,2.2)
        title_size = 10
        label_size = 9
    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax, cbar=False)
    plt.xlabel('Predicted', fontsize=label_size)
    plt.ylabel('Actual', fontsize=label_size)
    plt.title('Confusion Matrix', fontsize=title_size, fontweight='bold')
    st.pyplot(fig)

def tampil_form_prediksi():
    st.markdown("<h3 style='color:#1e3c72;'>🎯 7. Prediksi Obesitas</h3>", unsafe_allow_html=True)
    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        with col1:
            berat = st.number_input("Berat Badan (kg)", 40, 200, 70)
            tinggi = st.number_input("Tinggi Badan (m)", 1.3, 2.2, 1.65, step=0.01)
            fcvc = st.number_input("Frekuensi Konsumsi Sayur (0-3)", 0.0, 3.0, 2.0)
            umur = st.number_input("Umur (tahun)", 5, 100, 25)
        with col2:
            favc = st.selectbox("Sering Makan Tinggi Kalori", df["FAVC"].unique())
            riwayat = st.selectbox("Riwayat Keluarga Obesitas", df["family_history_with_overweight"].unique())
            makan = st.number_input("Jumlah Makan per Hari (1–5)", 1, 5, 3)
            mtrans = st.selectbox("Transportasi Utama", df["MTRANS"].unique())
        submitted = st.form_submit_button("🔍 Prediksi Sekarang")
    if submitted:
        form = pd.DataFrame([{
            "Weight": berat,
            "Height": tinggi,
            "FCVC": fcvc,
            "FAVC": favc,
            "family_history_with_overweight": riwayat,
            "NCP": makan,
            "Age": umur,
            "MTRANS": mtrans
        }])
        pred = model.predict(form)[0]
        hasil = label_encoder.inverse_transform([pred])[0]
        st.markdown(
            f"<div style='background-color:#d4edda;padding:16px 10px;border-radius:8px;border:1px solid #c3e6cb;color:#155724;margin-top:10px;'>"
            f"<span style='font-size:1.2rem;'>Hasil Prediksi: <b>{hasil}</b></span> ✅"
            "</div>",
            unsafe_allow_html=True
        )

        # Interpretasi hasil prediksi dengan saran ramah
        saran = {
            'Overweight_Level_II': 'Berat badan Anda berada pada tingkat overweight level II. Disarankan untuk mulai memperbaiki pola makan dan meningkatkan aktivitas fisik secara teratur.',
            'Obesity_Type_I': 'Anda termasuk dalam kategori obesitas tipe I. Segera konsultasikan ke dokter atau ahli gizi untuk mendapatkan penanganan yang tepat.',
            'Overweight_Level_I': 'Berat badan Anda termasuk overweight level I. Jaga pola makan dan lakukan olahraga rutin agar berat badan tetap terkontrol.',
            'Normal_Weight': 'Berat badan Anda normal. Pertahankan pola hidup sehat dan aktivitas fisik secara rutin.',
            'Obesity_Type_II': 'Anda termasuk dalam kategori obesitas tipe II. Sangat disarankan untuk segera berkonsultasi dengan tenaga medis dan menerapkan pola hidup sehat.',
            'Obesity_Type_III': 'Anda termasuk dalam kategori obesitas tipe III. Segera lakukan konsultasi ke dokter untuk penanganan medis lebih lanjut.',
            'Insufficient_Weight': 'Berat badan Anda kurang. Perbanyak asupan nutrisi dan konsultasikan ke dokter atau ahli gizi jika perlu.'
        }
        saran_pred = saran.get(hasil, 'Jaga selalu kesehatan Anda.')
        st.markdown(
            f"<div style='background:#f8f9fa;border-left:5px solid #1e3c72;padding:12px 10px 8px 10px;margin-top:8px;border-radius:6px;'>"
            f"<b>Interpretasi:</b> Berdasarkan data yang Anda masukkan, model memprediksi kategori obesitas sebagai <b>{hasil.replace('_', ' ')}</b>.<br>"
            f"<i>{saran_pred}</i>"
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)  # Jarak antar interpretasi dan tabel

        # Tampilkan probabilitas tiap kategori, urut dari terbesar
        probs = model.predict_proba(form)[0]
        prob_df = pd.DataFrame({
            "Kategori": label_encoder.inverse_transform(range(len(probs))),
            "Probabilitas": probs
        })
        prob_df = prob_df.sort_values(by="Probabilitas", ascending=False)
        prob_df["Probabilitas"] = prob_df["Probabilitas"].apply(lambda x: f"{x:.4f}")
        st.table(prob_df.reset_index(drop=True))

# Tampilkan sesuai pilihan sidebar
if sidebar_option == "Tampilkan Semua":
    tampil_dataset()
    tampil_cleansing()
    tampil_fitur_target()
    tampil_normalisasi()
    # Tampilkan classification report & confusion matrix berdampingan
    colcr, colcm = st.columns(2)
    with colcr:
        tampil_classification_report()
    with colcm:
        tampil_confusion_matrix()
    tampil_form_prediksi()
elif sidebar_option == "📁 Dataset Awal":
    tampil_dataset()
elif sidebar_option == "🧹 Data Setelah Cleansing":
    tampil_cleansing()
elif sidebar_option == "🔎 Fitur & Target":
    tampil_fitur_target()
elif sidebar_option == "⚙️ Data Setelah Normalisasi":
    tampil_normalisasi()
elif sidebar_option == "📄 Classification Report":
    tampil_classification_report()
elif sidebar_option == "🧩 Confusion Matrix":
    tampil_confusion_matrix()
elif sidebar_option == "🎯 Form Prediksi":
    tampil_form_prediksi()

