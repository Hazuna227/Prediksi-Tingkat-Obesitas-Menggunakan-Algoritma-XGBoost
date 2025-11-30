# streamlit run ObesityDataSet_raw_and_data_sinthetic_Andika_Putra_Apriyatna_Figo_Firnanda.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Dashboard Prediksi Obesitas", layout="wide")

# Sidebar Navigasi dengan ikon dan pemisah

# Sidebar: header, radio, tips
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
        "🖼️ Visualisasi Korelasi & Feature Importance",
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
    st.markdown("<h3 style='color:#1e3c72;'>📁 Dataset Awal</h3>", unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True, height=220)


def tampil_fitur_target():
    st.markdown("<h3 style='color:#1e3c72;'>🔎 Fitur & Target</h3>", unsafe_allow_html=True)
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
    st.markdown("<h3 style='color:#1e3c72;'>⚙️ Data Setelah Normalisasi & Encoding</h3>", unsafe_allow_html=True)
    st.dataframe(df_norm.head(), use_container_width=True, height=220)
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

def tampil_heatmap():
    st.markdown("<h4 style='color:#1e3c72;margin-top:18px;'>Heatmap Korelasi Semua Fitur</h4>", unsafe_allow_html=True)
    st.image('heatmap_korelasi.png', caption='Heatmap Korelasi Semua Fitur', use_container_width=True)

def tampil_feature_importance():
    st.markdown("<h4 style='color:#1e3c72;margin-top:18px;'>Feature Importance (XGBoost)</h4>", unsafe_allow_html=True)
    st.image('feature_importance_xgb.png', caption='Feature Importance (XGBoost)', use_container_width=True)

def tampil_classification_report():
    st.markdown("<h3 style='color:#1e3c72;'>📄 Classification Report</h3>", unsafe_allow_html=True)
    st.dataframe(report_df, use_container_width=True, height=390)

def tampil_confusion_matrix():
    st.markdown("<h3 style='color:#1e3c72;'>🧩 Confusion Matrix</h3>", unsafe_allow_html=True)
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
    st.markdown("<h3 style='color:#1e3c72;'>🎯 Prediksi Obesitas</h3>", unsafe_allow_html=True)

    col_form, col_hasil = st.columns([1,1])

    with col_form:
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

    with col_hasil:
        # Untuk menjaga tinggi sejajar dengan form input, gunakan container dengan min-height
        container_height = 410  # px, sesuaikan dengan tinggi form input
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
            warna_kat = {
                'Normal_Weight': '#d4edda',
                'Insufficient_Weight': '#fff3cd',
                'Overweight_Level_I': '#ffeeba',
                'Overweight_Level_II': '#ffeeba',
                'Obesity_Type_I': '#f8d7da',
                'Obesity_Type_II': '#f5c6cb',
                'Obesity_Type_III': '#f1b0b7',
            }
            warna = warna_kat.get(hasil, '#e2e3e5')
            warna_teks = '#155724' if hasil == 'Normal_Weight' else '#721c24' if 'Obesity' in hasil else '#856404'
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
            # Hitung probabilitas prediksi
            proba = model.predict_proba(form)[0]
            kelas = label_encoder.classes_
            idx_pred = list(kelas).index(hasil)
            prob_pred = proba[idx_pred]
            # Format persentase
            prob_str = f"{prob_pred*100:.2f}%"
            # Warna background untuk probabilitas (bukan card)
            warna_prob = '#e3f2fd' if prob_pred >= 0.7 else '#fff3cd' if prob_pred >= 0.4 else '#f8d7da'
            warna_prob_teks = '#1565c0' if prob_pred >= 0.7 else '#856404' if prob_pred >= 0.4 else '#721c24'
            st.markdown(f"""
                <div style='display:flex;flex-direction:column;gap:14px;min-height:{container_height}px;justify-content:flex-start;'>
                    <div style='background:{warna};padding:16px 10px 12px 10px;color:{warna_teks};text-align:center;'>
                        <div style='font-size:1.1rem;font-weight:bold;letter-spacing:1px;margin-bottom:6px;'>HASIL PREDIKSI KATEGORI</div>
                        <div style='font-size:1.5rem;font-weight:bold;margin-top:2px;margin-bottom:2px;'>{hasil.replace('_', ' ')}</div>
                    </div>
                    <div style='background:#f8f9fa;padding:14px 10px 10px 10px;'>
                        <b>Interpretasi:</b> {saran_pred}
                    </div>
                    <div style='background:{warna_prob};padding:14px 10px 10px 10px;color:{warna_prob_teks};border-radius:7px;'>
                        <b>Tingkat Keyakinan Model:</b> <span style='font-size:1.15rem;font-weight:bold'>{prob_str}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='display:flex;flex-direction:column;gap:14px;min-height:{container_height}px;justify-content:flex-start;'>
                    <div style='background:#f8f9fa;padding:16px 10px 12px 10px;color:#888;text-align:center;'>
                        <div style='font-size:1.1rem;font-weight:bold;letter-spacing:1px;margin-bottom:6px;'>HASIL PREDIKSI KATEGORI</div>
                        <div style='font-size:1.2rem;font-weight:500;margin-top:2px;margin-bottom:2px;'>Belum ada prediksi.<br>Silakan isi form dan klik prediksi.</div>
                    </div>
                    <div style='background:#f8f9fa;padding:14px 10px 10px 10px;color:#888;'>
                        <b>Interpretasi:</b> Hasil interpretasi akan muncul setelah Anda melakukan prediksi.
                    </div>
                    <div style='background:#f8f9fa;padding:14px 10px 10px 10px;color:#888;border-radius:7px;'>
                        <b>Tingkat Keyakinan Model:</b> -
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Tampilkan sesuai pilihan sidebar
if sidebar_option == "Tampilkan Semua":
    tampil_form_prediksi()
    tampil_dataset()
    tampil_fitur_target()
    tampil_normalisasi()
    col1, col2 = st.columns(2)
    with col1:
        tampil_heatmap()
    with col2:
        tampil_feature_importance()
    colcr, colcm = st.columns(2)
    with colcr:
        tampil_classification_report()
    with colcm:
        tampil_confusion_matrix()
elif sidebar_option == "🖼️ Visualisasi Korelasi & Feature Importance":
    st.markdown("<h3 style='color:#1e3c72;'>🖼️ Visualisasi Korelasi & Feature Importance</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        tampil_heatmap()
    with col2:
        tampil_feature_importance()
elif sidebar_option == "📁 Dataset Awal":
    tampil_dataset()
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
