import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sistem Deteksi Dropout",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-card-risk {
        background-color: #FFEBEE;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF5252;
        color: #B71C1C;
    }
    .result-card-safe {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        color: #1B5E20;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        data = joblib.load('pso_dropout_model_final.pkl')
        return data
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None

# --- UI UTAMA ---
st.markdown('<div class="main-header">🎓 Sistem Peringatan Dini Dropout Siswa</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Implementasi Random Forest dengan Optimasi Particle Swarm Optimization (PSO)</div>', unsafe_allow_html=True)

model_data = load_model()

if model_data:
    model = model_data['model']
    features = model_data['features'] 
    
    # --- SIDEBAR: INPUT DATA (FULL CUSTOM) ---
    st.sidebar.header("📝 Input Profil Siswa")
    
    input_data = {}

    with st.sidebar.form("input_form"):
        st.write("Silakan isi data berikut:")

        # Kita cek setiap fitur yang ada di model, lalu buat inputan khususnya
        
        # 1. UMUR (Limit 15-22)
        if 'Age' in features:
            input_data['Age'] = st.number_input(
                "Umur Siswa", 
                min_value=15, 
                max_value=22, 
                value=17,
                help="Batasan umur: 15 sampai 22 tahun"
            )

        # 2. PENDIDIKAN IBU
        if 'Mother_Education' in features:
            st.write("---")
            edu_options = {0: "Tidak Sekolah", 1: "SD", 2: "SMP", 3: "SMA", 4: "Kuliah"}
            val = st.selectbox(
                "Pendidikan Terakhir Ibu",
                options=list(edu_options.keys()),
                format_func=lambda x: edu_options[x]
            )
            input_data['Mother_Education'] = val

        # 3. PENDIDIKAN AYAH
        if 'Father_Education' in features:
            edu_options = {0: "Tidak Sekolah", 1: "SD", 2: "SMP", 3: "SMA", 4: "Kuliah"}
            val = st.selectbox(
                "Pendidikan Terakhir Ayah",
                options=list(edu_options.keys()),
                format_func=lambda x: edu_options[x]
            )
            input_data['Father_Education'] = val

        # 4. TRAVEL TIME
        if 'Travel_Time' in features:
            st.write("---")
            # 1: <15 min, 2: 15-30 min, 3: 30-60 min, 4: >1 hour
            time_opts = {1: "< 15 Menit", 2: "15 - 30 Menit", 3: "30 - 60 Menit", 4: "> 1 Jam"}
            val = st.select_slider(
                "Berapa lama waktu yang ditempuh untuk sampai ke kelas?",
                options=[1, 2, 3, 4],
                format_func=lambda x: time_opts[x],
                value=2
            )
            input_data['Travel_Time'] = val

        # 5. STUDY TIME (Input Angka, Max 4)
        if 'Study_Time' in features:
            # User minta input angka. Jika > 4 maka dianggap 4.
            # Kita set max_value=4 di UI agar user tidak bisa input lebih dari itu.
            val = st.number_input(
                "Berapa jam waktu yang dihabiskan untuk belajar? (Skala 1-4)",
                min_value=1,
                max_value=4,
                value=2,
                help="jika lebih dari 4 jam, masukkan 4."
            )
            input_data['Study_Time'] = val

        # 6. FAILURES (Input Angka, Max 4)
        if 'Number_of_Failures' in features:
            val = st.number_input(
                "Berapa kali gagal kelas?",
                min_value=0,
                max_value=4,
                value=0,
                help="Jika lebih dari 4 kali, masukkan 4."
            )
            input_data['Number_of_Failures'] = val

        # 7. ADDRESS (Urban/Rural)
        if 'Address_U' in features:
            st.write("---")
            # Address_U = 1 (Urban/Kota), 0 (Rural/Desa)
            opt = st.radio(
                "Apakah siswa tinggal di perkotaan?",
                options=[1, 0],
                format_func=lambda x: "Ya (Kota)" if x == 1 else "Tidak (Desa)",
                horizontal=True
            )
            input_data['Address_U'] = opt

        # 8. GROUP SKALA 1-5 (Health, Family, dll)
        # Kita cek satu-satu fitur skala yang mungkin muncul
        scale_features_map = {
            'Family_Relationship': 'Kualitas Hubungan Keluarga',
            'Free_Time': 'Waktu Luang sepulang sekolah',
            'Going_Out': 'Frekuensi Keluar Main/Nongkrong',
            'Health_Status': 'Status Kesehatan',
            'Weekday_Alcohol_Consumption': 'Konsumsi Alkohol (Hari Kerja)',
            'Weekend_Alcohol_Consumption': 'Konsumsi Alkohol (Akhir Pekan)'
        }
        
        # Tampilkan divider jika ada fitur skala
        if any(f in features for f in scale_features_map.keys()):
             st.write("---")
             st.write("**Indikator Sosial & Kesehatan (Skala 1-5)**")

        for f_name, f_label in scale_features_map.items():
            if f_name in features:
                input_data[f_name] = st.slider(
                    f_label,
                    min_value=1, max_value=5, value=3,
                    help="1 = Sangat Rendah/Buruk, 5 = Sangat Tinggi/Baik"
                )

        # 9. SISANYA (Boolean / Lainnya)
        # Fitur biner seperti Wants Higher Education, Ekskul, dll
        binary_map = {
            'Wants_Higher_Education_yes': 'Ingin Lanjut Kuliah?',
            'Extra_Curricular_Activities_yes': 'Mengikuti Ekstrakurikuler?',
            'Internet_Access_yes': 'Memiliki Akses Internet?',
            'In_Relationship_yes': 'Memiliki Pacar?'
        }
        
        # Cek apakah ada fitur biner tersisa
        remaining_features = [f for f in features if f not in input_data]
        
        if remaining_features:
            st.write("---")
            for col in remaining_features:
                # Gunakan label bahasa Indonesia jika ada di mapping, jika tidak pakai nama asli
                label = binary_map.get(col, col.replace('_', ' '))
                
                input_data[col] = st.radio(
                    label,
                    options=[0, 1],
                    format_func=lambda x: "Ya" if x == 1 else "Tidak",
                    horizontal=True
                )

        st.markdown("---")
        submit = st.form_submit_button("🔍 Analisis Risiko")

    # --- BAGIAN KANAN: HASIL ---
    col1, col2 = st.columns([2, 1])

    with col1:
        if submit:
            # 1. Konversi input ke DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Kita paksa urutan kolom input_df agar sama persis dengan 'features' model
            input_df = input_df[features]
            # ----------------------------------------
            
            # 2. Prediksi (Sekarang pasti aman)
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card-risk">
                    <h2>⚠️ PERINGATAN: BERISIKO DROPOUT</h2>
                    <p>Berdasarkan profil yang diinput, siswa ini memiliki kemiripan pola dengan siswa yang putus sekolah.</p>
                    <h1>Probabilitas Risiko: {probability*100:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 💡 Saran Tindakan:")
                st.warning("Segera lakukan pemanggilan konseling untuk mendalami masalah siswa.")
            else:
                st.markdown(f"""
                <div class="result-card-safe">
                    <h2>✅ STATUS: AMAN</h2>
                    <p>Profil siswa menunjukkan indikasi positif untuk melanjutkan studi.</p>
                    <h1>Probabilitas Risiko: {probability*100:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.info("👈 Silakan lengkapi data profil siswa di panel sebelah kiri untuk memulai analisis.")

    with col2:
        st.markdown("### ℹ️ Informasi")
        st.write("Sistem ini mendeteksi risiko dini berdasarkan:")
        st.markdown("""
        - **Akademik:** Kegagalan kelas, waktu belajar.
        - **Sosial:** Hubungan keluarga, pergaulan.
        - **Kesehatan:** Konsumsi alkohol, kondisi fisik.
        """)

else:
    st.warning("Menunggu file model (.pkl)...")