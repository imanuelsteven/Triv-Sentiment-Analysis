import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from typing import List, Tuple
import os

# =====================================================================================
# INISIALISASI NLTK YANG SANGAT ROBUST (STRATEGI BARU)
# =====================================================================================

def download_nltk_resources():
    """
    Fungsi untuk memeriksa dan mengunduh resource NLTK yang diperlukan.
    Ini adalah fallback jika nltk.txt gagal atau tidak terdeteksi tepat waktu.
    """
    try:
        # Cek stopwords
        nltk.data.find('corpora/stopwords')
        print("Resource 'stopwords' sudah ada.")
    except LookupError:
        print("Resource 'stopwords' tidak ditemukan, mengunduh sekarang...")
        nltk.download('stopwords')

    try:
        # Cek punkt
        nltk.data.find('tokenizers/punkt')
        print("Resource 'punkt' sudah ada.")
    except LookupError:
        print("Resource 'punkt' tidak ditemukan, mengunduh sekarang...")
        nltk.download('punkt')

# Panggil fungsi download di awal eksekusi skrip
download_nltk_resources()

# =====================================================================================
# KONFIGURASI DAN INISIALISASI AWAL (setelah NLTK dipastikan ada)
# =====================================================================================

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Analisis Sentimen Aplikasi TRIV",
    page_icon="🎯",
    layout="wide"
)

# --- Caching untuk efisiensi ---

@st.cache_resource
def get_stemmer():
    """Membuat dan meng-cache Sastrawi Stemmer."""
    factory = StemmerFactory()
    return factory.create_stemmer()

@st.cache_data
def get_stopwords_list() -> set:
    """Menggabungkan dan meng-cache daftar stopwords Indonesia & Inggris."""
    # Baris ini sekarang seharusnya aman untuk dijalankan
    stop_words_id = set(stopwords.words('indonesian'))
    stop_words_en = set(stopwords.words('english'))
    return stop_words_id.union(stop_words_en)

# ... (sisa kode Anda dari @st.cache_resource def load_models() dst. tidak perlu diubah) ...

@st.cache_resource
def load_models() -> Tuple[object, object]:
    """Memuat TF-IDF vectorizer dan model SVM dari file."""
    try:
        tfidf_vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
        model = joblib.load('model/model_svm_sentimen.pkl')
        return tfidf_vectorizer, model
    except FileNotFoundError:
        st.error("File model tidak ditemukan. Pastikan 'tfidf_vectorizer.pkl' dan 'model_svm_sentimen.pkl' ada di dalam folder 'model/'.")
        return None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None, None

# --- Konstanta ---

STEMMER = get_stemmer()
STOP_WORDS = get_stopwords_list()

# Kamus slang words yang diperluas dan dirapikan
SLANG_WORDS = {
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "nggak": "tidak", "tdk": "tidak", "gausah": "tidak usah",
    "bgt": "banget", "bngt": "banget", "bagu": "bagus",
    "dgn": "dengan", "trs": "terus", "krn": "karena", "kalo": "kalau", "kl": "kalau",
    "udh": "sudah", "sm": "sama", "smpe": "sampai", "sampe": "sampai",
    "jg": "juga", "tp": "tapi", "bkn": "bukan", "hrs": "harus", "dlm": "dalam",
    "cmn": "cuma", "cuma": "hanya", "aja": "saja",
    "dpt": "dapat", "dapet": "dapat", "bs": "bisa", "bsa": "bisa",
    "sy": "saya", "gua": "saya", "gw": "saya", "w": "saya", "k": "aku",
    "lu": "kamu", "loe": "kamu", "lo": "kamu",
    "skt": "sakit", "ny": "nya", "bru": "baru", "hr": "hari",
    "kcw": "kecewa", "fix": "perbaiki", "bug": "eror", "moga": "semoga",
    "anjing": "sial", "asu": "buruk", "parah": "buruk",
    "pake": "pakai", "make": "menggunakan", "bikin": "membuat",
    "btw": "ngomong-ngomong", "mksh": "terima kasih", "makasih": "terima kasih", "trims": "terima kasih",
    "rekomen": "rekomendasi", "mantap": "bagus", "ntap": "mantap",
    "cepet": "cepat", "bener": "benar", "gitu": "begitu", "bgtu": "begitu", "gmn": "bagaimana",
    "ngerti": "mengerti", "ribet": "rumit", "topup": "isi saldo", "ui": "tampilan",
    "iseng": "coba-coba", "sigap": "responsif", "dapetin": "mendapatkan",
    "ajarin": "mengajarkan", "cobain": "mencoba", "ngasih": "memberi", "ngambil": "mengambil",
    "nunggu": "menunggu", "ngulang": "mengulang",
    "ok": "oke", "okey": "oke", "oke": "baik", "sumpah": "sungguh",
    "yg": "yang", "tggjwabnya": "tanggung jawab",
    # Kata yang tidak memiliki arti signifikan bisa dihilangkan
    "sih": "", "deh": "", "dong": "", "kok": "", "nih": "", "tuh": "", "lho": ""
}

SENTIMENT_MAPPING = {0: "Negatif", 1: "Positif"}
EMOJI_MAPPING = {"Positif": "😊", "Negatif": "😞"}


# =====================================================================================
# FUNGSI-FUNGSI PREPROCESSING TEKS
# =====================================================================================

def clean_text(text: str) -> str:
    """Membersihkan teks dari URL, mention, hashtag, angka, dan tanda baca."""
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9]+|#[A-Za-z0-9]+|RT[\s]+|https?:\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.strip()
    return re.sub(r'\s+', ' ', text)

def fix_slangwords(text: str) -> str:
    """Mengganti kata-kata slang dengan kata baku berdasarkan kamus."""
    words = text.split()
    return ' '.join([SLANG_WORDS.get(word, word) for word in words])

def tokenize_text(text: str) -> List[str]:
    """Memecah teks menjadi token (kata)."""
    return word_tokenize(text)

def remove_stopwords(tokens: List[str]) -> List[str]:
    """Menghapus stopwords dari daftar token."""
    return [word for word in tokens if word not in STOP_WORDS and len(word) > 1]

def stem_text(tokens: List[str]) -> List[str]:
    """Melakukan stemming pada setiap token."""
    return [STEMMER.stem(word) for word in tokens]

def preprocess_text(text: str) -> str:
    """Menjalankan pipeline preprocessing lengkap."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    cleaned = clean_text(text)
    slang_fixed = fix_slangwords(cleaned)
    tokens = tokenize_text(slang_fixed)
    filtered_tokens = remove_stopwords(tokens)
    stemmed_tokens = stem_text(filtered_tokens)
    
    return ' '.join(stemmed_tokens)


# =====================================================================================
# FUNGSI PREDIKSI
# =====================================================================================

def predict_sentiment(text: str, vectorizer: object, model: object) -> Tuple[str, float]:
    """Memproses teks dan memprediksi sentimennya."""
    if not text.strip():
        return "Netral", 0.0

    try:
        processed_text = preprocess_text(text)
        if not processed_text:
            return "Netral", 0.0
            
        text_vector = vectorizer.transform([processed_text])
        prediction = model.predict(text_vector)[0]
        
        # Dapatkan probabilitas jika model mendukung
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(text_vector)[0]
            confidence = max(probability)
        else:
            confidence = 0.5 # Default confidence jika tidak ada proba

        sentiment = SENTIMENT_MAPPING.get(prediction, "Tidak Diketahui")
        return sentiment, confidence
    
    except Exception as e:
        st.warning(f"Terjadi kesalahan saat prediksi: {e}")
        return "Error", 0.0


# =====================================================================================
# UI APLIKASI STREAMLIT
# =====================================================================================

def main():
    st.title("🎯 Analisis Sentimen Ulasan Aplikasi TRIV")
    st.markdown("---")

    # --- Sidebar ---
    with st.sidebar:
        st.header("ℹ️ Informasi Aplikasi")
        st.markdown("""
        Aplikasi ini mendeteksi sentimen (Positif/Negatif) dari ulasan untuk aplikasi TRIV.
        
        **Teknologi yang Digunakan:**
        - **TF-IDF**: Untuk mengubah teks menjadi fitur numerik.
        - **SVM**: Model machine learning untuk klasifikasi.
        - **Sastrawi**: Untuk stemming Bahasa Indonesia.
        - **NLTK**: Untuk tokenisasi dan stopwords.
        
        **Cara Penggunaan:**
        1. Pilih tab yang sesuai (Analisis Tunggal atau Batch).
        2. Masukkan teks atau unggah file CSV.
        3. Klik tombol analisis dan lihat hasilnya.
        """)

    # --- Memuat Model ---
    tfidf_vectorizer, model = load_models()

    if not tfidf_vectorizer or not model:
        st.error("Aplikasi tidak dapat berjalan karena model gagal dimuat.")
        return

    st.success("✅ Model berhasil dimuat! Aplikasi siap digunakan.")
    
    # --- Tabs untuk Fitur ---
    tab1, tab2, tab3 = st.tabs(["📝 Analisis Teks Tunggal", "📊 Analisis File (Batch)", "📈 Info Model"])

    # --- Tab 1: Analisis Teks Tunggal ---
    with tab1:
        st.header("Analisis Satu Ulasan")
        user_input = st.text_area(
            "Masukkan teks ulasan di sini:",
            height=150,
            placeholder="Contoh: Aplikasi ini bagus banget, gampang dipake dan fiturnya lengkap!"
        )

        if st.button("🔍 Analisis Sentimen", type="primary"):
            if user_input.strip():
                with st.spinner("Menganalisis..."):
                    sentiment, confidence = predict_sentiment(user_input, tfidf_vectorizer, model)
                    emoji = EMOJI_MAPPING.get(sentiment, "🤔")

                st.subheader("📊 Hasil Analisis")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Sentimen Prediksi", value=f"{emoji} {sentiment}")
                with col2:
                    st.metric(label="Tingkat Keyakinan", value=f"{confidence:.2%}")

                with st.expander("👀 Lihat Detail Proses Preprocessing"):
                    st.write(f"**Teks Asli:** {user_input}")
                    cleaned = clean_text(user_input)
                    st.write(f"**1. Case Folding & Cleaning:** `{cleaned}`")
                    slang_fixed = fix_slangwords(cleaned)
                    st.write(f"**2. Koreksi Slang:** `{slang_fixed}`")
                    tokens = tokenize_text(slang_fixed)
                    st.write(f"**3. Tokenisasi:** `{tokens}`")
                    filtered = remove_stopwords(tokens)
                    st.write(f"**4. Hapus Stopwords:** `{filtered}`")
                    stemmed = stem_text(filtered)
                    st.write(f"**5. Stemming:** `{stemmed}`")
                    final_text = ' '.join(stemmed)
                    st.write(f"**Teks Akhir (Input ke Model):** `{final_text}`")
            else:
                st.warning("⚠️ Mohon masukkan teks untuk dianalisis.")

    # --- Tab 2: Analisis Batch ---
    with tab2:
        st.header("Analisis Beberapa Ulasan dari File CSV")
        uploaded_file = st.file_uploader(
            "Unggah file CSV Anda. Pastikan ada kolom bernama 'text'.",
            type=['csv']
        )

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns:
                    st.error("❌ File CSV harus memiliki kolom bernama 'text'.")
                else:
                    st.write("📋 Preview Data:", df.head())
                    if st.button("🚀 Analisis Semua Data dari File"):
                        with st.spinner("Sedang memproses semua data... Ini mungkin memakan waktu."):
                            # Menggunakan list comprehension untuk efisiensi
                            sentiments = [predict_sentiment(str(text), tfidf_vectorizer, model) for text in df['text']]
                            
                            # Memisahkan hasil menjadi dua kolom
                            df['sentiment'], df['confidence'] = zip(*sentiments)
                            
                        st.success("✅ Analisis Selesai!")
                        st.dataframe(df)
                        
                        csv_result = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Unduh Hasil Analisis (CSV)",
                            data=csv_result,
                            file_name="hasil_analisis_sentimen.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat membaca file: {e}")

    # --- Tab 3: Informasi Model ---
    with tab3:
        st.header("Informasi Teknis Model")
        try:
            vocab_size = len(tfidf_vectorizer.vocabulary_)
            max_features = tfidf_vectorizer.max_features
            
            st.write(f"- **Tipe Vectorizer**: `{type(tfidf_vectorizer).__name__}`")
            st.write(f"- **Ukuran Vocabulary**: `{vocab_size:,}` kata unik")
            st.write(f"- **Max Features**: `{max_features if max_features else 'Tidak dibatasi'}`")
            st.write(f"- **Tipe Model Klasifikasi**: `{type(model).__name__}`")
        except Exception as e:
            st.warning("Tidak dapat menampilkan detail model.")


if __name__ == '__main__':
    main()