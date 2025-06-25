import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize  # Tokenisasi teks
from nltk.corpus import stopwords  # Daftar kata-kata berhenti dalam teks
import nltk
nltk.download('punkt')  # Mengunduh dataset yang diperlukan untuk tokenisasi teks.
nltk.download('stopwords')  # Mengunduh dataset yang berisi daftar kata-kata berhenti (stopwords) dalam berbagai bahasa.

# Download NLTK data jika belum ada
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Konfigurasi halaman
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)

# Inisiasi Stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Inisiasi Stopwords
stop_words = set(stopwords.words('indonesian')).union(stopwords.words('english'))

# Dictionary slang words
slangwords = {
    "gak": "tidak", "ga": "tidak", "gk" : "tidak", 'bagu' : 'bagus',"tdk": "tidak", "bgt": "banget", "dgn": "dengan",
    "trs": "terus", "krn": "karena", "udh": "sudah", "sm": "sama", "jg": "juga",
    "tp": "tapi", "bkn": "bukan", "hrs": "harus", "dlm": "dalam", "bgt": "banget",
    "cmn": "cuma", "dpt": "dapat", "sy": "saya", "lg": "lagi", "sblm": "sebelum",
    "skt": "sakit", "ny": "nya", "smpe": "sampai", "bru": "baru", "bs": "bisa",
    "bsa": "bisa", "dpt": "dapat", "tdk": "tidak", "kl": "kalau", "sgt": "sangat", "bug" : "eror",
    "gua" : "saya", "gw" : "saya", "w" : "saya", "anjing" : "sial", "lu" : "kamu", "loe" : "kamu",
    "kcw" : "kecewa", "hr" : "hari", "k" : "aku", "fix" : "perbaiki", "bug" : "eror", "moga" : "semoga", "triv": "triv", "bgt": "banget","ny": "nya",
    "udh": "sudah","pake": "pakai","pakai": "pakai","cuma": "hanya",
    "cmn": "cuma","aja": "saja","ga": "tidak","gak": "tidak","gk": "tidak","nggak": "tidak","gausah": "tidak usah","kalo": "kalau","krn": "karena","bgt": "banget",
    "bngt": "banget","sm": "sama","smpe": "sampai","sampe": "sampai","sy": "saya","gue": "saya",  "gw": "saya","gua": "saya","w": "saya", "lu": "kamu","loe": "kamu",
    "lo": "kamu","trs": "terus","dgn": "dengan", "dpt": "dapat","bsa": "bisa","bs": "bisa","lg": "lagi","tdk": "tidak", "kl": "kalau","jg": "juga","tp": "tapi","bkn": "bukan",
    "hrs": "harus","dlm": "dalam", "skt": "sakit", "bru": "baru", "btw": "ngomong-ngomong", "sih": "", "deh": "", "eh": "","dong": "","makasih": "terima kasih", 
    "mksh": "terima kasih","trims": "terima kasih","trm": "terima", "rekomen": "rekomendasi","fix": "perbaiki","bug": "eror", "anjing": "sial","mantap": "bagus",
    "cepet": "cepat", "bgtu": "begitu", "bener": "benar","gitu": "begitu","gmn": "bagaimana",  "tuh": "","lho": "","kok": "","nih": "","ntap": "mantap",
    "ngerti": "mengerti","ribet": "rumit","make": "menggunakan", "topup": "isi saldo", "e-wallet": "dompet digital","emoney": "uang elektronik",
    "akun": "akun", "iseng": "coba-coba", "sigap": "responsif", "dapetin": "mendapatkan","bikin": "membuat","ajarin": "mengajarkan",  "cobain": "mencoba",
    "ngasih": "memberi", "ngambil": "mengambil","dapet": "dapat","nunggu": "menunggu","ngulang": "mengulang", "kalo": "kalau",
    "moga": "semoga",  "btw": "ngomong-ngomong",  "parah": "buruk", "ok": "oke", "okey": "oke","oke": "baik", "sumpah": "sungguh", "ui" : "tampilan",
    "asu" : "buruk", 'yg' : 'yang', "tggjwabnya" : "tanggung jawab"
}

def clean_text(text):
    """Membersihkan teks dari karakter yang tidak perlu dan melakukan case folding."""
    if pd.isna(text):
        return ""
    
    text = text.lower()  # Case folding
    text = re.sub(r'@[A-Za-z0-9]+', ' ', text)  # Hapus mention
    text = re.sub(r'#[A-Za-z0-9]+', ' ', text)  # Hapus hashtag
    text = re.sub(r'RT[\s]', ' ', text)  # Hapus RT
    text = re.sub(r"http\S+", ' ', text)  # Hapus link
    text = re.sub(r'\d+', ' ', text)  # Hapus angka
    text = re.sub(r'[^\w\s]', ' ', text)  
    text = text.strip()  # Hapus spasi di awal dan akhir
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text)
    return text

def fix_slangwords(text):
    """Mengganti kata slang dengan kata baku."""
    words = text.split()
    fixed_words = []
 
    for word in words:
        if word.lower() in slangwords:
            fixed_words.append(slangwords[word.lower()])
        else:
            fixed_words.append(word)
 
    fixed_text = ' '.join(fixed_words)
    return fixed_text

def tokenize_text(text):
    """Memecah teks menjadi token (kata-kata)."""
    return word_tokenize(text)

def remove_stopwords(tokens):
    """Menghapus stopwords dalam bahasa Indonesia dan Inggris."""
    return [word for word in tokens if word not in stop_words]

def stem_text(tokens):
    """Melakukan stemming pada kata-kata dalam teks menggunakan Sastrawi."""
    return [stemmer.stem(word) for word in tokens]

def preprocess_text(text):
    """Melakukan preprocessing lengkap dari pembersihan hingga stemming."""
    text = clean_text(text)
    text = fix_slangwords(text)
    tokens = tokenize_text(text)
    filtered_tokens = remove_stopwords(tokens)
    stemmed_tokens = stem_text(filtered_tokens)
    return ' '.join(stemmed_tokens)  # Mengembalikan hasil sebagai teks kembali

# Fungsi untuk memuat model
@st.cache_resource
def load_models():
    """
    Memuat TF-IDF vectorizer dan model klasifikasi dari direktori model
    """
    try:
        # Muat TF-IDF vectorizer
        tfidf_vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
        
        # Muat model klasifikasi
        model = joblib.load('model/model_svm_sentimen.pkl')
        
        return tfidf_vectorizer, model
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        st.error("Pastikan file model ada di folder 'model/'")
        return None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Fungsi untuk prediksi sentimen
def predict_sentiment(text, tfidf_vectorizer, model):
    """
    Memprediksi sentimen dari teks input dengan preprocessing lengkap
    """
    try:
        # Preprocessing teks menggunakan fungsi yang sudah ada
        processed_text = preprocess_text(text)
        
        if not processed_text.strip():
            return "Neutral", 0.0
        
        # Transformasi teks menggunakan TF-IDF
        text_vector = tfidf_vectorizer.transform([processed_text])
        
        # Prediksi sentimen
        prediction = model.predict(text_vector)[0]
        
        # Dapatkan probabilitas prediksi (jika model mendukung)
        try:
            probability = model.predict_proba(text_vector)[0]
            confidence = max(probability)
        except:
            confidence = 0.0
        
        return prediction, confidence
    
    except Exception as e:
        st.error(f"Error dalam prediksi: {e}")
        return "Error", 0.0

# Fungsi untuk mapping label sentimen
def map_sentiment_label(prediction):
    """
    Mapping label sentimen ke format yang mudah dibaca
    """
    sentiment_mapping = {
        0: "Negatif",
        1: "Positif",
        "negative": "Negatif",
        "positive": "Positif"
    }
    
    return sentiment_mapping.get(prediction, str(prediction))

# Fungsi untuk mendapatkan emoji berdasarkan sentimen
def get_sentiment_emoji(sentiment):
    """
    Mengembalikan emoji berdasarkan sentimen
    """
    emoji_mapping = {
        "Positif": "😊",
        "Negatif": "😞",
    }
    return emoji_mapping.get(sentiment, "🤔")

# Header aplikasi
st.title("🎯 Aplikasi Deteksi Sentimen Aplikasi TRIV")
st.markdown("---")

# Sidebar untuk informasi
st.sidebar.header("ℹ️ Informasi Aplikasi")
st.sidebar.markdown("""
**Aplikasi ini menggunakan:**
- TF-IDF Vectorizer untuk ekstraksi fitur
- Model SVM untuk klasifikasi sentimen
- Preprocessing lengkap (cleaning, slang conversion, stemming, stopwords removal)
- Sastrawi stemmer untuk bahasa Indonesia

**Cara penggunaan:**
1. Masukkan teks yang ingin dianalisis
2. Klik tombol 'Analisis Sentimen'
3. Lihat hasil prediksi dari model!

**Preprocessing meliputi:**
- Case folding dan pembersihan karakter
- Konversi kata slang ke kata baku
- Tokenisasi dan stopwords removal
- Stemming menggunakan Sastrawi
""")

# Muat model
tfidf_vectorizer, model = load_models()

if tfidf_vectorizer is not None and model is not None:
    st.success("✅ Model berhasil dimuat!")
    
    # Tabs untuk berbagai fungsi
    tab1, tab2, tab3 = st.tabs(["📝 Analisis Teks Tunggal", "📊 Analisis Batch", "📈 Statistik"])
    
    with tab1:
        st.header("Analisis Sentimen Teks Tunggal")
        
        # Input teks
        user_input = st.text_area(
            "Masukkan teks yang ingin dianalisis:",
            height=150,
            placeholder="Contoh: Produk ini sangat bagus dan berkualitas tinggi!"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            analyze_button = st.button("🔍 Analisis Sentimen", type="primary")
        
        if analyze_button and user_input.strip():
            with st.spinner("Menganalisis sentimen..."):
                prediction, confidence = predict_sentiment(user_input, tfidf_vectorizer, model)
                sentiment_label = map_sentiment_label(prediction)
                emoji = get_sentiment_emoji(sentiment_label)
                
                # Tampilkan hasil
                st.markdown("### 📊 Hasil Analisis Sentimen")
                
                st.metric("", f"{emoji} {sentiment_label}")
                

                
                # Progress bar untuk confidence
                st.progress(confidence)
                
                # Tampilkan teks yang sudah diproses
                processed_text = preprocess_text(user_input)
                with st.expander("👀 Lihat Proses Preprocessing"):
                    st.write(f"**Teks asli:** {user_input}")
                    
                    # Tampilkan langkah-langkah preprocessing
                    cleaned = clean_text(user_input)
                    st.write(f"**Setelah cleaning:** {cleaned}")
                    
                    slang_fixed = fix_slangwords(cleaned)
                    st.write(f"**Setelah konversi slang:** {slang_fixed}")
                    
                    tokens = tokenize_text(slang_fixed)
                    st.write(f"**Setelah tokenisasi:** {tokens}")
                    
                    filtered_tokens = remove_stopwords(tokens)
                    st.write(f"**Setelah hapus stopwords:** {filtered_tokens}")
                    
                    stemmed_tokens = stem_text(filtered_tokens)
                    st.write(f"**Setelah stemming:** {stemmed_tokens}")
                    
                    st.write(f"**Hasil akhir:** {processed_text}")
        
        elif analyze_button and not user_input.strip():
            st.warning("⚠️ Mohon masukkan teks yang ingin dianalisis!")
    
    with tab2:
        st.header("Analisis Batch dari File")
        
        uploaded_file = st.file_uploader(
            "Upload file CSV dengan kolom 'text'",
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'text' in df.columns:
                    st.write("📋 Preview data:")
                    st.dataframe(df.head())
                    
                    if st.button("🚀 Analisis Semua Data"):
                        with st.spinner("Menganalisis data..."):
                            results = []
                            progress_bar = st.progress(0)
                            
                            for i, text in enumerate(df['text']):
                                prediction, confidence = predict_sentiment(str(text), tfidf_vectorizer, model)
                                sentiment_label = map_sentiment_label(prediction)
                                
                                results.append({
                                    'text': text,
                                    'processed_text': preprocess_text(str(text)),
                                    'sentiment': sentiment_label,
                                    'confidence': confidence
                                })
                                
                                progress_bar.progress((i + 1) / len(df))
                            
                            # Buat DataFrame hasil
                            results_df = pd.DataFrame(results)
                            
                            st.success("✅ Analisis selesai!")
                            st.dataframe(results_df)
                            
                            # Download hasil
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Hasil (CSV)",
                                data=csv,
                                file_name="hasil_analisis_sentimen.csv",
                                mime="text/csv"
                            )
                else:
                    st.error("❌ File CSV harus memiliki kolom 'text'")
                    
            except Exception as e:
                st.error(f"❌ Error membaca file: {e}")
    
    with tab3:
        st.header("Statistik dan Informasi Model")
        
        # Informasi model
        st.subheader("🔧 Informasi Model")
        
        try:
            # Coba dapatkan informasi tentang model
            model_info = {
                "Tipe Model yang Digunakan ": type(model).__name__,
                "Tipe Ekstraksi Fitur / Vectorizer": type(tfidf_vectorizer).__name__,
            }
            
            # Coba dapatkan vocabulary size
            try:
                vocab_size = len(tfidf_vectorizer.vocabulary_)
                model_info["Ukuran Vocabulary"] = f"{vocab_size:,}"
            except:
                pass
            
            # Coba dapatkan max features
            try:
                max_features = tfidf_vectorizer.max_features
                if max_features:
                    model_info["Max Features"] = f"{max_features:,}"
            except:
                pass
            
            for key, value in model_info.items():
                st.write(f"**{key}:** {value}")
                
        except Exception as e:
            st.write("Informasi model tidak dapat ditampilkan")
        
        # Contoh penggunaan
        st.subheader("📝 Contoh Penggunaan")
        
        examples = [
            "Aplikasi ini bagus banget, gampang dipake dan fiturnya lengkap!",
            "Pelayanan customer service nya parah bgt, gak responsif sama sekali.",
            "Barang udh sampe, packaging rapi dan sesuai deskripsi.",
            "Harganya mahal bgt untuk kualitas kayak gini, kecewa banget deh.",
            "Biasa aja sih, gak ada yang istimewa tapi ya lumayan lah."
        ]
        
        for i, example in enumerate(examples, 1):
            if st.button(f"Test Contoh {i}", key=f"example_{i}"):
                prediction, confidence = predict_sentiment(example, tfidf_vectorizer, model)
                sentiment_label = map_sentiment_label(prediction)
                emoji = get_sentiment_emoji(sentiment_label)
                
                st.write(f"**Teks:** {example}")
                st.write(f"**Hasil:** {emoji} {sentiment_label}")
                st.markdown("---")

else:
    st.error("❌ Gagal memuat model! Pastikan file model ada di direktori yang benar.")
    
    st.markdown("""
    ### 📋 Checklist File:
    - [ ] `model/tfidf_vectorizer.pkl` - File TF-IDF vectorizer
    - [ ] `model/model_svm_sentimen.pkl` - File model SVM
    
    **Pastikan struktur direktori seperti ini:**
    ```
    your_app/
    ├── streamlit_app.py
    └── model/
        ├── tfidf_vectorizer.pkl
        └── model_svm_sentimen.pkl
    ```
    
    **Dependencies yang diperlukan:**
    ```bash
    pip install streamlit pandas numpy scikit-learn nltk Sastrawi joblib
    ```
    """)
