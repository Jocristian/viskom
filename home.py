import streamlit as st

# ⛔ set_page_config harus langsung setelah import st
st.set_page_config(page_title="Home - Face Exercise App", layout="wide")

import sys
import os

# Redirect sys.stderr to devnull to avoid WinError 6
sys.stderr = open(os.devnull, 'w')



from streamlit_webrtc import webrtc_streamer
import streamlit.components.v1 as components
from face_classification.src.video_camera import EmotionProcessor

import os

# di file utama (misal app.py)
import threading
import json

st.markdown("""
    <style>
        body {
            font-family: 'Poppins', sans-serif;
        }
        .main {
            background-color: #f0f0f0;
            padding: 2rem;
        }
        .banner {
            background-image: url('https://plus.unsplash.com/premium_photo-1722728055718-20684f6bddbb?q=80&w=2070&auto=format&fit=crop');
            background-size: cover;
            background-position: center;
            padding: 4rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .card {
            background-color: gray;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        .emotion-game {
            background-color: white;
            background-color: gray;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
        }
        .emotion-game img {
            border-radius: 12px;
            width: 300px;
            height: auto;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .btn-restart {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            margin-top: 1rem;
            cursor: pointer;
        }
        .btn-restart:hover {
            background-color: #0056b3;
        }
    </style>
""", unsafe_allow_html=True)

# Banner
st.markdown("""
<div class="banner">
    <h1>Face Exercise & Real-Time Emotion Detection</h1>
    <p>Latihan wajah interaktif untuk meningkatkan kesehatan kulit dan ekspresi Anda!</p>
</div>
""", unsafe_allow_html=True)

# Description
st.markdown("""
<div class="card">
    <h2>🧠 Real-Time Emotion Detection</h2>
    <p>Aktifkan kamera Anda dan deteksi ekspresi wajah secara langsung menggunakan model klasifikasi emosi.</p>
</div>
""", unsafe_allow_html=True)

import streamlit as st

st.markdown("""
## Insight Cara Kerja Deteksi Emosi dengan `EmotionProcessor`

Kode ini mengimplementasikan sebuah pemroses video (`EmotionProcessor`) yang menggunakan `streamlit-webrtc` untuk melakukan deteksi emosi secara *real-time* dari *feed* video. Berikut adalah breakdown cara kerjanya:

### 1. Inisialisasi dan Konfigurasi Model

* **Import *Libraries***: Kode ini mengimpor berbagai *library* penting seperti `streamlit_webrtc` untuk integrasi *webcam*, `av` untuk memproses *frame* video, `cv2` (OpenCV) untuk manipulasi gambar, `numpy` untuk operasi numerik, dan `keras.models` untuk memuat model *deep learning*.
* **Path Direktori dan Model**: Kode ini secara dinamis menentukan *path* ke direktori `src` dan `trained_models`. Ini memastikan fleksibilitas kode meskipun dijalankan dari direktori yang berbeda.
* **Pemuatan Model**:
    * Model deteksi wajah (`haarcascade_frontalface_default.xml`) dimuat menggunakan fungsi `load_detection_model`. Model ini bertugas untuk menemukan lokasi wajah dalam *frame* video.
    * Model klasifikasi emosi (`fer2013_mini_XCEPTION.102-0.66.hdf5`) dimuat menggunakan `load_model` dari Keras. Model ini telah dilatih untuk mengklasifikasikan emosi berdasarkan input gambar wajah.
    * Label emosi (`fer2013`) dimuat menggunakan `get_labels`, yang kemungkinan memetakan angka prediksi model ke dalam label emosi seperti 'angry', 'happy', 'sad', dll.
    * Ukuran input yang diharapkan oleh model emosi (`emotion_target_size`) dan *offset* untuk memotong wajah dari gambar (`emotion_offsets`) juga didefinisikan.

### 2. Kelas `EmotionProcessor`

Kelas ini merupakan inti dari pemrosesan video dan deteksi emosi. Ia mewarisi dari `VideoProcessorBase` yang disediakan oleh `streamlit-webrtc`.

* **`__init__(self)`**:
    * `frame_window`: Menentukan jumlah *frame* terakhir yang akan digunakan untuk mengambil keputusan emosi yang paling sering muncul (menggunakan *mode*). Ini membantu dalam menstabilkan prediksi emosi dan mengurangi *noise*.
    * `emotion_window`: Sebuah *list* yang menyimpan label emosi dari *frame-frame* terakhir.
    * `set_emotion_callback`: Sebuah *callback function* yang kemungkinan digunakan untuk mengirimkan emosi yang terdeteksi ke bagian lain dari aplikasi Streamlit (walaupun dalam kode ini implementasinya terlihat sederhana).

* **`update_emotion(self, emotion_mode)`**: Fungsi ini dipanggil untuk memperbarui emosi terakhir yang terdeteksi. Dalam kode ini, ia memanggil `self.set_emotion_callback` dengan emosi yang terdeteksi.

* **`recv(self, frame: av.VideoFrame) -> av.VideoFrame`**: Fungsi ini adalah jantung dari pemrosesan setiap *frame* video yang diterima. Berikut langkah-langkahnya:
    1.  **Konversi *Frame***: *Frame* video dari `streamlit-webrtc` dikonversi menjadi format NumPy array (`bgr24`) dan kemudian menjadi *grayscale* untuk deteksi wajah dan RGB untuk visualisasi.
    2.  **Deteksi Wajah**: Fungsi `detect_faces` dipanggil dengan model deteksi wajah dan gambar *grayscale* untuk menemukan koordinat wajah dalam *frame*.
    3.  **Iterasi Wajah**: Untuk setiap wajah yang terdeteksi:
        * **Ekstraksi Wajah**: Koordinat wajah disesuaikan menggunakan `apply_offsets` dan area wajah diekstrak dari gambar *grayscale*.
        * **Preprocessing Wajah**: Area wajah di-*resize* sesuai dengan input yang diharapkan oleh model emosi (`emotion_target_size`). Penanganan *error* dilakukan jika *resize* gagal (mungkin karena wajah terlalu kecil atau tidak terdefinisi dengan baik). Kemudian, wajah yang telah di-*resize* di-*preprocess* menggunakan `preprocess_input` (kemungkinan normalisasi atau penskalaan piksel) dan diubah dimensinya agar sesuai dengan input model Keras (menambahkan dimensi *batch* dan *channel*).
        * **Prediksi Emosi**: Model emosi (`emotion_classifier`) melakukan prediksi pada wajah yang telah diproses. Hasil prediksi adalah *array* probabilitas untuk setiap kelas emosi.
        * **Penentuan Label Emosi**: Probabilitas maksimum dan indeks kelas yang sesuai ditemukan menggunakan `np.max` dan `np.argmax`. Label emosi yang sesuai diambil dari `emotion_labels`.
        * **Penyimpanan Emosi**: Label emosi saat ini ditambahkan ke `self.emotion_window`. Jika panjang jendela melebihi `frame_window`, emosi terlama akan dihapus.
        * **Penentuan Emosi Dominan**: Emosi yang paling sering muncul dalam `self.emotion_window` dihitung menggunakan `statistics.mode`. Jika jendela masih kosong atau terjadi *error* dalam perhitungan *mode*, emosi saat ini digunakan sebagai emosi dominan.
        * **Pembaruan Emosi Global**: Fungsi `self.update_emotion` dipanggil untuk menyimpan atau mengirimkan emosi yang terdeteksi.
        * **Visualisasi**: Berdasarkan emosi dominan, warna untuk *bounding box* dan teks ditentukan. Fungsi `draw_bounding_box` dan `draw_text` digunakan untuk menggambar kotak di sekitar wajah dan menampilkan label emosi pada *frame* RGB.
    4.  **Konversi Kembali *Frame***: Gambar RGB yang telah diproses dikonversi kembali ke format BGR dan dikembalikan sebagai `av.VideoFrame`.

### 3. Penggunaan dalam Streamlit

Meskipun cuplikan kode ini tidak menunjukkan penggunaan langsung dalam aplikasi Streamlit, kelas `EmotionProcessor` ini dirancang untuk digunakan dengan komponen `webrtc_streamer` dari `streamlit-webrtc`. Anda akan membuat instance dari `EmotionProcessor` dan meneruskannya ke `webrtc_streamer` sebagai `video_processor_factory`. Setiap *frame* yang ditangkap dari *webcam* akan diproses oleh instance `EmotionProcessor`, dan *frame* yang telah diannotasi (dengan *bounding box* dan label emosi) akan ditampilkan di aplikasi Streamlit.

### Model yang Digunakan

* **Model Deteksi Wajah (Haar Cascade)**: `haarcascade_frontalface_default.xml` adalah sebuah *classifier* berbasis fitur Haar yang dilatih untuk mendeteksi wajah frontal. Metode ini relatif cepat tetapi mungkin kurang akurat dalam kondisi pencahayaan yang buruk atau pose wajah yang non-frontal.
* **Model Klasifikasi Emosi (Mini-XCEPTION)**: `fer2013_mini_XCEPTION.102-0.66.hdf5` adalah sebuah model *Convolutional Neural Network* (CNN) yang lebih dalam dan kompleks, dirancang khusus untuk tugas klasifikasi emosi pada dataset FER2013. Arsitektur XCEPTION dikenal karena efisiensinya dan kinerjanya yang baik dalam tugas-tugas visi komputer. Model ini kemungkinan telah dilatih pada tujuh emosi dasar (marah, jijik, takut, bahagia, sedih, terkejut, dan netral).

Secara keseluruhan, kode ini menggabungkan teknik deteksi wajah klasik dengan model *deep learning* modern untuk mencapai deteksi emosi *real-time* dari *feed* video. Penggunaan jendela *frame* untuk menentukan emosi dominan membantu dalam menghasilkan prediksi yang lebih stabil.
""")