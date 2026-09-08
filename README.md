# Local Real-Time Face Recognition, Age, Gender, & Expression Analysis Pipeline

Sistem pengenalan wajah (*face recognition*) real-time berkinerja tinggi yang berjalan secara lokal, dioptimalkan khusus untuk ekosistem Apple Silicon (M4 Pro) dengan pemanfaatan akselerasi *hardware* Neural Engine/GPU serta integrasi basis data enterprise menggunakan PostgreSQL dan ekstensi `pgvector`.

---

## 🚀 Fitur Utama
* **Face Detection & Embedding:** Menggunakan model SOTA ArcFace untuk mengekstrak vektor identitas wajah 512-dimensi.
* **Demografi Real-Time:** Estimasi umur dan jenis kelamin secara instan.
* **Expression Detection:** Analisis ekspresi (Senyum vs Netral) berbasis geometri landmark.
* **Vector Database:** Pencocokan kemiripan wajah menggunakan *Cosine Similarity* berkecepatan milidetik via ekstensi `pgvector` di PostgreSQL.
* **Hardware Acceleration:** Didukung penuh oleh eksekusi CoreML pada chip Apple Silicon.

---

## 🧠 Algoritma & Stack Teknologi

| Komponen | Teknologi / Algoritma | Keterangan |
| :--- | :--- | :--- |
| **Face Recognition Engine** | InsightFace (`buffalo_l` pack) | ArcFace Loss (ResNet-100 backbone) |
| **Akselerasi Perangkat Keras** | CoreML Execution Provider | Dioptimalkan untuk Apple Silicon (M1/M2/M3/M4) |
| **Database & Vector Search** | PostgreSQL + `pgvector` | HNSW Index untuk pencarian *Cosine Distance* |
| **Computer Vision** | OpenCV (`cv2`) | Manajemen *feed* kamera & rendering antarmuka UI |
| **Bahasa & Driver** | Python 3.11+, `psycopg` (v3) | Koneksi basis data dan pemrosesan array vektor |

---

## 📊 Dataset & Sumber Referensi Model
Model *pretrained* yang digunakan (`buffalo_l` dari InsightFace) dilatih menggunakan dataset skala besar standar industri:
* **MS-Celeb-1M / Glint360K:** Dataset masif untuk pelatihan *embedding* pengenalan wajah (*ArcFace*).
* **IMDB-WIKI / UTKFace (Referensi Arsitektur):** Digunakan sebagai basis pelatihan multi-task untuk prediksi umur dan gender.

---

## 🗄️ Skema Database PostgreSQL (`pgvector`)

Basis data dirancang untuk menyimpan vektor *embedding* 512 dimensi dengan indeks pencarian cepat HNSW:

```sql
-- Mengaktifkan ekstensi vector
CREATE EXTENSION IF NOT EXISTS vector;

-- Membuat tabel penyimpanan wajah terdaftar
CREATE TABLE registered_faces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_name VARCHAR(255) NOT NULL,
    face_embedding VECTOR(512) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Membuat indeks HNSW untuk Cosine Similarity yang cepat
CREATE INDEX idx_face_embedding ON registered_faces 
USING hnsw (face_embedding vector_cosine_ops);
