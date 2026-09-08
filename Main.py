import cv2
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from insightface.app import FaceAnalysis

# --- 1. SETUP DATABASE ---
conn = psycopg.connect("dbname=postgres", autocommit=True)
register_vector(conn)

def recognize_face_sql(target_embedding, threshold=0.45):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT user_name, 1 - (face_embedding <=> %s::vector) AS similarity
            FROM registered_faces
            WHERE 1 - (face_embedding <=> %s::vector) >= %s
            ORDER BY face_embedding <=> %s::vector ASC LIMIT 1;
            """,
            (target_embedding, target_embedding, threshold, target_embedding)
        )
        result = cur.fetchone()
        return (result[0], result[1]) if result else ("Unknown", 0.0)

# --- 2. SETUP INSIGHTFACE (Apple Silicon CoreML) ---
# Meminta buffalo_l mendeteksi landmark 2d (106 titik) secara otomatis
app = FaceAnalysis(name='buffalo_l', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# --- 3. FUNGSI ANALISIS EKSPRESI BERBASIS LANDMARK 5 TITIK ---
def estimate_expression(landmark_5):
    """
    Menghitung ekspresi sederhana berdasarkan jarak relatif sudut mulut 
    terhadap hidung menggunakan 5 titik landmark standard InsightFace.
    """
    if landmark_5 is None or len(landmark_5) < 5:
        return "Netral"
    
    # Landmark 5 titik InsightFace:
    # 0: Mata Kiri, 1: Mata Kanan, 2: Hidung, 3: Sudut Mulut Kiri, 4: Sudut Mulut Kanan
    nose = landmark_5[2]
    left_mouth = landmark_5[3]
    right_mouth = landmark_5[4]
    
    # Hitung lebar mulut
    mouth_width = np.linalg.norm(left_mouth - right_mouth)
    
    # Hitung jarak dari hidung ke tengah mulut sebagai referensi skala wajah
    mouth_center = (left_mouth + right_mouth) / 2
    nose_to_mouth = np.linalg.norm(nose - mouth_center)
    
    # Rasio senyum (Lebar mulut terhadap jarak hidung-mulut)
    ratio = mouth_width / (nose_to_mouth + 1e-6)
    
    # Ambang batas empiris untuk deteksi senyuman
    if ratio > 1.65:
        return "Senyum (Happy)"
    else:
        return "Netral (Calm)"

# --- 4. REAL-TIME WEBCAM LOOP ---
cap = cv2.VideoCapture(0)
is_registered = True

print("Tekan 'q' untuk keluar dari jendela.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        
        # A. Identitas & Akurasi Cosine Similarity
        name, confidence = recognize_face_sql(embedding, threshold=0.45)
        sim_percent = confidence * 100

        # B. Atribut Demografi
        gender = 'Laki-laki' if face.gender == 1 else 'Perempuan'
        age = int(face.age)

        # C. Deteksi Ekspresi dari Landmark 5 Titik (.kps)
        expression = "Netral"
        if hasattr(face, 'kps'):
            expression = estimate_expression(face.kps)

        # D. Visualisasi Antarmuka (UI)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        
        # Format Teks Informasi
        label_id = f"{name} ({sim_percent:.1f}%)"
        label_attr = f"{gender}, {age} Thn"
        label_exp = f"Ekspresi: {expression}"
        
        font_scale = 0.8
        font_thickness = 2
        
        cv2.putText(frame, label_id, (bbox[0], bbox[1] - 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
        cv2.putText(frame, label_attr, (bbox[0], bbox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), font_thickness)
        cv2.putText(frame, label_exp, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 200, 255), font_thickness)

    cv2.imshow('M4 Pro Face Recognition & Expression', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()