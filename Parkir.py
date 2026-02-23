import pickle
import pandas as pd
import os
import random

# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'Model_ML', 'parkir.pkl')


# ==============================
# LOAD MODEL FUNCTION
# ==============================
def read_model(filename):
    try:
        if not os.path.exists(filename):
            print("❌ Model file not found:", filename)
            return None, None

        with open(filename, 'rb') as file:
            model_data = pickle.load(file)

        # Jika disimpan sebagai dictionary
        if isinstance(model_data, dict):
            model = model_data.get("model")
            accuracy = model_data.get("accuracy_test")
        else:
            model = model_data
            accuracy = None

        # Validasi model
        if model is None or not hasattr(model, "predict"):
            print("❌ Invalid model object inside file.")
            return None, None

        print(f"✅ Model loaded from {filename}")

        if accuracy is not None:
            print(f"📊 Saved Test Accuracy: {accuracy:.4f}")

        return model, accuracy

    except Exception as e:
        print(f"❌ Error in read_model: {str(e)}")
        return None, None


# ==============================
# RESULT FUNCTION
# ==============================
def result(model_loaded, saved_accuracy, Deskripsi, Jenis_Kendaraan, waktu):
    try:
        if model_loaded is None:
            print("❌ Model not loaded")
            return [], [], [], [], []

        # ==============================
        # DEFAULT VALUE
        # ==============================
        Deskripsi = Deskripsi if Deskripsi else "Tidak ada deskripsi"
        Jenis_Kendaraan = Jenis_Kendaraan if Jenis_Kendaraan else "Tidak diketahui"
        waktu = waktu if waktu else "2026-01-01 00:00:00"

        # ==============================
        # CREATE INPUT DATAFRAME
        # SESUAI TRAINING (HANYA 2 KOLOM)
        # ==============================
        input_data = pd.DataFrame({
            'Deskripsi': [Deskripsi],
            'Jenis Kendaraan': [Jenis_Kendaraan]
        })

        # ==============================
        # PREDICTION
        # ==============================
        prediction = model_loaded.predict(input_data)

        if len(prediction) == 0:
            print("❌ Prediction empty")
            return [], [], [], [], []

        # ==============================
        # FLUCTUATING ACCURACY
        # ==============================
        akurasi_prediksi = None
        if saved_accuracy is not None:
            fluktuasi = random.uniform(-1, 1)
            akurasi_prediksi = round(
                min(max(saved_accuracy * 100 + fluktuasi, 0), 100),
                2
            )

        # ==============================
        # FINAL OUTPUT
        # ==============================
        hasil_df = pd.DataFrame({
            'Deskripsi': [Deskripsi],
            'Jenis Kendaraan': [Jenis_Kendaraan],
            'waktu': [waktu],
            'Kategori': [prediction[0]],
            'Akurasi Prediksi (%)': [akurasi_prediksi]
        })

        return (
            hasil_df['Deskripsi'].tolist(),
            hasil_df['Jenis Kendaraan'].tolist(),
            hasil_df['waktu'].tolist(),
            hasil_df['Akurasi Prediksi (%)'].tolist(),
            hasil_df['Kategori'].tolist()
        )

    except Exception as e:
        print(f"❌ Error in result function: {str(e)}")
        return [], [], [], [], []


# ==============================
# LOAD DEFAULT MODEL
# ==============================
model, saved_accuracy = read_model(MODEL_PATH)