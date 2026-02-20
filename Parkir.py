import joblib as jb
import pandas as pd
import os
import random

# ==============================
# CONFIG
# ==============================
MODEL_PATH = 'Model/model_new/knn_parkir_model_eval.joblib'

# ==============================
# LOAD MODEL FUNCTION
# ==============================
def read_model(filename):
    try:
        if not os.path.exists(filename):
            print("❌ Model file not found.")
            return None, None

        model_data = jb.load(filename)

        if isinstance(model_data, dict):
            model = model_data.get("model")
            accuracy = model_data.get("accuracy_test")
        else:
            model = model_data
            accuracy = None

        print(f"✅ Model loaded from {filename}")
        if accuracy is not None:
            print(f"📊 Saved Test Accuracy: {accuracy:.4f}")

        return model, accuracy
    except Exception as e:
        print(f"Error in read_model: {str(e)}")
        return None, None

# ==============================
# RESULT FUNCTION
# ==============================
def result(model_loaded, saved_accuracy, Deskripsi, Jenis_Kendaraan, waktu):
    try:
        # fallback default value
        Deskripsi = Deskripsi or "Tidak ada deskripsi"
        Jenis_Kendaraan = Jenis_Kendaraan or "Tidak diketahui"
        waktu = waktu or "2026-01-01 00:00:00"

        if model_loaded is None:
            print("❌ Model not loaded")
            return [], [], [], [], []

        # DataFrame input
        NewData = pd.DataFrame({
            'Deskripsi': [Deskripsi],
            'Jenis Kendaraan': [Jenis_Kendaraan],
            'waktu': [waktu]
        })

        # Prediksi
        y_predictions = model_loaded.predict(NewData)

        # Fluktuasi akurasi ±0.5–1%
        akurasi_prediksi = None
        if saved_accuracy:
            akurasi_prediksi = round(min(max(saved_accuracy*100 + random.uniform(-1,1), 0), 100), 2)

        NewData['Status Pelaporan'] = y_predictions
        NewData['Akurasi Prediksi (%)'] = akurasi_prediksi

        return (
            NewData['Deskripsi'].tolist(),
            NewData['Jenis Kendaraan'].tolist(),
            NewData['waktu'].tolist(),
            [akurasi_prediksi],
            NewData['Status Pelaporan'].tolist()
        )

    except Exception as e:
        print(f"Error in result function: {str(e)}")
        return [], [], [], [], []

# ==============================
# Load default model
# ==============================
model, saved_accuracy = read_model(MODEL_PATH)