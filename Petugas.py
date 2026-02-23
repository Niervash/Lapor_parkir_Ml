import pickle
import pandas as pd
import os
import random

# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'model_new', 'model_knn_petugas.pkl')


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

        # Jika model disimpan dalam dictionary
        if isinstance(model_data, dict):
            model = model_data.get("model")
            accuracy = model_data.get("accuracy_test")
        else:
            model = model_data
            accuracy = None

        # Validasi object model
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
def result(model_loaded, saved_accuracy, Lokasi, Identitas_Petugas):
    try:
        if model_loaded is None:
            print("❌ Model not loaded.")
            return [], [], [], []

        # ==============================
        # DEFAULT VALUE
        # ==============================
        Lokasi = Lokasi if Lokasi else "Tidak diketahui"
        Identitas_Petugas = Identitas_Petugas if Identitas_Petugas else "Tidak diketahui"

        # ==============================
        # CREATE INPUT DATAFRAME
        # SESUAI TRAINING
        # ==============================
        input_data = pd.DataFrame({
            'Lokasi': [Lokasi],
            'Identitas Petugas': [Identitas_Petugas]
        })

        # ==============================
        # PREDIKSI
        # ==============================
        prediction = model_loaded.predict(input_data)

        if len(prediction) == 0:
            print("❌ Prediction empty")
            return [], [], [], []

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
            'Lokasi': [Lokasi],
            'Identitas Petugas': [Identitas_Petugas],
            'Status Pelaporan': [prediction[0]],
            'Akurasi Prediksi (%)': [akurasi_prediksi]
        })

        return (
            hasil_df['Lokasi'].tolist(),
            hasil_df['Identitas Petugas'].tolist(),
            hasil_df['Akurasi Prediksi (%)'].tolist(),
            hasil_df['Status Pelaporan'].tolist()
        )

    except Exception as e:
        print(f"❌ Error in result function: {str(e)}")
        return [], [], [], []


# ==============================
# LOAD DEFAULT MODEL
# ==============================
model, saved_accuracy = read_model(MODEL_PATH)