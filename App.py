# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, jsonify, make_response, request 
import Petugas as pt
import Parkir as pl
import os
import pandas as pd
from dotenv import load_dotenv

# Flask constructor takes the name of the current module (__name__) as an argument.
app = Flask(__name__)

# Configuration
FILENAMEPETUGAS = os.getenv('PETUGAS_MODEL_PATH', 'Model/Model_Ml/Petugas_Model.pkl')
FILENAMEPARKIR = os.getenv('PARKIR_MODEL_PATH', 'Model/Model_Ml/Parkir_Model.pkl')  # Corrected filename
Lokasi = ['Pusat Perbelanjaan', 'Gedung Perkantoran', 'Gedung Perkantoran']
Petugas = ['Ada', 'Tidak Ada', 'Ada']

@app.route('/')
def hello_world():
    return 'Hello World'

# Petugas Parkir
@app.route('/Petugas_parkir', methods=['POST'])
def petugas_parkir():
    global FILENAMEPETUGAS
    try:
        # Ambil data dari request JSON
        data = request.json
        lokasi = data.get('Lokasi')
        identitas_petugas = data.get('Identitas_Petugas')

        # Pastikan lokasi dan identitas_petugas tidak kosong
        if not lokasi or not identitas_petugas:
            return make_response(jsonify({'error': 'Lokasi dan Identitas Petugas diperlukan'}), 400)

        # Panggil fungsi result untuk mendapatkan prediksi
        lokasi, identitas_petugas, y_predictions = pt.result(FILENAMEPETUGAS, lokasi, identitas_petugas)

        # Siapkan response data
        response_data = {
            'Lokasi': lokasi,
            'Identitas_Petugas': identitas_petugas,
            'Status Pelaporan': y_predictions
        }
        return make_response(jsonify(response_data), 200)
    except Exception as e:
        return make_response(jsonify({'error': str(e)}), 500)

@app.route('/Petugas_parkir/CM')
def confusion_matrix():
    try:
        model = pt.read_model(FILENAMEPETUGAS)

        # Create dummy test data (replace with actual test data)
        y_test = ['Ada', 'Tidak Ada']  # Replace with actual test labels
        X_test = pd.DataFrame({
            'Lokasi': ['Pusat Perbelanjaan', 'Gedung Perkantoran'],
            'Identitas Petugas': ['Ada', 'Tidak Ada']
        })

        cm = pt.confusion_matrix(model, X_test, y_test)
        return make_response(jsonify({'Confusion Matrix': cm.tolist()}), 200)
    except Exception as e:
        return make_response(jsonify({'error': str(e)}), 500)

@app.route('/Petugas_parkir/CR')
def classification_report():
    try:
        model = pt.read_model(FILENAMEPETUGAS)

        # Create dummy test data (replace with actual test data)
        y_test = ['Ada', 'Tidak Ada']  # Replace with actual test labels
        X_test = pd.DataFrame({
            'Lokasi': ['Pusat Perbelanjaan', 'Gedung Perkantoran'],
            'Identitas Petugas': ['Ada', 'Tidak Ada']
        })

        report = pt.classification_report(model, X_test, y_test)
        return make_response(jsonify({'Classification Report': report}), 200)
    except Exception as e:
        return make_response(jsonify({'error': str(e)}), 500)

# Parkir 
@app.route('/Parkir_Liar', methods=['POST'])
def parkir_Liar():
    global FILENAMEPARKIR
    try:
        # Extract JSON data from the request
        data = request.get_json()
        deskripsi_masalah = data.get('Deskripsi_Masalah', [])
        jenis_kendaraan = data.get('Jenis_Kendaraan', [])
        waktu = data.get('Waktu', [])

        # Call the Parkir result function
        deskripsi_masalah, jenis_kendaraan, y_predictions = pl.result(FILENAMEPARKIR, jenis_kendaraan, deskripsi_masalah, waktu)

        response_data = {
            'Deskripsi Masalah': deskripsi_masalah,
            'Jenis Kendaraan': jenis_kendaraan,
            'Waktu': waktu,
            'Status Pelaporan': y_predictions
        }
        return make_response(jsonify(response_data), 200)
    except Exception as e:
        return make_response(jsonify({'error': str(e)}), 500)

# Main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application on the local development server.
    app.run(debug=True)  # Set debug=False in production