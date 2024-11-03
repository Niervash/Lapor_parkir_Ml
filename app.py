# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, jsonify, make_response, request 
import joblib as jb
import Petugas as pt
import Parkir as pl
import os
import pandas as pd


# Flask constructor takes the name of the current module (__name__) as an argument.
app = Flask(__name__)

# Configurationally define
FILENAMEPETUGAS = 'Model/Model_Ml/Petugas_Model.pkl'
FILENAMEPARKIR = 'Model/Model_Ml/Parkir_liar_Model.pkl'
    
@app.route('/')
def hello_world():
    return 'Hello World !'

# Petugas Parkir
@app.route('/Petugas_parkir', methods=['POST'])
def petugas_parkir():
    global FILENAMEPETUGAS
    lokasi = []
    identitas_petugas = []
    try:
        # Ambil data dari request JSON
        data = request.json
        lokasi = data.get('Lokasi')
        identitas_petugas = data.get('Identitas_Petugas')
        print(data)    

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

# Parkir 
@app.route('/Parkir_Liar', methods=['POST'])
def parkir_Liar():
    global FILENAMEPARKIR
    deskripsi_masalah = []
    jenis_kendaraan = []
    waktu = []
    
    try:
        # Extract JSON data from the request
        data = request.json
        deskripsi_masalah = data.get('Deskripsi_Masalah')
        jenis_kendaraan = data.get('Jenis_Kendaraan')
        waktu = data.get('Waktu')


        # Panggil fungsi result untuk mendapatkan prediksi
        lokasi, identitas_petugas, y_predictions = pt.result(FILENAMEPETUGAS, lokasi, identitas_petugas)

        # Call the Parkir result function
        Deskripsi_Masalah, Jenis_Kendaraan, waktu, y_predictions  = pl.result(FILENAMEPARKIR, jenis_kendaraan, deskripsi_masalah, waktu)
        
        response_data = {
            'Deskripsi Masalah': Deskripsi_Masalah,
            'Jenis Kendaraan': Jenis_Kendaraan,
            'waktu': waktu,
            'Status Pelaporan': y_predictions
        }

        return make_response(jsonify(response_data), 200)
    except Exception as e:
        return make_response(jsonify({'error': str(e)}), 500)
    
    
# Main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application on the local development server.
    app.run(debug=False)  # Set debug=False in production
