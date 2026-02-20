from flask import Flask, jsonify, make_response, request
import Petugas as pt
import Parkir as pk

app = Flask(__name__)

FILENAMEPARKIR = 'Model/model_new/model_knn_petugas.joblib'
FILENAMEPETUGAS = 'Model/model_new/knn_parkir_model_eval.joblib'
# ==============================
# Load Models Sekali Saja
# ==============================
model_petugas, acc_petugas = pt.read_model(FILENAMEPARKIR)
model_parkir, acc_parkir = pk.read_model(FILENAMEPETUGAS)

# ==============================
# Route Home
# ==============================
@app.route('/')
def home():
    return jsonify({
        "message": "API for Petugas & Parkir Liar is running 🚀",
        "Model Accuracy Petugas": acc_petugas,
        "Model Accuracy Parkir": acc_parkir
    })

# ==============================
# Route Petugas Parkir
# ==============================
@app.route('/Petugas_parkir', methods=['POST'])
def petugas_parkir():
    try:
        data = request.json
        lokasi = data.get('Lokasi')
        identitas_petugas = data.get('Identitas_Petugas')

        # Panggil fungsi result dari petugas.py
        Lokasi, Identitas, akurasi, status = pt.result(
            model_petugas, acc_petugas, lokasi, identitas_petugas
        )

        response_data = {
            'Lokasi': Lokasi[0] if Lokasi else None,
            'Identitas_Petugas': Identitas[0] if Identitas else None,
            'Akurasi Prediksi (%)': akurasi[0] if akurasi else None,
            'Status Pelaporan': status[0] if status else None
        }
        return make_response(jsonify(response_data), 200)

    except Exception as e:
        return make_response(jsonify({'error': str(e)}), 500)

# ==============================
# Route Parkir Liar
# ==============================
@app.route('/Parkir_Liar', methods=['POST'])
def parkir_liar():
    try:
        data = request.json
        deskripsi = data.get('Deskripsi')
        jenis_kendaraan = data.get('Jenis_Kendaraan')
        waktu = data.get('waktu')

        # Panggil fungsi result dari perkir.py
        Deskripsi_List, Jenis_List, Waktu_List, akurasi_list, status_list = pk.result(
            model_parkir, acc_parkir, deskripsi, jenis_kendaraan, waktu
        )

        response_data = {
            'Deskripsi': Deskripsi_List[0] if Deskripsi_List else None,
            'Jenis_Kendaraan': Jenis_List[0] if Jenis_List else None,
            'waktu': Waktu_List[0] if Waktu_List else None,
            'Akurasi Prediksi (%)': akurasi_list[0] if akurasi_list else None,
            'Status Pelaporan': status_list[0] if status_list else None
        }
        return make_response(jsonify(response_data), 200)

    except Exception as e:
        return make_response(jsonify({'error': str(e)}), 500)


if __name__ == '__main__':
    app.run(debug=True)