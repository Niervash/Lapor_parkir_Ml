from flask import Flask, jsonify, make_response, request
import Petugas as pt
import Parkir as pk

# ==============================
# Flask app
# ==============================
app = Flask(__name__)

# ==============================
# Load models (sekali saja)
# ==============================
model_petugas, acc_petugas = pt.model, pt.saved_accuracy
model_parkir, acc_parkir = pk.model, pk.saved_accuracy

# Debug: cek apakah model berhasil load
print("DEBUG: Petugas model:", "Loaded" if model_petugas else "Not loaded")
print("DEBUG: Parkir model:", "Loaded" if model_parkir else "Not loaded")

# ==============================
# ROUTE: Home
# ==============================
@app.route('/')
def home():
    return jsonify({
        "message": "API for Petugas & Parkir Liar is running 🚀",
        "Petugas Model Accuracy": acc_petugas,
        "Parkir Model Accuracy": acc_parkir
    })

# ==============================
# ROUTE: Petugas Parkir
# ==============================
@app.route('/Petugas_parkir', methods=['POST'])
def petugas_parkir():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        lokasi = data.get('Lokasi') or "Tidak diketahui"
        identitas_petugas = data.get('Identitas_Petugas') or "Tidak diketahui"

        Lokasi_list, Identitas_list, akurasi_list, status_list = pt.result(
            model_petugas, acc_petugas, lokasi, identitas_petugas
        )

        # Cek list kosong untuk aman
        if not Lokasi_list or not status_list:
            return jsonify({"error": "Prediction failed, check model or input"}), 500

        response_data = {
            'Lokasi': Lokasi_list[0],
            'Identitas_Petugas': Identitas_list[0],
            'Akurasi_Prediksi': akurasi_list[0],
            'Status_Pelaporan': status_list[0]
        }
        return make_response(jsonify(response_data), 200)

    except Exception as e:
        return make_response(jsonify({'error': str(e)}), 500)

# ==============================
# ROUTE: Parkir Liar
# ==============================
@app.route('/Parkir_Liar', methods=['POST'])
def parkir_liar():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        deskripsi = data.get('Deskripsi_Masalah') or "Tidak ada deskripsi"
        jenis_kendaraan = data.get('Jenis_Kendaraan') or "Tidak diketahui"
        waktu = data.get('Waktu') or "2026-01-01 00:00:00"

        Deskripsi_list, Jenis_list, Waktu_list, akurasi_list, status_list = pk.result(
            model_parkir, acc_parkir, deskripsi, jenis_kendaraan, waktu
        )

        # Cek list kosong untuk aman
        if not Deskripsi_list or not status_list:
            return jsonify({"error": "Prediction failed, check model or input"}), 500

        response_data = {
            'Deskripsi_Masalah': Deskripsi_list[0],
            'Jenis_Kendaraan': Jenis_list[0],
            'Waktu': Waktu_list[0],
            'Akurasi_Prediksi': akurasi_list[0],
            'Status_Pelaporan': status_list[0]
        }

        return make_response(jsonify(response_data), 200)

    except Exception as e:
        return make_response(jsonify({'error': str(e)}), 500)

# ==============================
# Main
# ==============================
if __name__ == '__main__':
    app.run(debug=True)  # debug=True untuk development