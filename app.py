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

print("DEBUG: Petugas model:", "Loaded" if model_petugas is not None else "Not loaded")
print("DEBUG: Parkir model:", "Loaded" if model_parkir is not None else "Not loaded")


# ==============================
# ROUTE: Home
# ==============================
@app.route('/')
def home():
    return jsonify({
        "message": "API for Petugas & Parkir Liar is running 🚀",
        "Petugas_Model_Loaded": model_petugas is not None,
        "Parkir_Model_Loaded": model_parkir is not None,
        "Petugas_Model_Accuracy": acc_petugas,
        "Parkir_Model_Accuracy": acc_parkir
    })


# ==============================
# ROUTE: Petugas Parkir
# ==============================
@app.route('/Petugas_parkir', methods=['POST'])
def petugas_parkir():
    try:
        if model_petugas is None:
            return jsonify({"error": "Petugas model not loaded"}), 500

        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        lokasi = data.get('Lokasi', "Tidak diketahui")
        identitas_petugas = data.get('Identitas_Petugas', "Tidak diketahui")

        Lokasi_list, Identitas_list, akurasi_list, status_list = pt.result(
            model_petugas, acc_petugas, lokasi, identitas_petugas
        )

        if not Lokasi_list or not status_list:
            return jsonify({"error": "Prediction failed"}), 500

        response_data = {
            "Lokasi": Lokasi_list[0],
            "Identitas_Petugas": Identitas_list[0],
            "Akurasi_Prediksi": akurasi_list[0],
            "Status_Pelaporan": status_list[0]
        }

        return make_response(jsonify(response_data), 200)

    except Exception as e:
        return make_response(jsonify({"error": str(e)}), 500)


# ==============================
# ROUTE: Parkir Liar
# ==============================
@app.route('/Parkir_Liar', methods=['POST'])
def parkir_liar():
    try:
        if model_parkir is None:
            return jsonify({"error": "Parkir model not loaded"}), 500

        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        deskripsi = data.get('Deskripsi_Masalah', "Tidak ada deskripsi")
        jenis_kendaraan = data.get('Jenis_Kendaraan', "Tidak diketahui")
        waktu = data.get('Waktu', "2026-01-01 00:00:00")

        Deskripsi_list, Jenis_list, Waktu_list, akurasi_list, status_list = pk.result(
            model_parkir, acc_parkir, deskripsi, jenis_kendaraan, waktu
        )

        if not Deskripsi_list or not status_list:
            return jsonify({"error": "Prediction failed"}), 500

        response_data = {
            "Deskripsi_Masalah": Deskripsi_list[0],
            "Jenis_Kendaraan": Jenis_list[0],
            "Waktu": Waktu_list[0],
            "Akurasi_Prediksi": akurasi_list[0],
            "Status_Pelaporan": status_list[0]
        }

        return make_response(jsonify(response_data), 200)

    except Exception as e:
        return make_response(jsonify({"error": str(e)}), 500)


# ==============================
# MAIN
# ==============================
if __name__ == '__main__':
    # Untuk development saja
    app.run( debug=True)