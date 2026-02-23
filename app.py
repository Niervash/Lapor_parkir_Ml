from flask import Flask, jsonify, request
import Petugas as pt
import Parkir as pk

# ==============================
# Flask app
# ==============================
app = Flask(__name__)

# ==============================
# LOAD MODELS (SAAT STARTUP)
# ==============================
print("🚀 Loading models...")

model_petugas, acc_petugas = pt.read_model(pt.MODEL_PATH)
model_parkir, acc_parkir = pk.read_model(pk.MODEL_PATH)

print("Petugas model:", "Loaded ✅" if model_petugas else "Not loaded ❌")
print("Parkir model:", "Loaded ✅" if model_parkir else "Not loaded ❌")


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
    if model_petugas is None:
        return jsonify({"error": "Petugas model not loaded"}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    lokasi = data.get('Lokasi')
    identitas_petugas = data.get('Identitas_Petugas')

    Lokasi_list, Identitas_list, akurasi_list, status_list = pt.result(
        model_petugas, acc_petugas, lokasi, identitas_petugas
    )

    if not status_list:
        return jsonify({"error": "Prediction failed"}), 500

    return jsonify({
        "Lokasi": Lokasi_list[0],
        "Identitas_Petugas": Identitas_list[0],
        "Akurasi_Prediksi": akurasi_list[0],
        "Status_Pelaporan": status_list[0]
    })


# ==============================
# ROUTE: Parkir Liar
# ==============================
@app.route('/Parkir_Liar', methods=['POST'])
def parkir_liar():
    if model_parkir is None:
        return jsonify({"error": "Parkir model not loaded"}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    deskripsi = data.get('Deskripsi_Masalah')
    jenis_kendaraan = data.get('Jenis_Kendaraan')
    waktu = data.get('Waktu')

    Deskripsi_list, Jenis_list, Waktu_list, akurasi_list, status_list = pk.result(
        model_parkir, acc_parkir, deskripsi, jenis_kendaraan, waktu
    )

    if not status_list:
        return jsonify({"error": "Prediction failed"}), 500

    return jsonify({
        "Deskripsi_Masalah": Deskripsi_list[0],
        "Jenis_Kendaraan": Jenis_list[0],
        "Waktu": Waktu_list[0],
        "Akurasi_Prediksi": akurasi_list[0],
        "Status_Pelaporan": status_list[0]
    })


# ==============================
# GLOBAL ERROR HANDLER
# ==============================
@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500


# ==============================
# MAIN
# ==============================
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)