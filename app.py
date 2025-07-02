from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load('model_dt.pkl')

def kategorikan_performa(score):
    if score >= 75:
        return "Sangat Baik"
    elif score >= 70:
        return "Baik"
    elif score >= 65:
        return "Cukup Baik"
    elif score >= 60:
        return "Cukup"
    elif score >= 55:
        return "Kurang Baik"
    elif score >= 50:
        return "Kurang"
    elif score >= 45:
        return "Sangat Kurang"
    else:
        return "Sangat Buruk"

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        pendidikan = data.get("pendidikan")
        pengalaman = data.get("pengalaman_kerja")
        psikotes = data.get("skor_psikotes")

        if None in (pendidikan, pengalaman, psikotes):
            return jsonify({"error": "Semua field harus diisi"}), 400

        # Fitur interaksi
        edu_x_exp = pendidikan * pengalaman
        exp_x_score = pengalaman * psikotes

        X = np.array([[pendidikan, pengalaman, psikotes, edu_x_exp, exp_x_score]])
        score = round(model.predict(X)[0], 2)
        kategori = kategorikan_performa(score)

        return jsonify({
            "prediksi_performa": score,
            "kategori_performa": kategori
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
