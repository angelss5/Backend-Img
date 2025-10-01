from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # <- Esto permite peticiones desde Angular


# ✅ Cargar el modelo solo una vez
modelo = tf.keras.models.load_model("modelo_gatos_perros_transfer.h5")
clases = ['gatos', 'perros']

def predecir(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = modelo.predict(img_array)

    if pred.shape[1] == 1:  # Binary
        clase_predicha = clases[int(pred[0][0] > 0.5)]
        confianza = pred[0][0] if pred[0][0] > 0.5 else 1 - pred[0][0]
    else:  # Categorical
        clase_predicha = clases[np.argmax(pred)]
        confianza = np.max(pred)

    return clase_predicha, float(confianza*100)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400

    file = request.files["file"]
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    label, precision = predecir(filepath)

    os.remove(filepath)  # borrar archivo temporal

    return jsonify({"label": label, "precision": precision})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
