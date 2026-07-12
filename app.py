# ==========================================================
# APP.PY (PART 1/4)
# COPY DARI BARIS PERTAMA
# ==========================================================

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import json
from PIL import Image

# ==========================================================
# INISIALISASI
# ==========================================================

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = tf.keras.models.load_model("brain_tumor_model.keras")

# ==========================================================
# LOAD NAMA KELAS
# ==========================================================

with open("class_names.json", "r") as f:

    class_names = json.load(f)

# ==========================================================
# LABEL INDONESIA
# ==========================================================

label_indonesia = {

    "glioma": "Glioma",

    "meningioma": "Meningioma",

    "pituitary": "Tumor Pituitari",

    "notumor": "Tidak Ada Tumor"

}

# ==========================================================
# DESKRIPSI
# ==========================================================

description = {

    "glioma":
    "Glioma merupakan tumor otak yang berasal dari sel glial. Tumor ini dapat bersifat jinak maupun ganas tergantung tingkat pertumbuhannya.",

    "meningioma":
    "Meningioma merupakan tumor yang berkembang pada selaput pelindung otak (meninges). Sebagian besar kasus bersifat jinak.",

    "pituitary":
    "Tumor pituitari merupakan tumor yang berkembang pada kelenjar pituitari dan dapat memengaruhi produksi hormon dalam tubuh.",

    "notumor":
    "Tidak ditemukan indikasi adanya tumor pada citra MRI yang dianalisis."

}

# ==========================================================
# PREPROCESS IMAGE
# ==========================================================

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((224,224))

    image = np.array(image)

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    return image

# ==========================================================
# PREDICT
# ==========================================================

def predict(image_path):

    image = preprocess_image(image_path)

    prediction = model.predict(image, verbose=0)

    index = np.argmax(prediction)

    confidence = float(np.max(prediction) * 100)

    probabilities = {}

    for i, score in enumerate(prediction[0]):

        probabilities[class_names[i]] = round(float(score * 100),2)

    return class_names[index], round(confidence,2), probabilities

# ==========================================================
# HALAMAN UTAMA
# ==========================================================

@app.route("/")

def home():

    return render_template("index.html")


# ==========================================================
# PREDIKSI
# ==========================================================

@app.route("/predict", methods=["POST"])

def upload():

    if "file" not in request.files:

        return render_template(
            "index.html",
            error="Silakan pilih gambar terlebih dahulu."
        )

    file = request.files["file"]

    if file.filename == "":

        return render_template(
            "index.html",
            error="Silakan pilih gambar terlebih dahulu."
        )

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    prediction, confidence, probabilities = predict(filepath)

    prediction_display = label_indonesia.get(
        prediction,
        prediction
    )

    probabilities_display = {}

    for key, value in probabilities.items():

        probabilities_display[
            label_indonesia.get(key, key)
        ] = value

    description_text = description.get(
        prediction,
        "Deskripsi tidak tersedia."
    )

    confidence = round(confidence, 2)

    return render_template(

        "index.html",

        prediction=prediction_display,

        confidence=confidence,

        probabilities=probabilities_display,

        image="/" + filepath.replace("\\", "/"),

        description=description_text

    )



# ==========================================================
# ERROR HANDLER
# ==========================================================

@app.errorhandler(404)

def not_found(error):

    return render_template(
        "index.html",
        error="Halaman tidak ditemukan."
    ),404


@app.errorhandler(500)

def internal_error(error):

    return render_template(
        "index.html",
        error="Terjadi kesalahan pada sistem."
    ),500


# ==========================================================
# FUNGSI TAMBAHAN
# ==========================================================

def allowed_file(filename):

    allowed_extensions = {

        "png",

        "jpg",

        "jpeg"

    }

    return "." in filename and filename.rsplit(".",1)[1].lower() in allowed_extensions


def format_probability(probabilities):

    hasil = {}

    for key, value in probabilities.items():

        hasil[key] = round(float(value), 2)

    return hasil


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )