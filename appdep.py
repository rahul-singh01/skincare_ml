import numpy as np
import pandas as pd
import tensorflow as tf
import os

import io
from tensorflow import keras
import cv2
from PIL import Image

from flask import Flask, request, jsonify

model = keras.models.load_model("./EfficientNetB2-Skin-87.h5")


def transform_image(pillow_image):
    data = np.asarray(pillow_image)
    data = np.array([data])
    data = tf.image.resize(data,[224, 224])
    return data

def predict(x):
    predictions = model.predict(x)
    pred0 = predictions[0]
    label0 = np.argmax(pred0)
    labels = ['Eczema','Warts Molluscum and other Viral Infections','Melanoma','Basal Cell Carcinoma','Melanocytic Nevi','Benign Keratosis-like Lesions',
              'Psoriasis pictures Lichen Planus and related diseases','Seborrheic Keratoses and other Benign Tumors', 'Tinea Ringworm Candidiasis and other Fungal Infections']
    return [labels[label0],float(pred0[label0])]

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            image_bytes = file.read()
            pillow_img = Image.open(io.BytesIO(image_bytes))
            tensor = transform_image(pillow_img)
            prediction = predict(tensor)
            data = {"prediction": prediction[0], "probability":prediction[1]}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    if request.method == "GET":
        return jsonify({
            "status" : 200,
        })


if __name__ == "__main__":
    app.run(debug=True, port=8000)