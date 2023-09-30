from flask import Flask, request, jsonify, abort
from flask_cors import cross_origin
import numpy as np
from keras.models import load_model
import tensorflow as tf
import string
import re
import tensorflow_datasets as tfds
import os
from flask_cors import CORS
import sys

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
sess = tf.compat.v1.Session(config=config)



app = Flask(__name__)
CORS(app)


model_path = os.path.join(os.path.dirname(__file__), 'dcnn_model.h5')
encorder = os.path.join(os.path.dirname(__file__), 'tokenized_encoder')

# model_path = "C:/Users/Athindu/Downloads/sentiment (1)/sentiment/API/dcnn_model.h5"


ensemble_model = tf.keras.models.load_model(model_path,compile=False)


def clean_text(input_text):
    # Replace URLs in the text
    input_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', input_text)
    
    # Remove numbers
    input_text = re.sub(r'\d+', '', input_text)

    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation.replace('?', ''))
    input_text = input_text.translate(translator)

    return input_text


def tokenize_text(text):
    encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
        encorder)
    seq = encoder.encode(text)
    return seq


def checkForBullying(text):
    encoded_text = tokenize_text(text)
    output = ensemble_model(np.array([encoded_text]), training=False).numpy()
    labels = ['0', '1']
    highest_index = np.argmax(output)
    predicted_label = labels[highest_index]

    # Calculate the sum of the output array
    sum_output = sum(output[0])

    for i in range(len(output[0])):
        output[0][i] = output[0][i] / sum_output * 100
        print(f"Class {i}: {output[0][i]:.2f}%")
    print("")
    print("Highest match: Class ", highest_index)

    return predicted_label, output


@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    text = request.get_data(as_text=True)
    print(text)
    cleaned_text = clean_text(text)
    if not cleaned_text:
        abort(400, 'Error: Text is empty.')
    predicted_label, output = checkForBullying(cleaned_text)
    output = [float(x) for x in output[0]]
    print(predicted_label)
    return jsonify({'label': predicted_label, 'output': output})


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=400)