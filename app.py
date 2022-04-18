from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

model_dir = "./model"
model = tf.keras.models.load_model(model_dir)
# model = pickle.load(open('catsdogs_cnn_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        print(f)
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        img_arr = cv2.imread(file_path)
        img_arr = cv2.resize(img_arr, (100, 100))
        prediction = img_arr.shape
        print(prediction)
        img_arr = img_arr.reshape(1, 100, 100, 3)
        prediction = img_arr.shape
        print(prediction)
        prediction=model.predict(img_arr)
        if prediction[0][0]<0.5:
            print("cat")
            return render_template('index.html',prediction_text="Cat")
        else:
            print("dog")
            return render_template('index.html',prediction_text="Dog")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

