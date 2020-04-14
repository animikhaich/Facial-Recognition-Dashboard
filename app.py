from flask import Flask, redirect, jsonify, request, send_file, url_for, render_template
from utils.mtcnn import MTCNN
from PIL import Image
from datetime import datetime
import numpy as np
import zipfile
import hashlib
import sys
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Some Hardcoded Values
ROOT_DATA_DIR = 'data'
mtcnn = MTCNN()

# Flask Config
app = Flask(__name__)
app.config['DEBUG'] = True


@app.route('/', methods=['GET'])
def root():
    return redirect(url_for('home'))


@app.route('/home', methods=['POST'])
def home():
    return render_template('home.html')


@app.route('/upload', methods=['POST'])
def upload():
    # List of Uploaded files
    uploaded_files = request.files.getlist("files")
    faces_list = list()
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert('RGB')
        image = np.asarray(image)
        detected_faces = mtcnn.detect_faces(image)
        faces_list.append(detected_faces)
    return jsonify(faces_list)


@app.route('/generate_data')
def gen_data_redirect():
    return redirect(url_for('upload'))


if __name__ == '__main__':
    app.run(host='localhost', port='5000', use_reloader=False, threaded=False)
