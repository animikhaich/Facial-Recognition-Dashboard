from flask import Flask, redirect, jsonify, request, send_file, url_for, render_template
from utils.mtcnn import MTCNN
from utils.utils import *
from PIL import Image
from datetime import datetime
import numpy as np
import zipfile
import hashlib
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Some Hardcoded Values
ROOT_DATA_DIR = 'static/data'
CROPPED_FACES_DIR = os.path.join(ROOT_DATA_DIR, 'cropped_faces')
RECOGNIZED_FACES_DIR = os.path.join(ROOT_DATA_DIR, 'recognized_faces')
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


@app.route('/upload', methods=['POST', 'GET'])
def upload_and_detect():
    # Handle Uploaded Files
    if request.method == 'POST':
        # Making sure Folder exists
        if not os.path.isdir(CROPPED_FACES_DIR):
            os.makedirs(CROPPED_FACES_DIR)

        # Get uploaded files
        uploaded_files = request.files.getlist("files")

        face_num = 0

        # Detect Faces for each Uploaded Image
        faces_list = list()
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert('RGB')
            image = np.asarray(image)

            # Detect Faces using MTCNN
            detected_faces = mtcnn.detect_faces(image)

            height, width, _ = image.shape
            for detected_face in detected_faces:
                x1, y1, x2, y2 = fix_coordinates(
                    detected_face['box'], width, height
                )
                cropped_face = image[y1:y2, x1:x2]
                try:
                    cropped_face = Image.fromarray(cropped_face)
                except:
                    continue
                saved_image_path = os.path.join(
                    CROPPED_FACES_DIR,
                    str(face_num) + uploaded_file.filename
                )
                cropped_face.save(saved_image_path)
                faces_list.append('/' + saved_image_path)
                face_num += 1

        return jsonify(faces_list)

    # Display Page
    if request.method == 'GET':
        return render_template('upload.html')


@app.route('/generate_data')
def gen_data_redirect():
    return redirect(url_for('upload'))


if __name__ == '__main__':
    app.run(host='localhost', port='5000', use_reloader=False, threaded=False)
