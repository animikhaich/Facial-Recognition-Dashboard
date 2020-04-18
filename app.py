from flask import Flask, redirect, jsonify, request, send_file, url_for, render_template
from utils.mtcnn import MTCNN
from utils.utils import *
from utils.vgg_face import VGGFaceRecognizer
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import numpy as np
import zipfile
import hashlib
import shutil
import sys
import os

# Configure Environment for GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Some Hardcoded Values
ROOT_DATA_DIR = 'static/data'
CROPPED_FACES_DIR = os.path.join(ROOT_DATA_DIR, 'cropped_faces')
RECOGNIZED_FACES_DIR = os.path.join(ROOT_DATA_DIR, 'recognized_faces')
name_face_map = dict()
mtcnn = MTCNN()
face_recognizer = VGGFaceRecognizer(model='senet50')

# Flask Config
app = Flask(__name__)
app.config['DEBUG'] = True


@app.route('/', methods=['GET'])
@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/upload_faces', methods=['POST', 'GET'])
def upload_faces():
    # Display Page
    if request.method == 'GET':
        return render_template('upload_faces.html')

    # Delete Existing Images
    try:
        shutil.rmtree(CROPPED_FACES_DIR)
    except:
        pass

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


@app.route('/label_faces', methods=['POST', 'GET'])
def label_faces():
    global name_face_map

    # Display Page
    if request.method == 'GET':
        return render_template('label_faces.html')

    # Get Form Data
    name_face_map = request.json

    face_list = list()
    label_list = list()
    for image_path, label in name_face_map.items():
        image = Image.open(image_path[1:]).convert('RGB')

        # Create the List to pass to FR Module
        face_list.append(image)
        label_list.append(label)

    # Register Faces - Extract and store ground truth Facial Features
    face_recognizer.register_faces(face_list, label_list)

    return url_for('upload_pictures')


@app.route('/upload_pictures', methods=['POST', 'GET'])
def upload_pictures():

    # Display Page
    if request.method == 'GET':
        return render_template('upload_pictures.html')

    # Delete Existing Images
    try:
        shutil.rmtree(RECOGNIZED_FACES_DIR)
    except:
        pass

    # Making sure Folder exists
    if not os.path.isdir(RECOGNIZED_FACES_DIR):
        os.makedirs(RECOGNIZED_FACES_DIR)

    # Get uploaded files
    uploaded_files = request.files.getlist("files")

    # Detect Faces for each Uploaded Image
    recognized_image_list = list()
    for uploaded_file in uploaded_files:
        image_PIL = Image.open(uploaded_file).convert('RGB')
        image = np.asarray(image_PIL)

        # Detect Faces using MTCNN
        detected_faces = mtcnn.detect_faces(image)

        height, width, _ = image.shape

        # TODO: From here down
        for detected_face in detected_faces:
            x1, y1, x2, y2 = fix_coordinates(
                detected_face['box'], width, height
            )
            cropped_face = image[y1:y2, x1:x2]
            try:
                cropped_face = Image.fromarray(cropped_face)
            except:
                continue

            face_name = face_recognizer.recognize(cropped_face, thresh=0.25)
            if face_name:
                draw = ImageDraw.Draw(image_PIL)
                draw.rectangle((x1, y1, x2, y2))
                text_w, text_h = draw.textsize(face_name)
                text_x = int(x1 + (x2 - x1) / 2 - text_w / 2)
                text_y = int(y2 + 0.1 * (y2 - y1))
                draw.text((text_x, text_y), face_name, fill='red')

        image_path = os.path.join(RECOGNIZED_FACES_DIR, uploaded_file.filename)
        image_PIL.save(image_path)
        recognized_image_list.append('/'+image_path)

    return jsonify(recognized_image_list)


if __name__ == '__main__':
    app.run(
        host='localhost',
        port='5000',
        use_reloader=False,
        threaded=False
    )
