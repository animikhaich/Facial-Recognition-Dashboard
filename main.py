from flask import Flask, redirect, jsonify, request, send_file, url_for, render_template
from keras.backend.tensorflow_backend import set_session
from utils.mtcnn import MTCNN
from utils.utils import *
from utils.vgg_face import VGGFaceRecognizer
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from datetime import datetime
import numpy as np
import zipfile
import hashlib
import shutil
import sys
import os
import time

# Configure Environment for GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

# Some Hardcoded Values
ROOT_DATA_DIR = 'static/data'
CROPPED_FACES_DIR = os.path.join(ROOT_DATA_DIR, 'cropped_faces')
RECOGNIZED_FACES_DIR = os.path.join(ROOT_DATA_DIR, 'recognized_faces')
mtcnn = MTCNN()
face_recognizer = VGGFaceRecognizer(model='senet50')

# Flask Config
app = Flask(__name__)
app.config['DEBUG'] = False

# In App Variables
content = list()


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('home.html')

    return redirect(url_for('upload_faces'))


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html', title="About")


@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html', title="Contact")


@app.route('/upload_faces', methods=['POST', 'GET'])
def upload_faces():
    global content, graph
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
    uploaded_files = request.files.getlist("file")

    face_id = 0

    # Detect Faces for each Uploaded Image
    content = list()
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert('RGB')
        image = np.asarray(image)

        # Detect Faces using MTCNN
        with graph.as_default():
            set_session(sess)
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
                rem_punctuation(f"{face_id}_{uploaded_file.filename}")
            )
            cropped_face.save(saved_image_path)
            content.append(
                {
                    'id': face_id,
                    'image': f"/{saved_image_path}"
                }
            )
            face_id += 1

    return redirect(url_for('label_faces'))


@app.route('/label_faces', methods=['POST', 'GET'])
def label_faces():
    global content, graph

    # Display Page
    if request.method == 'GET':
        return render_template('label_faces.html', title="Label", table_contents=content)

    face_list = list()
    label_list = list()
    for element in content:
        face_id = element.get('id')
        image_path = element.get('image')
        label = request.form.get(f"face-name-{face_id}")
        print(face_id, image_path, label)

        if not label:
            continue

        # Create the List to pass to FR Module
        image = Image.open(image_path[1:]).convert('RGB')
        face_list.append(image)
        label_list.append(label)

    # Register Faces - Extract and store ground truth Facial Features
    with graph.as_default():
        set_session(sess)
        face_recognizer.register_faces(face_list, label_list)

    return redirect(url_for('upload_pictures'))


@app.route('/upload_pictures', methods=['POST', 'GET'])
def upload_pictures():
    global graph

    # Display Page
    if request.method == 'GET':
        return render_template('upload_pictures.html', title="Upload")

    # Delete Existing Images
    try:
        shutil.rmtree(RECOGNIZED_FACES_DIR)
    except:
        pass

    # Making sure Folder exists
    if not os.path.isdir(RECOGNIZED_FACES_DIR):
        os.makedirs(RECOGNIZED_FACES_DIR)

    # Get uploaded files
    uploaded_files = request.files.getlist("file")

    # Detect Faces for each Uploaded Image
    recognized_image_list = list()
    for uploaded_file in uploaded_files:
        image_PIL = Image.open(uploaded_file).convert('RGB')
        image = np.asarray(image_PIL)

        # Detect Faces using MTCNN
        with graph.as_default():
            set_session(sess)
            detected_faces = mtcnn.detect_faces(image)

        height, width, _ = image.shape
        font_size = int(np.mean([width, height]) * 0.03)

        # TODO: From here down
        draw = ImageDraw.Draw(image_PIL)
        for detected_face in detected_faces:
            x1, y1, x2, y2 = fix_coordinates(
                detected_face['box'], width, height
            )
            cropped_face = image[y1:y2, x1:x2]
            try:
                cropped_face = Image.fromarray(cropped_face)
            except:
                continue

            with graph.as_default():
                set_session(sess)
                face_name = face_recognizer.recognize(
                    cropped_face, thresh=0.25)
            if face_name:
                draw.rectangle((x1, y1, x2, y2))
                font = ImageFont.truetype(
                    "static/assets/fonts/Roboto-Regular.ttf", font_size)

                text_w, text_h = draw.textsize(face_name)
                text_x = int(x1)
                text_y = int(y1 - font_size)
                draw.text((text_x, text_y), face_name, fill='red', font=font)

        image_path = os.path.join(
            RECOGNIZED_FACES_DIR, rem_punctuation(
                f"{time.time()}_{uploaded_file.filename}")
        )
        image_PIL.save(image_path)
        recognized_image_list.append('/'+image_path)

    return redirect(url_for('display_results'))


@app.route('/display_results', methods=['GET'])
def display_results():
    # Get List of images in Recognized Faces Directory
    images = os.listdir(RECOGNIZED_FACES_DIR)
    image_paths = ['/'+os.path.join(RECOGNIZED_FACES_DIR, im_path).replace(' ', '%20')
                   for im_path in images]

    return render_template('display_results.html', image_paths=image_paths)


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port='5000',
        use_reloader=False,
        threaded=True
    )
