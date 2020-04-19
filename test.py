from flask import Flask, redirect, jsonify, request, send_file, url_for, render_template
from utils.utils import *
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import numpy as np
import zipfile
import hashlib
import shutil
import sys
import os

# Some Hardcoded Values
ROOT_DATA_DIR = 'static/data'
CROPPED_FACES_DIR = os.path.join(ROOT_DATA_DIR, 'cropped_faces')
RECOGNIZED_FACES_DIR = os.path.join(ROOT_DATA_DIR, 'recognized_faces')

# Flask Config
app = Flask(__name__)
app.config['DEBUG'] = True


@app.route('/', methods=['GET'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if request.form['next_button'] == "Let's Begin!":
            return redirect('upload_faces')

    return render_template('home.html')


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html', title="About")


@app.route('/upload_faces', methods=['POST', 'GET'])
def upload_faces():
    # Display Page
    if request.method == 'GET':
        return render_template('upload_faces.html', title="Upload")

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

    faces_list = list()
    return jsonify(faces_list)


@app.route('/label_faces', methods=['POST', 'GET'])
def label_faces():
    # Display Page
    if request.method == 'GET':
        return render_template('label_faces.html', title="Label")

    # Get Form Data
    name_face_map = request.json

    face_list = list()
    label_list = list()
    for image_path, label in name_face_map.items():
        image = Image.open(image_path[1:]).convert('RGB')

        # Create the List to pass to FR Module
        face_list.append(image)
        label_list.append(label)

    return redirect('upload_pictures')


@app.route('/upload_pictures', methods=['POST', 'GET'])
def upload_pictures():

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
    uploaded_files = request.files.getlist("files")

    # Detect Faces for each Uploaded Image
    recognized_image_list = list()

    return jsonify(recognized_image_list)


if __name__ == '__main__':
    app.run(
        host='localhost',
        port='5000',
        use_reloader=True,
        threaded=True
    )