from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import load_img
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Set environment variable to make sure that TF does not eat up all GPU memory
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class VGGFaceRecognizer:
    """
    Easy Face Recognizer Class, built on top of RC Mali's VGG Face Implementation
    URL (Keras VGG Face): https://github.com/rcmalli/keras-vggface
    License: MIT

    Author: Animikh Aich
    GitHub: https://github.com/animikhaich
    LinkedIn: https://www.linkedin.com/in/animikh-aich/
    """

    def __init__(self, model='senet50'):
        """
        __init__ Model Initializer

        - Sets the flags for GPU memory growth (if available)
        - Selects and Initializes the Model

        Args:
            model (str, optional): Choose Base Feature Extractor: VGG16, ResNet50, and SeNet50. Defaults to 'senet50'.
        """

        # Allow Memory Growth for Tensorflow GPU
        if tf.test.is_gpu_available():
            # For TF 1x
            if int(tf.__version__.split('.')[0]) == 1:
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                session = tf.Session(config=config)
                set_session(session)

            # For TF 2x
            else:
                gpu_devices = tf.config.experimental.list_physical_devices(
                    'GPU'
                )
                tf.config.experimental.set_memory_growth(gpu_devices[0], True)

        # Some Instance Variables
        self.model = model
        self.face_dict = dict()

        # initialize VGG Face using VGGFace Model
        self.recognizer = VGGFace(include_top=False, model=model)

    def calculate_similarity(self, vector_1, vector_2):
        """
        calculate_similarity

        Calculate the Cosine Similarity of Similarity Feature Vectors
        Flattens the Vectors for VGG16 base due to higher high dimension vector

        Args:
            vector_1 (numpy array): Feature Vector of Face 1
            vector_2 (numpy array): Feature Vector of Face 2

        Returns:
            float: Similarity Score, Lower means more similar
        """
        # VGG creates higher dimensional vector, and hence needs to be flattened
        if self.model == 'vgg16':
            vector_1 = vector_1.flatten()
            vector_2 = vector_2.flatten()

        # Remove Extra Dimension
        vector_1 = np.squeeze(vector_1)
        vector_2 = np.squeeze(vector_2)

        # Calculate the cosine similarity
        return cosine(vector_1, vector_2)

    def register_faces(self, face_list, name_list):
        """
        register_faces 

        Given a list of faces and names, register them and save them to the database
        These are the faces that a query face will be compared against.
        Every time this is called, it clear the existing database

        Args:
            face_list (list): List of PIL Face Images
            name_list (list): List of Strings containing the name of each face - Unique

        Returns:
            dict: Dictionary containing the name of the face and it's corresponding feature vector
        """
        self.face_dict = dict()
        for face, name in zip(face_list, name_list):
            self.face_dict[name] = self.feature_extractor(face)

        return self.face_dict

    def add_faces(self, face_list, name_list):
        """
        add_faces 

        Given a list of faces and names, register them and save them to the database
        These are the faces that a query face will be compared against.
        Every time this is called, it appends to existing faces in the DB

        Args:
            face_list (list): List of PIL Face Images
            name_list (list): List of Strings containing the name of each face - Unique

        Returns:
            dict: Dictionary containing the name of the face and it's corresponding feature vector
        """
        for face, name in zip(face_list, name_list):
            self.face_dict[name] = self.feature_extractor(face)

        return self.face_dict

    def feature_extractor(self, face):
        """
        feature_extractor 

        Primary CNN based feature extractor for faces. Uses pre-trained model by RC Mali for the same.
        You can choose the architecture while creating the instance of the class

        Args:
            face (PIL Image): PIL image of a cropped out face

        Returns:
            list: Returns the feature vector that is extracted from the given face
        """
        # Resize the face, face is a PIL image
        face = face.resize((224, 224), Image.ANTIALIAS)
        face = np.asarray(face).astype(np.float)
        face = np.expand_dims(face, axis=0)

        if self.model == 'vgg16':
            face = preprocess_input(face, version=1)
        else:
            face = preprocess_input(face, version=2)

        return self.recognizer.predict(face)

    def recognize(self, face, thresh=0.25):
        """
        recognize 

        Given a Query Face, this function compares it against the existing faces and returns a match, if any

        Args:
            face (PIL Image): Cropped face in the form of a PIL image
            thresh (float, optional): Cutoff Threshold used to discard unmatched faces. Defaults to 0.25.

        Returns:
            string: Name of the matched face, if recognized, else None
        """
        # Extract features of the Query Face
        query_features = self.feature_extractor(face)
        temp_sim_dict = dict()

        # Compare the Query Face features with the existing database faces
        for key in self.face_dict.keys():
            db_face_features = self.face_dict[key]
            score = self.calculate_similarity(
                db_face_features, query_features
            )
            temp_sim_dict[key] = score

        try:
            # Return None if no face matches
            if min(temp_sim_dict.values()) > thresh:
                return None
        except:
            return None

        # Find most similar face
        most_similar_face = min(temp_sim_dict, key=temp_sim_dict.get)

        return most_similar_face
