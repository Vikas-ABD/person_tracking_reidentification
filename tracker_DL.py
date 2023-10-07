import cv2
import numpy as np
from model import CustomCNNModel
import tensorflow as tf

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Store the features of detected persons along with their IDs
        self.detected_persons = {}

        # Threshold for re-identification
        self.similarity_threshold = 0.7

        # Load your pre-trained deep learning model and remove the last three layers
        # Load your pre-trained deep learning model
        self.model = CustomCNNModel()

        # Build the model
        self.model.build_model()

        # Load the pre-trained weights
        self.model.load_weights("person_classification_model.h5")  # Replace with your model weights path

        # Define a new model that excludes the last three layers
        self.feature_extractor = tf.keras.Model(
            inputs=self.model.model.input,
            outputs=self.model.model.layers[-4].output  # Remove the last three layers
        )

        # Initialize the ID count based on existing IDs
        if self.detected_persons:
            self.id_count = max(self.detected_persons.keys()) + 1
        else:
            self.id_count = 0

    def extract_features(self, image):
        # Preprocess the image for your model (resize, normalize, etc.)
        image = cv2.resize(image, (640, 640))  # Adjust the size as needed
        image = image / 255.0  # Normalize pixel values (if needed)

        # Expand dimensions to match the model input shape (batch size of 1)
        image = np.expand_dims(image, axis=0)

        # Extract features using the model
        features = self.feature_extractor.predict(image)

        return features

    def update(self, objects_rect, frame):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get features of new objects
        new_objects_features = []

        # Initialize a list for lost persons
        lost_persons = []

        for rect in objects_rect:
            x, y, w, h = rect
            person_roi = frame[y:y+h, x:x+w]

            # Extract features from the new object
            features = self.extract_features(person_roi)
            new_objects_features.append(features)

            # Check for similarities with detected persons
            matched_id = self.match_person(features)

            if matched_id is not None:
                # Assign the same ID as the matched person
                objects_bbs_ids.append([x, y, w, h, matched_id])
            else:
                # New person is detected, assign a new ID
                new_id = self.id_count  # Use the current ID count
                self.center_points[new_id] = (x + w // 2, y + h // 2)
                objects_bbs_ids.append([x, y, w, h, new_id])
                self.detected_persons[new_id] = features
                self.id_count += 1  # Increment the ID count

        # Update the last person features
        self.detected_persons = {k: v for k, v in self.detected_persons.items() if k in [bb[4] for bb in objects_bbs_ids]}

        return objects_bbs_ids

    def match_person(self, features):
        for person_id, person_features in self.detected_persons.items():
            similarity = self.compare_features(person_features, features)
            if similarity > self.similarity_threshold:
                return person_id
        return None

    def compare_features(self, features1, features2):
        # Flatten the feature arrays
        features1 = features1.flatten()
        features2 = features2.flatten()

        # Compute the dot product
        dot_product = np.dot(features1, features2)

        # Compute the norms
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)

        # Compute the similarity
        similarity = dot_product / (norm1 * norm2)

        return similarity

