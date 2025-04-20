
import numpy as np
import joblib
import tensorflow as tf
import os

def load_model():
    """Loads the trained model"""
    model_path = 'models/plant_classifier.h5'
    encoder_path = 'models/label_encoder.joblib'

    if os.path.exists(model_path) and os.path.exists(encoder_path):
        print("Loading trained model...")
        model = tf.keras.models.load_model(model_path)
        label_encoder = joblib.load(encoder_path)
        return model, label_encoder
    else:
        raise FileNotFoundError("No trained model found. Please train the model first.")

def predict(model_tuple, processed_image):
    """Makes a prediction with improved confidence handling"""
    model, label_encoder = model_tuple

    # Add batch dimension
    image_batch = np.expand_dims(processed_image, 0)

    # Get prediction probabilities
    probabilities = model.predict(image_batch)[0]
    max_prob = np.max(probabilities)

    # Lower the confidence threshold since we have a well-trained model
    confidence_threshold = 0.45

    if max_prob >= confidence_threshold:
        prediction = label_encoder.inverse_transform([np.argmax(probabilities)])[0]
        confidence = max_prob * 100
    else:
        prediction = "Unknown"
        confidence = 0.0

    return prediction, confidence
