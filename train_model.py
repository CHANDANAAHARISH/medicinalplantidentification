
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import joblib
from image_processor import preprocess_image

def load_training_data(data_dir):
    """Load training images and labels from data directory"""
    images = []
    labels = []

    print("Loading training data...")
    for plant_class in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, plant_class)
        if not os.path.isdir(class_dir):
            continue

        print(f"Processing {plant_class} images...")
        class_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for i, image_file in enumerate(class_files):
            if i % 100 == 0:
                print(f"Processed {i}/{len(class_files)} {plant_class} images")

            image_path = os.path.join(class_dir, image_file)
            try:
                image = cv2.imread(image_path)
                if image is None:
                    continue

                processed_image = preprocess_image(image)
                images.append(processed_image)
                labels.append(plant_class)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

    return np.array(images), np.array(labels)

def create_efficientnet_model(num_classes):
    """Create EfficientNet model with enhanced architecture for better recall"""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    
    # Freeze early layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_model(data_dir='data/training'):
    """Train the plant identification model using EfficientNet"""
    print("Loading training data...")
    X, y = load_training_data(data_dir)

    if len(X) == 0:
        raise ValueError("No training data found! Please add some images first.")

    if len(np.unique(y)) < 2:
        raise ValueError("Need images from at least 2 different plant categories for training.")

    print(f"Loaded {len(X)} images from {len(np.unique(y))} classes")

    # Convert labels to categorical
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = tf.keras.utils.to_categorical(y_encoded)

    # Split with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Create and compile model
    model = create_efficientnet_model(len(np.unique(y)))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model with recall metric
    print("Training model with EfficientNet...")
    recall = tf.keras.metrics.Recall()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', recall]
    )
    
    # Use class weights to handle imbalanced data
    class_weights = {i: len(y_train) / (len(np.unique(y_encoded)) * np.sum(y_encoded == i)) 
                    for i in range(len(np.unique(y_encoded)))}
    
    # Use early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_recall',
        patience=5,
        mode='max',
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=30,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[early_stopping]
    )
    
    # Calculate final scores
    _, accuracy, recall_score = model.evaluate(X_val, y_val)
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall Score: {recall_score:.2f}")
    
    # Save metrics for reference
    with open('models/metrics.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"Recall Score: {recall_score:.2f}\n")

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/plant_classifier.h5')
    joblib.dump(label_encoder, 'models/label_encoder.joblib')
    print("Model saved to models/plant_classifier.h5")

    return model, label_encoder

if __name__ == "__main__":
    train_model()
