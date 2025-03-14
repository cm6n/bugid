import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Optional, Dict
import os
import tensorflow as tf
from tensorflow import keras

# Ensure TensorFlow Lite converter is available
try:
    from tensorflow import lite
except ImportError:
    # For older TensorFlow versions
    from tensorflow.lite.python import lite

class BugClassifier:
    """Handles training and prediction for bug sound classification."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.label_encoder = LabelEncoder()
        
        # Create TensorFlow model
        self._create_model()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _create_model(self):
        """Create a TensorFlow model for bug classification."""
        self.model = tf.keras.Sequential([
            
            tf.keras.layers.Input(shape=(16,)),  # 16 features from AudioProcessor
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')  # Will be adjusted based on number of classes
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """Train the classifier and return performance metrics.
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Proportion of dataset to use for testing
            
        Returns:
            dict: Classification report
        """
        # Encode labels
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        y_encoded = y_encoded.astype(np.int64)
        
        # Adjust output layer to match number of classes
        num_classes = len(self.label_encoder.classes_)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(16,)),  # 16 features from AudioProcessor
            tf.keras.layers.Dense(16, activation='relu', dtype=tf.float16),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu', dtype=tf.float16),
            tf.keras.layers.Dense(num_classes, activation='softmax', dtype=tf.float16)
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Split dataset
        X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model.run_eagerly = False
        self.model.fit(
            X_train, y_train_encoded,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test_encoded),
            verbose=0,
        )
        
        # Evaluate
        y_pred_encoded = np.argmax(self.model.predict(X_test), axis=1)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        y_test = self.label_encoder.inverse_transform(y_test_encoded)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return report
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict bug classes for new audio features."""
        if not hasattr(self, 'model') or not hasattr(self, 'label_encoder'):
            raise RuntimeError("Model not trained or loaded")
        
        predictions = np.argmax(self.model.predict(X), axis=1)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities for bug classes."""
        if not hasattr(self, 'model'):
            raise RuntimeError("Model not trained or loaded")
        
        return self.model.predict(X)
    
    def save_model(self, path: str):
        """Save trained model to disk as TensorFlow Lite model."""
        if not hasattr(self, 'model'):
            raise RuntimeError("No trained model to save")
        
        # Save label encoder classes
        model_dir = os.path.dirname(path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save the Keras model
        self.model.save(path)
    
    def load_model(self, path: str):
        """Load trained model from disk."""
        self.model = tf.keras.models.load_model(path)
