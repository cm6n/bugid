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
        """Save trained model to disk as TensorFlow Lite model.
        
        Note: The saved model requires the TensorFlow Lite Flex delegate
        to be linked when loading the model for inference, as it uses
        TensorFlow ops (BiasAdd, MatMul, Relu, Softmax) that aren't
        natively supported in TFLite.
        """
        if not hasattr(self, 'model'):
            raise RuntimeError("No trained model to save")
        
        # Save label encoder classes
        model_dir = os.path.dirname(path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save the label encoder classes
        label_encoder_path = os.path.splitext(path)[0] + '_classes.npy'
        np.save(label_encoder_path, self.label_encoder.classes_)

        # Convert the Keras model to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        # Enable TF Select to support operations not natively available in TFLite
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops
        ]
        # Optimize for size
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save the model
        with open(path, 'wb') as f:
            f.write(tflite_model)
        
        # Create a README file with instructions for loading the model
        readme_path = os.path.splitext(path)[0] + '_README.txt'
        with open(readme_path, 'w') as f:
            f.write("""
TensorFlow Lite Model with Flex Ops
===================================

This model uses TensorFlow operations that aren't natively supported in TFLite.
When loading this model for inference, you need to ensure the TensorFlow Lite
Flex delegate is linked.

In Python, you can load the model with:

```python
import tensorflow as tf

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(
    model_path="path/to/model.tflite",
    experimental_delegates=None  # Flex delegate is automatically loaded when needed
)
interpreter.allocate_tensors()
```

For Android/iOS or other platforms, see:
https://www.tensorflow.org/lite/guide/ops_select
""")

    
    def load_model(self, path: str):
        """Load trained model from disk.
        
        This method can load either:
        1. A Keras model (.h5 or SavedModel format)
        2. A TensorFlow Lite model (.tflite)
        
        For TFLite models with Flex ops, it ensures the Flex delegate is properly linked.
        """
        # Check if this is a TFLite model
        if path.endswith('.tflite'):
            # Load TFLite model with Flex delegate support
            try:
                # Create the interpreter with automatic Flex delegate loading
                self.interpreter = tf.lite.Interpreter(
                    model_path=path,
                    experimental_delegates=None  # Flex delegate is automatically loaded when needed
                )
                self.interpreter.allocate_tensors()
                
                # Get input and output details
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                
                # Create a wrapper function to mimic the Keras model predict API
                def predict_fn(x):
                    # Ensure input is the right shape and type
                    input_shape = self.input_details[0]['shape']
                    if len(x.shape) == 1:
                        x = np.expand_dims(x, axis=0)  # Add batch dimension
                    
                    # Set the input tensor
                    self.interpreter.set_tensor(self.input_details[0]['index'], x.astype(np.float32))
                    
                    # Run inference
                    self.interpreter.invoke()
                    
                    # Get the output tensor
                    output = self.interpreter.get_tensor(self.output_details[0]['index'])
                    return output
                
                # Create a simple object with a predict method to mimic Keras model API
                class ModelWrapper:
                    def __init__(self, predict_function):
                        self.predict = predict_function
                
                self.model = ModelWrapper(predict_fn)
                
            except Exception as e:
                raise RuntimeError(f"Error loading TFLite model: {str(e)}\n"
                                  f"Note: This model requires the TensorFlow Lite Flex delegate to be linked.")
        else:
            # Load regular Keras model
            self.model = tf.keras.models.load_model(path)
        
        # Load the label encoder classes
        label_encoder_path = os.path.splitext(path)[0] + '_classes.npy'
        if os.path.exists(label_encoder_path):
            self.label_encoder.classes_ = np.load(label_encoder_path)
        else:
            # For TFLite models, we can't infer classes from model structure
            # For Keras models, we can try to infer from the output layer
            if not path.endswith('.tflite') and hasattr(self.model, 'layers'):
                print(f"Warning: Label encoder classes file not found: {label_encoder_path}")
                print("Attempting to infer classes from model structure...")
                
                # Get the number of output classes from the model's output layer
                output_layer = self.model.layers[-1]
                num_classes = output_layer.units  # For Dense layer, units attribute gives the output dimension
                
                # Create generic class names (Class_0, Class_1, etc.)
                generic_classes = np.array([f"Class_{i}" for i in range(num_classes)])
                self.label_encoder.classes_ = generic_classes
                
                print(f"Inferred {num_classes} classes: {', '.join(generic_classes)}")
                print("Note: These are generic class names. For accurate class names, retrain the model with the updated code.")
            else:
                raise RuntimeError(f"Label encoder classes file not found: {label_encoder_path}")
