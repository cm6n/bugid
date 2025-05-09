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
        self.model = None
        
    
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
            tf.keras.layers.Input(shape=(43,)),  # 43 features from AudioProcessor (40 MFCCs + 3 spectral features)
            tf.keras.layers.Reshape((43, 1)),  # Reshape 2D input (batch_size, 43) to 3D (batch_size, 43, 1)
            tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax', dtype=tf.float32)  # Using float32 for better compatibility
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
            batch_size=4,
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
        TensorFlow ops that aren't natively supported in TFLite.
        """
        if not hasattr(self, 'model'):
            raise RuntimeError("No trained model to save")
        
        # Save label encoder classes
        model_dir = os.path.dirname(path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save the label encoder classes as text file with one class per line
        label_encoder_path = os.path.splitext(path)[0] + '_classes.txt'
        np.savetxt(label_encoder_path, self.label_encoder.classes_, fmt='%s')

        # Convert the Keras model to TensorFlow Lite with improved compatibility
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Configure for better compatibility with Android TFLite
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops
        ]
        
        # Set additional options for better compatibility
        converter.allow_custom_ops = True
        converter.experimental_new_converter = True
        
        # Force downgrade to TFLite 2.15.0 compatibility
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # Critical: Set the minimum TFLite version to ensure compatibility
        # This ensures ops like FULLY_CONNECTED use versions compatible with TFLite 2.15.0
        if hasattr(converter, 'target_spec') and hasattr(converter.target_spec, 'set_tflite_version'):
            converter.target_spec.set_tflite_version(2, 15, 0)
        
        # Use minimal optimizations to avoid quantization issues
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Commented out to avoid quantization issues
        
        # Convert model
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
        
        # Load the label encoder classes from text file
        label_encoder_path = os.path.splitext(path)[0] + '_classes.txt'
        if os.path.exists(label_encoder_path):
            self.label_encoder.classes_ = np.loadtxt(label_encoder_path, dtype=str, delimiter=',')
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


# Add a main function to allow direct execution of this script for retraining
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Retrain bug sound classification model with TFLite 2.15.0 compatibility')
    parser.add_argument('--output', type=str, default='animal_sound_model.tflite', 
                        help='Output path for the TFLite model')
    args = parser.parse_args()
    
    print("Creating dummy training data for model retraining...")
    # Create dummy data with 8 classes (matching NUM_CLASSES in MainActivity.kt)
    num_samples = 100
    num_features = 43  # Match the input shape in the model (40 MFCCs + 3 spectral features)
    
    # Generate random features
    X = np.random.rand(num_samples, num_features).astype(np.float32)
    
    # Generate random labels (animal names matching getAnimalName in MainActivity.kt)
    animals = ["Dog", "Cat", "Bird", "Fox", "Wolf", "Frog", "Cricket", "Owl"]
    y = np.random.choice(animals, num_samples)
    
    print(f"Created dummy dataset with {num_samples} samples, {num_features} features, and {len(animals)} classes")
    
    # Initialize and train the classifier
    print("Training model...")
    classifier = BugClassifier()
    classifier.train(X, y)
    
    # Save the model with TFLite 2.15.0 compatibility
    print(f"Saving TensorFlow Lite model to {args.output}")
    classifier.save_model(args.output)
    print("Model saved successfully with TFLite 2.15.0 compatibility")
    
    # Print instructions for deploying to Android
    print("\nNext steps:")
    print("1. Copy the generated TFLite model to your Android project's assets folder:")
    print(f"   cp {args.output} /path/to/AndroidStudioProjects/bugid/app/src/main/assets/")
    print("2. Rebuild and run your Android app")
