import unittest
import numpy as np
import os
import tempfile
import tensorflow as tf
from training.model import BugClassifier

class TestBugClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = BugClassifier()
        
        # Create synthetic dataset
        np.random.seed(42)
        self.X = np.random.rand(100, 16)  # 100 samples, 16 features
        self.y = np.array(['cricket', 'beetle'] * 50)  # Binary classification
        
    def test_train_and_predict(self):
        """Test model training and prediction."""
        # Train model
        report = self.classifier.train(self.X, self.y, test_size=0.2)
        
        # Check report structure
        self.assertIsInstance(report, dict)
        self.assertIn('accuracy', report)
        self.assertIn('weighted avg', report)
        
        # Test prediction
        X_test = np.random.rand(5, 16)
        predictions = self.classifier.predict(X_test)
        
        # Check predictions
        self.assertEqual(len(predictions), 5)
        for pred in predictions:
            self.assertIn(pred, ['cricket', 'beetle'])
            
        # Test prediction probabilities
        probabilities = self.classifier.predict_proba(X_test)
        self.assertEqual(probabilities.shape, (5, 2))  # 5 samples, 2 classes
        np.testing.assert_array_almost_equal(
            np.sum(probabilities, axis=1),
            np.ones(5)
        )
    
    def test_save_load_model(self):
        """Test model serialization."""
        # Train model
        self.classifier.train(self.X, self.y)
        
        # Save and load model
        with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as tmp:
            model_path = tmp.name
            
        try:
            # Save model
            self.classifier.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Check that classes file was also created
            classes_path = os.path.splitext(model_path)[0] + '_classes.txt'
            self.assertTrue(os.path.exists(classes_path))
            
            # Load model in new instance
            new_classifier = BugClassifier(model_path=model_path)
            
            # Compare predictions
            X_test = np.random.rand(5, 16)
            original_preds = self.classifier.predict(X_test)
            loaded_preds = new_classifier.predict(X_test)
            np.testing.assert_array_equal(original_preds, loaded_preds)
            
        finally:
            # Cleanup
            if os.path.exists(model_path):
                os.remove(model_path)
            classes_path = os.path.splitext(model_path)[0] + '_classes.txt'
            if os.path.exists(classes_path):
                os.remove(classes_path)
    
    def test_predict_without_model(self):
        """Test prediction without a trained model."""
        classifier = BugClassifier()
        X_test = np.random.rand(5, 16)
        
        with self.assertRaises(RuntimeError):
            classifier.predict(X_test)
            
if __name__ == '__main__':
    unittest.main()
