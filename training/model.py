import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from typing import Tuple, List, Optional
import os

class BugClassifier:
    """Handles training and prediction for bug sound classification."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> dict:
        """Train the classifier and return performance metrics.
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Proportion of dataset to use for testing
            
        Returns:
            dict: Classification report
        """
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return report
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict bug classes for new audio features."""
        if not hasattr(self, 'model'):
            raise RuntimeError("Model not trained or loaded")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities for bug classes."""
        if not hasattr(self, 'model'):
            raise RuntimeError("Model not trained or loaded")
        return self.model.predict_proba(X)
    
    def save_model(self, path: str):
        """Save trained model to disk."""
        if not hasattr(self, 'model'):
            raise RuntimeError("No trained model to save")
        joblib.dump(self.model, path)
    
    def load_model(self, path: str):
        """Load trained model from disk."""
        self.model = joblib.load(path)
