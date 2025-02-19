import unittest
import numpy as np
from cli_tool.audio import AudioProcessor
import os
import librosa

class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = AudioProcessor()
        
    def test_extract_features(self):
        """Test feature extraction from a synthetic signal."""
        # Create a synthetic signal (1 second of 440Hz sine wave)
        duration = 1.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Extract features
        features = self.processor.extract_features(signal)
        
        # Check feature vector properties
        self.assertIsInstance(features, np.ndarray)
        # Expected length: 13 MFCCs + centroid + bandwidth + rolloff = 16
        self.assertEqual(len(features), 16)
        
        # Check feature values are finite
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_process_files_empty_list(self):
        """Test processing an empty list of files."""
        features = self.processor.process_files([])
        self.assertEqual(features.shape[0], 0)
        
if __name__ == '__main__':
    unittest.main()
