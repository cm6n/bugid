import unittest
import numpy as np
from training.audio import AudioProcessor
import os

class TestRealAudioFeatures(unittest.TestCase):
    def setUp(self):
        self.processor = AudioProcessor(sample_rate=44100)
        # Path to the real audio file
        self.audio_file_path = os.path.join(
            os.path.dirname(__file__), 
            'audio_samples', 
            '492bws.wav'
        )
        
    def test_extract_features_from_real_audio(self):
        """Test feature extraction from a real audio file."""
        # Verify the file exists
        self.assertTrue(os.path.exists(self.audio_file_path), 
                        f"Audio file not found: {self.audio_file_path}")
        
        # Load the audio file
        signal, sr = self.processor.load_audio(self.audio_file_path)
        
        # Print basic audio information
        print(f"\nAudio file: {self.audio_file_path}")
        print(f"Sample rate: {sr} Hz")
        print(f"Duration: {len(signal) / sr:.2f} seconds")
        print(f"Number of samples: {len(signal)}")
        
        # Extract features
        features = self.processor.extract_features(signal)
        
        # Print all features with descriptive labels
        print("\nExtracted Audio Features:")
        print("-" * 40)
        
        # Print MFCC values (first 40 features)
        for i in range(40):
            print(f"MFCC {i+1}: {features[i]}")
        
        # Print spectral features (last 3 features)
        print(f"Spectral Centroid: {features[40]}")
        print(f"Spectral Bandwidth: {features[41]}")
        print(f"Spectral Rolloff: {features[42]}")
        
        # Verify feature vector properties
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 43)
        
        # Check feature values are finite
        self.assertTrue(np.all(np.isfinite(features)))
        
        # Process the file directly
        direct_features = self.processor.process_file(self.audio_file_path)
        
        # Print differences between direct processing and manual extraction
        print("\nDifferences between direct processing and manual extraction:")
        for i, (f1, f2) in enumerate(zip(features, direct_features)):
            diff = abs(f1 - f2)
            feature_name = ""
            if i < 40:
                feature_name = f"MFCC {i+1}"
            elif i == 40:
                feature_name = "Spectral Centroid"
            elif i == 41:
                feature_name = "Spectral Bandwidth"
            elif i == 42:
                feature_name = "Spectral Rolloff"
            
            print(f"{feature_name} diff: {diff}")
        
        # Verify both methods produce the same features (within tolerance)
        self.assertTrue(
            np.allclose(features, direct_features, rtol=1e-5, atol=1e-5),
            "Features from direct extraction and process_file don't match"
        )

if __name__ == '__main__':
    unittest.main()
