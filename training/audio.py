import librosa
import numpy as np
from typing import Tuple, List

class AudioProcessor:
    """Handles audio file processing and feature extraction."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return signal and sample rate."""
        try:
            signal, sr = librosa.load(file_path, sr=self.sample_rate)
            return signal, sr
        except Exception as e:
            raise RuntimeError(f"Error loading audio file: {str(e)}")
    
    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract relevant audio features for bug identification.
        
        Features extracted:
        - MFCCs (Mel-frequency cepstral coefficients)
        - Spectral centroid
        - Spectral bandwidth
        - Spectral rolloff
        """
        features = []
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=signal, sr=self.sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        features.extend(mfccs_mean)
        
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=signal, sr=self.sample_rate)
        features.append(np.mean(centroid))
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=self.sample_rate)
        features.append(np.mean(bandwidth))
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=signal, sr=self.sample_rate)
        features.append(np.mean(rolloff))
        
        return np.array(features)
    
    def process_file(self, file_path: str) -> np.ndarray:
        """Process a single audio file and extract features."""
        signal, _ = self.load_audio(file_path)
        return self.extract_features(signal)
    
    def process_files(self, file_paths: List[str]) -> np.ndarray:
        """Process multiple audio files and return feature matrix."""
        features_list = []
        for file_path in file_paths:
            features = self.process_file(file_path)
            features_list.append(features)
        return np.array(features_list)
