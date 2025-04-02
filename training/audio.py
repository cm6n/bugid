import librosa
import numpy as np
from typing import Tuple, List

class AudioProcessor:
    """Handles audio file processing and feature extraction."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return signal and sample rate."""
        try:
            signal, sr = librosa.load(file_path, sr=self.sample_rate)
            return signal, sr
        except Exception as e:
            raise RuntimeError(f"Error loading audio file: {str(e)}")
    
    def custom_spectral_bandwidth(self, signal: np.ndarray) -> float:
        """Calculate spectral bandwidth using a custom implementation.
        
        This implementation ensures consistency between direct signals and file-loaded signals.
        """
        # Use fixed parameters for consistency
        n_fft = 2048
        hop_length = 512
        
        # Calculate the magnitude spectrum
        D = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
        
        # Calculate power spectrum
        S = D**2
        
        # Calculate frequencies for each FFT bin
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)
        
        # Calculate spectral centroid for each frame
        centroids = np.sum(freqs.reshape(-1, 1) * S, axis=0) / (np.sum(S, axis=0) + 1e-8)
        
        # Calculate spectral bandwidth for each frame
        deviation = np.sqrt(np.sum(((freqs.reshape(-1, 1) - centroids)**2) * S, axis=0) / (np.sum(S, axis=0) + 1e-8))
        
        # Return mean bandwidth across frames
        return np.mean(deviation)
    
    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract relevant audio features for bug identification.
        
        Features extracted:
        - MFCCs (Mel-frequency cepstral coefficients)
        - Spectral centroid
        - Spectral bandwidth
        - Spectral rolloff
        """
        features = []
        
        # Normalize the signal to ensure consistent feature extraction
        signal = librosa.util.normalize(signal)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=signal, sr=self.sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        features.extend(mfccs_mean)
        
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=signal, sr=self.sample_rate)
        features.append(np.mean(centroid))
        
        # Spectral bandwidth using custom implementation
        bandwidth = self.custom_spectral_bandwidth(signal)
        features.append(bandwidth)
        
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
