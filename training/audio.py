import librosa
import numpy as np
from typing import Tuple, List

class AudioProcessor:
    """Handles audio file processing and feature extraction."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return signal and sample rate.
        
        If the audio is stereo, only the left channel will be used.
        """
        try:
            # librosa.load by default returns mono audio, but let's be explicit
            # mono=True will average all channels, but we want to use only the left channel
            # So we'll load with mono=False first to check if it's stereo
            signal_all_channels, sr = librosa.load(file_path, sr=self.sample_rate, mono=False)
            
            # Check if the audio is stereo (has 2 or more channels)
            if signal_all_channels.ndim > 1 and signal_all_channels.shape[0] >= 2:
                print(f"Audio file {file_path} has {signal_all_channels.shape[0]} channels. Using left channel only.")
                # Take only the left channel (first channel)
                signal = signal_all_channels[0]
            else:
                # If it's already mono, just use it as is
                signal = signal_all_channels
                
                # If loaded as mono but returned as 2D array with 1 row, flatten it
                if signal.ndim > 1:
                    signal = signal.flatten()
                    
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
    
    def get_mfccs(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate MFCC values using librosa.
        
        Returns MFCC coefficients to match the Android implementation.
        """
        # Use librosa to calculate MFCCs
        # Use n_fft and hop_length values that match the Android implementation
        n_fft = 2048
        hop_length = 512
        
        # Calculate MFCCs using librosa
        mfccs = librosa.feature.mfcc(
            y=signal, 
            sr=self.sample_rate, 
            n_mfcc=40,  # Get 40 coefficients to match Android
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Take the mean of each coefficient across all frames
        mfcc_means = np.mean(mfccs, axis=1)

        
        return mfcc_means
    
    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract relevant audio features for bug identification.
        
        Features extracted:
        - 40 MFCCs calculated using librosa
        - Spectral centroid
        - Spectral bandwidth
        - Spectral rolloff
        
        This implementation matches the Android implementation for consistency.
        """
        features = []
        
        # Calculate MFCCs using librosa
        mfccs = self.get_mfccs(signal)
        features.extend(mfccs)
        
        # Calculate features to match Android implementation
        
        # Spectral centroid - Android uses a weighted sum approach
        weighted_sum = 0.0
        total_energy = 0.0
        
        # Calculate zero-crossing rate, which correlates with frequency
        zero_crossings = 0
        for i in range(1, len(signal)):
            if (signal[i] > 0 and signal[i-1] <= 0) or (signal[i] <= 0 and signal[i-1] > 0):
                zero_crossings += 1
        
        # Estimate frequency based on zero crossings
        estimated_frequency = zero_crossings / len(signal) * self.sample_rate / 2
        
        # Use this as a base for our centroid calculation
        for i in range(len(signal)):
            magnitude = abs(signal[i])
            weighted_sum += i * magnitude
            total_energy += magnitude
        
        # Apply scaling factor to match Android implementation if needed. 1 is no scaling.
        scaling_factor = 1
        
        # Adjust the centroid based on estimated frequency and apply scaling
        centroid = scaling_factor * ((weighted_sum / (total_energy + 1e-8) + estimated_frequency) / 2) if total_energy > 0 else 0
        features.append(centroid)
        
        # Spectral bandwidth - Android uses a variance-based approach
        variance_sum = 0.0
        for i in range(len(signal)):
            magnitude = abs(signal[i])
            deviation = i - (centroid / scaling_factor)  # Use unscaled centroid for calculation
            variance_sum += deviation * deviation * magnitude
        
        # Apply scaling factor to match Android implementation (approximately 2x)
        bandwidth = scaling_factor * np.sqrt(variance_sum / (total_energy + 1e-8)) if total_energy > 0 else 0
        features.append(bandwidth)
        
        # Spectral rolloff - Android uses a cumulative energy approach with threshold 0.85
        rolloff_threshold = 0.85
        cumulative_energy = 0.0
        rolloff_bin = 0
        
        for i in range(len(signal)):
            cumulative_energy += abs(signal[i])
            if cumulative_energy >= rolloff_threshold * total_energy:
                rolloff_bin = i
                break
        
        rolloff = rolloff_bin / len(signal) if len(signal) > 0 else 0
        features.append(rolloff)
        
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
            print(".", end="")
        
        return np.array(features_list)
