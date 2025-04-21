import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import List, Optional, Tuple
from training.audio import AudioProcessor


def plot_features(audio_file: str, title: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot extracted features from an audio file.
    
    Args:
        audio_file: Path to the audio file
        title: Optional title for the plot
        figsize: Figure size as (width, height)
    """
    # Load audio and extract features
    processor = AudioProcessor()
    signal, sr = processor.load_audio(audio_file)
    features = processor.extract_features(signal)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot MFCCs (first 40 features)
    mfccs = features[:40]
    plt.subplot(2, 2, 1)
    plt.bar(range(len(mfccs)), mfccs)
    plt.title('MFCCs')
    plt.xlabel('Coefficient')
    plt.ylabel('Value')
    
    # Plot spectral features (last 3 features)
    spectral_features = features[40:]
    feature_names = ['Spectral Centroid', 'Spectral Bandwidth', 'Spectral Rolloff']
    
    plt.subplot(2, 2, 2)
    plt.bar(feature_names, spectral_features)
    plt.title('Spectral Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Plot waveform
    plt.subplot(2, 2, 3)
    plt.plot(signal)
    plt.title('Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    # Plot spectrogram
    plt.subplot(2, 2, 4)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    # Set main title if provided
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.show()


def plot_features_comparison(audio_files: List[str], labels: Optional[List[str]] = None, 
                            figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Compare extracted features from multiple audio files.
    
    Args:
        audio_files: List of paths to audio files
        labels: Optional list of labels for each audio file
        figsize: Figure size as (width, height)
    """
    if not audio_files:
        raise ValueError("No audio files provided")
    
    if labels is None:
        labels = [f"Audio {i+1}" for i in range(len(audio_files))]
    
    if len(labels) != len(audio_files):
        raise ValueError("Number of labels must match number of audio files")
    
    # Load audio and extract features
    processor = AudioProcessor()
    all_features = []
    
    for audio_file in audio_files:
        signal, _ = processor.load_audio(audio_file)
        features = processor.extract_features(signal)
        all_features.append(features)
    
    # Convert to numpy array
    all_features = np.array(all_features)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot MFCCs
    plt.subplot(2, 1, 1)
    for i, features in enumerate(all_features):
        plt.plot(features[:40], label=labels[i])
    plt.title('MFCCs Comparison')
    plt.xlabel('Coefficient')
    plt.ylabel('Value')
    plt.legend()
    
    # Plot spectral features
    plt.subplot(2, 1, 2)
    feature_names = ['Spectral Centroid', 'Spectral Bandwidth', 'Spectral Rolloff']
    x = np.arange(len(feature_names))
    width = 0.8 / len(audio_files)
    
    for i, features in enumerate(all_features):
        spectral_features = features[40:]
        plt.bar(x + i * width, spectral_features, width=width, label=labels[i])
    
    plt.title('Spectral Features Comparison')
    plt.xlabel('Feature')
    plt.xticks(x + width * (len(audio_files) - 1) / 2, feature_names)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_mfcc_2d(audio_file: str, title: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot the spectrogram and MFCC in 2D before taking the mean.
    
    Args:
        audio_file: Path to the audio file
        title: Optional title for the plot
        figsize: Figure size as (width, height)
    """
    # Load audio
    processor = AudioProcessor()
    signal, sr = processor.load_audio(audio_file)
    
    # Parameters for STFT and MFCC
    n_fft = 2048
    hop_length = 512
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot waveform
    plt.subplot(3, 1, 1)
    plt.plot(signal)
    plt.title('Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    # Plot spectrogram
    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    # Calculate and plot MFCC (before taking mean)
    plt.subplot(3, 1, 3)
    mfccs = librosa.feature.mfcc(
        y=signal, 
        sr=sr, 
        n_mfcc=40,
        n_fft=n_fft,
        hop_length=hop_length
    )
    librosa.display.specshow(mfccs, sr=sr, hop_length=hop_length, x_axis='time')
    plt.colorbar()
    plt.title('MFCC (before taking mean)')
    
    # Set main title if provided
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.show()


def get_raw_mfcc(audio_file: str) -> np.ndarray:
    """
    Get the raw MFCC values before taking the mean.
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        MFCC values as a 2D array (n_mfcc, n_frames)
    """
    # Load audio
    processor = AudioProcessor()
    signal, sr = processor.load_audio(audio_file)
    
    # Parameters for MFCC
    n_fft = 2048
    hop_length = 512
    
    # Calculate MFCC
    mfccs = librosa.feature.mfcc(
        y=signal, 
        sr=sr, 
        n_mfcc=40,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    return mfccs
