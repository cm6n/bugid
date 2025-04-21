import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
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


def plot_features_comparison(paths: List[str], labels: Optional[List[str]] = None, 
                            figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Compare extracted features from multiple audio files or directories.
    
    Args:
        paths: List of paths to audio files or directories
        labels: Optional list of labels for each path. If not provided:
               - For files: filename without extension is used
               - For directories: directory name is used
        figsize: Figure size as (width, height)
    """
    if not paths:
        raise ValueError("No paths provided")
    
    # Process paths to get audio files and labels
    audio_files = []
    final_labels = []
    
    for i, path in enumerate(paths):
        if os.path.isdir(path):
            # Path is a directory, find all audio files in it
            dir_audio_files = []
            for file in os.listdir(path):
                if file.endswith(('.wav', '.mp3', '.m4a')):
                    dir_audio_files.append(os.path.join(path, file))
            
            if not dir_audio_files:
                print(f"Warning: No audio files found in directory {path}")
                continue
                
            # Use the directory name as the label if not provided
            dir_label = os.path.basename(path)
            audio_files.extend(dir_audio_files)
            final_labels.extend([dir_label] * len(dir_audio_files))
        else:
            # Path is a file
            audio_files.append(path)
            # Use filename without extension as label if not provided
            if labels is None:
                file_label = os.path.splitext(os.path.basename(path))[0]
                final_labels.append(file_label)
    
    if not audio_files:
        raise ValueError("No valid audio files found in the provided paths")
    
    # If labels were provided, use them instead of the auto-generated ones
    if labels is not None:
        if len(labels) != len(paths):
            print(f"Warning: Number of labels ({len(labels)}) doesn't match number of paths ({len(paths)}). Using auto-generated labels.")
        else:
            # Expand labels for directories
            final_labels = []
            label_index = 0
            for path in paths:
                if os.path.isdir(path):
                    # Count audio files in this directory
                    count = sum(1 for f in os.listdir(path) if f.endswith(('.wav', '.mp3', '.m4a')))
                    if count > 0:
                        final_labels.extend([labels[label_index]] * count)
                else:
                    final_labels.append(labels[label_index])
                label_index += 1
    
    # Load audio and extract features
    processor = AudioProcessor()
    all_features = []
    processed_labels = []
    processed_filenames = []
    
    for i, audio_file in enumerate(audio_files):
        try:
            signal, _ = processor.load_audio(audio_file)
            features = processor.extract_features(signal)
            all_features.append(features)
            processed_labels.append(final_labels[i])
            processed_filenames.append(os.path.basename(audio_file))
            print(f"Processed {audio_file} with label '{final_labels[i]}'")
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
    
    if not all_features:
        raise ValueError("No features could be extracted from the provided audio files")
    
    # Convert to numpy array
    all_features = np.array(all_features)
    
    # Group features by unique labels
    unique_labels = list(set(processed_labels))
    print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
    
    # Create a figure for each unique label
    for label in unique_labels:
        # Get indices of features with this label
        indices = [i for i, l in enumerate(processed_labels) if l == label]
        label_features = all_features[indices]
        label_filenames = [processed_filenames[i] for i in indices]
        
        # Create plot for this label
        plt.figure(figsize=figsize)
        plt.suptitle(f'Features for {label}', fontsize=16)
        
        # Plot MFCCs
        plt.subplot(2, 1, 1)
        for i, features in enumerate(label_features):
            plt.plot(features[:40], label=label_filenames[i])
        plt.title('MFCCs')
        plt.xlabel('Coefficient')
        plt.ylabel('Value')
        if len(label_features) > 1:
            plt.legend()
        
        # Plot spectral features
        plt.subplot(2, 1, 2)
        feature_names = ['Spectral Centroid', 'Spectral Bandwidth', 'Spectral Rolloff']
        x = np.arange(len(feature_names))
        width = 0.8 / len(label_features) if len(label_features) > 0 else 0.8
        
        for i, features in enumerate(label_features):
            spectral_features = features[40:]
            plt.bar(x + i * width, spectral_features, width=width, label=label_filenames[i])
        
        plt.title('Spectral Features')
        plt.xlabel('Feature')
        plt.xticks(x + width * (len(label_features) - 1) / 2, feature_names)
        if len(label_features) > 1:
            plt.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        plt.show()
    
    # Create a summary plot comparing averages across labels
    if len(unique_labels) > 1:
        plt.figure(figsize=figsize)
        plt.suptitle('Summary Comparison Across Labels', fontsize=16)
        
        # Calculate average features for each label
        avg_features_by_label = {}
        for label in unique_labels:
            indices = [i for i, l in enumerate(processed_labels) if l == label]
            avg_features_by_label[label] = np.mean(all_features[indices], axis=0)
        
        # Plot average MFCCs
        plt.subplot(2, 1, 1)
        for label, avg_features in avg_features_by_label.items():
            plt.plot(avg_features[:40], label=label)
        plt.title('Average MFCCs by Label')
        plt.xlabel('Coefficient')
        plt.ylabel('Value')
        plt.legend()
        
        # Plot average spectral features
        plt.subplot(2, 1, 2)
        feature_names = ['Spectral Centroid', 'Spectral Bandwidth', 'Spectral Rolloff']
        x = np.arange(len(feature_names))
        width = 0.8 / len(unique_labels)
        
        for i, label in enumerate(unique_labels):
            spectral_features = avg_features_by_label[label][40:]
            plt.bar(x + i * width, spectral_features, width=width, label=label)
        
        plt.title('Average Spectral Features by Label')
        plt.xlabel('Feature')
        plt.xticks(x + width * (len(unique_labels) - 1) / 2, feature_names)
        plt.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
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
