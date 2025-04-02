import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import sys
from training.audio import AudioProcessor

# Redirect output to a file
output_file = open('diagnostic_results.txt', 'w')
sys.stdout = output_file

# Create a temporary directory
temp_dir = tempfile.mkdtemp()

# Create a synthetic signal (1 second of 440Hz sine wave)
duration = 1.0
sample_rate = 44100
t = np.linspace(0, duration, int(sample_rate * duration))
test_signal = 0.5 * np.sin(2 * np.pi * 440 * t)

# Create a test audio file using FLAC format instead of WAV
test_file_path = os.path.join(temp_dir, "test_audio.flac")
sf.write(test_file_path, test_signal, sample_rate, format='FLAC')

# Initialize the processor
processor = AudioProcessor(sample_rate=sample_rate)

# Method 1: Direct extraction from signal
direct_features = processor.extract_features(test_signal)

# Method 2: Load from file and extract
signal_from_file, _ = processor.load_audio(test_file_path)
file_features = processor.extract_features(signal_from_file)

# Method 3: Use process_file method
process_file_features = processor.process_file(test_file_path)

# Compare signals
print("Signal comparison:")
print(f"Original signal shape: {test_signal.shape}")
print(f"Signal from file shape: {signal_from_file.shape}")
print(f"Are signals identical? {np.array_equal(test_signal, signal_from_file)}")
print(f"Max difference between signals: {np.max(np.abs(test_signal - signal_from_file))}")
print(f"Mean difference between signals: {np.mean(np.abs(test_signal - signal_from_file))}")

# Compare features
print("\nFeature comparison:")
for i, (direct, file_feat, proc_file) in enumerate(zip(direct_features, file_features, process_file_features)):
    print(f"Feature {i}:")
    print(f"  Direct extraction: {direct}")
    print(f"  From loaded file: {file_feat}")
    print(f"  From process_file: {proc_file}")
    print(f"  Diff (direct vs loaded): {abs(direct - file_feat)}")
    print(f"  Diff (direct vs process_file): {abs(direct - proc_file)}")

# Focus on Feature 14 (index 13) - Spectral centroid
print("\nDetailed analysis of Feature 14 (Spectral centroid):")
print(f"Direct extraction: {direct_features[13]}")
print(f"From file: {file_features[13]}")
print(f"Difference: {abs(direct_features[13] - file_features[13])}")

# Directly calculate spectral centroid using librosa
direct_centroid = np.mean(librosa.feature.spectral_centroid(y=test_signal, sr=sample_rate))
file_centroid = np.mean(librosa.feature.spectral_centroid(y=signal_from_file, sr=sample_rate))

# Normalize signals for direct calculation
normalized_test_signal = librosa.util.normalize(test_signal)
normalized_file_signal = librosa.util.normalize(signal_from_file)

# Directly calculate spectral bandwidth using librosa with explicit parameters and normalization
direct_bandwidth = np.mean(librosa.feature.spectral_bandwidth(
    y=normalized_test_signal, 
    sr=sample_rate,
    n_fft=2048,
    hop_length=512,
    center=True,
    pad_mode='reflect'
))
file_bandwidth = np.mean(librosa.feature.spectral_bandwidth(
    y=normalized_file_signal, 
    sr=sample_rate,
    n_fft=2048,
    hop_length=512,
    center=True,
    pad_mode='reflect'
))

print("\nDirect librosa calculation with explicit parameters (bandwidth):")
print(f"Bandwidth from original signal: {direct_bandwidth}")
print(f"Bandwidth from file signal: {file_bandwidth}")
print(f"Difference: {abs(direct_bandwidth - file_bandwidth)}")

print("\nDirect librosa calculation:")
print(f"Centroid from original signal: {direct_centroid}")
print(f"Centroid from file signal: {file_centroid}")
print(f"Difference: {abs(direct_centroid - file_centroid)}")

# Clean up
for file in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, file))
os.rmdir(temp_dir)

# Close the output file
sys.stdout = sys.__stdout__
output_file.close()
print("Diagnostic complete. Results written to diagnostic_results.txt")
