import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import describe

def analyze_audio_files(file1_path, file2_path, sr=16000):
    """
    Compare two audio files and display their characteristics
    
    Args:
        file1_path (str): Path to first audio file (reference)
        file2_path (str): Path to second audio file (test)
        sr (int): Sample rate to use for analysis
    """
    # Load audio files
    y1, _ = librosa.load(file1_path, sr=sr)
    y2, _ = librosa.load(file2_path, sr=sr)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Waveform comparison
    ax1 = plt.subplot(3, 2, 1)
    librosa.display.waveshow(y1, sr=sr)
    ax1.set_title('Reference Audio Waveform')
    
    ax2 = plt.subplot(3, 2, 2)
    librosa.display.waveshow(y2, sr=sr)
    ax2.set_title('Test Audio Waveform')
    
    # 2. Spectrograms
    # Reference audio spectrogram
    D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
    ax3 = plt.subplot(3, 2, 3)
    librosa.display.specshow(D1, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    ax3.set_title('Reference Audio Spectrogram')
    
    # Test audio spectrogram
    D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)
    ax4 = plt.subplot(3, 2, 4)
    librosa.display.specshow(D2, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    ax4.set_title('Test Audio Spectrogram')
    
    # 3. MFCC comparison
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=20)
    ax5 = plt.subplot(3, 2, 5)
    librosa.display.specshow(mfcc1, sr=sr, x_axis='time')
    plt.colorbar()
    ax5.set_title('Reference Audio MFCCs')
    
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=20)
    ax6 = plt.subplot(3, 2, 6)
    librosa.display.specshow(mfcc2, sr=sr, x_axis='time')
    plt.colorbar()
    ax6.set_title('Test Audio MFCCs')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistical analysis
    print("\nAudio Statistics Comparison:")
    print("-" * 50)
    
    # Calculate statistics for both files
    stats1 = describe(y1)
    stats2 = describe(y2)
    
    print(f"{'Metric':<15} {'Reference':<15} {'Test':<15}")
    print("-" * 50)
    print(f"{'Duration':<15} {len(y1)/sr:<15.2f} {len(y2)/sr:<15.2f}")
    print(f"{'Mean':<15} {stats1.mean:<15.4f} {stats2.mean:<15.4f}")
    print(f"{'Variance':<15} {stats1.variance:<15.4f} {stats2.variance:<15.4f}")
    print(f"{'Skewness':<15} {stats1.skewness:<15.4f} {stats2.skewness:<15.4f}")
    print(f"{'Kurtosis':<15} {stats1.kurtosis:<15.4f} {stats2.kurtosis:<15.4f}")
    print(f"{'Min':<15} {stats1.minmax[0]:<15.4f} {stats2.minmax[0]:<15.4f}")
    print(f"{'Max':<15} {stats1.minmax[1]:<15.4f} {stats2.minmax[1]:<15.4f}")
    
    # Calculate RMS energy
    rms1 = librosa.feature.rms(y=y1)
    rms2 = librosa.feature.rms(y=y2)
    print(f"{'RMS Energy':<15} {np.mean(rms1):<15.4f} {np.mean(rms2):<15.4f}")
    
    # Calculate zero crossing rate
    zcr1 = librosa.feature.zero_crossing_rate(y1)
    zcr2 = librosa.feature.zero_crossing_rate(y2)
    print(f"{'Zero Cross Rate':<15} {np.mean(zcr1):<15.4f} {np.mean(zcr2):<15.4f}")
    
    # Frequency analysis
    spec1 = np.abs(librosa.stft(y1))
    spec2 = np.abs(librosa.stft(y2))
    spec_mean1 = np.mean(spec1, axis=1)
    spec_mean2 = np.mean(spec2, axis=1)
    
    # Plot frequency distribution
    plt.figure(figsize=(10, 5))
    plt.plot(spec_mean1, label='Reference', alpha=0.7)
    plt.plot(spec_mean2, label='Test', alpha=0.7)
    plt.title('Average Frequency Distribution')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    reference_file = "/Users/user/Desktop/ml_tech/healthy.wav"  # File from training set that works well
    test_file = "/Users/user/Desktop/ml_tech/NormalBreathSound.mp3"  # New file that doesn't work well
    
    analyze_audio_files(reference_file, test_file)