import librosa as lb
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import soundfile as sf

def analyze_audio(file_path):
    """
    Analyze key characteristics of an audio file.
    Returns a dictionary of audio features and statistics.
    """
    # Load audio
    y, sr = lb.load(file_path)
    
    # Basic statistics
    stats_dict = {
        'duration': len(y) / sr,
        'mean': np.mean(y),
        'std': np.std(y),
        'rms': np.sqrt(np.mean(y**2)),
        'peak': np.max(np.abs(y)),
        'zero_crossings': np.sum(lb.zero_crossings(y)),
    }
    
    # Spectral features
    spectral_centroid = lb.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = lb.feature.spectral_rolloff(y=y, sr=sr)
    mfccs = lb.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    stats_dict.update({
        'spectral_centroid_mean': np.mean(spectral_centroid),
        'spectral_rolloff_mean': np.mean(spectral_rolloff),
        'mfcc_means': np.mean(mfccs, axis=1)
    })
    
    return stats_dict, y, sr

def compare_audio_files(reference_path, test_path, output_dir="./"):
    """
    Compare two audio files and generate normalization parameters.
    """
    # Analyze both files
    ref_stats, ref_y, ref_sr = analyze_audio(reference_path)
    test_stats, test_y, test_sr = analyze_audio(test_path)
    
    # Create comparison plot
    plt.figure(figsize=(15, 10))
    
    # Waveform comparison
    plt.subplot(3, 1, 1)
    plt.title("Waveform Comparison")
    plt.plot(np.arange(len(ref_y))/ref_sr, ref_y, label='Reference', alpha=0.7)
    plt.plot(np.arange(len(test_y))/test_sr, test_y, label='Test', alpha=0.7)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    # Amplitude distribution
    plt.subplot(3, 1, 2)
    plt.title("Amplitude Distribution")
    plt.hist(ref_y, bins=100, alpha=0.5, density=True, label='Reference')
    plt.hist(test_y, bins=100, alpha=0.5, density=True, label='Test')
    plt.legend()
    plt.xlabel("Amplitude")
    plt.ylabel("Density")
    
    # Spectral comparison
    ref_spec = np.abs(lb.stft(ref_y))
    test_spec = np.abs(lb.stft(test_y))
    plt.subplot(3, 1, 3)
    plt.title("Average Frequency Spectrum")
    plt.plot(np.mean(ref_spec, axis=1), label='Reference', alpha=0.7)
    plt.plot(np.mean(test_spec, axis=1), label='Test', alpha=0.7)
    plt.legend()
    plt.xlabel("Frequency Bin")
    plt.ylabel("Magnitude")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/audio_comparison.png")
    
    # Calculate normalization parameters
    norm_params = {
        'amplitude_scale': ref_stats['rms'] / test_stats['rms'],
        'dc_offset': ref_stats['mean'] - test_stats['mean'],
        'duration_target': ref_stats['duration']
    }
    
    return norm_params

def normalize_audio(input_path, norm_params, output_path):
    """
    Apply normalization parameters to an audio file with enhanced processing.
    """
    # Load audio
    y, sr = lb.load(input_path)
    
    # Find the segment with highest energy (main respiratory sound)
    frame_length = int(sr * 0.5)  # 500ms frames
    energy = np.array([np.sum(y[i:i+frame_length]**2) for i in range(0, len(y)-frame_length, frame_length)])
    max_energy_frame = np.argmax(energy)
    
    # Center the highest energy segment
    start_idx = max_energy_frame * frame_length
    target_length = int(norm_params['duration_target'] * sr)
    half_target = target_length // 2
    center_idx = start_idx + frame_length // 2
    
    # Extract centered segment
    start = max(0, center_idx - half_target)
    end = min(len(y), center_idx + half_target)
    y = y[start:end]
    
    # Apply DC offset correction
    y = y - np.mean(y) + norm_params['dc_offset']
    
    # Apply amplitude scaling with compression
    y = np.sign(y) * np.power(np.abs(y), 0.9)  # Slight compression
    y = y * norm_params['amplitude_scale']
    
    # Handle duration with centered padding if needed
    current_length = len(y)
    if current_length < target_length:
        pad_left = (target_length - current_length) // 2
        pad_right = target_length - current_length - pad_left
        y = np.pad(y, (pad_left, pad_right), mode='constant')
    elif current_length > target_length:
        start_idx = (current_length - target_length) // 2
        y = y[start_idx:start_idx + target_length]
    
    # Save normalized audio
    sf.write(output_path, y, sr)
    return y, sr


def main():
    # Example usage
    reference_path = "/Users/user/Desktop/ml_tech/101_1b1_Al_sc_Meditron_1.wav"
    test_path = "/Users/user/Desktop/ml_tech/Wheeze-Asthma.mp3"
    output_dir = "/Users/user/Desktop/ml_tech/output"
    
    # Compare files and get normalization parameters
    norm_params = compare_audio_files(reference_path, test_path, output_dir)
    print("\nNormalization Parameters:")
    for key, value in norm_params.items():
        print(f"{key}: {value}")
    
    # Apply normalization
    normalized_y, sr = normalize_audio(test_path, norm_params, f"{output_dir}/normalized_test.wav")
    
    print("\nNormalization complete! Check the output directory for results.")

if __name__ == "__main__":
    main()