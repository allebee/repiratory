import librosa
import numpy as np
from tensorflow.keras.models import load_model

def segment_audio(x, sr, window_duration=5):
    """
    Segment audio into fixed-length windows
    
    Args:
        x (numpy.ndarray): Audio signal
        sr (int): Sample rate
        window_duration (int): Duration of each window in seconds
    
    Returns:
        list: List of audio segments
    """
    window_length = sr * window_duration
    segments = []
    
    # Calculate number of complete windows
    num_windows = len(x) // window_length
    
    for i in range(num_windows):
        start = i * window_length
        end = start + window_length
        segment = x[start:end]
        segments.append(segment)
    
    return segments

def preprocess_segment(x, sr_new=16000):
    """
    Preprocess a single audio segment
    
    Args:
        x (numpy.ndarray): Audio segment
        sr_new (int): Target sample rate
    
    Returns:
        numpy.ndarray: Preprocessed features
    """
    # Extract MFCC features
    feature = librosa.feature.mfcc(y=x, sr=sr_new)
    
    # Reshape for model input (1, 20, 157, 1)
    feature = feature.reshape(1, 20, 157, 1)
    
    return feature

def preprocess_audio(audio_file, sr_new=16000, window_duration=5):
    """
    Preprocess audio file for model inference with windowing
    
    Args:
        audio_file (str): Path to audio file
        sr_new (int): Target sample rate
        window_duration (int): Duration of each window in seconds
    
    Returns:
        list: List of preprocessed features for each window
    """
    # Load and resample audio
    x, sr = librosa.load(audio_file, sr=sr_new)
    
    # Segment audio into windows
    segments = segment_audio(x, sr_new, window_duration)
    
    # Preprocess each segment
    processed_segments = []
    for segment in segments:
        processed = preprocess_segment(segment, sr_new)
        processed_segments.append(processed)
    
    return processed_segments

def aggregate_predictions(predictions, method='mean'):
    """
    Aggregate predictions from multiple windows
    
    Args:
        predictions (list): List of prediction arrays
        method (str): Aggregation method ('mean' or 'max')
    
    Returns:
        numpy.ndarray: Aggregated predictions
    """
    if method == 'mean':
        return np.mean(predictions, axis=0)
    elif method == 'max':
        return np.max(predictions, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

def predict_lung_disease(audio_file, model_path='/Users/user/Desktop/ml_tech/prediction_lung_disease_model.keras', 
                        window_duration=5, aggregation_method='mean'):
    """
    Predict lung disease from audio file with windowing
    
    Args:
        audio_file (str): Path to audio file
        model_path (str): Path to saved model
        window_duration (int): Duration of each window in seconds
        aggregation_method (str): Method to aggregate predictions ('mean' or 'max')
    
    Returns:
        dict: Prediction results
    """
    # Class labels from training
    classes = ['Asthma', 'Bronchiectasis', 'Bronchiolitis', 'COPD', 
              'Healthy', 'LRTI', 'Pneumonia', 'URTI']
    
    try:
        # Load model
        model = load_model(model_path)
        
        # Preprocess audio with windowing
        processed_segments = preprocess_audio(audio_file, window_duration=window_duration)
        
        if not processed_segments:
            return {'error': 'Audio file is too short'}
        
        # Get predictions for each segment
        segment_predictions = []
        for segment in processed_segments:
            pred = model.predict(segment, verbose=0)
            segment_predictions.append(pred[0])
        
        # Aggregate predictions from all segments
        final_predictions = aggregate_predictions(segment_predictions, 
                                               method=aggregation_method)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(final_predictions)
        confidence = final_predictions[predicted_class_idx]
        
        # Get predictions per segment
        segment_results = []
        for i, pred in enumerate(segment_predictions):
            idx = np.argmax(pred)
            segment_results.append({
                'segment': i,
                'predicted_class': classes[idx],
                'confidence': float(pred[idx])
            })
        
        return {
            'predicted_class': classes[predicted_class_idx],
            'confidence': float(confidence),
            'all_probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(classes, final_predictions)
            },
            'segment_predictions': segment_results,
            'num_segments': len(processed_segments)
        }
        
    except Exception as e:
        return {
            'error': f'Prediction failed: {str(e)}'
        }

# Example usage
if __name__ == "__main__":
    # Example audio file path
    audio_file = "/Users/user/Desktop/ml_tech/healthy.wav"
    
    # Get prediction
    result = predict_lung_disease(audio_file, aggregation_method='mean')
    
    if 'error' not in result:
        print(f"Predicted Disease: {result['predicted_class']}")
        print(f"Overall Confidence: {result['confidence']:.2%}")
        print(f"\nNumber of segments analyzed: {result['num_segments']}")
        print("\nPredictions by segment:")
        for segment in result['segment_predictions']:
            print(f"Segment {segment['segment']}: "
                  f"{segment['predicted_class']} "
                  f"({segment['confidence']:.2%})")
        print("\nAll class probabilities:")
        for disease, prob in result['all_probabilities'].items():
            print(f"{disease}: {prob:.2%}")
    else:
        print(result['error'])