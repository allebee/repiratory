import streamlit as st
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import librosa as lb
import tempfile
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration with custom theme
st.set_page_config(
    page_title="AI Respiratory Disease Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        margin-top: 1rem;
    }
    .upload-header {
        text-align: center;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .disclaimer {
        font-size: 0.9rem;
        color: #6c757d;
        padding: 1rem;
        border-left: 3px solid #ffc107;
        background: #fff8e1;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with app information
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/stethoscope.png", width=96)
    st.title("О приложении")
    st.markdown("""
    ### Как это работает
    1. Загрузите аудиофайл с записью дыхания
    2. AI модель проанализирует звуковые паттерны
    3. Получите результаты анализа с вероятностями
    
    ### Поддерживаемые заболевания:
    - 🌬️ Астма
    - 🫁 Бронхоэктаз
    - 🔬 Бронхиолит
    - 🫧 ХОБЛ
    - 🦠 ИНДП
    - 🌡️ ИВДП
    - 🫀 Пневмония
    """)
    
    st.markdown("---")
    st.markdown("### История анализов")
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    for item in st.session_state.history[-5:]:  # Show last 5 analyses
        st.markdown(f"""
        📊 **{item['date']}**  
        Диагноз: {item['diagnosis']}  
        Вероятность: {item['probability']:.1f}%
        """)

@st.cache_resource
def load_classification_model():
    """Load the trained model"""
    try:
        # Define possible model paths
        model_paths = [
            'model2.h5',
            'models/model2.h5',
            os.path.join(os.getcwd(), 'model2.h5')
        ]
        
        # Try each path
        for path in model_paths:
            if os.path.exists(path):
                return load_model(path)
        
        st.error("Model file not found. Please ensure the model is uploaded to the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def create_gauge_chart(value, title):
    """Create a gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def get_features_for_neural_network(path):
    """Extract audio features with progress tracking"""
    with st.spinner('Извлечение аудио характеристик...'):
        sound_arr, sample_rate = lb.load(path)
        mfcc = lb.feature.mfcc(y=sound_arr, sr=sample_rate)
        cstft = lb.feature.chroma_stft(y=sound_arr, sr=sample_rate)
        mspec = lb.feature.melspectrogram(y=sound_arr, sr=sample_rate)
        return mfcc, cstft, mspec

def classify_audio(audio_path, model):
    """Classify audio with enhanced error handling and progress tracking"""
    try:
        # Feature extraction
        mfcc_test, croma_test, mspec_test = get_features_for_neural_network(audio_path)
        
        # Prepare input arrays
        mfcc, cstft, mspec = [], [], []
        mfcc.append(mfcc_test)
        cstft.append(croma_test)
        mspec.append(mspec_test)

        mfcc_test = np.array(mfcc)
        cstft_test = np.array(cstft)
        mspec_test = np.array(mspec)

        # Model prediction
        with st.spinner('AI анализирует данные...'):
            result = model.predict({
                "mfcc": mfcc_test,
                "croma": cstft_test,
                "mspec": mspec_test
            })

        # Process results
        disease_array = ['Астма', 'Бронхоэктаз', 'Бронхиолит', 'Хроническая обструктивная болезнь лёгких', 
                        'Здоров', 'Инфекции нижних дыхательных путей', 'Пневмония', 'Инфекционное заболевание верхних дыхательных путей']
        result = result.flatten()
        
        # Get top-3 predictions
        sorted_indices = np.argsort(result)[::-1][:3]
        
        return [
            {
                'disorder': disease_array[idx],
                'probability': result[idx] * 100
            }
            for idx in sorted_indices
        ]
    
    except Exception as e:
        st.error(f"Ошибка при обработке аудиофайла: {str(e)}")
        return None

def main():
    # Main page header
    st.title("🫁 AI Анализ Респираторных Заболеваний")
    st.markdown("""
    Загрузите аудиозапись дыхательных звуков для анализа с помощью искусственного интеллекта.
    Система определит вероятность различных респираторных заболеваний на основе паттернов дыхания.
    """)

    # Model loading
    try:
        model = load_classification_model()
    except Exception as e:
        st.error("❌ Ошибка загрузки модели. Пожалуйста, проверьте наличие файла модели.")
        return

    # File uploader with clear instructions
    st.markdown("""
    <div class="upload-header">
        <h3>📤 Загрузка аудиофайла</h3>
        <p>Поддерживается формат WAV</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=['wav'])

    if uploaded_file is not None:
        # Create columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.audio(uploaded_file, format='audio/wav')
        
        with col2:
            st.markdown("""
            **Информация о файле:**  
            - Имя: {}  
            - Размер: {:.2f} MB
            """.format(
                uploaded_file.name,
                uploaded_file.size / (1024*1024)
            ))

        # Analysis button
        if st.button("🔬 Начать анализ", key="analyze_button"):
            # Progress tracking
            progress_text = "Операция в процессе. Пожалуйста, подождите."
            my_bar = st.progress(0, text=progress_text)

            # Temporary file handling
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name

            # Analysis process with visual feedback
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            
            # Get classification results
            results = classify_audio(temp_path, model)
            os.unlink(temp_path)

            if results:
                # Update session history
                st.session_state.history.append({
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'diagnosis': results[0]['disorder'],
                    'probability': results[0]['probability']
                })

                # Display results in an organized layout
                st.markdown("### 📊 Результаты анализа")
                
                # Primary result with gauge chart
                st.markdown("""
                <div class="result-card">
                    <h4>Основной диагноз</h4>
                </div>
                """, unsafe_allow_html=True)
                
                primary = results[0]
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.plotly_chart(create_gauge_chart(
                        primary['probability'],
                        primary['disorder']
                    ), use_container_width=True)
                
                with col2:
                    st.markdown(f"""
                    ### {primary['disorder']}
                    Вероятность: **{primary['probability']:.1f}%**
                    
                    *Основное предполагаемое заболевание*
                    """)

                # Secondary results
                st.markdown("#### Дополнительные возможные диагнозы:")
                cols = st.columns(len(results[1:]))
                for idx, result in enumerate(results[1:]):
                    with cols[idx]:
                        st.markdown(f"""
                        <div class="result-card">
                            <h5>{result['disorder']}</h5>
                            <p>Вероятность: {result['probability']:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Disclaimer
                st.markdown("""
                <div class="disclaimer">
                    ⚠️ <strong>Важное примечание:</strong><br>
                    Данный инструмент предназначен только для образовательных целей и не заменяет 
                    профессиональную медицинскую диагностику. Пожалуйста, обратитесь к квалифицированному 
                    медицинскому специалисту для получения точного диагноза.
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()