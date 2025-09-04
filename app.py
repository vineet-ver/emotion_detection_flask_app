from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os

# Define preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

app = Flask(__name__)

# Load the model and preprocessing objects
def load_model():
    try:
        model = tf.keras.models.load_model('emotion_model.h5')
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open('label_encoder.pkl', 'rb') as handle:
            label_encoder = pickle.load(handle)
        print("Model, tokenizer, and label encoder loaded successfully.")
        return model, tokenizer, label_encoder
    except Exception as e:
        print(f"Error loading model or preprocessing objects: {e}")
        return None, None, None

# Load model on startup
model, tokenizer, label_encoder = load_model()

# Use the max_len from your training
max_len = 66

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None or label_encoder is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
    
    try:
        # Get text from form or JSON
        if request.form:
            text = request.form['text']
        else:
            data = request.get_json()
            if not data or 'text' not in data:
                return jsonify({'error': 'No text provided'}), 400
            text = data['text']
        
        if not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Preprocess the text
        cleaned_text = preprocess_text(text)
        
        # Convert to sequence and pad
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        
        # Make prediction
        prediction = model.predict(padded_sequence)
        predicted_emotion_encoded = np.argmax(prediction, axis=1)[0]
        predicted_emotion = label_encoder.inverse_transform([predicted_emotion_encoded])[0]
        
        # Get confidence scores
        confidence_scores = {label_encoder.classes_[i]: float(prediction[0][i]) 
                           for i in range(len(label_encoder.classes_))}
        
        return jsonify({
            'text': text,
            'emotion': predicted_emotion,
            'confidence': confidence_scores,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    return predict()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)