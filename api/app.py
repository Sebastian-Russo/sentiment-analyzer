"""
Sentiment Analysis API
Analyzes text sentiment: positive or negative

POST /predict
Body: {"text": "This movie was amazing!"}
Returns: {"sentiment": "positive", "confidence": 95.2}
"""
from flask import Flask, request, jsonify
import pickle
import re

app = Flask(__name__)

# Load model when server starts
print("Loading model...")
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/baseline_sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("✓ Model loaded")

def clean_text(text):
    """Clean text for prediction (same as training)"""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment of text

    POST /predict
    Body: {"text": "Your review here"}

    Returns: {"sentiment": "positive", "confidence": 85.5, "probabilities": {...}}
    """
    try:
        # Get text from request
        data = request.get_json()

        if 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400

        text = data['text']

        if not text or not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400

        # Clean text
        text_clean = clean_text(text)

        # Convert to TF-IDF
        text_tfidf = vectorizer.transform([text_clean])

        # Predict
        prediction = model.predict(text_tfidf)[0]
        probabilities = model.predict_proba(text_tfidf)[0]

        # Format response
        sentiment = "positive" if prediction == 1 else "negative"
        confidence = float(probabilities[prediction] * 100)

        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'confidence': round(confidence, 1),
            'probabilities': {
                'negative': round(float(probabilities[0]) * 100, 1),
                'positive': round(float(probabilities[1]) * 100, 1)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_predict():
    """
    Predict sentiment for multiple texts at once

    POST /batch
    Body: {"texts": ["Review 1", "Review 2", "Review 3"]}

    Returns: {"results": [{...}, {...}, {...}]}
    """
    try:
        data = request.get_json()

        if 'texts' not in data:
            return jsonify({'error': 'Missing "texts" field'}), 400

        texts = data['texts']

        if not isinstance(texts, list):
            return jsonify({'error': '"texts" must be a list'}), 400

        if len(texts) > 100:
            return jsonify({'error': 'Maximum 100 texts per batch'}), 400

        # Process each text
        results = []
        for text in texts:
            # Clean
            text_clean = clean_text(text)

            # Predict
            text_tfidf = vectorizer.transform([text_clean])
            prediction = model.predict(text_tfidf)[0]
            probabilities = model.predict_proba(text_tfidf)[0]

            sentiment = "positive" if prediction == 1 else "negative"
            confidence = float(probabilities[prediction] * 100)

            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': round(confidence, 1)
            })

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'Logistic Regression + TF-IDF',
        'accuracy': '86.1%'
    })

@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    return jsonify({
        'name': 'Sentiment Analysis API',
        'version': '1.0',
        'endpoints': {
            'POST /predict': 'Analyze sentiment of single text',
            'POST /batch': 'Analyze sentiment of multiple texts',
            'GET /health': 'Health check'
        },
        'example': {
            'endpoint': '/predict',
            'method': 'POST',
            'body': {
                'text': 'This movie was amazing!'
            },
            'response': {
                'sentiment': 'positive',
                'confidence': 95.2,
                'probabilities': {
                    'negative': 4.8,
                    'positive': 95.2
                }
            }
        }
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎬 Sentiment Analysis API")
    print("="*60)
    print("Running on: http://localhost:5000")
    print("Endpoints:")
    print("  GET  /         - API documentation")
    print("  POST /predict  - Analyze single text")
    print("  POST /batch    - Analyze multiple texts")
    print("  GET  /health   - Health check")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5001)
