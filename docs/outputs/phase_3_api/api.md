$ export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Now test
/usr/bin/curl http://localhost:5001/health

/usr/bin/curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely amazing!"}'

/usr/bin/curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible waste of time."}'
Command 'sed' is available in the following places
 * /bin/sed
 * /usr/bin/sed
The command could not be located because '/usr/bin:/bin' is not included in the PATH environment variable.
sed: command not found
{
  "accuracy": "86.1%",
  "model": "Logistic Regression + TF-IDF",
  "status": "healthy"
}
{
  "confidence": 59.9,
  "probabilities": {
    "negative": 40.1,
    "positive": 59.9
  },
  "sentiment": "positive",
  "text": "This movie was absolutely amazing!"
}
{
  "confidence": 92.7,
  "probabilities": {
    "negative": 92.7,
    "positive": 7.3
  },
  "sentiment": "negative",
  "text": "Terrible waste of time."
}

### API WORKS PERFECTLY!

Results
Health Check:
json{
  "status": "healthy",
  "model": "Logistic Regression + TF-IDF",
  "accuracy": "86.1%"
}
Positive Review:
json{
  "text": "This movie was absolutely amazing!",
  "sentiment": "positive",
  "confidence": 59.9,
  "probabilities": {
    "negative": 40.1,
    "positive": 59.9
  }
}
✓ Correctly predicted positive (though not super confident - 59.9%)
Negative Review:
json{
  "text": "Terrible waste of time.",
  "sentiment": "negative",
  "confidence": 92.7,
  "probabilities": {
    "negative": 92.7,
    "positive": 7.3
  }
}
✓ Correctly predicted negative with high confidence (92.7%)!

🎓 Sentiment Analysis Project Complete!
What You Built:

✅ Downloaded 25,000 IMDB movie reviews
✅ Cleaned and preprocessed text data
✅ Trained baseline model (TF-IDF + Logistic Regression) - 86.1% accuracy
✅ Trained BERT transformer model - 84.0% accuracy
✅ Deployed best model as Flask API on port 5001

What You Learned:

Text preprocessing (cleaning, tokenization, stopwords)
TF-IDF vectorization
Bag-of-words vs transformers
When simple models beat complex ones
NLP-specific concepts (n-grams, embeddings, context)
API deployment for NLP models
