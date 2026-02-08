"""
Phase 2.1: Baseline Sentiment Model
Simple approach: TF-IDF + Logistic Regression

ANALOGY:
Instead of complex understanding, just count words:
- If review has "amazing", "excellent", "loved" → Probably positive
- If review has "terrible", "waste", "boring" → Probably negative
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load cleaned data
print("Loading data...")
train_df = pd.read_csv('data/processed/imdb_train_clean.csv')

# For faster training, let's use a subset first (optional)
# Comment out these lines to use full dataset
SAMPLE_SIZE = 5000  # Use 5k for speed, or set to None for full dataset
if SAMPLE_SIZE:
    train_df = train_df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"Using {SAMPLE_SIZE} samples for faster training")
else:
    print(f"Using all {len(train_df)} samples")

# Split into features and target
X_text = train_df['text_clean']
y = train_df['label']  # 0=negative, 1=positive

print(f"\nDataset: {len(X_text)} reviews")
print(f"Positive: {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
print(f"Negative: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")

# Split train/validation (80/20)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_text, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"\nTrain: {len(X_train)} reviews")
print(f"Validation: {len(X_val)} reviews")

# Step 1: Convert text to numbers using TF-IDF
print("\n" + "="*60)
print("STEP 1: TEXT → NUMBERS (TF-IDF)")
print("="*60)

# TfidfVectorizer converts text to TF-IDF features
# max_features: Only keep top 5000 most important words
# min_df: Word must appear in at least 5 documents
# max_df: Ignore words that appear in >70% of documents (too common)
# ngram_range: Use single words and word pairs
vectorizer = TfidfVectorizer(
    max_features=5000,      # Top 5000 words (vocabulary limit)
    min_df=5,               # Word must appear in 5+ documents
    max_df=0.7,             # Ignore if appears in >70% of documents
    ngram_range=(1, 2),     # Unigrams and bigrams
    strip_accents='unicode'
)

# Fit on training data and transform
# fit_transform() learns vocabulary from train, converts to numbers
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform validation data using same vocabulary
# transform() uses already-learned vocabulary
X_val_tfidf = vectorizer.transform(X_val)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Feature matrix shape: {X_train_tfidf.shape}")
print(f"  {X_train_tfidf.shape[0]} reviews × {X_train_tfidf.shape[1]} features")

# Show some vocabulary
vocab_sample = list(vectorizer.vocabulary_.items())[:10]
print(f"\nSample vocabulary:")
for word, index in vocab_sample:
    print(f"  '{word}' → feature {index}")

# Step 2: Train Logistic Regression
print("\n" + "="*60)
print("STEP 2: TRAIN LOGISTIC REGRESSION")
print("="*60)
print("Training...")

# Logistic Regression - same as churn prediction!
# max_iter=1000 gives enough iterations to converge
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

print("✓ Model trained")

# Step 3: Make predictions
print("\n" + "="*60)
print("STEP 3: EVALUATE")
print("="*60)

# Predict on validation set
y_pred = model.predict(X_val_tfidf)

# Calculate metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"              Negative  Positive")
print(f"Actual Negative  {cm[0,0]:5d}     {cm[0,1]:5d}")
print(f"       Positive  {cm[1,0]:5d}     {cm[1,1]:5d}")

# Detailed report
print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_val, y_pred, target_names=['Negative', 'Positive']))

# Step 4: Inspect important features
print("\n" + "="*60)
print("MOST PREDICTIVE WORDS")
print("="*60)

# Get feature names and coefficients
# .coef_[0] gets weights for each word
feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = model.coef_[0]

# Top positive words (strong positive sentiment)
top_positive_idx = np.argsort(coefficients)[-20:]
print("\nTop 20 POSITIVE words:")
for idx in reversed(top_positive_idx):
    print(f"  {feature_names[idx]:20s} → {coefficients[idx]:+.3f}")

# Top negative words (strong negative sentiment)
top_negative_idx = np.argsort(coefficients)[:20]
print("\nTop 20 NEGATIVE words:")
for idx in top_negative_idx:
    print(f"  {feature_names[idx]:20s} → {coefficients[idx]:+.3f}")

# Step 5: Test on examples
print("\n" + "="*60)
print("TEST ON SAMPLE REVIEWS")
print("="*60)

test_reviews = [
    "This movie was absolutely amazing! Best film I've ever seen.",
    "Terrible waste of time. Boring and predictable.",
    "It was okay. Not great, not terrible.",
]

for review in test_reviews:
    # Clean and vectorize
    review_clean = review.lower()
    review_tfidf = vectorizer.transform([review_clean])

    # Predict
    prediction = model.predict(review_tfidf)[0]
    probability = model.predict_proba(review_tfidf)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probability[prediction] * 100

    print(f"\nReview: {review}")
    print(f"  Prediction: {sentiment} ({confidence:.1f}% confident)")

# Save model
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

import os
os.makedirs('models', exist_ok=True)

# Save vectorizer and model
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('models/baseline_sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✓ Saved models/tfidf_vectorizer.pkl")
print("✓ Saved models/baseline_sentiment_model.pkl")

print("\n✓ Phase 2.1 complete! Baseline model trained.")
