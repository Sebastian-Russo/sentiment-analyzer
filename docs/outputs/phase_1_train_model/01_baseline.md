$ python3 src/models/01_baseline_model.py
Loading data...
Using 5000 samples for faster training

Dataset: 5000 reviews
Positive: 2485 (49.7%)
Negative: 2515 (50.3%)

Train: 4000 reviews
Validation: 1000 reviews

============================================================
STEP 1: TEXT → NUMBERS (TF-IDF)
============================================================
Vocabulary size: 5000
Feature matrix shape: (4000, 5000)
  4000 reviews × 5000 features

Sample vocabulary:
  'if' → feature 1909
  'you' → feature 4952
  'as' → feature 359
  'have' → feature 1723
  'very' → feature 4574
  'close' → feature 829
  'long' → feature 2427
  'relationship' → feature 3310
  'with' → feature 4846
  'world' → feature 4901

============================================================
STEP 2: TRAIN LOGISTIC REGRESSION
============================================================
Training...
✓ Model trained

============================================================
STEP 3: EVALUATE
============================================================
Accuracy:  0.861
Precision: 0.862
Recall:    0.857
F1-Score:  0.860

Confusion Matrix:
                Predicted
              Negative  Positive
Actual Negative    435        68
       Positive     71       426

============================================================
DETAILED CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

    Negative       0.86      0.86      0.86       503
    Positive       0.86      0.86      0.86       497

    accuracy                           0.86      1000
   macro avg       0.86      0.86      0.86      1000
weighted avg       0.86      0.86      0.86      1000


============================================================
MOST PREDICTIVE WORDS
============================================================

Top 20 POSITIVE words:
  great                → +3.487
  excellent            → +2.438
  best                 → +2.114
  wonderful            → +2.081
  love                 → +2.057
  fun                  → +1.812
  the best             → +1.708
  it is                → +1.692
  also                 → +1.663
  loved                → +1.614
  very                 → +1.614
  enjoyed              → +1.600
  classic              → +1.533
  amazing              → +1.527
  definitely           → +1.495
  today                → +1.494
  fantastic            → +1.438
  as                   → +1.424
  life                 → +1.390
  perfect              → +1.347

Top 20 NEGATIVE words:
  bad                  → -4.242
  worst                → -3.820
  the worst            → -2.953
  no                   → -2.495
  nothing              → -2.475
  awful                → -2.349
  terrible             → -2.331
  waste                → -2.258
  poor                 → -2.227
  boring               → -2.221
  even                 → -1.969
  script               → -1.870
  only                 → -1.838
  was                  → -1.828
  worse                → -1.779
  dull                 → -1.754
  better               → -1.684
  stupid               → -1.623
  instead              → -1.562
  horrible             → -1.545

============================================================
TEST ON SAMPLE REVIEWS
============================================================

Review: This movie was absolutely amazing! Best film I've ever seen.
  Prediction: Positive (73.3% confident)

Review: Terrible waste of time. Boring and predictable.
  Prediction: Negative (95.6% confident)

Review: It was okay. Not great, not terrible.
  Prediction: Negative (72.1% confident)

============================================================
SAVING MODEL
============================================================
✓ Saved models/tfidf_vectorizer.pkl
✓ Saved models/baseline_sentiment_model.pkl

✓ Phase 2.1 complete! Baseline model trained.


### What is Logistic Regression?
Logistic Regression = A classification algorithm that draws a line (or curve) to separate classes.
How it works:
Step 1: Assign weights to each word
python"great" → +3.487 (strong positive signal)
"bad" → -4.242 (strong negative signal)
"movie" → +0.05 (neutral)
Step 2: Calculate score for a review
pythonReview: "This movie is great"

Score = (weight_of_this × count_of_this) +
        (weight_of_movie × count_of_movie) +
        (weight_of_is × count_of_is) +
        (weight_of_great × count_of_great)

Score = (0.1 × 1) + (0.05 × 1) + (0.02 × 1) + (3.487 × 1)
Score = 3.667 (positive!)
Step 3: Convert score to probability
pythonIf score > 0 → Positive sentiment
If score < 0 → Negative sentiment
```

### **Why "Logistic"?**

It uses the **logistic function** (sigmoid) to convert any score into a probability between 0 and 1:
```
Score: -∞ to +∞  →  Probability: 0% to 100%

What is scikit-learn (sklearn)?
scikit-learn = The standard machine learning library for Python (classic ML, not deep learning).
We already used it in the churn predictor! Same library, different algorithms.


### Why This Simple Approach Works
Sentiment is often about specific words:

Positive reviews use: "great", "excellent", "loved", "amazing"
Negative reviews use: "bad", "worst", "awful", "boring"

Just counting these words gets you 86% accuracy!

Limitations of This Approach

No context: "not good" is treated as "not" + "good" (both words)
No word order: "Dog bites man" = "Man bites dog"
Sarcasm fails: "Oh great, another terrible movie" (has "great" but is negative)
Neutral reviews: Struggles with mixed sentiment

That's why we'll try BERT next - it understands context!
