$ python3 src/data/01_download_data.py
Downloading IMDB dataset...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

============================================================
DATASET OVERVIEW
============================================================
Dataset splits: dict_keys(['train', 'test', 'unsupervised'])
Train samples: 25000
Test samples: 25000

============================================================
SAMPLE DATA
============================================================
Features: {'text': Value('string'), 'label': ClassLabel(names=['neg', 'pos'])}

============================================================
EXAMPLES
============================================================

Example 1:
Sentiment: Negative
Text: I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ev...

Example 2:
Sentiment: Negative
Text: "I Am Curious: Yellow" is a risible and pretentious steaming pile. It doesn't matter what one's political views are because this film can hardly be taken seriously on any level. As for the claim that ...

Example 3:
Sentiment: Negative
Text: If only to avoid making this type of film in the future. This film is interesting as an experiment but tells no cogent story.<br /><br />One might feel virtuous for sitting thru it because it touches ...

============================================================
CONVERTING TO PANDAS
============================================================
Train DataFrame shape: (25000, 3)
Test DataFrame shape: (25000, 3)

============================================================
CLASS DISTRIBUTION
============================================================

Train set:
sentiment
negative    12500
positive    12500
Name: count, dtype: int64

Percentages:
sentiment
negative    50.0
positive    50.0
Name: proportion, dtype: float64

Test set:
sentiment
negative    12500
positive    12500
Name: count, dtype: int64

============================================================
SAVING DATA
============================================================
✓ Saved to data/raw/

============================================================
TEXT STATISTICS
============================================================
Average review length: 1325 characters
Shortest review: 52 characters
Longest review: 13704 characters

Average word count: 234 words
Shortest review: 10 words
Longest review: 2470 words

============================================================
POSITIVE VS NEGATIVE COMPARISON
============================================================

POSITIVE reviews:
  Average length: 1347 chars
  Average words: 237 words

NEGATIVE reviews:
  Average length: 1303 chars
  Average words: 231 words

✓ Phase 1.1 complete! Data downloaded and explored.


### Key Findings
Dataset:

25,000 training reviews
25,000 test reviews
Perfectly balanced: 50% positive, 50% negative (no class imbalance!)

Text Statistics:

Average review: 234 words (~1,325 characters)
Shortest: 10 words
Longest: 2,470 words (very detailed!)

Interesting:

Positive reviews are slightly longer (237 vs 231 words)
People write more when they're happy! 😊

### Why are there already positive and negative labels?

The Dataset Has Labels - That's the Point!
What We Have:
Review: "This movie was amazing!"  → Label: positive
Review: "Terrible waste of time"   → Label: negative
What We're Building:
New Review: "Best film I've ever seen!" → Model predicts: positive

The Process:
1. Training (What we're doing now):
Model learns: "When I see words like 'amazing', 'best', 'loved' → probably POSITIVE"
Model learns: "When I see words like 'terrible', 'waste', 'boring' → probably NEGATIVE"
We NEED the labels to teach the model!