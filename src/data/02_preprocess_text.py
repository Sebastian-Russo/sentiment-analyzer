"""
Phase 1.2: Text preprocessing exploration
Learn about cleaning and preparing text for ML
"""
import pandas as pd
import re
from collections import Counter

# Load data
print("Loading data...")
train_df = pd.read_csv('data/raw/imdb_train.csv')

print(f"Loaded {len(train_df)} reviews\n")

# Look at a sample review
print("="*60)
print("RAW REVIEW EXAMPLE")
print("="*60)
sample_review = train_df['text'].iloc[0]
print(sample_review[:500] + "...\n")

# Text preprocessing steps
print("="*60)
print("TEXT CLEANING STEPS")
print("="*60)

def clean_text(text):
    """
    Clean text for ML processing

    Steps:
    1. Lowercase
    2. Remove HTML tags
    3. Remove special characters
    4. Remove extra whitespace
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags (e.g., <br />, <p>, etc.)
    text = re.sub(r'<[^>]+>', '', text)

    # 3. Remove special characters (keep only letters, numbers, spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # 4. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Show cleaning effect
print("\nBefore cleaning:")
print(sample_review[:200])
print("\nAfter cleaning:")
cleaned = clean_text(sample_review)
print(cleaned[:200])

# Apply to all reviews
print("\n" + "="*60)
print("CLEANING ALL REVIEWS")
print("="*60)
train_df['text_clean'] = train_df['text'].apply(clean_text)
print("✓ All reviews cleaned")

# Analyze common words
print("\n" + "="*60)
print("MOST COMMON WORDS")
print("="*60)

# Combine all text
all_text = ' '.join(train_df['text_clean'])
words = all_text.split()

# Count word frequencies
word_counts = Counter(words)

print(f"\nTotal words: {len(words):,}")
print(f"Unique words: {len(word_counts):,}")

print("\nTop 20 most common words:")
for word, count in word_counts.most_common(20):
    print(f"  {word}: {count:,}")

# Stopwords analysis
print("\n" + "="*60)
print("STOPWORDS (common words with little meaning)")
print("="*60)

# Common English stopwords
stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
             'of', 'with', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
             'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
             'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
             'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which'}

# Filter out stopwords
meaningful_words = [word for word in words if word not in stopwords]
meaningful_counts = Counter(meaningful_words)

print(f"\nWords after removing stopwords: {len(meaningful_words):,}")
print(f"Unique meaningful words: {len(meaningful_counts):,}")

print("\nTop 20 meaningful words:")
for word, count in meaningful_counts.most_common(20):
    print(f"  {word}: {count:,}")

# Compare positive vs negative vocabulary
print("\n" + "="*60)
print("POSITIVE VS NEGATIVE VOCABULARY")
print("="*60)

# Get words for each sentiment
pos_text = ' '.join(train_df[train_df['sentiment'] == 'positive']['text_clean'])
neg_text = ' '.join(train_df[train_df['sentiment'] == 'negative']['text_clean'])

pos_words = Counter([w for w in pos_text.split() if w not in stopwords])
neg_words = Counter([w for w in neg_text.split() if w not in stopwords])

print("\nTop 10 words in POSITIVE reviews:")
for word, count in pos_words.most_common(10):
    print(f"  {word}: {count:,}")

print("\nTop 10 words in NEGATIVE reviews:")
for word, count in neg_words.most_common(10):
    print(f"  {word}: {count:,}")

# Save cleaned data
print("\n" + "="*60)
print("SAVING CLEANED DATA")
print("="*60)
train_df[['text_clean', 'sentiment', 'label']].to_csv('data/processed/imdb_train_clean.csv', index=False)
print("✓ Saved to data/processed/imdb_train_clean.csv")

print("\n✓ Phase 1.2 complete! Text preprocessing explored.")
