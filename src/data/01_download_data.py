"""
Phase 1.1: Download and explore IMDB sentiment dataset
"""
from datasets import load_dataset
import pandas as pd

print("Downloading IMDB dataset...")
# datasets library from Hugging Face - loads common ML datasets
# load_dataset() downloads and caches the data
dataset = load_dataset('imdb')

print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)
print(f"Dataset splits: {dataset.keys()}")
print(f"Train samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")

# Look at structure
print("\n" + "="*60)
print("SAMPLE DATA")
print("="*60)
print(f"Features: {dataset['train'].features}")

# Show examples
print("\n" + "="*60)
print("EXAMPLES")
print("="*60)
for i in range(3):
    example = dataset['train'][i]
    # Sentiment: 0 = negative, 1 = positive
    sentiment = "Positive" if example['label'] == 1 else "Negative"
    # Truncate long reviews for display
    text_preview = example['text'][:200] + "..." if len(example['text']) > 200 else example['text']
    print(f"\nExample {i+1}:")
    print(f"Sentiment: {sentiment}")
    print(f"Text: {text_preview}")

# Convert to pandas for easier manipulation
print("\n" + "="*60)
print("CONVERTING TO PANDAS")
print("="*60)

# Convert train and test to DataFrames
# .to_pandas() converts Hugging Face Dataset to pandas DataFrame
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Map labels to readable names
# 0 → 'negative', 1 → 'positive'
train_df['sentiment'] = train_df['label'].map({0: 'negative', 1: 'positive'})
test_df['sentiment'] = test_df['label'].map({0: 'negative', 1: 'positive'})

print(f"Train DataFrame shape: {train_df.shape}")
print(f"Test DataFrame shape: {test_df.shape}")

# Check class distribution
print("\n" + "="*60)
print("CLASS DISTRIBUTION")
print("="*60)
print("\nTrain set:")
print(train_df['sentiment'].value_counts())
print(f"\nPercentages:")
print(train_df['sentiment'].value_counts(normalize=True) * 100)

print("\nTest set:")
print(test_df['sentiment'].value_counts())

# Save to CSV
print("\n" + "="*60)
print("SAVING DATA")
print("="*60)
train_df.to_csv('data/raw/imdb_train.csv', index=False)
test_df.to_csv('data/raw/imdb_test.csv', index=False)
print("✓ Saved to data/raw/")

# Basic text statistics
print("\n" + "="*60)
print("TEXT STATISTICS")
print("="*60)

# Calculate text length (number of characters)
train_df['text_length'] = train_df['text'].str.len()

print(f"Average review length: {train_df['text_length'].mean():.0f} characters")
print(f"Shortest review: {train_df['text_length'].min()} characters")
print(f"Longest review: {train_df['text_length'].max()} characters")

# Word count
# .str.split() splits text into words, .str.len() counts them
train_df['word_count'] = train_df['text'].str.split().str.len()

print(f"\nAverage word count: {train_df['word_count'].mean():.0f} words")
print(f"Shortest review: {train_df['word_count'].min()} words")
print(f"Longest review: {train_df['word_count'].max()} words")

# Compare positive vs negative
print("\n" + "="*60)
print("POSITIVE VS NEGATIVE COMPARISON")
print("="*60)
for sentiment in ['positive', 'negative']:
    subset = train_df[train_df['sentiment'] == sentiment]
    print(f"\n{sentiment.upper()} reviews:")
    print(f"  Average length: {subset['text_length'].mean():.0f} chars")
    print(f"  Average words: {subset['word_count'].mean():.0f} words")

print("\n✓ Phase 1.1 complete! Data downloaded and explored.")
