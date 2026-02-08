$ python3 src/data/02_preprocess_text.py
Loading data...
Loaded 25000 reviews

============================================================
RAW REVIEW EXAMPLE
============================================================
I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attent...

============================================================
TEXT CLEANING STEPS
============================================================

Before cleaning:
I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ev

After cleaning:
i rented i am curiousyellow from my video store because of all the controversy that surrounded it when it was first released in 1967 i also heard that at first it was seized by us customs if it ever t

============================================================
CLEANING ALL REVIEWS
============================================================
✓ All reviews cleaned

============================================================
MOST COMMON WORDS
============================================================

Total words: 5,719,281
Unique words: 141,995

Top 20 most common words:
  the: 328,025
  and: 161,555
  a: 161,294
  of: 145,155
  to: 134,814
  is: 106,794
  in: 92,169
  it: 76,258
  this: 73,158
  i: 72,423
  that: 69,189
  was: 47,987
  as: 46,043
  with: 43,722
  for: 43,696
  movie: 41,803
  but: 40,984
  film: 37,461
  on: 33,332
  not: 30,009

============================================================
STOPWORDS (common words with little meaning)
============================================================

Words after removing stopwords: 3,677,477
Unique meaningful words: 141,948

Top 20 meaningful words:
  as: 46,043
  movie: 41,803
  film: 37,461
  not: 30,009
  his: 29,205
  one: 25,154
  its: 24,446
  all: 22,782
  by: 22,257
  who: 20,335
  from: 20,284
  like: 19,562
  so: 19,407
  her: 18,117
  just: 17,530
  about: 17,214
  out: 16,330
  if: 15,757
  some: 15,504
  there: 14,646

============================================================
POSITIVE VS NEGATIVE VOCABULARY
============================================================

Top 10 words in POSITIVE reviews:
  as: 25,842
  film: 19,596
  movie: 18,136
  his: 17,125
  not: 14,044
  one: 12,870
  its: 12,680
  by: 11,878
  all: 11,316
  who: 10,871

Top 10 words in NEGATIVE reviews:
  movie: 23,667
  as: 20,201
  film: 17,865
  not: 15,965
  one: 12,284
  his: 12,080
  its: 11,766
  all: 11,466
  like: 10,879
  so: 10,726

============================================================
SAVING CLEANED DATA
============================================================
✓ Saved to data/processed/imdb_train_clean.csv

✓ Phase 1.2 complete! Text preprocessing explored.



### Key Observations

Vocabulary Size:

5.7 million total words (with repeats)
142,000 unique words - that's a LOT of vocabulary!
After removing stopwords: Still 3.6 million meaningful words

Most Common Words:
Top words are mostly stopwords (the, and, a, of) - these don't tell us about sentiment.
After removing stopwords, we get domain-specific words:

movie, film (makes sense - it's movie reviews!)
not (important for negation: "not good")


Positive vs Negative - Interesting Pattern!
Notice the subtle difference:
POSITIVE reviews use:

film more (19,596 vs 17,865) - sounds more sophisticated/cinephile
Less not (14,044 vs 15,965) - less negation

NEGATIVE reviews use:

movie more (23,667 vs 18,136) - more casual/dismissive?
More not (15,965 vs 14,044) - more negation ("not good", "not worth")
More like and so - informal language

The insight: Sentiment isn't just about "good" vs "bad" words - it's also about writing style!




### Explain text preprocessing and NLP-specific concepts.

What Are Stopwords?
Stopwords = Common words that appear frequently but carry little meaning.
Examples:
the, a, an, and, or, but, in, on, at, to, for, of, with, is, was, are
I, you, he, she, it, we, they, this, that, these, those
Why Remove Them?
Without removing:
"The movie was the best movie I have ever seen"
Most common words: the, the, was, I → Useless!
After removing:
"movie best movie ever seen"
Most common words: movie, best, ever → Meaningful!
When NOT to Remove:
Some tasks need stopwords:

Translation: "I am" vs "I'm not" - stopwords matter!
Question answering: "Who is the president?" - need "the"
Modern transformers (BERT): They learn context, so stopwords help

For our simple model: Removing helps (reduces noise)
For BERT later: We'll keep them

What Are "Meaningful Words"?
Meaningful words = Words that carry semantic content (actual meaning).
Categories:
1. Content Words (meaningful)

Nouns: movie, actor, film, story
Verbs: loved, hated, watched, enjoyed
Adjectives: amazing, terrible, boring, excellent
Adverbs: extremely, very, badly

2. Function Words (stopwords - less meaningful)

Articles: a, an, the
Prepositions: in, on, at, to
Pronouns: I, you, he, she
Conjunctions: and, but, or


Common Text Preprocessing Steps
1. Lowercasing
python"The Movie Was AMAZING!" → "the movie was amazing!"
Why: "Movie" and "movie" should be the same word

2. Remove HTML/XML Tags
python"This is <br /> great!" → "This is great!"
"<p>Amazing</p>" → "Amazing"
Why: Web-scraped text often has HTML

3. Remove Special Characters
python"Best movie!!! #amazing @director" → "Best movie amazing director"
Why: Punctuation usually doesn't help (except sentiment: "!!!" = excited)

4. Remove URLs
python"Check out http://movie.com great film" → "Check out great film"

5. Remove Numbers (sometimes)
python"I watched it 5 times" → "I watched it times"
Depends on task: For sentiment, numbers might not matter. For financial analysis, they're crucial!

6. Tokenization
Breaking text into words (tokens)
python"I love this movie!" → ["I", "love", "this", "movie", "!"]
Why: ML models work with individual words, not sentences

7. Stemming
Reduce words to root form (crude)
python"running" → "run"
"movies" → "movi"  # Note: Not always a real word!
"better" → "better" # Doesn't change
Algorithm: Porter Stemmer (just chops off endings)

8. Lemmatization
Reduce words to root form (smart)
python"running" → "run"
"movies" → "movie"  # Actual word!
"better" → "good"   # Understands comparison
"is, are, was" → "be"
Algorithm: Uses dictionary + grammar rules
Stemming vs Lemmatization:
WordStemmingLemmatizationcaringcarcareranranrunbetterbettergood

9. Remove Emojis/Emoticons
python"Love this 😍 🎬" → "Love this"
OR keep them: Emojis are strong sentiment signals!

10. Expand Contractions
python"I'm" → "I am"
"won't" → "will not"
"should've" → "should have"

11. Handle Negations
python"not good" → "not_good"  # Treat as single token
"isn't great" → "not great" → "not_great"
Why: "not good" means the opposite of "good"!

NLP-Specific Concepts
1. Vocabulary
The set of all unique words in your dataset.
pythonVocabulary size = 142,000 unique words (from IMDB)
Vocabulary explosion: Text has MANY more unique values than tabular data

Tabular: "Contract" has 3 values (month-to-month, 1-year, 2-year)
Text: "Review" has 142,000 unique words!


2. N-grams
Sequences of N words together.
Unigrams (1-gram): Single words
python["I", "love", "this", "movie"]
Bigrams (2-grams): Pairs of words
python["I love", "love this", "this movie"]
Trigrams (3-grams): Triples
python["I love this", "love this movie"]
Why useful: Captures phrases

"not good" (bigram) vs "not" + "good" (unigrams) - different meaning!
"ice cream" vs "ice" + "cream"


3. Bag of Words (BoW)
Represent text as word counts (ignores order).
python"I love love this movie"
→ {I: 1, love: 2, this: 1, movie: 1}
Loses word order: "Dog bites man" vs "Man bites dog" look identical!

4. TF-IDF (Term Frequency-Inverse Document Frequency)
Weight words by importance.
Term Frequency (TF): How often word appears in THIS document
Inverse Document Frequency (IDF): How rare word is across ALL documents
python"movie" appears in 90% of reviews → Low IDF (not distinctive)
"masterpiece" appears in 2% of reviews → High IDF (distinctive!)
Score = TF × IDF
Common words get LOW scores, rare meaningful words get HIGH scores.

5. Word Embeddings
Represent words as vectors (numbers) that capture meaning.
python"king" → [0.2, 0.5, -0.1, 0.8, ...]  # 300 numbers
"queen" → [0.3, 0.4, -0.1, 0.7, ...]  # Similar numbers!
"car" → [-0.5, 0.1, 0.9, -0.2, ...]  # Different numbers
Magic: Similar words have similar numbers!
pythonking - man + woman ≈ queen  # Math with words!
```

**Popular embeddings:**
- Word2Vec
- GloVe
- FastText

---

### **6. Transformers (Modern NLP)**
Models like BERT that understand **context**.

**Old way (Bag of Words):**
```
"bank" always means the same thing
```

**Transformer way (Context-aware):**
```
"I went to the bank" → financial institution
"I sat by the river bank" → riverside
Same word, different meaning based on context!

Preprocessing Pipeline Example
pythonOriginal: "I'm LOVING this movie!!! 😍 http://imdb.com"

1. Lowercase: "i'm loving this movie!!! 😍 http://imdb.com"
2. Remove URLs: "i'm loving this movie!!! 😍"
3. Remove emojis: "i'm loving this movie!!!"
4. Expand contractions: "i am loving this movie!!!"
5. Remove punctuation: "i am loving this movie"
6. Tokenize: ["i", "am", "loving", "this", "movie"]
7. Remove stopwords: ["loving", "movie"]
8. Lemmatize: ["love", "movie"]

Final: ["love", "movie"]

What We Did in Our Script
pythondef clean_text(text):
    text = text.lower()           # Lowercase
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text
We did: Lowercase, remove HTML, remove punctuation, normalize spaces
We didn't do (yet): Stemming, lemmatization, stopword removal in the file

Summary
ConceptWhat It IsWhy It MattersStopwordsCommon words (the, a, is)Remove noiseTokenizationSplit into wordsModels need individual wordsStemming/LemmatizationReduce to root"running" = "run" = same meaningN-gramsWord sequencesCapture phrases ("not good")Bag of WordsWord countsSimple but effectiveTF-IDFWeight by importanceRare words matter moreEmbeddingsWords as vectorsCapture meaning mathematicallyTransformersContext-aware modelsUnderstand "bank" depends on context