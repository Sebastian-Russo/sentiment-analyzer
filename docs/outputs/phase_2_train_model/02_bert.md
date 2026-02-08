### Phase 2.2: Transformer Model (BERT)
Now we'll use a pre-trained BERT model - the state-of-the-art approach for NLP.

What is BERT?
BERT = Bidirectional Encoder Representations from Transformers
Key Difference from Logistic Regression:
Logistic Regression (what we just did):
"not good" → Counts "not" (0.0) + "good" (+2.0) = Positive ❌ WRONG!
BERT:
"not good" → Understands context → Negative ✓ CORRECT!
BERT reads the WHOLE sentence and understands how words relate to each other.

Pre-trained vs Training from Scratch
Training from scratch (what we could do):

Train on 25,000 IMDB reviews
Takes days/weeks
Needs expensive GPUs
Results: okay

Pre-trained BERT (what we'll do):

Google already trained BERT on billions of words
BERT already understands English
We just fine-tune on IMDB (teach it sentiment specifically)
Takes 10-30 minutes
Results: excellent!

It's like: Hiring an English professor and teaching them about movies vs teaching someone English from scratch AND movies.

### Output
$ python3 src/models/02_bert_model.py
Using device: cpu
Training on CPU (will be slower)

Loading data...
Using 1000 samples
Train: 800 | Validation: 200

============================================================
LOADING BERT
============================================================
Loading tokenizer...
✓ Tokenizer loaded

Tokenizing text...
✓ Text tokenized

Loading BERT model...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█| 199/199 [00:00<00:00, 984.33it/s, Materializing param=bert.pooler.dense.weig
BertForSequenceClassification LOAD REPORT from: bert-base-uncased
Key                                        | Status     |
-------------------------------------------+------------+-
cls.predictions.transform.dense.weight     | UNEXPECTED |
cls.seq_relationship.bias                  | UNEXPECTED |
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |
cls.predictions.bias                       | UNEXPECTED |
cls.seq_relationship.weight                | UNEXPECTED |
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |
cls.predictions.transform.dense.bias       | UNEXPECTED |
classifier.bias                            | MISSING    |
classifier.weight                          | MISSING    |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING       :those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
✓ Model loaded

============================================================
TRAINING BERT
============================================================
`logging_dir` is deprecated and will be removed in v5.2. Please set `TENSORBOARD_LOGGING_DIR` instead.
Training on 800 samples for 3 epochs...
This may take 10-30 minutes depending on your hardware...

  5%|█▍                          | 15/300 [01:39<30:06,  6.34s/it]{'loss': '0.6891', 'grad_norm': '7.696', 'learning_rate': '2.45e-05', 'epoch': '0.5'}
{'loss': '0.5729', 'grad_norm': '14.49', 'learning_rate': '4.95e-05', 'epoch': '1'}
{'eval_loss': '0.4949', 'eval_accuracy': '0.785', 'eval_f1': '0.7943', 'eval_precision': '0.7477', 'eval_recall': '0.8469', 'eval_runtime': '23.71', 'eval_samples_per_second': '8.435', 'eval_steps_per_second': '1.054', 'epoch': '1'}
Writing model shards: 100%|█████████| 1/1 [00:00<00:00,  3.44it/s]
 49%|█████████████▎             | 148/300 [10:07<09:08,  3.61s/it]

{'loss': '0.3834', 'grad_norm': '24.18', 'learning_rate': '3.775e-05', 'epoch': '1.5'}
{'loss': '0.3375', 'grad_norm': '10.69', 'learning_rate': '2.525e-05', 'epoch': '2'}
{'eval_loss': '0.4235', 'eval_accuracy': '0.845', 'eval_f1': '0.8473', 'eval_precision': '0.819', 'eval_recall': '0.8776', 'eval_runtime': '24.03', 'eval_samples_per_second': '8.322', 'eval_steps_per_second': '1.04', 'epoch': '2'}
Writing model shards: 100%|█████████| 1/1 [00:00<00:00,  3.50it/s]
{'loss': '0.1745', 'grad_norm': '0.1636', 'learning_rate': '1.275e-05', 'epoch': '2.5'}
{'loss': '0.1527', 'grad_norm': '0.2716', 'learning_rate': '2.5e-07', 'epoch': '3'}
{'eval_loss': '0.7234', 'eval_accuracy': '0.835', 'eval_f1': '0.8325', 'eval_precision': '0.8283', 'eval_recall': '0.8367', 'eval_runtime': '25.21', 'eval_samples_per_second': '7.933', 'eval_steps_per_second': '0.992', 'epoch': '3'}
Writing model shards: 100%|█████████| 1/1 [00:00<00:00,  3.52it/s]
There were missing keys in the checkpoint model loaded: ['bert.embeddings.LayerNorm.weight', 'bert.embeddings.LayerNorm.bias', 'bert.encoder.layer.0.attention.output.LayerNorm.weight', 'bert.encoder.layer.0.attention.output.LayerNorm.bias', 'bert.encoder.layer.0.output.LayerNorm.weight', 'bert.encoder.layer.0.output.LayerNorm.bias', 'bert.encoder.layer.1.attention.output.LayerNorm.weight', 'bert.encoder.layer.1.attention.output.LayerNorm.bias', 'bert.encoder.layer.1.output.LayerNorm.weight', 'bert.encoder.layer.1.output.LayerNorm.bias', 'bert.encoder.layer.2.attention.output.LayerNorm.weight', 'bert.encoder.layer.2.attention.output.LayerNorm.bias', 'bert.encoder.layer.2.output.LayerNorm.weight', 'bert.encoder.layer.2.output.LayerNorm.bias', 'bert.encoder.layer.3.attention.output.LayerNorm.weight', 'bert.encoder.layer.3.attention.output.LayerNorm.bias', 'bert.encoder.layer.3.output.LayerNorm.weight', 'bert.encoder.layer.3.output.LayerNorm.bias', 'bert.encoder.layer.4.attention.output.LayerNorm.weight', 'bert.encoder.layer.4.attention.output.LayerNorm.bias', 'bert.encoder.layer.4.output.LayerNorm.weight', 'bert.encoder.layer.4.output.LayerNorm.bias', 'bert.encoder.layer.5.attention.output.LayerNorm.weight', 'bert.encoder.layer.5.attention.output.LayerNorm.bias', 'bert.encoder.layer.5.output.LayerNorm.weight', 'bert.encoder.layer.5.output.LayerNorm.bias', 'bert.encoder.layer.6.attention.output.LayerNorm.weight', 'bert.encoder.layer.6.attention.output.LayerNorm.bias', 'bert.encoder.layer.6.output.LayerNorm.weight', 'bert.encoder.layer.6.output.LayerNorm.bias', 'bert.encoder.layer.7.attention.output.LayerNorm.weight', 'bert.encoder.layer.7.attention.output.LayerNorm.bias', 'bert.encoder.layer.7.output.LayerNorm.weight', 'bert.encoder.layer.7.output.LayerNorm.bias', 'bert.encoder.layer.8.attention.output.LayerNorm.weight', 'bert.encoder.layer.8.attention.output.LayerNorm.bias', 'bert.encoder.layer.8.output.LayerNorm.weight', 'bert.encoder.layer.8.output.LayerNorm.bias', 'bert.encoder.layer.9.attention.output.LayerNorm.weight', 'bert.encoder.layer.9.attention.output.LayerNorm.bias', 'bert.encoder.layer.9.output.LayerNorm.weight', 'bert.encoder.layer.9.output.LayerNorm.bias', 'bert.encoder.layer.10.attention.output.LayerNorm.weight', 'bert.encoder.layer.10.attention.output.LayerNorm.bias', 'bert.encoder.layer.10.output.LayerNorm.weight', 'bert.encoder.layer.10.output.LayerNorm.bias', 'bert.encoder.layer.11.attention.output.LayerNorm.weight', 'bert.encoder.layer.11.attention.output.LayerNorm.bias', 'bert.encoder.layer.11.output.LayerNorm.weight', 'bert.encoder.layer.11.output.LayerNorm.bias'].
There were unexpected keys in the checkpoint model loaded: ['bert.embeddings.LayerNorm.beta', 'bert.embeddings.LayerNorm.gamma', 'bert.encoder.layer.0.attention.output.LayerNorm.beta', 'bert.encoder.layer.0.attention.output.LayerNorm.gamma', 'bert.encoder.layer.0.output.LayerNorm.beta', 'bert.encoder.layer.0.output.LayerNorm.gamma', 'bert.encoder.layer.1.attention.output.LayerNorm.beta', 'bert.encoder.layer.1.attention.output.LayerNorm.gamma', 'bert.encoder.layer.1.output.LayerNorm.beta', 'bert.encoder.layer.1.output.LayerNorm.gamma', 'bert.encoder.layer.2.attention.output.LayerNorm.beta', 'bert.encoder.layer.2.attention.output.LayerNorm.gamma', 'bert.encoder.layer.2.output.LayerNorm.beta', 'bert.encoder.layer.2.output.LayerNorm.gamma', 'bert.encoder.layer.3.attention.output.LayerNorm.beta', 'bert.encoder.layer.3.attention.output.LayerNorm.gamma', 'bert.encoder.layer.3.output.LayerNorm.beta', 'bert.encoder.layer.3.output.LayerNorm.gamma', 'bert.encoder.layer.4.attention.output.LayerNorm.beta', 'bert.encoder.layer.4.attention.output.LayerNorm.gamma', 'bert.encoder.layer.4.output.LayerNorm.beta', 'bert.encoder.layer.4.output.LayerNorm.gamma', 'bert.encoder.layer.5.attention.output.LayerNorm.beta', 'bert.encoder.layer.5.attention.output.LayerNorm.gamma', 'bert.encoder.layer.5.output.LayerNorm.beta', 'bert.encoder.layer.5.output.LayerNorm.gamma', 'bert.encoder.layer.6.attention.output.LayerNorm.beta', 'bert.encoder.layer.6.attention.output.LayerNorm.gamma', 'bert.encoder.layer.6.output.LayerNorm.beta', 'bert.encoder.layer.6.output.LayerNorm.gamma', 'bert.encoder.layer.7.attention.output.LayerNorm.beta', 'bert.encoder.layer.7.attention.output.LayerNorm.gamma', 'bert.encoder.layer.7.output.LayerNorm.beta', 'bert.encoder.layer.7.output.LayerNorm.gamma', 'bert.encoder.layer.8.attention.output.LayerNorm.beta', 'bert.encoder.layer.8.attention.output.LayerNorm.gamma', 'bert.encoder.layer.8.output.LayerNorm.beta', 'bert.encoder.layer.8.output.LayerNorm.gamma', 'bert.encoder.layer.9.attention.output.LayerNorm.beta', 'bert.encoder.layer.9.attention.output.LayerNorm.gamma', 'bert.encoder.layer.9.output.LayerNorm.beta', 'bert.encoder.layer.9.output.LayerNorm.gamma', 'bert.encoder.layer.10.attention.output.LayerNorm.beta', 'bert.encoder.layer.10.attention.output.LayerNorm.gamma', 'bert.encoder.layer.10.output.LayerNorm.beta', 'bert.encoder.layer.10.output.LayerNorm.gamma', 'bert.encoder.layer.11.attention.output.LayerNorm.beta', 'bert.encoder.layer.11.attention.output.LayerNorm.gamma', 'bert.encoder.layer.11.output.LayerNorm.beta', 'bert.encoder.layer.11.output.LayerNorm.gamma'].
{'train_runtime': '1196', 'train_samples_per_second': '2.006', 'train_steps_per_second': '0.251', 'train_loss': '0.385', 'epoch': '3'}
100%|███████████████████████████| 300/300 [19:56<00:00,  3.99s/it]

✓ Training complete!

============================================================
FINAL EVALUATION
============================================================
100%|█████████████████████████████| 25/25 [00:26<00:00,  1.06s/it]
Accuracy:  0.840
Precision: 0.811
Recall:    0.878
F1-Score:  0.843

============================================================
TEST ON SAMPLE REVIEWS
============================================================

Review: This movie was absolutely amazing! Best film I've ever seen.
  Prediction: Positive (97.9% confident)

Review: Terrible waste of time. Boring and predictable.
  Prediction: Negative (83.1% confident)

Review: It was okay. Not great, not terrible.
  Prediction: Positive (76.2% confident)

Review: This is not good at all.
  Prediction: Negative (83.0% confident)

============================================================
SAVING MODEL
============================================================
Writing model shards: 100%|█████████| 1/1 [00:00<00:00,  3.29it/s]
✓ Saved to models/bert_sentiment/

✓ Phase 2.2 complete! BERT model trained.

============================================================
BASELINE vs BERT COMPARISON
============================================================
Logistic Regression: 86.1% accuracy
BERT:                84.0% accuracy
Improvement:         -2.1%


### Results Summary

Interesting results! BERT actually performed slightly worse than Logistic Regression on this small dataset.

ModelAccuracyPrecisionRecallF1-ScoreLogistic Regression86.1%86.2%85.7%86.0%BERT84.0%81.1%87.8%84.3%
Winner (slightly): Logistic Regression

Why BERT Didn't Win
1. Dataset Size

We only used 1,000 samples (800 train, 200 validation)
BERT shines with large datasets (10k+ samples)
Logistic Regression works better with small datasets

2. Training Time Constraint

CPU training = slow = limited epochs
BERT needs more training to converge
We stopped at 3 epochs (could go 5-10)

3. Simple Task

Sentiment is often word-based ("great" = positive, "bad" = negative)
Logistic Regression's word counting works well here
BERT's context understanding is overkill for this


BERT Did Better At:
Test Case Results:
"This is not good at all."

Logistic Regression would struggle (has "good" = positive word)
BERT: Negative (83% confident) ✓ Understood negation!

"It was okay. Not great, not terrible."

Both models struggled with neutral sentiment
BERT: Positive (76.2%)
LR: Negative (72.1%)
Neither perfect on neutral reviews


Key Insight: Simple Can Win!
This is an important ML lesson:

Simpler models (Logistic Regression) can outperform complex ones (BERT)
Especially with small datasets
BERT is powerful but needs more data to show its strength

Similar to churn prediction:

Logistic Regression beat XGBoost and Random Forest
Sometimes the simplest approach wins!


What We Learned
✅ Text preprocessing (cleaning, tokenization, stopwords)
✅ TF-IDF (converting text to numbers)
✅ Baseline model (86.1% with word counting)
✅ Transformers (BERT understands context but needs more data)
✅ Important lesson: More complex ≠ always better
