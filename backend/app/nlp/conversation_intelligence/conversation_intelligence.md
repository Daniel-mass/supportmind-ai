# SupportMind AI — Twitter Conversation Intelligence

## Customer Support on Twitter → Multi-Turn Conversation Complexity Prediction

This document describes the complete machine-learning preparation and training pipeline used to transform the **Customer Support on Twitter** dataset into a leakage-safe, transformer-ready dataset for SupportMind AI.

The objective is not to build a generic intent classifier.

The Twitter dataset was analyzed first and its actual structure was used to determine the appropriate SupportMind capability.

The final role of this dataset is:

> **Conversation Intelligence + Multi-Turn Conversation Complexity Prediction**

The trained model receives a customer's **first message only** and predicts whether that interaction is likely to remain a simple interaction or become a multi-turn support conversation.

---

# 1. Objective

SupportMind AI is an enterprise customer-support intelligence platform.

One of its responsibilities is understanding the likely complexity of an incoming support interaction.

For the Twitter dataset, the final machine-learning task is:

```text
Customer's first message
        │
        ▼
Tokenizer
        │
        ▼
DistilBERT Transformer
        │
        ▼
Classification Head
        │
        ├──────────────┐
        ▼              ▼
     SIMPLE        MULTI_TURN
```

The important design principle is:

> The model must make its prediction from the information available at the beginning of the interaction.

Therefore, future conversation messages cannot be provided to the model as input.

Future conversation structure is used only to construct the training target.

---

# 2. Dataset

## Raw Dataset

Dataset:

```text
Customer Support on Twitter
```

Raw CSV:

```text
C:\Users\danie\supportmind-ai\rawsupportminddata\Customer Support on Twitter\twcs\twcs.csv
```

Schema:

```text
tweet_id
author_id
inbound
created_at
text
response_tweet_id
in_response_to_tweet_id
```

---

# 3. Understanding the Raw Dataset

The Twitter dataset is fundamentally different from a normal tabular classification dataset.

There is no explicit column such as:

```text
intent
category
sentiment
escalation
resolution
conversation_complexity
```

Instead, the dataset contains a conversation graph.

Two fields are especially important:

```text
response_tweet_id
in_response_to_tweet_id
```

They allow conversations to be reconstructed.

---

# 4. Initial Dataset Audit

The first stage was to understand the dataset before deciding what model should be trained.

The complete dataset was loaded.

```python
import pandas as pd

twitter_path = (
    r"C:\Users\danie\supportmind-ai\rawsupportminddata"
    r"\Customer Support on Twitter\twcs\twcs.csv"
)

twitter_df = pd.read_csv(twitter_path)

print("Shape:", twitter_df.shape)
print("\nColumns:")
print(twitter_df.columns.tolist())

print("\nDtypes:")
print(twitter_df.dtypes)

print("\nMissing values:")
print(twitter_df.isna().sum())

print("\nDuplicate tweet IDs:")
print(twitter_df["tweet_id"].duplicated().sum())

print("\nUnique tweet IDs:")
print(twitter_df["tweet_id"].nunique())

print("\nUnique authors:")
print(twitter_df["author_id"].nunique())
```

### Result

```text
Shape: (2811774, 7)
```

Therefore:

```text
2,811,774 tweets
7 columns
```

The dataset contains no missing values in:

```text
tweet_id
author_id
inbound
created_at
text
```

Missingness exists in conversation-link columns.

---

# 5. Raw Dataset Missingness

The audit showed:

```text
response_tweet_id
1,040,629 missing
37.01%

in_response_to_tweet_id
794,335 missing
28.25%
```

These missing values are not automatically data-quality errors.

They have structural meaning.

For example:

```text
in_response_to_tweet_id = missing
```

can mean that a tweet is a conversation root.

Likewise:

```text
response_tweet_id = missing
```

can mean that no response tweet is explicitly linked from that record.

Therefore, the missing values were not blindly dropped.

---

# 6. Understanding `inbound`

The `inbound` column indicates message direction.

```text
True
```

represents an inbound/customer message.

```text
False
```

represents an outbound/company/support response.

Distribution:

```text
Customer / inbound:
1,537,843
54.69%

Company / outbound:
1,273,931
45.31%
```

This distinction is important because the final model is designed to predict complexity from a **customer's first message**.

---

# 7. Understanding Conversation Relationships

The dataset contains two complementary relationships.

## `in_response_to_tweet_id`

This identifies the parent tweet.

Conceptually:

```text
Child Tweet
     │
     └── in_response_to_tweet_id
             │
             ▼
        Parent Tweet
```

Example:

```text
Tweet A
   │
   └── Tweet B
          │
          └── Tweet C
```

Then:

```text
B.in_response_to_tweet_id = A
C.in_response_to_tweet_id = B
```

---

## `response_tweet_id`

This identifies responses generated from a tweet.

A tweet can have more than one response ID.

Therefore the field may look conceptually like:

```text
12345
```

or:

```text
12345,12346
```

or more IDs.

This means the dataset is not simply a flat sequence.

It is a graph/tree-like conversation structure.

---

# 8. Response Relationship Audit

The dataset was analyzed to determine how many tweets have response relationships.

```python
twitter_df["response_tweet_id"].notna().sum()
```

Result:

```text
1,771,145
```

Tweets with response IDs:

```text
1,771,145
```

The response-count distribution was then examined.

The first large groups were:

```text
1 response    1,548,719
2 responses     172,743
3 responses      31,306
4 responses       8,873
5 responses       3,204
...
```

There is also a very long tail.

The maximum observed response count was:

```text
1755
```

However, this raw count was **not selected as the final ML target**.

---

# 9. Why Raw Response Count Was Rejected

A tempting approach would have been:

```text
response_count > 1
        ↓
MULTI_TURN
```

This was rejected.

Reasons:

1. `response_tweet_id` can contain multiple IDs.
2. Conversation structure is not equivalent to a simple count.
3. Response counts are extremely imbalanced.
4. A large response count does not necessarily mean a specific semantic resolution.
5. The model should predict conversation complexity, not exact future response count.
6. Using raw response count directly would turn the task into a noisy count-prediction problem.

The project therefore uses **conversation structure** to construct the target.

---

# 10. Parent-Linkage Analysis

The parent relationship was analyzed:

```python
twitter_df["in_response_to_tweet_id"].notna().sum()
```

Result:

```text
2,017,439
```

Therefore:

```text
Tweets with parent:
2,017,439
71.75%

Tweets without parent:
794,335
28.25%
```

Tweets without parents are candidate conversation roots.

---

# 11. Conversation Root Analysis

Candidate roots:

```text
794,335
```

The root messages were analyzed by direction.

Result:

```text
Customer roots:
787,346
99.12%

Company roots:
6,989
0.88%
```

This was an important discovery.

Almost every conversation begins with a customer.

Therefore the dataset naturally supports:

```text
Customer first message
        ↓
Future conversation structure
        ↓
Conversation complexity target
```

This gave us a valid foundation for the final ML task.

---

# 12. Conversation Depth Analysis

Conversation depth was calculated by following:

```text
tweet
 ↓
parent
 ↓
parent
 ↓
...
 ↓
root
```

A diagnostic depth calculation was performed over the first 10,000 tweet IDs.

Important result:

```text
median depth = 1
mean depth   ≈ 2.92
75th percentile = 3
```

There was a long tail.

Some examples reached the configured traversal cap.

The traversal used:

```text
max_depth = 100
```

Therefore the result:

```text
depth = 100
```

does **not** mean the true maximum conversation depth is exactly 100.

It means traversal reached the diagnostic cap.

This distinction is important for avoiding incorrect claims about the dataset.

---

# 13. Customer and Company Message Length

Customer messages were separately analyzed.

Customer messages:

```text
1,537,843
```

Average:

```text
≈ 109.94 characters
≈ 18.73 words
```

Company messages:

```text
≈ 118.66 characters
≈ 20.34 words
```

The conclusion was:

> Message length alone cannot reliably distinguish customer and company messages.

More importantly, customer messages contain enough natural language for transformer-based modeling.

---

# 14. Customer Text Linguistic Audit

Customer text was analyzed using word and phrase statistics.

Common phrases included:

```text
i have
thank you
customer service
my phone
trying to
can you
can i
how do i
can you help
i need to
i want to
```

This confirmed that the dataset contains real support language.

However, these phrases are **features**, not labels.

For example:

```text
"refund"
"account"
"phone"
"delivery"
"payment"
```

cannot automatically be converted into an intent label.

Therefore no artificial intent labels were created.

---

# 15. spaCy Entity Diagnostic

spaCy was used only as an exploratory diagnostic.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
```

A sample of customer messages was processed.

The first 10,000 customer tweets contained:

```text
Average tokens ≈ 20.33
Average alphabetic tokens ≈ 15.19
Detected entities ≈ 12,608
```

Common entity labels included:

```text
ORG
CARDINAL
DATE
PERSON
GPE
TIME
MONEY
PRODUCT
```

However, many predictions were noisy.

Examples included:

```text
Twitter handles → ORG/PERSON/GPE
LTE → PERSON
Spectrum → PERSON
URLs → entities
Halloween → DATE/EVENT
```

Some genuine entities were detected correctly, but the labels are not ground-truth annotations.

Therefore:

> Twitter was NOT used to train a supervised NER model.

A pretrained NER component can still be used later in SupportMind runtime processing if needed.

---

# 16. Pretrained Sentiment Diagnostic

A pretrained sentiment model was also tested.

Model:

```text
distilbert-base-uncased-finetuned-sst-2-english
```

The first 1,000 customer tweets were evaluated.

Result:

```text
NEGATIVE: 793
79.30%

POSITIVE: 207
20.70%
```

Mean confidence:

```text
≈ 0.9723
```

This diagnostic was used to understand whether sentiment was strongly associated with conversation length.

It was not.

Therefore:

> The Twitter dataset was not used to train a sentiment classifier.

Sentiment can still be used as a standalone SupportMind intelligence signal using an existing/pretrained model.

---

# 17. Response Distribution Analysis

Customer tweets were grouped by future response count.

Approximate distribution:

```text
0 responses      234,014    15.22%
1 response     1,141,874    74.25%
2 responses      128,090     8.33%
3 responses       22,424     1.46%
4–5                8,090     0.53%
6–10               2,132     0.14%
11–20                680     0.04%
21–50                375     0.02%
51–100                98     0.01%
101+                  66     ~0.00%
```

This demonstrated that exact response count is heavily skewed.

Therefore:

```text
Exact response-count prediction
```

was rejected.

---

# 18. Final Twitter Dataset Role

After completing the dataset-first analysis, the Twitter dataset role was locked.

## Final role

```text
Customer Support on Twitter
            ↓
Conversation Intelligence
            +
Multi-Turn Conversation Complexity Prediction
```

---

# 19. Tasks Explicitly Rejected

The following tasks were considered but rejected for this dataset:

```text
Intent Classification             ❌
Supervised NER                    ❌
Twitter-trained Sentiment Model   ❌
Escalation Prediction             ❌
Resolution Prediction             ❌
Exact Response Count Prediction   ❌
```

Why?

Because the dataset does not contain trustworthy supervised labels for these tasks.

---

# 20. Final ML Target

The final target is:

```text
SIMPLE
MULTI_TURN
```

The model predicts whether the initial customer interaction develops into a multi-turn conversation.

Conceptually:

```text
First customer message
        │
        │
        ▼
Future conversation structure
        │
        ▼
Target
        │
        ├── SIMPLE
        │
        └── MULTI_TURN
```

The critical point is:

> Future information is used to construct the target but is never given to the model as an input feature.

---

# 21. Leakage Principle

This is one of the most important parts of the pipeline.

We want:

```text
INPUT
First customer message
```

and:

```text
TARGET
Future conversation complexity
```

We must NOT provide:

```text
total_nodes
customer_nodes
company_nodes
max_depth
future tweets
response_count
future message text
```

to the transformer.

If we did, the model could directly observe information that only becomes available after the interaction develops.

That would be target leakage.

---

# 22. Conversation-Level Modeling Dataset

The conversation graph was used to identify customer-root conversations.

The resulting modeling dataset contained:

```text
787,346 rows
3 columns
```

Columns:

```text
root_id
text
multi_turn
```

This means each modeling row represents the initial customer message associated with a conversation root.

---

# 23. Initial Modeling Dataset

The modeling dataset was created conceptually as:

```python
modeling_dataset = customer_root_df[
    [
        "root_id",
        "text",
        "multi_turn",
    ]
].copy()
```

The resulting shape:

```text
(787346, 3)
```

---

# 24. Leakage Columns Removed

During target construction, conversation-level information was available.

Examples:

```text
total_nodes
customer_nodes
company_nodes
max_depth
```

These are useful for understanding the conversation after the fact.

They are not valid model inputs.

Therefore they were removed from the final modeling feature set.

The final supervised dataset contains only:

```text
text
multi_turn
```

plus the temporary conversation identifier:

```text
root_id
```

which is later removed before tokenization.

---

# 25. Duplicate Text Analysis

Before splitting the dataset, exact text duplication was analyzed.

Result:

```text
Rows with duplicated text:
4,260

Rows with unique text:
783,086
```

There were:

```text
415 conflicting duplicate-text groups
```

containing:

```text
2,852 rows
```

A conflicting group means the exact same customer message can occur with different future conversation outcomes.

For example:

```text
same initial message
        │
        ├── SIMPLE conversation
        │
        └── MULTI_TURN conversation
```

This is legitimate.

It means the wording of an initial message is not always sufficient to determine future behavior.

We therefore did not arbitrarily remove all duplicate messages.

---

# 26. Why Duplicate Text Was Important

If duplicate messages were randomly split between train and validation/test, the same text could appear in both sets.

That could inflate evaluation results.

For example:

```text
TRAIN:
"my payment failed"

TEST:
"my payment failed"
```

The model would effectively see the exact same input during training.

Therefore a normal row-level random split was not sufficient.

This led to the group-aware split.

---

# 27. Group-Aware Split

The grouping unit was:

```text
exact text
```

Every identical text belongs to the same split.

Therefore:

```text
same text
    ↓
one split only
```

This prevents exact text leakage across:

```text
train
validation
test
```

---

# 28. Unique Text Groups

After grouping:

```text
Unique text groups:
784,060
```

Group-level target distribution:

```text
SIMPLE groups:
460,778

MULTI_TURN groups:
323,282

Conflicting groups:
415
```

---

# 29. Train / Validation / Test Group Split

The unique text groups were split using:

```python
from sklearn.model_selection import train_test_split
```

Configuration:

```python
test_size=0.20
random_state=42
stratify=group_target
```

The remaining 80% was split equally between validation and test.

Final unique-text groups:

```text
TRAIN:
627,248

VALIDATION:
78,406

TEST:
78,406
```

---

# 30. Group-Level Target Distribution

Training groups:

```text
SIMPLE       368,622
MULTI_TURN   258,626
```

Validation groups:

```text
SIMPLE        46,078
MULTI_TURN    32,328
```

Test groups:

```text
SIMPLE        46,078
MULTI_TURN    32,328
```

The class proportions are therefore preserved.

---

# 31. Mapping Groups Back to Original Rows

The split assignment was then mapped back to every original modeling row.

Final row counts:

```text
TRAIN         630,075
VALIDATION     78,675
TEST           78,596
UNKNOWN             0
```

The total is:

```text
630,075
+ 78,675
+ 78,596
= 787,346
```

Therefore:

```text
Original rows = split rows
```

No rows were lost.

---

# 32. Final Row-Level Target Distribution

Training:

```text
SIMPLE       370,706
MULTI_TURN   259,369

SIMPLE       58.84%
MULTI_TURN   41.16%
```

Validation:

```text
SIMPLE        46,253
MULTI_TURN    32,422

SIMPLE        58.79%
MULTI_TURN    41.21%
```

Test:

```text
SIMPLE        46,202
MULTI_TURN    32,394

SIMPLE        58.78%
MULTI_TURN    41.22%
```

The distribution is highly consistent across all splits.

---

# 33. Split Integrity Checks

Exact-text overlap was explicitly tested.

```text
Train ↔ Validation:
0

Train ↔ Test:
0

Validation ↔ Test:
0
```

Also:

```text
Unknown split rows:
0
```

And:

```text
Original rows:
787,346

Split rows:
787,346
```

Therefore the split passed all integrity checks.

## Split Status

```text
LOCKED
```

This split should not be regenerated casually.

---

# 34. Hugging Face Dataset Conversion

The pandas DataFrames were converted to Hugging Face `Dataset` objects.

```python
from datasets import Dataset

train_dataset = Dataset.from_pandas(
    train_df[["root_id", "text", "multi_turn"]],
    preserve_index=False,
)

validation_dataset = Dataset.from_pandas(
    val_df[["root_id", "text", "multi_turn"]],
    preserve_index=False,
)

test_dataset = Dataset.from_pandas(
    test_df[["root_id", "text", "multi_turn"]],
    preserve_index=False,
)
```

Result:

```text
Train:
630,075

Validation:
78,675

Test:
78,596
```

Columns:

```text
root_id
text
multi_turn
```

---

# 35. Why Hugging Face Dataset Was Used

The transformer training pipeline benefits from Hugging Face's dataset abstraction because it supports:

```text
batched processing
tokenization
efficient dataset storage
dataset mapping
Trainer integration
```

This allows the full dataset to be processed without converting the entire training workflow into a custom PyTorch data-loading system.

---

# 36. Transformer Selection

The chosen model is:

```text
distilbert-base-uncased
```

Reason:

* suitable for English text classification
* much smaller than BERT-large style architectures
* fast enough for the project's scale
* approximately 67M parameters
* appropriate for short customer-support messages
* suitable for a fit-for-purpose SupportMind model

The project does not require a huge transformer.

The goal is:

> A model that is useful for SupportMind's conversation-intelligence role.

Not:

> The theoretically highest-performing transformer possible.

---

# 37. Tokenizer

The tokenizer was loaded with:

```python
from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)
```

Tokenizer:

```text
BertTokenizer
```

Vocabulary:

```text
30,522
```

Maximum model length:

```text
512
```

---

# 38. Why No Manual NLP Cleaning Was Applied

The pipeline intentionally does NOT perform generic transformations such as:

```text
stopword removal
stemming
aggressive lemmatization
manual punctuation removal
manual regex normalization
manual token filtering
```

This is intentional.

A transformer tokenizer is designed to process natural language directly.

Aggressive preprocessing could remove information useful for conversation-complexity prediction.

For example:

```text
"can you help me?"

"can you please help me?"

"it still doesn't work"

"I've been waiting for 3 days"
```

contain conversational signals.

Therefore the raw meaningful language was preserved.

---

# 39. Token-Length Analysis

Token lengths were measured on the training data.

Statistics:

```text
Examples:
630,075

Mean:
35.718 tokens

Median:
34

75th percentile:
43

90th percentile:
55

95th percentile:
63

99th percentile:
80

99.5th percentile:
87

99.9th percentile:
106

Maximum:
256
```

Only:

```text
140 / 630,075
```

training examples exceeded 128 tokens.

Percentage:

```text
0.0222%
```

This is extremely small.

Therefore:

```python
max_length = 128
```

was selected.

---

# 40. Why `max_length=128`

128 tokens provides a strong trade-off:

```text
Short enough
    ↓
Lower GPU memory
    ↓
Faster training
    ↓
Suitable for RTX 3050 6GB
```

while retaining almost all customer messages without truncation.

Only:

```text
0.0222%
```

of training examples exceed the selected length.

Therefore the truncation trade-off is acceptable for this task.

## Decision

```text
max_length = 128
LOCKED
```

---

# 41. Tokenization Function

The tokenizer was applied using batched mapping.

```python
def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
        padding=False,
    )
```

The datasets were processed in batches:

```python
train_tokenized = train_dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=5000,
)

validation_tokenized = validation_dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=5000,
)

test_tokenized = test_dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=5000,
)
```

---

# 42. Why `padding=False` During Tokenization

Padding was deliberately disabled during preprocessing.

```python
padding=False
```

Instead, dynamic padding is performed later by the data collator.

This avoids storing unnecessary padding tokens.

Example:

```text
Message A → 20 tokens
Message B → 45 tokens
Message C → 70 tokens
```

Instead of storing all three as 70 tokens during preprocessing, dynamic batching can pad only to the longest sequence in the current batch.

This reduces computation and memory usage.

---

# 43. Tokenized Dataset Columns

After tokenization, the datasets contained:

```text
root_id
text
multi_turn
input_ids
token_type_ids
attention_mask
```

DistilBERT does not require:

```text
token_type_ids
```

so it is removed later.

---

# 44. Label Mapping

The target is binary.

The mapping was defined as:

```python
label2id = {
    "SIMPLE": 0,
    "MULTI_TURN": 1,
}

id2label = {
    0: "SIMPLE",
    1: "MULTI_TURN",
}
```

This is important because the model's classification head outputs two logits:

```text
logit 0 → SIMPLE
logit 1 → MULTI_TURN
```

---

# 45. Adding Transformer Labels

The original target:

```text
multi_turn
```

contains boolean values.

The Trainer expects:

```text
labels
```

Therefore the label column was created.

```python
def add_labels(batch):
    return {
        "labels": [
            label2id[value]
            for value in batch["multi_turn"]
        ]
    }
```

Applied to all splits:

```python
train_tokenized = train_tokenized.map(
    add_labels,
    batched=True,
)

validation_tokenized = validation_tokenized.map(
    add_labels,
    batched=True,
)

test_tokenized = test_tokenized.map(
    add_labels,
    batched=True,
)
```

---

# 46. Label Verification

The final labels were checked.

Training:

```text
SIMPLE:
370,706

MULTI_TURN:
259,369
```

Validation:

```text
SIMPLE:
46,253

MULTI_TURN:
32,422
```

Test:

```text
SIMPLE:
46,202

MULTI_TURN:
32,394
```

Unique encoded values:

```text
[0, 1]
```

Mapping:

```text
0 → SIMPLE
1 → MULTI_TURN
```

No invalid labels were found.

---

# 47. Removing Non-Model Columns

Before training, the following columns were removed:

```python
columns_to_remove = [
    "root_id",
    "text",
    "multi_turn",
    "token_type_ids",
]
```

Then:

```python
train_tokenized = train_tokenized.remove_columns(
    columns_to_remove
)

validation_tokenized = validation_tokenized.remove_columns(
    columns_to_remove
)

test_tokenized = test_tokenized.remove_columns(
    columns_to_remove
)
```

Final columns:

```text
input_ids
attention_mask
labels
```

This is the final transformer-ready representation.

---

# 48. What `input_ids` Mean

The text is converted into integer token IDs.

For example, conceptually:

```text
"my payment failed"
        ↓
Tokenizer
        ↓
[2026, 7909, ...]
```

The actual IDs depend on the DistilBERT vocabulary.

The transformer does not directly consume raw strings.

It consumes:

```text
input_ids
attention_mask
```

---

# 49. What `attention_mask` Means

The attention mask tells the transformer which positions contain actual tokens.

Conceptually:

```text
Real token → 1
Padding    → 0
```

Because dynamic padding is used, the attention mask changes according to the batch.

---

# 50. Final ML-Feedable Dataset

At the end of preprocessing, the dataset has been transformed from:

```text
Raw Twitter CSV
```

into:

```text
input_ids
attention_mask
labels
```

This is the actual machine-learning feedable representation.

The transformation is:

```text
RAW CSV
   │
   ▼
Conversation graph reconstruction
   │
   ▼
Customer conversation roots
   │
   ▼
Future conversation structure
   │
   ▼
SIMPLE / MULTI_TURN target
   │
   ▼
Leakage removal
   │
   ▼
Group-aware train/validation/test split
   │
   ▼
Hugging Face Dataset
   │
   ▼
DistilBERT tokenizer
   │
   ▼
input_ids
attention_mask
labels
```

---

# 51. Saving Tokenized Datasets

Tokenization is computationally expensive over hundreds of thousands of examples.

Therefore the tokenized datasets were persisted to disk.

Directory:

```text
C:\Users\danie\supportmind-ai\data\processed\conversation_intelligence\tokenized
```

Saved datasets:

```text
tokenized/
├── train/
├── validation/
└── test/
```

Code:

```python
from pathlib import Path

tokenized_dataset_dir = Path(
    r"C:\Users\danie\supportmind-ai\data\processed"
    r"\conversation_intelligence\tokenized"
)

tokenized_dataset_dir.mkdir(
    parents=True,
    exist_ok=True,
)

train_tokenized.save_to_disk(
    str(tokenized_dataset_dir / "train")
)

validation_tokenized.save_to_disk(
    str(tokenized_dataset_dir / "validation")
)

test_tokenized.save_to_disk(
    str(tokenized_dataset_dir / "test")
)
```

Result:

```text
Tokenized datasets saved successfully.
```

This means a Jupyter restart does not require tokenization to be repeated.

---

# 52. Dataset Recovery After Kernel Restart

The datasets can be restored with:

```python
from pathlib import Path
from datasets import load_from_disk

tokenized_dataset_dir = Path(
    r"C:\Users\danie\supportmind-ai\data\processed"
    r"\conversation_intelligence\tokenized"
)

train_tokenized = load_from_disk(
    str(tokenized_dataset_dir / "train")
)

validation_tokenized = load_from_disk(
    str(tokenized_dataset_dir / "validation")
)

test_tokenized = load_from_disk(
    str(tokenized_dataset_dir / "test")
)

print("Tokenized datasets restored.")
print("Train:", len(train_tokenized))
print("Validation:", len(validation_tokenized))
print("Test:", len(test_tokenized))
```

Expected:

```text
Train: 630075
Validation: 78675
Test: 78596
```

---

# 53. GPU Environment

Initially the Jupyter environment had CPU-only PyTorch.

Initial diagnostic:

```text
PyTorch: 2.12.1+cpu
CUDA available: False
```

The system GPU itself was verified separately.

GPU:

```text
NVIDIA GeForce RTX 3050 6GB Laptop GPU
```

VRAM:

```text
6 GB
```

The CUDA-enabled PyTorch build was then installed.

Final environment:

```text
PyTorch:
2.11.0+cu128

CUDA:
True

GPU:
NVIDIA GeForce RTX 3050 6GB Laptop GPU

VRAM:
6.0 GB
```

Verification:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024**3),
            2,
        ),
        "GB",
    )
```

---

# 54. Trainer Dependency

The first attempt to create `TrainingArguments` failed because Accelerate was not available to the Transformers Trainer environment.

Error:

```text
ImportError:
Using the `Trainer` with `PyTorch` requires
accelerate>=1.1.0
```

Accelerate was installed:

```powershell
python -m pip install -U "accelerate>=1.1.0"
```

Installed version:

```text
Accelerate 1.15.0
```

The Jupyter kernel was restarted to clear stale dependency detection.

After restart:

```text
TrainingArguments created successfully.
```

Therefore the environment problem was resolved.

---

# 55. Training Configuration

Final configuration:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=(
        r"C:\Users\danie\supportmind-ai"
        r"\backend\app\nlp\conversation_intelligence"
        r"\training_checkpoints"
    ),

    num_train_epochs=2,

    learning_rate=2e-5,

    weight_decay=0.01,

    warmup_steps=1970,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=32,

    gradient_accumulation_steps=2,

    fp16=True,

    eval_strategy="epoch",

    save_strategy="epoch",

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="macro_f1",

    greater_is_better=True,

    logging_steps=500,

    seed=42,

    data_seed=42,

    dataloader_num_workers=2,

    dataloader_pin_memory=True,

    report_to="none",

    run_name=(
        "supportmind_conversation_complexity_distilbert"
    ),
)
```

---

# 56. Why `warmup_steps=1970`

The training dataset contains:

```text
630,075 examples
```

Effective batch size:

```text
16 × 2
= 32
```

Approximate optimizer steps per epoch:

```text
630,075 / 32
≈ 19,690
```

For two epochs:

```text
≈ 39,380 optimizer steps
```

Approximately 5% warmup:

```text
39,380 × 0.05
≈ 1,969
```

Therefore:

```python
warmup_steps=1970
```

was selected.

This replaced the deprecated:

```python
warmup_ratio=0.05
```

configuration.

---

# 57. Effective Batch Size

Physical GPU batch:

```text
16
```

Gradient accumulation:

```text
2
```

Therefore:

```text
Effective batch size
=
16 × 2
=
32
```

This allows the model to behave approximately like it is training with batches of 32 while keeping the instantaneous GPU batch smaller for the RTX 3050 6GB GPU.

---

# 58. Why FP16

The GPU supports half-precision computation.

Therefore:

```python
fp16=True
```

was enabled.

Advantages:

```text
Lower GPU memory usage
Faster tensor operations
Larger practical batch capacity
```

This is useful for training DistilBERT on a 6GB laptop GPU.

---

# 59. Model Initialization

The model was loaded using:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)
```

Model:

```text
distilbert-base-uncased
```

Classification labels:

```text
0 → SIMPLE
1 → MULTI_TURN
```

---

# 60. Expected Missing Classification Head Warning

When loading the base DistilBERT checkpoint, Transformers reported that:

```text
pre_classifier.*
classifier.*
```

were newly initialized.

This is expected.

The base checkpoint contains the pretrained language model but not our specific:

```text
SIMPLE vs MULTI_TURN
```

classification head.

Therefore the classification head must be learned during fine-tuning.

This is not a failed model load.

---

# 61. Model Size

Total parameters:

```text
66,955,010
```

All parameters are trainable.

Approximate FP32 model size:

```text
255.41 MB
```

The model is therefore small enough for the available GPU compared with much larger transformer architectures.

---

# 62. Data Collator

Dynamic padding was configured using:

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)
```

The collator performs padding at batch time.

This is more efficient than padding every dataset example to 128 tokens permanently.

---

# 63. Batch Validation

A four-example batch was created to validate the final dataset representation.

Expected structure:

```text
input_ids
attention_mask
labels
```

Observed:

```text
input_ids shape:
torch.Size([4, 52])

attention_mask shape:
torch.Size([4, 52])

labels shape:
torch.Size([4])
```

Labels:

```text
[1, 0, 0, 0]
```

Maximum sequence length in the batch:

```text
52
```

This verified:

```text
Dynamic padding     ✅
Labels               ✅
Attention masks      ✅
Input IDs            ✅
Expected columns     ✅
```

---

# 64. Evaluation Metrics

The model is evaluated using:

```text
Accuracy
Precision — MULTI_TURN
Recall — MULTI_TURN
F1 — MULTI_TURN
Macro Precision
Macro Recall
Macro F1
```

Metric function:

```python
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return {
        "accuracy": accuracy,

        "precision_multi_turn": precision,

        "recall_multi_turn": recall,

        "f1_multi_turn": f1,

        "macro_precision": macro_precision,

        "macro_recall": macro_recall,

        "macro_f1": macro_f1,
    }
```

---

# 65. Why Macro-F1 Matters

The dataset is moderately imbalanced:

```text
SIMPLE:
≈ 59%

MULTI_TURN:
≈ 41%
```

Accuracy alone can hide poor performance on one class.

Macro-F1 calculates the F1 score for both classes and averages them.

Therefore:

```text
Macro-F1
```

is an important overall metric.

For SupportMind, we also care strongly about:

```text
MULTI_TURN recall
```

because failing to identify a potentially complex interaction can be operationally important.

---

# 66. Metric Function Validation

Before training, the metrics function was tested with synthetic predictions.

Output:

```text
accuracy:
1.0

precision_multi_turn:
1.0

recall_multi_turn:
1.0

f1_multi_turn:
1.0

macro_precision:
1.0

macro_recall:
1.0

macro_f1:
1.0
```

This confirms that the metric implementation itself behaves correctly.

---

# 67. Trainer Creation

The Hugging Face Trainer was created:

```python
from transformers import Trainer

trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_tokenized,

    eval_dataset=validation_tokenized,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,
)
```

Transformers version:

```text
5.13.0
```

The current Transformers API uses:

```python
processing_class=tokenizer
```

rather than relying on older Trainer argument conventions.

---

# 68. Trainer Validation

Trainer successfully initialized.

Result:

```text
Training examples:
630075

Validation examples:
78675

Model device:
cuda:0
```

This confirms that the Trainer recognized the CUDA device.

---

# 69. Smoke Test

Before launching full training, a tiny evaluation was performed.

Only:

```text
32 examples
```

were used.

This is NOT final training.

It is an engineering smoke test.

Purpose:

```text
Dataset
   ↓
Data Collator
   ↓
Trainer
   ↓
CUDA
   ↓
DistilBERT
   ↓
Forward Pass
   ↓
Loss
   ↓
Metrics
```

---

# 70. Smoke Test Code

```python
import torch

print("Before smoke test:")

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "Allocated:",
    round(
        torch.cuda.memory_allocated()
        / (1024**2),
        2,
    ),
    "MB"
)

print(
    "Reserved:",
    round(
        torch.cuda.memory_reserved()
        / (1024**2),
        2,
    ),
    "MB"
)

smoke_test_dataset = (
    train_tokenized.select(range(32))
)

print(
    "\nRunning smoke test on 32 examples..."
)

smoke_output = trainer.evaluate(
    eval_dataset=smoke_test_dataset,
)

print(
    "\nSmoke test completed successfully."
)

print("Metrics:")

for key, value in smoke_output.items():
    print(f"{key}: {value}")
```

---

# 71. Smoke Test Hardware Result

Before the test:

```text
GPU:
NVIDIA GeForce RTX 3050 6GB Laptop GPU

Allocated:
256.5 MB

Reserved:
292.0 MB
```

The test completed successfully.

Therefore:

```text
CUDA execution       ✅
Trainer evaluation   ✅
Data collator        ✅
DistilBERT forward   ✅
Metrics              ✅
GPU memory           ✅
```

---

# 72. Smoke Test Metrics

The 32-example smoke test produced:

```text
eval_loss:
0.701202392578125

eval_accuracy:
0.375

eval_precision_multi_turn:
0.4

eval_recall_multi_turn:
0.8571428571428571

eval_f1_multi_turn:
0.5454545454545454

eval_macro_precision:
0.2

eval_macro_recall:
0.42857142857142855

eval_macro_f1:
0.2727272727272727
```

These values must NOT be interpreted as the final model performance.

The model has not yet been trained.

The classification head is effectively untrained at this stage.

The smoke test only verifies that the pipeline works.

---

# 73. Current Training State

At this point:

```text
Data preparation                     ✅
Conversation reconstruction          ✅
Target construction                  ✅
Leakage prevention                   ✅
Duplicate-text analysis              ✅
Group-aware splitting                ✅
Split integrity validation           ✅
Hugging Face conversion              ✅
Tokenizer                             ✅
Token-length analysis                ✅
max_length=128                       ✅
Full tokenization                    ✅
Label encoding                       ✅
Final ML-ready columns               ✅
Tokenized dataset persistence        ✅
CUDA environment                     ✅
DistilBERT initialization            ✅
Data collator                        ✅
Metrics                              ✅
Trainer                              ✅
GPU smoke test                       ✅
```

Full model training:

```text
⏳ NOT YET COMPLETED
```

---

# 74. Current Final Dataset Sizes

The final tokenized datasets contain:

```text
TRAIN:
630,075

VALIDATION:
78,675

TEST:
78,596
```

No arbitrary reduction was made for final modelling.

The full training dataset will be used.

---

# 75. Why We Are Not Training on a Small Sample

A small subset was used only for:

```text
pipeline smoke testing
```

It is not a modelling dataset.

The final model must train on:

```text
630,075 examples
```

because reducing the dataset would unnecessarily discard available training signal.

The project requirement is:

> Use the complete relevant dataset for final modelling.

---

# 76. Final Training Architecture

The final architecture is:

```text
                    CUSTOMER SUPPORT ON TWITTER
                               │
                               ▼
                    Conversation Graph
                               │
                               ▼
                     Customer Root Tweets
                               │
                               ▼
                     Target Construction
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
                 SIMPLE              MULTI_TURN
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    Group-Aware Split
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
          TRAIN             VALIDATION          TEST
        630,075              78,675            78,596
             │
             ▼
        DistilBERT Tokenizer
             │
             ▼
       input_ids + attention_mask
             │
             ▼
       DistilBERT Encoder
             │
             ▼
       Classification Head
             │
             ▼
       SIMPLE / MULTI_TURN
```

---

# 77. Final Model Configuration

```text
Model:
distilbert-base-uncased

Task:
Binary sequence classification

Classes:
SIMPLE
MULTI_TURN

Maximum sequence length:
128

Training examples:
630,075

Validation examples:
78,675

Test examples:
78,596

Epochs:
2

Learning rate:
2e-5

Weight decay:
0.01

Warmup:
1,970 steps

Train batch:
16

Evaluation batch:
32

Gradient accumulation:
2

Effective batch:
32

Precision:
FP16

Evaluation:
Every epoch

Checkpoint saving:
Every epoch

Best model metric:
macro_f1

Random seed:
42
```

---

# 78. Output Directories

## Training Checkpoints

```text
C:\Users\danie\supportmind-ai\backend\app\nlp\conversation_intelligence\training_checkpoints
```

## Final Model

The final model must be saved exactly here:

```text
C:\Users\danie\supportmind-ai\backend\app\nlp\conversation_intelligence\models\supportmind_conversation_complexity
```

---

# 79. Planned Full Training

The next stage is the actual full training run:

```python
train_result = trainer.train()
```

This will use:

```text
630,075 training examples
```

for:

```text
2 epochs
```

with:

```text
effective batch size = 32
```

The model will be evaluated at the end of each epoch.

---

# 80. Full Training Code

Once the smoke test has passed, full training can be started with:

```python
print("=" * 60)
print("STARTING FULL SUPPORTMIND TRAINING")
print("=" * 60)

print(
    f"Training examples : {len(train_tokenized):,}"
)

print(
    f"Validation examples: "
    f"{len(validation_tokenized):,}"
)

print(
    f"Epochs             : "
    f"{training_args.num_train_epochs}"
)

print(
    f"Learning rate      : "
    f"{training_args.learning_rate}"
)

print(
    f"Effective batch    : "
    f"{16 * 2}"
)

print(
    f"FP16               : "
    f"{training_args.fp16}"
)

print(
    f"GPU                : "
    f"{torch.cuda.get_device_name(0)}"
)

print("=" * 60)
print("Training started...")
print("=" * 60)

train_result = trainer.train()

print("=" * 60)
print("FULL TRAINING COMPLETED")
print("=" * 60)

print("Training metrics:")

for key, value in train_result.metrics.items():
    print(f"{key}: {value}")
```

---

# 81. Important Distinction: Smoke Test vs Final Training

## Smoke Test

```text
32 examples
```

Purpose:

```text
Engineering validation
```

It answers:

> Does the complete training/evaluation pipeline work?

---

## Final Training

```text
630,075 examples
```

Purpose:

```text
Actual model learning
```

It answers:

> Can DistilBERT learn the relationship between the customer's first message and future conversation complexity?

These must not be confused.

---

# 82. Final Evaluation After Training

After training, the locked test set will be evaluated.

Expected evaluation metrics:

```text
Accuracy
MULTI_TURN precision
MULTI_TURN recall
MULTI_TURN F1
Macro precision
Macro recall
Macro F1
```

The test set must remain untouched during model development.

It is used only for final evaluation.

---

# 83. Additional Post-Training Analysis

After final training, the following should be inspected:

```text
Confusion matrix
Classification report
MULTI_TURN recall
MULTI_TURN F1
Macro-F1
```

Potentially:

```text
ROC-AUC
```

may also be calculated if useful.

The purpose is not to endlessly tune the model.

The model only needs to be:

> **FIT for SupportMind's Conversation Intelligence role.**

---

# 84. Final Model Saving

After the model has been evaluated, the final model and tokenizer should be saved to:

```text
C:\Users\danie\supportmind-ai\backend\app\nlp\conversation_intelligence\models\supportmind_conversation_complexity
```

Example:

```python
from pathlib import Path

final_model_dir = Path(
    r"C:\Users\danie\supportmind-ai\backend\app"
    r"\nlp\conversation_intelligence\models"
    r"\supportmind_conversation_complexity"
)

final_model_dir.mkdir(
    parents=True,
    exist_ok=True,
)

trainer.save_model(
    str(final_model_dir)
)

tokenizer.save_pretrained(
    str(final_model_dir)
)

print(
    "Final model and tokenizer saved successfully."
)

print(
    "Location:",
    final_model_dir
)
```

---

# 85. Expected Final Model Contents

The final directory should contain Hugging Face model artifacts such as:

```text
supportmind_conversation_complexity/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── vocab.txt
├── special_tokens_map.json
└── ...
```

The exact auxiliary files depend on the Transformers version and tokenizer serialization.

---

# 86. Inference Concept

Once saved, the model can process a new customer message.

Example:

```text
Customer:
"My payment has failed again and I still haven't received my refund."
```

Pipeline:

```text
Raw customer message
        │
        ▼
DistilBERT tokenizer
        │
        ▼
input_ids
attention_mask
        │
        ▼
DistilBERT
        │
        ▼
Classification Head
        │
        ▼
Probability
        │
        ├── SIMPLE
        │
        └── MULTI_TURN
```

The prediction is about likely interaction complexity.

It is NOT:

```text
intent
sentiment
resolution
escalation
```

---

# 87. How This Fits SupportMind

The conversation-complexity model becomes one signal in SupportMind's broader intelligence system.

Conceptually:

```text
Incoming Customer Message
             │
             ▼
    Conversation Intelligence
             │
       ┌─────┴─────┐
       ▼           ▼
   Complexity    Other NLP
   Prediction    Signals
       │
       ▼
   Coordinator
       │
       ├── Retrieval
       ├── Resolution
       ├── Escalation
       └── Evaluation
```

The model does not independently decide the final support action.

It provides intelligence to downstream components.

---

# 88. What This Model Does NOT Claim

The model does not claim:

```text
"This ticket will be resolved."
```

It does not claim:

```text
"This customer will escalate."
```

It does not claim:

```text
"This is a refund request."
```

It does not claim:

```text
"The customer is angry."
```

It predicts:

```text
Whether the interaction is likely to become multi-turn.
```

This distinction prevents the model from being assigned a role that the dataset cannot support.

---

# 89. Data Leakage Controls Summary

The following leakage controls were implemented:

```text
Future conversation messages
        → NOT model input

Future response count
        → NOT model input

Conversation depth
        → NOT model input

Total conversation nodes
        → NOT model input

Customer/company node counts
        → NOT model input

Exact duplicate text across splits
        → PREVENTED

Test set during training
        → NOT USED

Validation set during training
        → USED ONLY FOR MODEL SELECTION/EVALUATION
```

The model sees only information available from the initial customer message.

---

# 90. Reproducibility

Important reproducibility values:

```text
Random seed:
42

Data seed:
42

Model:
distilbert-base-uncased

max_length:
128

Train batch:
16

Gradient accumulation:
2

Learning rate:
2e-5

Epochs:
2

Warmup steps:
1970
```

The split itself is also locked.

---

# 91. Locked Decisions

The following decisions are considered locked unless a genuine engineering or data-quality issue is discovered.

```text
Twitter dataset role:
Conversation Intelligence
+ Multi-Turn Conversation Complexity

Target:
SIMPLE / MULTI_TURN

Input:
First customer message only

Model:
distilbert-base-uncased

max_length:
128

No aggressive manual NLP preprocessing:
YES

Group-aware split:
YES

Split seed:
42

Full training dataset:
630,075 rows

Validation:
78,675 rows

Test:
78,596 rows

Training epochs:
2

Learning rate:
2e-5

Effective batch:
32

FP16:
YES

Best model metric:
macro_f1
```

---

# 92. Why the Pipeline Is Considered Production-Oriented

The pipeline avoids several common mistakes.

## Mistake 1 — Choosing the model before understanding the dataset

Avoided.

We analyzed the Twitter dataset first.

---

## Mistake 2 — Creating artificial intent labels

Avoided.

No unsupported intent labels were invented.

---

## Mistake 3 — Training NER on noisy pretrained entity predictions

Avoided.

spaCy entities were treated as diagnostics, not ground truth.

---

## Mistake 4 — Using future conversation data as model input

Avoided.

Only the first customer message is input.

---

## Mistake 5 — Random row-level splitting

Avoided.

Exact duplicate texts are grouped into the same split.

---

## Mistake 6 — Using exact response count as the target

Avoided.

The target represents conversation complexity instead.

---

## Mistake 7 — Aggressive text cleaning

Avoided.

Transformer-compatible natural language is preserved.

---

## Mistake 8 — Training on an arbitrary sample

Avoided.

The final training run uses the complete 630,075-example training set.

---

## Mistake 9 — Endless model tuning

Avoided.

The project objective is a fit-for-purpose SupportMind model, not benchmark optimization for its own sake.

---

# 93. Complete Pipeline Summary

The entire transformation can be summarized as:

```text
2,811,774 raw tweets
        │
        ▼
Dataset audit
        │
        ▼
Conversation relationship analysis
        │
        ▼
Parent/response graph
        │
        ▼
Conversation roots
        │
        ▼
787,346 customer-root interactions
        │
        ▼
Future conversation structure
        │
        ▼
SIMPLE / MULTI_TURN target
        │
        ▼
Remove future/leakage information
        │
        ▼
784,060 unique text groups
        │
        ▼
Group-aware split
        │
        ├───────────────┬───────────────┐
        ▼               ▼               ▼
     630,075          78,675          78,596
      TRAIN           VALIDATION        TEST
        │
        ▼
Hugging Face Dataset
        │
        ▼
DistilBERT tokenizer
        │
        ▼
max_length = 128
        │
        ▼
Dynamic padding
        │
        ▼
Label encoding
        │
        ▼
Remove metadata/text columns
        │
        ▼
input_ids
attention_mask
labels
        │
        ▼
DataCollatorWithPadding
        │
        ▼
DistilBERT
        │
        ▼
Binary Classification Head
        │
        ▼
SIMPLE / MULTI_TURN
```

---

# 94. Current Status

```text
┌─────────────────────────────────────────────┐
│ SUPPORTMIND TWITTER ML PIPELINE             │
├─────────────────────────────────────────────┤
│ Dataset analysis              COMPLETE      │
│ Task selection               LOCKED        │
│ Conversation reconstruction  COMPLETE      │
│ Target construction          COMPLETE      │
│ Leakage controls             COMPLETE      │
│ Group-aware split            LOCKED        │
│ HF Dataset conversion        COMPLETE      │
│ Tokenization                 COMPLETE      │
│ Label preparation             COMPLETE      │
│ Dataset persistence           COMPLETE      │
│ CUDA setup                    COMPLETE      │
│ DistilBERT setup              COMPLETE      │
│ Trainer                       COMPLETE      │
│ Smoke test                    PASSED        │
│                                             │
│ FULL TRAINING                 NEXT          │
│ FINAL TEST EVALUATION         AFTER TRAIN  │
│ MODEL SAVING                  AFTER TEST   │
└─────────────────────────────────────────────┘
```

---

# 95. Final Principle

The most important design decision in this pipeline is:

> **The dataset determines the model task. The model does not determine what the dataset is supposed to mean.**

The Twitter dataset was not forced into the original tentative SupportMind ML plan.

Instead:

```text
Analyze dataset
       ↓
Understand available signals
       ↓
Identify trustworthy target
       ↓
Construct target without leakage
       ↓
Create appropriate ML input
       ↓
Choose model appropriate to task
       ↓
Train on full relevant data
       ↓
Evaluate fit-for-purpose
```

This makes the resulting Conversation Intelligence model defensible as an actual SupportMind component rather than simply another portfolio classification model.
