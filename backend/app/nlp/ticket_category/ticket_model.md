# SupportMind AI — IT Help Desk Ticket Category Classification

## Overview

This module implements the **IT Help Desk Ticket Category Classification** component of SupportMind AI.

The objective is to automatically classify incoming IT support tickets into their appropriate operational category using the ticket's **subject and description**.

This classifier is designed as an NLP component inside SupportMind's broader customer-support intelligence pipeline. It is not intended to solve the ticket, retrieve knowledge-base information, or generate the final response. Its responsibility is to provide a reliable **initial category signal** that downstream SupportMind components can use for routing, analytics, retrieval, and resolution workflows.

### SupportMind role

```text
Incoming IT Support Ticket
          │
          ▼
   Subject + Description
          │
          ▼
   Text Preprocessing
          │
          ▼
 Word TF-IDF + Character TF-IDF
          │
          ▼
 Logistic Regression
          │
          ▼
 Predicted Ticket Category
          │
          ├──► Ticket Routing
          ├──► Agent Dashboard
          ├──► Support Analytics
          ├──► Similar Ticket Retrieval
          └──► Downstream Intelligence
```

---

# 1. Dataset

## Dataset

**IT-helpdesk-synthetic-ticketshg.csv**

Location:

```text
rawsupportminddata/
└── IT-helpdesk-synthetic-ticketshg.csv
```

The dataset contains synthetic IT help-desk support requests.

### Original dataset

The original dataset contained:

* **500 rows**
* **7 columns**

```text
id
subject
description
priority
category
createdAt
requesterEmail
```

### Dataset characteristics

The dataset contains support requests covering multiple operational categories such as:

* Software
* Account
* Network
* Security
* Communication
* Hardware
* RemoteWork
* Training
* Infrastructure
* Licensing
* Performance

The target distribution was highly imbalanced, with **Software** representing the largest category and several categories containing very few examples.

### Original category distribution

| Category       | Count |
| -------------- | ----: |
| Software       |   231 |
| Account        |    71 |
| Network        |    68 |
| Security       |    42 |
| Communication  |    24 |
| Hardware       |    22 |
| RemoteWork     |    20 |
| Training       |    12 |
| Infrastructure |     7 |
| Licensing      |     2 |
| Performance    |     1 |

This imbalance was an important modelling consideration.

---

# 2. Modelling Objective

The SupportMind task selected for this dataset is:

> **IT Support Ticket Category Classification**

### Input

The model receives:

```text
Ticket Subject + Ticket Description
```

### Target

```text
category
```

### Example

```text
Subject:
VPN connection problem

Description:
I am unable to connect to the company VPN from home.
The connection keeps timing out.
```

Expected output:

```text
RemoteWork
```

The objective is not to predict every attribute in the dataset. The dataset was specifically evaluated to determine which SupportMind NLP capability it could realistically support.

---

# 3. Dataset Analysis

Before modelling, the dataset was systematically analyzed to determine whether the available fields contained useful predictive information.

The analysis focused specifically on the proposed SupportMind modelling objective rather than performing unnecessary exploratory analysis.

## 3.1 Structural validation

The dataset was checked for:

* dataset shape
* column names
* data types
* missing values
* complete duplicate rows
* duplicate identifiers
* target distribution

Results:

* No missing values
* No complete duplicate rows
* Ticket IDs were unique
* Text fields were populated
* Category labels were available for supervised learning

Therefore, the dataset provided a usable supervised classification target.

---

# 4. Target Distribution Analysis

The category distribution revealed significant class imbalance.

The majority class was:

```text
Software
```

while categories such as:

```text
Licensing
Performance
Infrastructure
```

had very few observations.

This meant that plain accuracy could be misleading.

For example, a model could perform well by primarily learning the majority class while performing poorly on smaller operational categories.

Therefore, model evaluation focused on:

* Accuracy
* Macro-F1
* Weighted-F1
* Per-class precision
* Per-class recall
* Per-class F1

Macro-F1 was particularly important because it gives every category equal importance regardless of its frequency.

---

# 5. Feature Analysis

Several fields were examined to determine which could provide legitimate predictive information.

## Selected modelling features

The primary text signal was:

```text
subject + description
```

These fields describe the actual support issue and therefore contain information relevant to category prediction.

## Excluded fields

The following fields were intentionally excluded from the model:

### `id`

Ticket identifiers do not represent the semantic content of the issue.

Using them would not provide meaningful generalization.

### `requesterEmail`

Requester identity is not an appropriate feature for general ticket categorization.

It can also introduce identity-related or proxy-based leakage.

### `createdAt`

Creation timestamps do not represent the semantic category of a ticket.

### `priority`

Priority was not selected as an input feature because priority represents an operational attribute rather than the underlying category of the issue.

The classifier should learn the category from the actual ticket content.

---

# 6. Text Signal Analysis

The combined text field was created from:

```text
subject + description
```

The resulting text characteristics were analyzed before selecting the preprocessing and modelling strategy.

### Combined text statistics

Approximate statistics:

* Mean length: **66.48 words**
* Median length: **52 words**
* Maximum length: **614 words**
* Mean characters: **453.42**
* Median characters: **352.5**
* Maximum characters: **3545**

The dataset therefore contained enough textual information for conventional NLP classification.

There were also:

* 489 unique subjects
* 487 subjects mapping to only one category
* 19 duplicated subjects
* 4 duplicated descriptions

The data showed evidence of being synthetic/template-driven, which was considered during modelling and interpretation.

---

# 7. Synthetic Data Consideration

The dataset is synthetic rather than naturally collected enterprise support data.

All 500 original records were generated on:

```text
2024-10-28
```

within a short time period.

This means the dataset is useful for validating the SupportMind classification architecture, but the resulting model should not be interpreted as representative of every real-world enterprise help-desk environment.

The objective was therefore to build a **fit-for-purpose SupportMind component**, rather than claim universal production accuracy.

---

# 8. Target-Specific Preprocessing

Preprocessing was designed specifically around the ticket-category classification task.

The goal was not to aggressively clean the text.

Instead, the preprocessing pipeline removed information that could interfere with generalization while preserving the actual semantic content required for category prediction.

## 8.1 Combined text

The subject and description were combined:

```text
subject + description
```

This provided the classifier with both:

* short high-level issue summaries
* detailed issue descriptions

---

## 8.2 Email removal

Email addresses were removed because requester identity should not contribute to category prediction.

Example:

```text
john.smith@company.com
```

was removed from the modelling text.

---

## 8.3 URL removal

URLs were removed because they generally represent external references rather than the semantic category of the support issue.

Example:

```text
https://example.com/ticket/123
```

was removed.

---

## 8.4 Ticket/request identifier removal

Ticket-like identifiers were removed from the text.

Patterns such as:

```text
Ticket #12345
Request 1023
Incident INC-1024
Case ABC-123
```

were treated as metadata rather than useful category information.

---

## 8.5 Whitespace normalization

Repeated whitespace was normalized to produce a clean modelling representation.

The resulting preprocessing function produced:

```text
clean_text
```

which became the input to the NLP pipeline.

---

# 9. Synthetic Minority-Class Support

The original dataset contained only **one Performance example**.

That made it impossible to meaningfully evaluate the Performance category using a conventional train/validation/test split.

Three carefully constructed synthetic Performance examples were therefore added to provide minimal training signal:

```text
SYN_PERF_001
SYN_PERF_002
SYN_PERF_003
```

These examples represented realistic Performance-related issues such as:

* application performance degradation
* slow system behaviour
* slow application response

The original Performance record was preserved.

The resulting dataset contained:

```text
503 rows
```

with:

```text
Performance = 4
```

This augmentation was deliberately limited. The objective was not to artificially inflate the dataset, but to prevent a category with only one original observation from being completely unusable during model development.

---

# 10. Train / Validation / Test Strategy

Because of the severe class imbalance, a conventional fully stratified split was not possible for categories with extremely small sample counts.

The split strategy therefore separated the sufficiently represented categories from extremely rare categories.

Categories with at least 10 real observations were stratified.

Very rare categories were retained in the training set.

The synthetic examples were also kept in training.

### Final split

```text
Training   : 405
Validation : 49
Test       : 49
```

### Training distribution

The training set contained:

```text
Software        185
Account          57
Network          54
Security         34
Communication    19
Hardware         17
RemoteWork       16
Training         10
Infrastructure    7
Performance       4
Licensing         2
```

Validation and test sets were evaluated using the naturally available examples for sufficiently represented categories.

This strategy ensured that rare categories did not create artificially unstable validation behaviour while still allowing the main categories to be evaluated on unseen examples.

---

# 11. Modelling Approach

The modelling pipeline was deliberately evaluated in stages.

The initial objective was to establish whether conventional sparse-text classification was suitable for the dataset.

The final approach was then improved based on the observed behaviour of the data.

The important modelling progression was:

```text
Raw Ticket Text
      │
      ▼
Target-specific preprocessing
      │
      ▼
Word-level representation
      │
      ▼
Character-level representation
      │
      ▼
Combined feature space
      │
      ▼
Class-balanced Logistic Regression
      │
      ▼
Ticket Category
```

---

# 12. Why Word TF-IDF Alone Was Not Enough

Word-level TF-IDF represents text using word and word-pair patterns.

For example:

```text
vpn connection
password reset
email access
software installation
network connectivity
```

This is useful for capturing semantic vocabulary.

However, support tickets also contain useful sub-word patterns.

Examples include:

```text
password-reset
vpn
VPN
login
log-in
authentication
auth
Outlook
OneDrive
Teams
```

Character-level information can capture similarities between related terms even when the exact word token is different.

This is particularly useful in technical support data where:

* abbreviations
* product names
* spelling variations
* partial words
* technical terminology
* identifiers
* morphological patterns

can carry useful classification information.

---

# 13. Word-Level TF-IDF

The final word representation used:

```python
TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95,
    sublinear_tf=True,
)
```

### Configuration rationale

#### Unigrams + bigrams

```text
ngram_range=(1, 2)
```

captures both individual terms and short phrases.

For example:

```text
vpn
connection
vpn connection
```

can all contribute to classification.

#### Sublinear TF

```text
sublinear_tf=True
```

reduces the dominance of extremely frequent terms by applying a logarithmic term-frequency transformation.

#### `strip_accents="unicode"`

Provides normalized text representation for accented characters.

#### `max_df=0.95`

Prevents extremely common terms from dominating the feature space.

---

# 14. Character-Level TF-IDF

A second feature representation was introduced using character n-grams:

```python
TfidfVectorizer(
    lowercase=True,
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_df=0.98,
    sublinear_tf=True,
)
```

This captures local character patterns rather than only complete words.

For example, related technical terms can share character patterns even when their complete word tokens differ.

This provides additional robustness for technical support language.

---

# 15. Word + Character Feature Fusion

The final representation combines both feature spaces:

```text
Word TF-IDF
      +
Character TF-IDF
      │
      ▼
Combined sparse feature matrix
```

Feature dimensions:

```text
Word features       : 11,253
Character features  : 11,492
Combined features   : 22,745
```

The final model therefore has access to both:

### Word-level information

```text
"vpn connection"
"password reset"
"email access"
"software installation"
```

and:

### Character-level information

```text
partial words
technical terminology
word fragments
spelling patterns
product names
abbreviations
```

This was the key feature-engineering improvement over a basic word-only TF-IDF implementation.

---

# 16. Class-Balanced Logistic Regression

The final classifier is:

```text
Logistic Regression
```

with:

```python
class_weight="balanced"
```

The balanced weighting compensates for the highly imbalanced category distribution.

Instead of allowing the majority class to dominate the optimization process, minority categories receive proportionally greater importance during training.

The final regularization parameter was selected as:

```text
C = 2
```

---

# 17. Model Configuration

Final classification configuration:

```python
LogisticRegression(
    C=2,
    class_weight="balanced",
    max_iter=2000,
    random_state=42,
)
```

The model operates on:

```text
Word TF-IDF + Character TF-IDF
```

features.

---

# 18. Model Development

Multiple modelling approaches were evaluated during development, including:

* TF-IDF-based Logistic Regression
* class-balanced classification
* word + character feature fusion
* SVM
* DistilBERT transformer classification

The purpose of these experiments was to determine which modelling family actually fitted the dataset.

The final solution was selected based on validation/test evidence and practical suitability rather than choosing a more complex architecture simply because it was a Transformer.

The resulting sparse NLP pipeline provided the strongest fit for this small, synthetic, highly imbalanced dataset.

---

# 19. Hyperparameter Improvement

After establishing the word + character representation, the Logistic Regression regularization parameter was evaluated.

The tested values included:

```text
C = 0.25
C = 0.5
C = 1
C = 2
C = 4
```

The selected value was:

```text
C = 2
```

because it provided the strongest validation balance across:

* accuracy
* macro-F1
* weighted-F1

without introducing unnecessary model complexity.

---

# 20. Character Feature Weight Analysis

The contribution of the character representation was also evaluated.

Character feature weighting was tested at different levels.

The selected configuration used:

```text
Character feature weight = 1.0
```

This established that character-level information was not merely decorative: it contributed meaningfully to the final classification representation.

The final feature space therefore uses equal contribution from:

```text
Word features
+
Character features
```

---

# 21. Final Model Evaluation

The final model was evaluated on the held-out test set.

### Test set

```text
49 tickets
```

### Results

| Metric      |      Score |
| ----------- | ---------: |
| Accuracy    | **77.55%** |
| Macro-F1    | **69.02%** |
| Weighted-F1 | **77.56%** |

The classifier correctly predicted:

```text
38 / 49
```

test tickets.

---

# 22. Per-Class Performance

Final test classification results:

| Category      | Precision | Recall |   F1 | Support |
| ------------- | --------: | -----: | ---: | ------: |
| Account       |      0.50 |   0.60 | 0.55 |       5 |
| Communication |      0.33 |   0.33 | 0.33 |       3 |
| Hardware      |      1.00 |   1.00 | 1.00 |       1 |
| Network       |      0.88 |   1.00 | 0.93 |       7 |
| RemoteWork    |      1.00 |   1.00 | 1.00 |       3 |
| Security      |      0.83 |   1.00 | 0.91 |       5 |
| Software      |      0.86 |   0.75 | 0.80 |      24 |
| Training      |      0.00 |   0.00 | 0.00 |       1 |

Overall:

```text
Accuracy     = 0.7755
Macro-F1     = 0.6902
Weighted-F1  = 0.7756
```

The macro-F1 is lower than weighted-F1 because several minority classes have very small test support.

This is expected given the dataset size and imbalance.

---

# 23. Error Analysis

The incorrect predictions were manually inspected to understand where the model struggled.

Examples included tickets where the semantic boundary between categories was naturally ambiguous.

Examples:

```text
Software → Communication
Account → Software
Communication → Training
Training → Security
Software → Account
Software → Network
```

Several errors involved overlapping enterprise technologies such as:

```text
Teams
OneDrive
Outlook
email
Zoom
Asana
```

These technologies can appear in multiple operational contexts.

For example:

```text
software problem
        vs
communication problem
        vs
account/access problem
        vs
network problem
```

can share similar vocabulary.

This confirms that the remaining errors are not simply caused by the absence of a larger vocabulary. Some tickets contain genuinely overlapping semantic signals.

---

# 24. Why the Final Model Is Considered Fit

The objective of this component is not to achieve a theoretically perfect classification score.

The purpose is to provide a useful category signal inside the SupportMind architecture.

The final model provides:

* approximately **77.5% test accuracy**
* approximately **69.0% macro-F1**
* strong performance on several important categories
* lightweight inference
* deterministic behaviour
* low computational requirements
* easy deployment
* interpretable sparse features
* no dependency on a large generative model

For this small synthetic dataset, the final model is considered **fit for its SupportMind role**.

Further model complexity or extensive hyperparameter tuning was not justified by the available data.

---

# 25. Transformer Evaluation

A lightweight Transformer classifier using:

```text
DistilBERT
```

was also evaluated during development.

The purpose was to verify whether contextual Transformer representations provided enough additional value to justify their substantially greater computational cost.

The Transformer approach did not provide a competitive result on this dataset.

Therefore, the final implementation uses the simpler:

```text
Word TF-IDF
+
Character TF-IDF
+
Balanced Logistic Regression
```

architecture.

This is an intentional engineering decision rather than a limitation of the project.

---

# 26. SVM Evaluation

An SVM-based classifier was also evaluated using the same combined word + character feature representation.

It did not outperform the selected Logistic Regression configuration.

Therefore, Logistic Regression was retained as the final classifier.

---

# 27. Final Architecture

The final trained model consists of four persisted artifacts:

```text
supportmind_ticket_category/
│
├── classifier.joblib
├── word_tfidf_vectorizer.joblib
├── char_tfidf_vectorizer.joblib
└── label_encoder.joblib
```

Model directory:

```text
backend/app/nlp/ticket_category/models/supportmind_ticket_category
```

---

# 28. Persisted Components

## Classifier

```text
classifier.joblib
```

Contains the trained:

```text
LogisticRegression
```

classifier.

---

## Word Vectorizer

```text
word_tfidf_vectorizer.joblib
```

Contains the fitted word-level TF-IDF vocabulary and transformation configuration.

---

## Character Vectorizer

```text
char_tfidf_vectorizer.joblib
```

Contains the fitted character-level TF-IDF vocabulary and transformation configuration.

---

## Label Encoder

```text
label_encoder.joblib
```

Maps the model's numerical class IDs back to the original category names.

The final model contains:

```text
11 classes
```

---

# 29. Model Reload Verification

The saved model was reloaded independently from disk.

The following components were successfully restored:

```text
LogisticRegression
TfidfVectorizer
TfidfVectorizer
LabelEncoder
```

The reloaded model was evaluated again on the test set.

Results were identical to the original in-memory model:

```text
Predictions identical : True

Accuracy     : 0.7755
Macro-F1     : 0.6902
Weighted-F1  : 0.7756
```

This verifies that the complete inference pipeline was persisted correctly.

---

# 30. Fresh Inference Verification

The reloaded model was also tested on previously unseen support-ticket examples.

Examples included:

```text
Password/account recovery
Office Wi-Fi disconnecting
Suspicious credentials
Slow application
Work-from-home VPN
License assistance
Outlook emails stuck
Shared server unavailable
Project workspace
```

The model successfully produced:

```text
predicted category
confidence
```

for new input text using only the persisted artifacts.

This confirms that the saved pipeline can be used independently for inference.

---

# 31. Inference Pipeline

At runtime, a new ticket follows:

```text
New Ticket
    │
    ▼
Subject + Description
    │
    ▼
Target-specific Cleaning
    │
    ├───────────────┐
    ▼               ▼
Word TF-IDF     Character TF-IDF
    │               │
    └───────┬───────┘
            ▼
     Feature Combination
            │
            ▼
 Logistic Regression
            │
            ▼
   Numerical Prediction
            │
            ▼
     Label Encoder
            │
            ▼
   Ticket Category
```

---

# 32. SupportMind Integration

The classifier is intended to become one component of the SupportMind NLP layer.

Potential downstream consumers include:

### Ticket routing

```text
Incoming Ticket
      ↓
Category Classification
      ↓
Route to appropriate support workflow
```

### Similar-ticket retrieval

The predicted category can be combined with semantic retrieval to improve historical ticket filtering.

```text
Category
   +
Semantic Similarity
   ↓
Relevant Historical Tickets
```

### Knowledge retrieval

Category can provide additional context to the retrieval pipeline.

```text
Ticket
  ↓
Category
  ↓
Relevant Knowledge Domain
  ↓
Knowledge Retrieval
```

### Support analytics

Category predictions can contribute to:

* ticket volume analysis
* category trends
* workload analysis
* operational dashboards
* recurring issue analysis

---

# 33. Engineering Decisions

Several engineering principles guided the implementation.

## Dataset-first modelling

The model was selected based on the actual signal available in the dataset.

The goal was not to force every SupportMind capability onto the dataset.

---

## Target-driven preprocessing

Preprocessing was designed around:

```text
Subject + Description → Category
```

rather than applying generic text-cleaning rules without considering the modelling target.

---

## Leakage prevention

Fields that could introduce irrelevant identity or metadata signals were excluded.

The classifier learns from the actual ticket content.

---

## Complexity proportional to data

Although Transformer models were considered, the final solution uses sparse linear NLP because it fits the dataset better.

A more complex model is not automatically a better model.

---

## Evaluation beyond accuracy

Because of class imbalance, the model was evaluated using:

```text
Accuracy
Macro-F1
Weighted-F1
Precision
Recall
Per-class F1
```

This prevents majority-class performance from hiding minority-class behaviour.

---

# 34. Limitations

The dataset has several limitations.

### Small dataset

Only 500 original records were available.

Even after controlled augmentation, the final dataset contained only:

```text
503 records
```

This limits the amount of information available for learning minority categories.

### Synthetic data

The dataset is synthetic and may not represent the linguistic diversity of real enterprise support tickets.

### Minority classes

Some categories have extremely low support.

For example:

```text
Licensing
Performance
Infrastructure
```

are underrepresented.

### Category overlap

Some enterprise applications can belong to different operational categories depending on context.

For example:

```text
Outlook
Teams
Zoom
OneDrive
Asana
```

may occur in multiple types of support issues.

### Confidence interpretation

Model probabilities should be treated as classifier confidence rather than absolute certainty.

Low-confidence predictions should be candidates for downstream review or fallback handling.

---

# 35. Final Status

## Ticket Category Classification

```text
STATUS: COMPLETE
```

### Final pipeline

```text
Subject + Description
        ↓
Target-specific preprocessing
        ↓
Word TF-IDF
        +
Character TF-IDF
        ↓
Feature fusion
        ↓
Class-balanced Logistic Regression
        ↓
Ticket Category
```

### Final test performance

```text
Accuracy     : 77.55%
Macro-F1     : 69.02%
Weighted-F1  : 77.56%
```

### Model

```text
Logistic Regression
```

### Feature representation

```text
Word TF-IDF
+
Character TF-IDF
```

### Model directory

```text
backend/app/nlp/ticket_category/models/supportmind_ticket_category
```

### Persisted artifacts

```text
classifier.joblib
word_tfidf_vectorizer.joblib
char_tfidf_vectorizer.joblib
label_encoder.joblib
```

---

# 36. Conclusion

The IT Help Desk dataset was not treated as a generic machine-learning dataset.

The modelling process followed:

```text
Dataset Audit
     ↓
Target Identification
     ↓
Feature Signal Analysis
     ↓
Leakage Prevention
     ↓
Target-Specific Preprocessing
     ↓
Class Imbalance Handling
     ↓
Word-Level NLP
     ↓
Character-Level NLP
     ↓
Feature Fusion
     ↓
Model Comparison
     ↓
Hyperparameter Selection
     ↓
Error Analysis
     ↓
Final Test Evaluation
     ↓
Model Persistence
     ↓
Reload Verification
     ↓
Fresh Inference
```

The resulting **Word TF-IDF + Character TF-IDF + Class-Balanced Logistic Regression** pipeline provides a lightweight, reproducible and fit-for-purpose ticket-category classification component for SupportMind AI.
