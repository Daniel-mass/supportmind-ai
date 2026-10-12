# SupportMind AI — Priority Detection

## Overview

The **Priority Detection** component is a supervised machine learning model within SupportMind AI that predicts the priority level of a Help Desk issue from structured issue information available at issue creation.

The model is designed as an **early structured priority classifier**, not as an NLP model. The Help Desk `issues.csv` dataset does not contain a suitable free-text issue description field, so the model uses structured ticket metadata and creation-time context.

The component is intentionally lightweight and fit-for-purpose. It is not intended to replace human priority decisions in all cases. Its predictions can be used as an operational signal for SupportMind's Intelligence Layer.

---

## ML Task

### Objective

Predict the priority assigned to a Help Desk issue.

### Target

```text
issue_priority
```

The source dataset contains seven values:

```text
unknown
Medium
High
Highest
Blocker
Low
Lowest
```

The `unknown` value is not a valid supervised target class and was excluded from model training.

The final classification problem contains six classes:

```text
Lowest
Low
Medium
High
Highest
Blocker
```

---

## Dataset

### Source

```text
C:\Users\danie\supportmind-ai\rawsupportminddata\Help Desk Tickets\Help Desk Tickets\issues.csv
```

### Original dataset

```text
Rows    : 66,691
Columns : 58
```

### Original priority distribution

| Priority |   Rows | Percentage |
| -------- | -----: | ---------: |
| unknown  | 33,965 |     50.93% |
| Medium   | 24,788 |     37.17% |
| High     |  4,554 |      6.83% |
| Highest  |  2,084 |      3.12% |
| Blocker  |    656 |      0.98% |
| Low      |    560 |      0.84% |
| Lowest   |     84 |      0.13% |

After removing `unknown`:

```text
Labelled rows: 32,726
```

---

## Issue Identity and Deduplication

The `issue_num` column alone is not a globally unique issue identifier because the same issue number can occur in different projects.

The effective issue key was therefore defined as:

```text
(issue_proj, issue_num)
```

The dataset contained:

```text
Project + issue_num keys : 32,665
```

Four project/issue keys appeared more than once.

Inspection showed that those repeated rows represented multiple operational records for the same issue. Importantly, none of those keys contained conflicting priority labels.

To avoid treating repeated operational records as independent ML samples, all four duplicate issue keys were removed.

Final modelling population:

```text
Unique issue-level examples: 32,661
```

---

## Target Distribution After Cleaning

| Priority |   Rows |
| -------- | -----: |
| Medium   | 24,786 |
| High     |  4,493 |
| Highest  |  2,084 |
| Blocker  |    654 |
| Low      |    560 |
| Lowest   |     84 |

The dataset is strongly imbalanced, particularly for `Lowest`, `Low`, and `Blocker`.

For this reason, **macro-F1** is treated as an important evaluation metric in addition to accuracy and weighted-F1.

---

## Leakage Analysis

A major requirement of this model is that priority must be predicted from information available around issue creation rather than from information produced later in the issue lifecycle.

The following fields were therefore excluded from the predictive feature set because they can contain future or post-processing information:

```text
issue_priority
started
ended
issue_resolution_date
issue_resolution
issue_status
last_change_date
issue_assignee
issue_comments_count
wf_*
wfe_*
wf_total_time
processing_steps
```

The model does not use final workflow outcomes, resolution information, or later processing activity to predict priority.

This prevents the model from learning the target indirectly from future ticket outcomes.

---

## Feature Engineering

The final feature set contains seven features.

### Categorical features

```text
issue_proj
issue_type
```

### Creation-time numerical features

`issue_created` was converted into:

```text
created_year
created_month
created_dayofweek
created_hour
```

### Additional ticket-level feature

```text
issue_contr_count
```

Final feature set:

```text
[
    "issue_proj",
    "issue_type",
    "created_year",
    "created_month",
    "created_dayofweek",
    "created_hour",
    "issue_contr_count",
]
```

---

## Timestamp Preprocessing

The original `issue_created` values contained mixed timestamp formats.

They were parsed using:

```python
pd.to_datetime(
    series,
    utc=True,
    errors="coerce",
    format="mixed",
)
```

No actual timestamp parsing failures were found.

Creation-time features were then derived:

```python
created_year = issue_created.dt.year
created_month = issue_created.dt.month
created_dayofweek = issue_created.dt.dayofweek
created_hour = issue_created.dt.hour
```

Final feature table contained:

```text
32,661 rows
7 predictive features
0 missing values
```

---

## Train/Test Split

A stratified train/test split was used so that the priority distribution was preserved across both partitions.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)
```

Result:

```text
Training rows : 26,128
Test rows     : 6,533
```

---

## Models Evaluated

Three modelling approaches were evaluated.

### 1. Majority-Class Baseline

The baseline predicts:

```text
Medium
```

for every test example.

Result:

```text
Accuracy : 75.89%
Macro-F1 : 14.38%
```

The high accuracy is caused by the strong class imbalance and does not represent useful priority classification performance.

---

### 2. Balanced Logistic Regression

The initial structured baseline used:

* One-hot encoded categorical features
* Numeric creation-time features
* `class_weight="balanced"`
* Logistic Regression

Result:

```text
Accuracy    : 39.98%
Macro-F1    : 23.49%
Weighted-F1 : 49.04%
```

The model also reached the iteration limit during optimization.

This approach was rejected.

---

### 3. ExtraTrees Classifier

The final model uses:

```python
ExtraTreesClassifier(
    n_estimators=400,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
)
```

Categorical preprocessing:

```python
OneHotEncoder(
    handle_unknown="ignore",
    min_frequency=2,
)
```

The preprocessing and classifier are stored together in a single scikit-learn `Pipeline`.

---

## Final Test Results

### ExtraTrees Priority Detection

```text
Accuracy    : 0.7326
Macro-F1    : 0.3966
Weighted-F1 : 0.7635
```

Equivalent percentages:

```text
Accuracy    : 73.26%
Macro-F1    : 39.66%
Weighted-F1 : 76.35%
```

### Classification Report

| Priority | Precision | Recall |   F1 |
| -------- | --------: | -----: | ---: |
| Blocker  |      0.19 |   0.44 | 0.26 |
| High     |      0.46 |   0.48 | 0.47 |
| Highest  |      0.34 |   0.50 | 0.40 |
| Low      |      0.18 |   0.52 | 0.27 |
| Lowest   |      0.06 |   0.35 | 0.10 |
| Medium   |      0.95 |   0.81 | 0.87 |

The model performs strongest on the dominant `Medium` class and provides meaningful but weaker detection of minority priorities.

Because the dataset contains only 84 `Lowest` examples, performance for that class is unstable and should not be interpreted as highly reliable.

---

## Rejected Ordinal Approach

An ordinal ExtraTrees regression approach was also tested because priority levels have a natural ordering:

```text
Lowest < Low < Medium < High < Highest < Blocker
```

Results:

```text
Accuracy    : 78.40%
Macro-F1    : 33.12%
Weighted-F1 : 77.64%
```

Although ordinal prediction produced higher accuracy, it produced worse macro-F1 and weaker minority-class performance.

Therefore the ordinal approach was rejected.

The standard multiclass ExtraTrees classifier was selected as the final model.

---

## Final Model Artifact

The complete trained pipeline is saved at:

```text
C:\Users\danie\supportmind-ai\backend\app\nlp\priority_detection\priority_detection_model.joblib
```

This file contains:

```text
OneHotEncoder
        +
ExtraTreesClassifier
```

The model can therefore be loaded directly for inference without separately reconstructing the preprocessing pipeline.

---

## Model Metadata

Metadata is stored at:

```text
C:\Users\danie\supportmind-ai\backend\app\nlp\priority_detection\model_metadata.json
```

The metadata records:

* task
* dataset
* target
* excluded target value
* issue key
* feature columns
* model configuration
* training/test sizes
* evaluation metrics
* class labels

---

## Reload Verification

The saved model was reloaded from disk and evaluated on the same held-out test set.

Verification result:

```text
Predictions identical : True
Accuracy              : 73.26%
Macro-F1              : 39.66%
Weighted-F1           : 76.35%
```

This confirms that the serialized model reproduces the original model predictions exactly.

---

## Fresh Inference Verification

The reloaded model was also tested on previously unseen synthetic issue records.

Example input features:

```text
issue_proj
issue_type
created_year
created_month
created_dayofweek
created_hour
issue_contr_count
```

The model successfully produced:

```text
predicted_priority
confidence
probability distribution across all classes
```

Example predictions included:

```text
Ticket       → High
Deployment   → Blocker
Bug          → Medium
Ticket       → High
```

Confidence values for the examples were relatively low:

```text
0.2706
0.2512
0.3540
0.3506
```

Therefore, downstream SupportMind components should treat model confidence as an important uncertainty signal.

Low-confidence predictions should not be presented as certain operational facts.

---

## Production Interpretation

This model provides a **priority classification signal** from structured issue information.

It should be treated as one evidence source within SupportMind rather than as an absolute authority.

Recommended downstream behavior:

```text
Priority model
      ↓
predicted priority + confidence
      ↓
Intelligence Layer
      ↓
workflow / operational context
      ↓
manager / agent intelligence
```

The model should not overwrite confirmed operational priority unless the application explicitly supports human review and correction.

---

## Current Status

```text
Status: COMPLETE / LOCKED
```

Completed:

* Dataset inspection
* Target definition
* Removal of `unknown`
* Issue-level deduplication
* Leakage analysis
* Feature selection
* Timestamp preprocessing
* Stratified train/test split
* Majority baseline
* Logistic Regression baseline
* ExtraTrees model
* Ordinal model comparison
* Final model selection
* Model serialization
* Reload verification
* Fresh inference verification

Final model:

```text
ExtraTreesClassifier
```

Final test performance:

```text
Accuracy    : 73.26%
Macro-F1    : 39.66%
Weighted-F1 : 76.35%
```

The Help Desk ML scope is intentionally limited to **Priority Detection**. Further Help Desk predictive models are not part of the current SupportMind roadmap.
