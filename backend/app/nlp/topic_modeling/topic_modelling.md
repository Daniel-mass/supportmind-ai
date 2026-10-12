# SupportMind AI — BERTopic Topic Modeling

## Overview

This module implements topic modeling for the SupportMind AI platform.

The purpose of this component is to discover recurring support issues from historical customer-support ticket descriptions.

SupportMind is not designed to be only a chatbot.

It is an enterprise customer-support intelligence platform that transforms support signals into actionable intelligence for customers, support agents, managers, and operations teams.

Topic modeling is one of the intelligence capabilities inside this platform.

It helps SupportMind answer questions such as:

* What problems are customers repeatedly experiencing?
* Which technical issues are appearing frequently?
* Are software-update-related problems increasing?
* Are customers repeatedly reporting battery problems?
* Are account-access problems recurring?
* Which issues are associated with particular products?
* Which support problems deserve operational attention?
* Which emerging issue clusters should managers investigate?

This module uses BERTopic to discover these recurring issue patterns without requiring manually labeled topic data.

---

## Role in SupportMind

The high-level SupportMind intelligence pipeline is:

```text
Customer / Support Signals
          |
          v
     Ingestion Layer
          |
          v
     NLP Processing
          |
          v
     Intelligence Layer
          |
          +----------------------+
          |                      |
          v                      v
   Similar Tickets          Topic Modeling
                                 |
                                 v
                         Recurring Issues
                                 |
                                 v
                         Operations Intelligence
```

The BERTopic component belongs to the Intelligence Layer.

It does not replace classification, retrieval, sentiment analysis, urgency prediction, or escalation models.

Each ML component has a separate responsibility.

BERTopic specifically answers:

> "What recurring issue patterns exist in the support corpus?"

---

# 1. What Is Topic Modeling?

Topic modeling is an unsupervised machine-learning technique used to discover groups of documents that discuss similar subjects.

In SupportMind, each document is a customer-support ticket description.

For example, a collection of tickets might contain:

```text
My device suddenly loses battery very quickly.

The battery used to last much longer.

My device battery drains within a few hours.
```

Topic modeling can recognize that these documents are related and place them into a common topic.

The topic might contain keywords such as:

```text
battery
life
sudden
decrease
charging
```

The topic itself is not manually predefined.

The model discovers the structure from the ticket corpus.

---

# 2. Why SupportMind Uses Topic Modeling

Traditional supervised classification requires labeled training data.

For example:

```text
Ticket -> Battery Issue
Ticket -> Account Access
Ticket -> Network Problem
Ticket -> Payment Problem
```

A classifier needs enough reliable labels to learn these mappings.

Our selected historical SupportMind dataset does not provide reliable supervised labels for every desired NLP task.

Therefore, forcing a classifier onto this dataset would not be appropriate.

Instead, the dataset is used for capabilities where it provides meaningful signal.

For this dataset, the selected capabilities are:

```text
Historical Tickets
      |
      +----> Semantic Similarity
      |
      +----> BERTopic
      |
      +----> Issue Clustering
```

BERTopic is therefore used because it can discover recurring support issues without requiring manually labeled topic classes.

---

# 3. Why Not Use This Dataset for Every Model?

A major design principle of SupportMind is:

> A dataset should only train or support a model when it provides an appropriate training signal for that model.

The dataset was evaluated before selecting the ML tasks.

Experiments showed that the text did not provide reliable supervised signal for several candidate tasks.

For example:

```text
Description -> Ticket Type
```

produced approximately random-level classification performance.

Similarly:

```text
Description -> Ticket Subject
```

was approximately random.

Priority prediction from description was also approximately random.

Therefore, these models were not forced into the pipeline.

The dataset is instead used for:

```text
1. Semantic similarity
2. Topic modeling
3. Issue clustering
```

This prevents unnecessary models from being created simply to increase model count.

---

# 4. Dataset Used

The canonical dataset used for this BERTopic implementation is:

```text
supportmind_final_clean_dataset.csv
```

Location:

```text
data/
└── processed/
    └── supportmind_bertopic_cleaned_dataset.csv
```

The original dataset contains:

```text
8,469 tickets
17 columns
```

Important fields include:

```text
Ticket ID
Customer Name
Customer Email
Customer Age
Customer Gender
Product Purchased
Date of Purchase
Ticket Type
Ticket Subject
Ticket Description
Ticket Status
Resolution
Ticket Priority
Ticket Channel
First Response Time
Time to Resolution
Customer Satisfaction Rating
```

The `Resolution` field is completely empty.

There are also substantial missing values in operational fields.

Therefore, the dataset was not treated as a generic supervised-learning dataset.

---

# 5. BERTopic Target

The target of this modeling work is:

```text
Recurring support issue discovery
```

The model is not intended to predict:

```text
Ticket Type
Ticket Subject
Priority
Sentiment
Urgency
Escalation
NER entities
Resolution
```

Those tasks require different data and potentially different models.

The BERTopic target is:

```text
Ticket Description
       |
       v
Semantic representation
       |
       v
Topic discovery
       |
       v
Recurring support issues
```

---

# 6. Data Quality Analysis

Before training, the ticket descriptions were analyzed specifically for topic-modeling problems.

The analysis identified several characteristics.

The corpus contained repeated support boilerplate.

For example:

```text
Please assist.
```

appeared thousands of times.

Another repeated opening was:

```text
I'm having an issue with the PRODUCT.
```

There were also contaminated fragments originating from unrelated data.

Examples included:

```text
Product ID
#PRODUCT
product.product
if (product_name == ...)
```

There were also fragments resembling:

```text
website metadata
marketplace listings
code
supplier information
seller information
shipping information
```

These fragments could create artificial topics.

Therefore, preprocessing was designed specifically for the BERTopic target.

---

# 7. Target-Driven Preprocessing Principle

The preprocessing strategy follows this rule:

> Remove information that creates artificial topic structure while preserving information that describes the actual customer issue.

We did not perform arbitrary generic cleaning.

We did not aggressively remove every unusual word.

We did not normalize the corpus simply because it looked messy.

Every cleaning rule was evaluated against its impact on issue-topic discovery.

---

# 8. Preprocessing Rule 1 — Generic Boilerplate

Several sentences were identified as generic support-request boilerplate.

Examples:

```text
Please assist.

Thank you.

What should I do?

Can you please help me with this?

Can you please provide a solution?

How can I resolve this issue?

What does it mean?

Could you please guide me through the steps?

What can be done to resolve this issue?

How can I troubleshoot this network problem?

What steps should I take to troubleshoot this issue?

Are there any updates or fixes available?

How can I regain access to my account?

How can I recover my account?

Can you guide me on how to retrieve them?

I need them urgently.
```

These sentences generally provide little information about the actual issue.

They were removed from the BERTopic input.

Validation showed:

```text
Original documents: 8,469

Tickets with at least one sentence removed: 8,022

Total sentences removed: 9,500

Empty documents after Rule 1: 0
```

The rule therefore removed repetitive support language while retaining issue-bearing sentences.

---

# 9. Preprocessing Rule 2 — Contamination Removal

The corpus contained injected fragments unrelated to the actual support issue.

Examples included:

```text
supplier inventory information
seller transaction information
website metadata
source-code fragments
marketplace fragments
malformed product metadata
```

The cleaning strategy was deliberately conservative.

We did not delete an entire sentence simply because it contained a suspicious keyword.

Instead, confirmed contamination fragments were removed.

This was important because some contaminated sentences also contained legitimate customer-support information.

The objective was:

```text
Contamination
     |
     v
Remove confirmed artificial fragment
     |
     v
Preserve genuine issue statement
```

---

# 10. Product Masking

Product identity is useful for SupportMind.

However, product names can dominate unsupervised topic discovery.

For example:

```text
Nintendo Switch
Sony PlayStation
Dyson Vacuum Cleaner
iPhone
Microsoft Office
```

could cause BERTopic to create topics primarily based on product identity rather than issue type.

That would be undesirable.

Therefore, product names were masked in the topic-modeling text.

For example:

```text
I'm having an issue with the Dyson Vacuum Cleaner.
```

becomes:

```text
I'm having an issue with the PRODUCT.
```

The original product information remains available separately.

This gives us two dimensions:

```text
Issue Topic
    +
Product
```

This allows SupportMind to later answer questions such as:

```text
Which issues are recurring?
```

and:

```text
Which products are affected by this recurring issue?
```

---

# 11. Product Masking Strategy

The authoritative `Product Purchased` column was used to obtain the product vocabulary.

Products were sorted by length before constructing the replacement pattern.

This prevents shorter product names from incorrectly matching portions of longer product names.

The masking process is conceptually:

```python
products = sorted(products, key=len, reverse=True)

product_pattern = re.compile(
    r"\b(?:" + "|".join(
        re.escape(product)
        for product in products
    ) + r")\b",
    flags=re.IGNORECASE
)

def mask_product_names(text):
    return product_pattern.sub("PRODUCT", str(text))
```

Validation showed:

```text
Documents: 8,469

Documents containing product names before masking: 8,469

Product information retained separately: Yes
```

---

# 12. Preprocessing Rule 3 — Generic Product Opening

After product masking, the most common generic opening became:

```text
I'm having an issue with the PRODUCT.
```

This sentence does not tell BERTopic what the issue actually is.

It only identifies the existence of a problem.

Another generic pattern was:

```text
I'm facing a problem with my PRODUCT.
```

These exact generic openings were removed.

The issue-bearing sentences were retained.

---

# 13. Empty Document Handling

One ticket contained only:

```text
I'm having an issue with the Xbox.
Please assist.
```

After:

```text
Rule 1
+
product masking
+
generic opening removal
```

there was no actual issue information remaining.

That document was therefore excluded from BERTopic training.

It was not deleted from the master dataset.

This distinction is important.

The master dataset remains complete.

Only the BERTopic training corpus excludes documents that contain no meaningful issue information.

---

# 14. Final BERTopic Training Corpus

The final training corpus contains:

```text
Original dataset: 8,469

Excluded invalid topic documents: 9

Final BERTopic documents: 8,460
```

The nine excluded tickets were identified as documents that did not contain useful topic-modeling information after the target-driven preprocessing process.

The final corpus contains:

```text
8,460 documents
0 empty documents
```

---

# 15. Final Modeling Text

The final BERTopic input column is:

```text
bertopic_issue_text_final_v2
```

This represents the issue-focused text used for topic discovery.

The final training list is created using:

```python
documents_final = (
    bertopic_training_data_final[
        "bertopic_issue_text_final_v2"
    ]
    .astype(str)
    .tolist()
)
```

---

# 16. Embedding Model

The embedding model used is:

```text
BAAI/bge-base-en-v1.5
```

It is loaded through:

```text
FastEmbed
```

The embedding dimension is:

```text
768
```

The embedding model converts each issue description into a numerical vector.

Conceptually:

```text
Ticket text
    |
    v
BGE embedding model
    |
    v
768-dimensional vector
```

Semantically similar issue descriptions should produce nearby representations.

---

# 17. Why BGE Base?

BGE was selected because semantic similarity is central to SupportMind.

The model provides a strong general-purpose sentence/document embedding baseline without requiring a large custom fine-tuned model.

It is also practical for local execution.

The selected embedding model is therefore suitable for:

```text
semantic similarity
topic discovery
historical ticket retrieval
```

within the project's current scope.

---

# 18. Embedding Cache

The final embeddings were generated once and cached.

File:

```text
data/processed/
└── supportmind_bertopic_embeddings.npy
```

Embedding shape:

```text
(8460, 768)
```

Data type:

```text
float32
```

Caching avoids recomputing embeddings every time the BERTopic experiment or notebook is executed.

---

# 19. UMAP

BERTopic uses UMAP to reduce the dimensionality of the embedding space before clustering.

The final configuration is:

```python
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)
```

Configuration:

```text
n_neighbors = 15
n_components = 5
min_dist = 0.0
metric = cosine
random_state = 42
```

---

# 20. Why UMAP?

The original embeddings contain 768 dimensions.

Clustering directly in that space is not always ideal.

UMAP creates a lower-dimensional representation while attempting to preserve useful neighborhood relationships.

The pipeline becomes:

```text
768D BGE embeddings
        |
        v
       UMAP
        |
        v
     5D space
        |
        v
    HDBSCAN
```

The goal is better clustering of semantically similar support issues.

---

# 21. HDBSCAN

The clustering algorithm is:

```text
HDBSCAN
```

Configuration:

```python
hdbscan_model = HDBSCAN(
    min_cluster_size=30,
    min_samples=10,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)
```

Parameters:

```text
min_cluster_size = 30
min_samples = 10
metric = euclidean
cluster_selection_method = eom
prediction_data = True
```

---

# 22. Why HDBSCAN?

Support issues do not necessarily form equally sized clusters.

Some issues may occur hundreds of times.

Other recurring issues may occur only several dozen times.

HDBSCAN is useful because it can:

```text
discover clusters
handle different densities
identify outliers
avoid forcing every document into a cluster
```

This is appropriate for support-ticket issue discovery.

---

# 23. CountVectorizer

BERTopic uses a CountVectorizer for topic representation.

The final configuration is:

```python
vectorizer_model = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
)
```

Configuration:

```text
stop_words = english
ngram_range = (1, 2)
min_df = 5
```

---

# 24. Why Unigrams and Bigrams?

Support issues often contain meaningful multi-word phrases.

Examples:

```text
battery life
software update
factory reset
network connection
password reset
credit card
data loss
error message
```

Using:

```text
ngram_range=(1, 2)
```

allows BERTopic to consider both individual words and two-word phrases.

This improves interpretability of topic representations.

---

# 25. Why min_df=5?

Extremely rare words can produce noisy topic representations.

A minimum document frequency of five prevents terms that occur only in a tiny number of documents from dominating the representation.

This is a practical topic-representation constraint rather than an attempt to maximize a benchmark score.

---

# 26. BERTopic Initialization

The final BERTopic model is initialized as:

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model=None,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    verbose=True,
)
```

The embedding model is set to:

```python
embedding_model=None
```

because embeddings are generated externally using FastEmbed and passed directly to BERTopic.

This provides explicit control over embedding generation and caching.

---

# 27. Final Training

Training is performed using:

```python
topics, probabilities = topic_model.fit_transform(
    documents_final,
    embeddings=embeddings,
)
```

The model therefore receives:

```text
documents_final
+
precomputed BGE embeddings
```

The resulting topic assignments are stored in:

```text
topics
```

and probability information is stored in:

```text
probabilities
```

---

# 28. Final Model Result

The selected model discovered:

```text
8,460 training documents

59 actual topics

1,379 outlier documents

7,081 assigned documents
```

The outlier rate is:

```text
16.30%
```

The assigned rate is:

```text
83.70%
```

---

# 29. Topic Size Distribution

The topic sizes were:

```text
Minimum: 31
Median: 77
Mean: 120.02
Maximum: 393
```

There were:

```text
0 topics smaller than 30 documents
```

This is useful for SupportMind because every discovered topic has enough historical tickets to represent a recurring issue group.

---

# 30. Topic Diversity

The model was evaluated using the top ten words from each topic.

Results:

```text
Topics evaluated: 59

Total topic words: 590

Unique topic words: 353

Topic diversity: 0.5983
```

This means approximately:

```text
59.83%
```

of the top topic-word positions correspond to unique words.

Some vocabulary overlap is expected in customer-support data.

Common words such as:

```text
issue
problem
product
update
account
software
```

can naturally occur across multiple topics.

---

# 31. Topic Coherence

C_V coherence was calculated using the actual BERTopic corpus and topic representations.

Result:

```text
C_V coherence = 0.7972
```

This indicates that the discovered topic representations are generally coherent.

The coherence score is supporting evidence.

It is not treated as the sole acceptance criterion.

---

# 32. Fit-for-Purpose Evaluation

The model was not evaluated against an arbitrary academic threshold.

Instead, the evaluation considered:

```text
Topic usefulness
Topic size
Assignment coverage
Topic diversity
Topic coherence
Representative documents
SupportMind business usefulness
```

The model is considered:

```text
FIT FOR SUPPORTMIND
```

---

# 33. Why We Are Not Continuing Hyperparameter Tuning

The purpose of this component is not to produce a perfect research benchmark.

The objective is to provide useful issue intelligence.

The current model already produces meaningful topics such as:

```text
error messages
firmware updates
software updates
battery problems
network connectivity
account access
charging problems
hardware problems
peripheral compatibility
data recovery
password reset
application crashes
cache/data issues
factory reset
intermittent problems
```

These are directly useful to SupportMind.

Therefore, additional hyperparameter tuning is not currently justified.

---

# 34. Example Discovered Topic

One discovered topic contains keywords associated with error messages.

Representative examples include:

```text
I've noticed a peculiar error message
popping up on my PRODUCT screen.
```

This is useful because SupportMind can identify a recurring error-message issue without the topic having been manually defined beforehand.

---

# 35. Example — Battery Topic

Another topic contains:

```text
battery
life
sudden
decrease
used
longer
```

Representative issue:

```text
I've noticed a sudden decrease in battery life
on my PRODUCT. It used to last much longer.
```

This becomes a recurring issue cluster.

---

# 36. Example — Account Topic

Another topic identifies account/login problems.

Representative examples include:

```text
I'm facing issues logging into my PRODUCT account.
```

and:

```text
my account is locked
```

This gives SupportMind an operational issue category around account access.

---

# 37. Example — Password Topic

Another topic identifies:

```text
password
reset
account
```

Representative tickets describe:

```text
forgotten passwords
password reset failures
```

This is distinct from general account-access issues.

---

# 38. Example — Connectivity Topic

Topics were also discovered around:

```text
internet connection
network
Wi-Fi
connection failures
```

This enables SupportMind to distinguish different connectivity-related recurring problems.

---

# 39. Example — Data Loss Topics

The model discovered multiple data-loss-related topic groups.

Examples include:

```text
deleted files
data recovery
lost data
missing documents
```

These topics may have some semantic overlap.

We intentionally did not merge them automatically.

The topic model is retained because downstream business analysis can determine whether these represent useful operational distinctions.

---

# 40. Example — Software Issues

The model identified several software-related topics.

Examples include:

```text
software bugs
application crashes
software glitches
cache/data issues
specific feature problems
```

These can support operational investigations into software defects.

---

# 41. Topic IDs

BERTopic produces numeric topic IDs.

For example:

```text
Topic 1
Topic 2
Topic 3
...
Topic 59
```

These IDs are model-generated identifiers.

They should not be treated as permanent business labels.

A production layer can map them to human-readable labels.

For example:

```text
Topic 8
    ->
Battery Life Degradation
```

The mapping should be stored as a separate artifact.

---

# 42. Topic Representation

Each topic contains a ranked list of representative words.

Conceptually:

```text
Topic ID
    |
    +-- keyword 1
    +-- keyword 2
    +-- keyword 3
    +-- keyword 4
    +-- keyword 5
```

These keywords are used to create human-readable topic summaries.

---

# 43. Representative Documents

BERTopic also provides representative documents.

These documents are useful for validating the meaning of a topic.

For example:

```python
docs = topic_model.get_representative_docs(topic_id)
```

Representative documents are more informative than topic keywords alone.

They allow an analyst to answer:

> "Do these documents actually describe the same support problem?"

---

# 44. Why Representative Documents Matter

Topic keywords can sometimes look coherent while the underlying documents are mixed.

Therefore, SupportMind topic validation considers:

```text
keywords
+
representative documents
```

rather than keywords alone.

This prevents blindly trusting automatically generated labels.

---

# 45. Outliers

BERTopic uses:

```text
Topic -1
```

for documents that it cannot confidently assign to a discovered topic.

Our final result contains:

```text
1,379 outliers
```

or:

```text
16.30%
```

of the training corpus.

---

# 46. Why Outliers Are Acceptable

Not every customer-support ticket belongs to a recurring issue group.

Some tickets may be:

```text
rare
unique
ambiguous
poorly described
contaminated
semantically isolated
```

Forcing all of them into an existing topic could create worse topic quality.

Therefore, outliers are retained as a legitimate model outcome.

---

# 47. How Outliers Can Be Used Later

Outliers can potentially support:

```text
rare issue detection
new issue monitoring
emerging issue discovery
manual review
future topic updates
```

A future production process could monitor whether outlier tickets begin forming a new cluster.

---

# 48. SupportMind Operations Use Case

The primary downstream consumer of BERTopic is the Operations/Manager intelligence layer.

The pipeline can be:

```text
Historical tickets
       |
       v
BERTopic
       |
       v
Topic assignments
       |
       v
Topic volume
       |
       v
Recurring issue dashboard
```

Managers can see:

```text
Issue
Ticket count
Affected products
Channels
Priority distribution
Status distribution
Trend over time
```

---

# 49. Topic × Product Analysis

Because products were masked only from the topic text, the original product remains available.

Therefore, SupportMind can combine:

```text
Topic ID
+
Product Purchased
```

For example:

```text
Battery Issue
    |
    +-- iPhone
    +-- Samsung Galaxy
    +-- Fitbit Versa
    +-- Garmin Forerunner
```

This is significantly more useful than product-specific topics.

---

# 50. Topic × Channel Analysis

Topics can also be combined with:

```text
Ticket Channel
```

For example:

```text
Network Issue

Email       120
Chat        95
Phone       180
Social      76
```

This can reveal where particular support issues are being reported.

---

# 51. Topic × Priority Analysis

Topic IDs can also be joined with ticket priority.

Example:

```text
Security Concern

Critical    82
High        61
Medium      44
Low         12
```

This does not mean BERTopic predicts priority.

Instead, the existing priority metadata is used for downstream analytics.

---

# 52. Topic × Status Analysis

Similarly:

```text
Topic
+
Ticket Status
```

can show whether recurring issues remain unresolved.

Example:

```text
Software Update Issue

Open      140
Pending    92
Closed    118
```

This can support operational investigation.

---

# 53. Topic Trends

If ticket timestamps are available and reliable, SupportMind can aggregate topic assignments over time.

Example:

```text
Date
+
Topic
+
Ticket count
```

This enables:

```text
recurring issue monitoring
emerging issue monitoring
issue spikes
product incident detection
```

BERTopic itself does not automatically become a forecasting model.

The topic assignments are the input to the analytics layer.

---

# 54. Emerging Issue Detection

A future SupportMind workflow can monitor:

```text
Topic volume
+
time
+
outlier volume
+
new semantic clusters
```

A sudden increase in a topic can indicate:

```text
product defect
software regression
service outage
policy problem
support-process issue
```

This is one of the major reasons topic modeling is valuable in an enterprise support platform.

---

# 55. Relationship With Similar-Ticket Retrieval

BERTopic and semantic similarity solve different problems.

Similarity asks:

> "Which historical tickets are similar to this ticket?"

Topic modeling asks:

> "What recurring issues exist across the corpus?"

Therefore:

```text
Incoming ticket
       |
       +----> Similarity Retrieval
       |          |
       |          +--> Similar historical tickets
       |
       +----> Topic Model
                  |
                  +--> Recurring issue group
```

Both capabilities can operate together.

---

# 56. Relationship With RAG

BERTopic is not a replacement for RAG.

RAG retrieves relevant knowledge from sources such as:

```text
policies
FAQs
product documentation
troubleshooting guides
incident reports
```

BERTopic discovers issue patterns from historical support signals.

Therefore:

```text
Historical Tickets
       |
       +--> BERTopic
       |
       +--> Similar Ticket Retrieval

Enterprise Knowledge
       |
       +--> RAG
```

These capabilities complement each other.

---

# 57. Relationship With Agents

SupportMind uses agents later in the architecture.

The BERTopic component can provide structured intelligence to agents.

For example:

```text
Agent receives ticket
        |
        v
Issue understanding
        |
        v
Topic assignment
        |
        v
Historical retrieval
        |
        v
Knowledge retrieval
        |
        v
Resolution reasoning
```

The agent does not need to rediscover recurring issue patterns from raw tickets.

It can consume topic information as structured context.

---

# 58. Relationship With Root Cause Analysis

Topic modeling does not automatically prove root cause.

A topic such as:

```text
Software Update Problems
```

does not prove:

```text
Software update X caused defect Y.
```

Root cause analysis requires additional evidence.

Potential future evidence sources include:

```text
incident reports
ticket timestamps
product versions
deployment records
logs
knowledge-base documents
engineering incidents
```

BERTopic can identify the recurring symptom cluster that should be investigated.

---

# 59. Model Capability After Training

After training, the model can provide:

```text
Topic discovery
Topic assignment
Topic representations
Representative documents
Topic sizes
Outlier identification
Topic-document relationships
```

It can also support analysis of:

```text
recurring issues
issue clusters
topic trends
topic-product relationships
topic-channel relationships
topic-priority relationships
```

---

# 60. What the Model Does Not Do

The model does not automatically:

```text
diagnose customers
recommend trades
predict escalation
predict urgency
detect sentiment
extract entities
generate resolutions
prove root causes
```

Those are separate capabilities.

This separation keeps the architecture interpretable.

---

# 61. Model Version

The current selected model should be treated as:

```text
SupportMind BERTopic v1
```

The exact configuration is part of the reproducibility record.

Model components:

```text
Embedding:
BAAI/bge-base-en-v1.5

Embedding provider:
FastEmbed

Dimensionality:
768

UMAP:
15 neighbors
5 components
cosine distance
min_dist 0.0

HDBSCAN:
min_cluster_size 30
min_samples 10
EOM selection

Vectorizer:
English stop words
1-2 grams
min_df 5

BERTopic:
0.17.4
```

---

# 62. Reproducibility

The model uses:

```python
random_state=42
```

for UMAP.

Embeddings are cached.

The preprocessing rules are implemented explicitly.

The final training corpus is persisted.

This allows the model to be reproduced without repeating exploratory preprocessing decisions.

---

# 63. Project Structure

The intended module structure is:

```text
backend/
└── app/
    └── nlp/
        └── topic_modeling/
            ├── __init__.py
            ├── bertopic_model.py
            ├── topic_service.py
            ├── notebook/
            │   └── notebook_modelling_eda.ipynb
            └── models/
                └── supportmind_bertopic/
```

Supporting processed data:

```text
data/
└── processed/
    ├── supportmind_bertopic_cleaned_dataset.csv
    └── supportmind_bertopic_embeddings.npy
```

---

# 64. Notebook Role

The notebook is used for:

```text
EDA
preprocessing validation
model experiments
evaluation
topic inspection
```

The notebook is not intended to become the production runtime.

Production logic should eventually live inside:

```text
bertopic_model.py
topic_service.py
```

---

# 65. Model Service Role

`topic_service.py` should provide a controlled interface around the model.

The service can eventually expose operations such as:

```text
load_model()
assign_topic()
get_topic_info()
get_topic_keywords()
get_representative_documents()
```

This separates model implementation from the FastAPI application layer.

---

# 66. Production Inference

A future ticket-processing pipeline can look like:

```text
New Ticket
    |
    v
Text preprocessing
    |
    v
Product masking
    |
    v
Embedding
    |
    v
BERTopic assignment
    |
    v
Topic ID
    |
    v
Topic metadata
```

The resulting topic information can be stored alongside the ticket.

---

# 67. Topic Assignment Storage

A future processed ticket record can contain:

```text
Ticket ID
Topic ID
Topic Label
Topic Probability
Product
Channel
Priority
Status
Timestamp
```

This becomes an analytics-ready representation.

---

# 68. Example Processed Record

Conceptually:

```json
{
  "ticket_id": 12345,
  "topic_id": 8,
  "topic_label": "Battery Life Issue",
  "product": "iPhone",
  "channel": "Chat",
  "priority": "High",
  "status": "Open"
}
```

The topic label is a human-readable layer built on top of the BERTopic topic ID.

---

# 69. Dashboard Integration

The Next.js SupportMind dashboards can consume topic analytics through FastAPI.

Example:

```text
Next.js Dashboard
        |
        v
FastAPI
        |
        v
Topic Service
        |
        v
BERTopic artifacts
        |
        v
Topic analytics
```

The dashboard can display:

```text
Top recurring issues
Issue volume
Issue trends
Affected products
Issue/channel relationships
Issue/priority relationships
Emerging issue signals
```

---

# 70. Manager Dashboard

The manager dashboard can use topic modeling for:

```text
Top support issues
Recurring issue volume
Unresolved issue clusters
Issue trends
Product-specific issue concentration
```

Example:

```text
Top recurring issues

1. Error Messages
2. Software Updates
3. Battery Life
4. Account Access
5. Network Connectivity
```

---

# 71. Operations Dashboard

The Operations dashboard can go deeper.

It can combine:

```text
Topic
+
Product
+
Priority
+
Status
+
Channel
+
Time
```

This enables operational intelligence rather than simple ticket counting.

---

# 72. Emerging Issue Workflow

A future workflow could be:

```text
Daily ticket ingestion
        |
        v
Topic assignment
        |
        v
Aggregate topic counts
        |
        v
Compare against historical baseline
        |
        v
Detect unusual increase
        |
        v
Flag emerging issue
        |
        v
Operations review
```

This is a downstream analytics workflow.

It is not part of the BERTopic training algorithm itself.

---

# 73. Model Monitoring

The model should eventually be monitored for:

```text
topic distribution drift
outlier rate
topic-size changes
topic representation changes
embedding distribution changes
```

A large increase in outliers could indicate:

```text
new issue types
new products
new terminology
data-quality problems
distribution shift
```

---

# 74. Retraining Strategy

The BERTopic model should not be retrained on every incoming ticket.

A future production strategy can use scheduled retraining.

For example:

```text
Historical corpus
      |
      v
Periodic retraining
      |
      v
New BERTopic version
      |
      v
Evaluation
      |
      v
Model registry
```

The exact schedule can be determined later based on ticket volume and topic drift.

---

# 75. Model Versioning

Each production model should have a version.

Example:

```text
supportmind_bertopic/
├── v1/
├── v2/
└── v3/
```

Each version should preserve:

```text
model
configuration
preprocessing version
embedding model
evaluation results
topic metadata
```

This allows historical topic assignments to remain reproducible.

---

# 76. Current Evaluation Summary

The selected BERTopic model produced:

```text
Documents:
8,460

Topics:
59

Outliers:
1,379

Outlier rate:
16.30%

Assigned:
7,081

Assigned rate:
83.70%

Minimum topic size:
31

Median topic size:
77

Mean topic size:
120.02

Maximum topic size:
393

Topic diversity:
0.5983

C_V coherence:
0.7972
```

---

# 77. Final Model Decision

The model is:

```text
FIT FOR PURPOSE
```

for SupportMind's topic-modeling role.

It provides meaningful recurring issue discovery.

It has sufficient topic coverage.

It avoids tiny clusters.

It has strong topic coherence.

It provides useful issue-oriented topic representations.

It is therefore suitable for the next phase of implementation.

---

# 78. Why We Stop Here

The model does not need to be perfect.

SupportMind requires:

```text
Useful
Interpretable
Reproducible
Operationally relevant
```

topic intelligence.

It does not require:

```text
Perfect topic separation
Zero outliers
Maximum theoretical coherence
Endless hyperparameter optimization
```

The current model satisfies the project requirement.

---

# 79. Final Locked Configuration

```text
Model:
BERTopic

BERTopic version:
0.17.4

Embedding:
BAAI/bge-base-en-v1.5

Embedding provider:
FastEmbed

Embedding dimension:
768

UMAP:
n_neighbors=15
n_components=5
min_dist=0.0
metric="cosine"
random_state=42

HDBSCAN:
min_cluster_size=30
min_samples=10
metric="euclidean"
cluster_selection_method="eom"
prediction_data=True

Vectorizer:
stop_words="english"
ngram_range=(1, 2)
min_df=5

Training documents:
8,460

Discovered topics:
59

Outliers:
1,379

Outlier rate:
16.30%

Topic diversity:
0.5983

C_V coherence:
0.7972

Status:
LOCKED / FIT FOR SUPPORTMIND
```

---

# 80. Final SupportMind Architecture

The topic-modeling component fits into the larger system as:

```text
                 SUPPORTMIND AI
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
   Ingestion      Knowledge       Historical
      Layer          Layer          Tickets
        |              |              |
        |              |              |
        v              v              v
      NLP          Qdrant/Neo4j   BERTopic
        |              |              |
        |              |              v
        |              |        Recurring Issues
        |              |              |
        +--------------+--------------+
                       |
                       v
                Intelligence Layer
                       |
          +------------+------------+
          |            |            |
          v            v            v
      Retrieval     Topics      Issue Clusters
          |            |            |
          +------------+------------+
                       |
                       v
                  AutoGen Agents
                       |
                       v
                    FastAPI
                       |
                       v
                  Next.js UI
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
     Customer        Agent         Manager/
     Dashboard     Dashboard      Operations
```

---

# 81. Key Design Principle

The most important principle behind this implementation is:

```text
Dataset
   |
   v
Understand available signal
   |
   v
Select appropriate ML capability
   |
   v
Target-driven preprocessing
   |
   v
Train model
   |
   v
Evaluate usefulness
   |
   v
Deploy only if fit
```

The project does not create models simply because a dataset contains columns that appear predictive.

Each model exists because it solves a real SupportMind problem.

---

# 82. Summary

SupportMind's BERTopic implementation transforms historical customer-support descriptions into recurring issue intelligence.

It uses:

```text
BGE embeddings
+
UMAP
+
HDBSCAN
+
CountVectorizer
+
BERTopic
```

The model was trained on:

```text
8,460 issue-focused support tickets
```

and discovered:

```text
59 recurring issue topics
```

with:

```text
83.70% topic assignment coverage
```

and:

```text
0.7972 C_V coherence
```

The resulting topics include meaningful support problems such as:

```text
battery issues
account access
password reset
network connectivity
charging
hardware problems
software updates
firmware problems
data recovery
application crashes
error messages
peripheral compatibility
```

The model therefore provides the recurring-issue discovery capability required by SupportMind.

The model is now considered:

```text
FINAL
FIT FOR PURPOSE
LOCKED
```

The next implementation step is to convert the trained model output into production-ready topic artifacts and service interfaces.
# BERTopic — Model Artifact Preparation & Validation

## Overview

After completing the targeted preprocessing and training of the BERTopic model, the next stage was to prepare the trained model and its outputs for integration into the SupportMind AI platform.

The objective of this stage was to:

1. Generate topic assignments for individual support tickets.
2. Attach topic metadata to each ticket.
3. Extract representative keywords for every discovered topic.
4. Create a compact topic-level summary.
5. Persist the generated datasets as reusable artifacts.
6. Save the trained BERTopic model.
7. Verify that the saved model can be successfully reloaded.
8. Validate that the reloaded model can perform inference on new documents.

The final BERTopic training corpus contained **8,460 valid support tickets** after excluding only documents that contained no meaningful issue information after the targeted preprocessing.

---

# Step 1 — Generate Ticket-Level Topic Assignments

## Objective

The first step was to associate every ticket in the final BERTopic training corpus with the topic discovered by BERTopic.

The model had already generated:

* `topics` — topic ID assigned to each document.
* `documents_final` — final cleaned text used for BERTopic training.

The topic assignment output was combined with the original ticket information.

### Implementation

```python
topic_results = bertopic_training_data_final[
    [
        "Ticket ID",
        "Ticket Subject",
        "bertopic_issue_text_final_v2",
    ]
].copy()

topic_results = topic_results.merge(
    df[
        [
            "Ticket ID",
            "Product Purchased",
        ]
    ],
    on="Ticket ID",
    how="left",
    validate="one_to_one",
)

topic_results["Topic ID"] = topics
```

## Output

The resulting dataset contained:

| Column                       | Purpose                             |
| ---------------------------- | ----------------------------------- |
| Ticket ID                    | Original support ticket identifier  |
| Ticket Subject               | Original ticket subject             |
| bertopic_issue_text_final_v2 | Final text used for topic modelling |
| Product Purchased            | Product associated with the ticket  |
| Topic ID                     | BERTopic-assigned topic             |

### Validation

The generated dataset contained:

* **8,460 rows**
* **5 columns**
* No missing `Product Purchased` values
* Topic counts matched the trained BERTopic model

The topic distribution included:

* `-1` → outlier/unassigned documents
* `0–58` → discovered support issue topics

Therefore, the ticket-level topic assignment was validated successfully.

---

# Step 2 — Attach Topic Metadata

## Objective

A numeric topic ID alone is not useful for downstream applications.

Therefore, each topic was enriched with metadata generated by BERTopic.

The metadata included:

* Topic ID
* Topic size
* Automatically generated topic name

### Implementation

```python
topic_info = topic_model.get_topic_info()

topic_metadata = topic_info[
    [
        "Topic",
        "Count",
        "Name",
    ]
].copy()

topic_metadata = topic_metadata.rename(
    columns={
        "Topic": "Topic ID",
        "Count": "Topic Size",
        "Name": "Topic Name",
    }
)

topic_results = topic_results.merge(
    topic_metadata,
    on="Topic ID",
    how="left",
    validate="many_to_one",
)
```

## Result

The enriched ticket dataset contained:

```text
Ticket ID
Ticket Subject
bertopic_issue_text_final_v2
Product Purchased
Topic ID
Topic Size
Topic Name
```

### Validation

Validation confirmed:

* **8,460 ticket records**
* Topic metadata successfully attached to every ticket
* No missing topic metadata

For example, a ticket could now contain:

```text
Ticket ID: 1
Product: GoPro Hero
Topic ID: 31
Topic Size: 71
```

The `Topic Name` generated by BERTopic is treated as a machine-generated description rather than a final business label.

---

# Step 3 — Extract Topic Keywords

## Objective

BERTopic produces representative words and phrases for every discovered topic.

These keywords are useful for:

* Understanding what each topic represents.
* Operations dashboards.
* Topic interpretation.
* Emerging issue analysis.
* Human-readable reporting.

The top **10 keywords** were extracted for every discovered topic.

Topic `-1` was excluded because it represents BERTopic outliers rather than a meaningful discovered topic.

### Implementation

```python
topic_keywords = []

for topic_id in topic_model.get_topics():
    if topic_id == -1:
        continue

    words_with_scores = topic_model.get_topic(topic_id)

    for rank, (word, score) in enumerate(
        words_with_scores[:10],
        start=1
    ):
        topic_keywords.append(
            {
                "Topic ID": topic_id,
                "Rank": rank,
                "Word": word,
                "Score": score,
            }
        )

topic_keywords_df = pd.DataFrame(topic_keywords)
```

## Result

The output contained:

* **59 discovered topics**
* **10 keywords per topic**
* **590 topic-keyword records**

Each keyword has:

* Topic ID
* Rank
* Word/phrase
* BERTopic relevance score

Example:

```text
Topic ID: 1

1. error message
2. screen says
3. popping
...
```

These keywords provide an interpretable representation of the discovered support issue.

---

# Step 4 — Create Compact Topic Summary

## Objective

The keyword-level dataset is useful for detailed analysis, but downstream systems also need a compact topic-level representation.

Therefore, the top 10 keywords were combined into a single topic summary.

### Implementation

```python
topic_summary = (
    topic_keywords_df
    .sort_values(["Topic ID", "Rank"])
    .groupby("Topic ID")
    .agg(
        Top_Keywords=("Word", lambda x: ", ".join(x))
    )
    .reset_index()
)

topic_summary = topic_summary.merge(
    topic_metadata[
        ["Topic ID", "Topic Size", "Topic Name"]
    ],
    on="Topic ID",
    how="left",
    validate="one_to_one",
)

topic_summary = topic_summary[
    [
        "Topic ID",
        "Topic Size",
        "Topic Name",
        "Top_Keywords",
    ]
].sort_values("Topic ID")
```

## Final Structure

The topic summary contains:

| Column       | Description                              |
| ------------ | ---------------------------------------- |
| Topic ID     | Unique discovered topic                  |
| Topic Size   | Number of tickets belonging to the topic |
| Topic Name   | BERTopic-generated name                  |
| Top Keywords | Top 10 representative keywords           |

This provides a compact representation suitable for dashboards and Operations intelligence.

---

# Step 5 — Save BERTopic Output Artifacts

## Objective

The generated topic information should not need to be regenerated every time the SupportMind application starts.

Therefore, the outputs were persisted as reusable CSV artifacts.

### Output directory

```text
C:\Users\danie\supportmind-ai\data\processed\bertopic
```

### Implementation

```python
from pathlib import Path

output_dir = Path(
    r"C:\Users\danie\supportmind-ai\data\processed\bertopic"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

topic_results_path = (
    output_dir /
    "bertopic_ticket_topic_assignments.csv"
)

topic_summary_path = (
    output_dir /
    "bertopic_topic_summary.csv"
)

topic_keywords_path = (
    output_dir /
    "bertopic_topic_keywords.csv"
)

topic_results.to_csv(
    topic_results_path,
    index=False
)

topic_summary.to_csv(
    topic_summary_path,
    index=False
)

topic_keywords_df.to_csv(
    topic_keywords_path,
    index=False
)
```

## Persisted Artifacts

Three reusable datasets were generated:

```text
data/
└── processed/
    └── bertopic/
        ├── bertopic_ticket_topic_assignments.csv
        ├── bertopic_topic_summary.csv
        └── bertopic_topic_keywords.csv
```

### Purpose

These artifacts provide persistent access to:

* Ticket → Topic relationships
* Topic statistics
* Topic keywords

without retraining BERTopic.

---

# Step 6 — Save the Trained BERTopic Model

## Objective

The trained BERTopic model itself must be persisted so that SupportMind can perform topic inference on future support tickets.

A dedicated model directory was created:

```text
C:\Users\danie\supportmind-ai\
backend\app\nlp\topic_modeling\
models\supportmind_bertopic
```

### Implementation

```python
from pathlib import Path

model_dir = Path(
    r"C:\Users\danie\supportmind-ai\backend\app\nlp\topic_modeling\models\supportmind_bertopic"
)

model_dir.mkdir(
    parents=True,
    exist_ok=True
)
```

The trained model was then saved using BERTopic's `safetensors` serialization:

```python
topic_model.save(
    str(model_dir),
    serialization="safetensors",
    save_ctfidf=True,
)
```

## Why `safetensors`?

`safetensors` provides a structured and safer format for storing model-related tensor data.

The saved model contains the learned BERTopic components required for topic representation and inference.

---

# Step 7 — Reload the Persisted Model

## Objective

Saving a model successfully does not guarantee that it can be restored correctly.

Therefore, the saved model was explicitly reloaded from disk.

### Implementation

```python
from bertopic import BERTopic

loaded_topic_model = BERTopic.load(
    str(model_dir)
)
```

## Validation

The model successfully reloaded.

The reloaded model contained:

```text
59 discovered topics
```

Topic keyword inspection also matched the original trained model.

For example, the keywords and scores for Topic 1 remained consistent after reloading.

### Important Note

The model was trained with:

```python
embedding_model=None
```

because embeddings were generated separately using:

```text
BAAI/bge-base-en-v1.5
```

and supplied directly during training.

Therefore, BERTopic displayed an informational warning during loading indicating that an embedding model was not explicitly stored.

This is **not a model failure**.

For production inference, SupportMind must use the same embedding model:

```text
BAAI/bge-base-en-v1.5
```

to generate embeddings for new support text before passing them to the loaded BERTopic model.

---

# Step 8 — Validate Inference After Reload

## Objective

The final validation step was to ensure that the persisted model could actually classify new support text into previously learned topics.

Five documents from the final modelling corpus were used as an inference test.

### Implementation

```python
test_documents = documents_final[:5]

inferred_topics, inferred_probabilities = (
    loaded_topic_model.transform(
        test_documents,
        embeddings=embeddings[:5],
    )
)
```

## Result

The reloaded model successfully generated topic predictions:

```text
Test 1 → Topic 31
Test 2 → Topic 4
Test 3 → Topic 14
Test 4 → Topic 40
Test 5 → Topic 8
```

The predictions were semantically consistent with the underlying support issues.

Examples included:

| Issue pattern            | Inferred topic |
| ------------------------ | -------------: |
| Intermittent issue       |        Topic 4 |
| Device not turning on    |       Topic 14 |
| Unresolved support issue |       Topic 40 |
| Battery-life decrease    |        Topic 8 |

## Final Validation

This confirmed that:

```text
Trained Model
      ↓
Saved to Disk
      ↓
Reloaded
      ↓
New Text + BGE Embedding
      ↓
BERTopic Transform
      ↓
Valid Topic Prediction
```

The model persistence and inference pipeline therefore passed validation.

---

# Final BERTopic Artifact Structure

After Steps 1–8, the BERTopic component produced the following structure:

```text
supportmind-ai/
│
├── data/
│   └── processed/
│       ├── supportmind_bertopic_cleaned_dataset.csv
│       ├── supportmind_bertopic_embeddings.npy
│       │
│       └── bertopic/
│           ├── bertopic_ticket_topic_assignments.csv
│           ├── bertopic_topic_summary.csv
│           └── bertopic_topic_keywords.csv
│
└── backend/
    └── app/
        └── nlp/
            └── topic_modeling/
                ├── __init__.py
                ├── bertopic_model.py
                ├── topic_service.py
                ├── notebook/
                │   └── notebook_modelling_eda.ipynb
                │
                └── models/
                    └── supportmind_bertopic/
```

# Step 1–8 Completion Status

| Step | Component                        | Status     |
| ---- | -------------------------------- | ---------- |
| 1    | Ticket-level topic assignments   | ✅ Complete |
| 2    | Topic metadata                   | ✅ Complete |
| 3    | Topic keywords                   | ✅ Complete |
| 4    | Topic summary                    | ✅ Complete |
| 5    | Output artifact persistence      | ✅ Complete |
| 6    | BERTopic model persistence       | ✅ Complete |
| 7    | Model reload validation          | ✅ Complete |
| 8    | Post-reload inference validation | ✅ Complete |

The BERTopic model and its supporting artifacts are therefore ready for integration into the SupportMind intelligence pipeline.







Now — how does this actually help the final SupportMind system?

This is the key architectural point:

A customer does not directly “talk to BERTopic.”

Instead:

Customer Query
      ↓
NLP / Query Understanding
      ↓
SupportMind Agent
      ↓
Retrieval + Knowledge + Intelligence
      ↓
Response

BERTopic sits more on the intelligence/analytics side.

Example 1 — Customer has a battery problem

Customer asks:

“My Dyson vacuum battery suddenly dies after a few minutes.”

The live ticket can be embedded and passed through the BERTopic pipeline.

BERTopic may identify it as something like:

Topic 8
Battery-life / sudden decrease

Then SupportMind can use that topic as an issue signal.

The system can combine:

Customer Query
      ↓
Topic = Battery-life issue
      ↓
Similar historical tickets
      ↓
Troubleshooting knowledge
      ↓
Current incident information
      ↓
Resolution recommendation

So BERTopic helps the agent understand:

“What broader support issue does this customer's problem belong to?”

Example 2 — Customer asks about an intermittent problem

“Sometimes my PlayStation works perfectly, but sometimes it suddenly stops responding.”

BERTopic can map this to an intermittent/unexpected behavior topic.

That topic can then help retrieval find other tickets with similar issue patterns.

The agent isn't simply searching for the exact sentence.

Instead:

"works sometimes, fails sometimes"
             ↓
       Issue Topic
             ↓
Historical tickets with same issue pattern
             ↓
Relevant troubleshooting information

That's much more useful for SupportMind.

Example 3 — Operations discovers an emerging problem

This is actually where BERTopic becomes much more valuable.

Imagine SupportMind receives:

Day 1 → 12 update-related tickets
Day 2 → 27
Day 3 → 51
Day 4 → 103

The Operations dashboard can identify:

Topic: Software/Firmware Update
Volume: rapidly increasing
Affected products: multiple

SupportMind can then surface:

Emerging Issue Detected: Support tickets related to recent software/firmware updates are increasing rapidly.

That is an Operations intelligence capability, not a chatbot capability.

Example 4 — Manager asks

“What are the major issues customers are facing this week?”

This is a very good BERTopic-related query.

The manager/operations agent can query the topic analytics:

Topic ID
Topic Size
Topic Keywords
Product distribution
Time distribution

and generate a summary such as:

1. Software update issues
2. Battery-life problems
3. Account/login problems
4. Connection problems
5. Charging issues

BERTopic provides the underlying issue clusters.

Example 5 — Manager asks

“What issues are increasing the fastest?”

Again, this is a strong use case.

The system compares topic volumes over time:

Topic A
Monday     21
Tuesday    29
Wednesday  48
Thursday   76
Friday     103

That becomes an emerging issue signal.

But BERTopic should NOT answer these questions directly

For example:

“What is your refund policy?”

❌ BERTopic isn't the right component.

Use:

Question
 ↓
Retrieval
 ↓
Qdrant / Knowledge Base
 ↓
Refund Policy
 ↓
Answer

Similarly:

“How do I reset my account password?”

❌ BERTopic shouldn't generate the answer.

Use:

Intent
 ↓
Knowledge Retrieval
 ↓
Account Management Guide
 ↓
Answer

And:

“What is the status of my ticket?”

❌ BERTopic isn't needed.

Use:

Customer identity/ticket ID
 ↓
Ticket database
 ↓
Current ticket status
 ↓
Response
So where does BERTopic fit among our agents?

This is the important final architecture:

                    SUPPORTMIND
                         │
              ┌──────────┴──────────┐
              │                     │
        Customer/Agent         Intelligence
          Interaction              Layer
              │                     │
              │              ┌──────┴──────┐
              │              │             │
              │          BERTopic      Clustering
              │              │             │
              │              └──────┬──────┘
              │                     │
              │              Recurring Issues
              │              Emerging Issues
              │              Issue Trends
              │                     │
              └──────────────┬──────┘
                             ↓
                       Agent Layer
                             │
                ┌────────────┼────────────┐
                ↓            ↓            ↓
           Retrieval     Resolution    Escalation
             Agent         Agent         Agent
                │            │            │
                └────────────┼────────────┘
                             ↓
                        Final Response
The biggest distinction

BERTopic tells SupportMind what kinds of problems are happening.

Retrieval tells SupportMind what information is relevant.

Resolution Agent determines how to respond.

Escalation Agent determines whether a human/manager needs to become involved.

Operations Agent uses BERTopic to understand recurring and emerging problems across the organization.

So the final customer journey can look like:

Customer: “My battery suddenly dies after 10 minutes.”

↓

Issue understanding

Battery-life issue

↓

BERTopic

Battery-related recurring issue

↓

Similar-ticket retrieval

Find historical tickets with similar symptoms

↓

Knowledge retrieval

Troubleshooting guide
Warranty/policy
Known incidents

↓

Resolution Agent

Generate evidence-backed troubleshooting response

↓

Customer receives answer

Meanwhile, the same BERTopic signal can go to Operations:

Battery issue
   ↓
+120 similar tickets
   ↓
Mostly affecting Product X
   ↓
Increasing 38% this week
   ↓
Potential emerging product issue

That is exactly why BERTopic has a place in SupportMind even though it isn't a chatbot model.

And importantly, this keeps our original principle intact: we aren't adding BERTopic just to increase the model count. It has a specific role — discovering and monitoring recurring/emerging support issue themes.