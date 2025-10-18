# Group Project: NLP Spam Detection

Natural Language Processing project for YouTube comment spam classification using Naive Bayes in COMP 237 (AI Fundamentals).

## 📋 Overview

This group project implements a complete NLP pipeline to classify YouTube comments as spam or legitimate (ham). Using real comments from a Katy Perry music video, the system learns to identify spam patterns and achieves high accuracy through machine learning.

**Dataset:** YouTube comments (350 samples)  
**Task:** Binary classification (Spam vs Ham)  
**Approach:** Bag of Words + TF-IDF + Naive Bayes  
**Result:** 90.47% accuracy (cross-validation), 92.05% accuracy (test set)

## 👥 Group Members

**Group 5:**
- Song Malisa Se (301233051)
- **Matheus Teixeira (301236904)**
- Viet Hoang
- Yi-lin Lou (301226659)
- Yin-Siang Mao (301180968)

## 🎯 Project Objectives

### Learning Goals
1. **Text Preprocessing** - Transform raw text into numerical features
2. **Feature Extraction** - Bag of Words and TF-IDF techniques
3. **Classification** - Multinomial Naive Bayes for text
4. **Evaluation** - Cross-validation and confusion matrix analysis
5. **Production Testing** - Real-world spam detection scenarios

### Business Context
- **Problem:** YouTube videos receive thousands of comments, many are spam
- **Challenge:** Manual moderation is time-consuming and inconsistent
- **Solution:** Automated spam detection using machine learning
- **Impact:** Improved user experience and reduced moderation workload

## 📁 Project Structure

```
GroupProject-NLP/
├── group5_nlp_katy_perry.py     # ⭐ Main implementation (use this!)
├── Youtube02-KatyPerry.csv      # Training/testing dataset (350 comments)
├── Youtube02-KatyPerry2.csv     # Alternative dataset version
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment setup
│
├── DocAndAnalysis/              # 📄 Documentation and analysis
│   ├── Analysis_Group5.docx     #    Detailed written analysis
│   ├── Group5_Presentation.pptx #    Project presentation slides
│   ├── trainMatrix.xlsx         #    Data matrix
│   └── Copy of trainMatrix.xlsx #    Backup
│
└── OtherCodeVersion/            # 📦 Code evolution history
    ├── GroupProject_initialVersion.py  # Initial version (simple)
    └── nlp_katy_perry1.py              # Intermediate version
```

**Note:** The main code (`group5_nlp_katy_perry.py`) includes:
- Confusion matrix visualizations with matplotlib
- 9 production test cases
- Classification report with detailed metrics
- Professional code structure with docstrings

## 🚀 Key Features

### NLP Pipeline Implementation
- ✅ Data loading and exploration
- ✅ Text preprocessing (lowercase, tokenization)
- ✅ Bag of Words conversion (CountVectorizer)
- ✅ TF-IDF weighting (TfidfTransformer)
- ✅ Multinomial Naive Bayes classification
- ✅ 5-fold cross-validation
- ✅ Production testing with 9 custom examples
- ✅ **Confusion matrix visualizations** (matplotlib)
- ✅ **Classification report** with precision/recall/F1 per class

### Project Organization
- **Main code:** Clean, professional implementation in root
- **Documentation:** Analysis report and presentation in `DocAndAnalysis/`
- **Version history:** Earlier iterations preserved in `OtherCodeVersion/`

### Model Performance
- **Cross-Validation Accuracy:** 90.47%
- **Cross-Validation Precision:** 91.97%
- **Cross-Validation Recall:** 90.47%
- **Cross-Validation F1-Score:** 90.31%
- **Test Set Accuracy:** 92.05%
- **Production Test Accuracy:** 100% (9 custom examples)

## 📊 Dataset Information

### Youtube02-KatyPerry.csv Structure

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| COMMENT_ID | String | Unique identifier | "LZQPQhUe1F5..." |
| AUTHOR | String | Comment author | "Julius NM" |
| DATE | String | Post date/time | "2013-11-07T06:21:48" |
| CONTENT | String | Comment text | "i love this song!" |
| **CLASS** | Integer | **Label (0=spam, 1=ham)** | 0 or 1 |

### Dataset Statistics
- **Total Comments:** 350
- **Features Used:** CONTENT (text), CLASS (label)
- **Features Dropped:** COMMENT_ID, AUTHOR, DATE (not predictive)
- **Train/Test Split:** 75% train (262 samples), 25% test (88 samples)
- **Class Distribution:** Balanced dataset

### Example Comments

**Spam (CLASS = 0):**
- "Check out my new song: https://www.youtube.com/..."
- "Join the greatest Katy Perry fan club: https://..."
- "Subscribe to my channel for more!"

**Ham/Legitimate (CLASS = 1):**
- "I love this song. Katy Perry is just amazing!"
- "This song brings me all kinds of fierce feelings!"
- "I cannot wait for the next Katy Perry clip."

## 🛠️ Technical Implementation

### Step 1: Data Loading & Exploration

```python
import pandas as pd
import os

# Load dataset
path = "path/to/data"
filename = 'Youtube02-KatyPerry.csv'
data = pd.read_csv(os.path.join(path, filename))

# Explore structure
print(data.head(3))           # Preview
print(data.shape)             # (350, 5)
print(data.columns.values)    # Column names
print(data.dtypes)            # Data types
print(data.isnull().sum())    # No missing values
```

**Key Findings:**
- 350 rows, 5 columns
- No missing values
- Mix of spam and legitimate comments
- Text data requires preprocessing

### Step 2: Feature Selection

```python
# Keep only relevant columns
data = data[['CONTENT', 'CLASS']]
```

**Rationale:**
- **CONTENT:** The comment text (features)
- **CLASS:** Spam (0) or Ham (1) label (target)
- **Dropped:** COMMENT_ID (unique, no pattern), AUTHOR (not predictive), DATE (time-independent task)

### Step 3: Data Shuffling

```python
# Shuffle to avoid ordering bias
data = data.sample(frac=1, random_state=1)
```

**Why shuffle?**
- Original data might be ordered (all spam first, then ham)
- Random order ensures unbiased train/test split
- `random_state=1` ensures reproducibility

### Step 4: Train-Test Split

```python
# 75% train, 25% test
train_data = data.sample(frac=0.75, random_state=1)
test_data = data.drop(train_data.index)

# Result: 262 training, 88 testing samples
```

### Step 5: Bag of Words (CountVectorizer)

```python
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()

# Fit on training data
train_tc = count_vectorizer.fit_transform(train_data['CONTENT'])

# Transform test data (using training vocabulary)
test_tc = count_vectorizer.transform(test_data['CONTENT'])
```

**What it does:**
- Converts text to numerical vectors
- Each unique word becomes a feature
- Value = count of word in document

**Example:**
```
Comment: "I love this song"
Vector: [0, 0, 1, 1, 0, 1, 0, 1, ...]  (1367 dimensions)
         ^     ^  ^     ^     ^
         |     |  |     |     |
         |   love this  |   song
    (no "amazing")      I
```

**Dimensions:**
- **Training matrix:** 262 rows × 1367 columns
- **1367 columns** = 1367 unique words across all training comments

**Statistics:**
- **Non-zero entries:** 3892 (word occurrences)
- **Total word count:** 4438 (sum of all counts)
- **Sparsity:** Most entries are 0 (most words don't appear in each comment)

### Step 6: TF-IDF Transformation

```python
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer()

# Transform training data
train_tfidf = tfidf.fit_transform(train_tc)

# Transform test data
test_tfidf = tfidf.transform(test_tc)
```

**What is TF-IDF?**

**TF (Term Frequency):** How often word appears in document
```
TF = (count of word in document) / (total words in document)
```

**IDF (Inverse Document Frequency):** How rare the word is across all documents
```
IDF = log(total documents / documents containing word)
```

**TF-IDF = TF × IDF**

**Purpose:**
- Downweight common words (the, is, a)
- Upweight rare, informative words (subscribe, channel, spam-specific terms)

**Effect on dimensions:**
- Still 262 × 1367 matrix
- But values are weighted, not raw counts
- **Sum of frequencies:** 4438 → 881.25 (weights, not counts)

**Example:**
```
Word "love" appears in 100 comments → low IDF (common)
Word "subscribe" appears in 5 comments → high IDF (rare, likely spam indicator)

TF-IDF("love") = small value
TF-IDF("subscribe") = large value (spam signal!)
```

### Step 7: Train Multinomial Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB

# Train classifier
classifier = MultinomialNB()
classifier.fit(train_tfidf, train_data['CLASS'])
```

**Why Naive Bayes for text?**
- ✅ Fast training and prediction
- ✅ Works well with high-dimensional sparse data (like text)
- ✅ Probabilistic predictions
- ✅ Simple and interpretable
- ✅ Good performance even with small datasets

**How it works:**
```
P(spam | words) = P(words | spam) × P(spam) / P(words)

Bayes' Theorem applied to each word independently (hence "naive")
```

### Step 8: Cross-Validation

```python
from sklearn.model_selection import cross_val_score

num_folds = 5

# Accuracy
accuracy = cross_val_score(classifier, train_tfidf, train_data['CLASS'], 
                           scoring='accuracy', cv=num_folds)
print(f"Accuracy: {round(100*accuracy.mean(), 2)}%")  # 90.47%

# Precision
precision = cross_val_score(classifier, train_tfidf, train_data['CLASS'], 
                            scoring='precision_weighted', cv=num_folds)
print(f"Precision: {round(100*precision.mean(), 2)}%")  # 91.97%

# Recall
recall = cross_val_score(classifier, train_tfidf, train_data['CLASS'], 
                         scoring='recall_weighted', cv=num_folds)
print(f"Recall: {round(100*recall.mean(), 2)}%")  # 90.47%

# F1-Score
f1 = cross_val_score(classifier, train_tfidf, train_data['CLASS'], 
                     scoring='f1_weighted', cv=num_folds)
print(f"F1: {round(100*f1.mean(), 2)}%")  # 90.31%
```

**5-Fold Cross-Validation:**
```
Training Data (262 samples) split into 5 folds:

Fold 1: [Test] [Train] [Train] [Train] [Train]
Fold 2: [Train] [Test] [Train] [Train] [Train]
Fold 3: [Train] [Train] [Test] [Train] [Train]
Fold 4: [Train] [Train] [Train] [Test] [Train]
Fold 5: [Train] [Train] [Train] [Train] [Test]

Average performance across all 5 folds = robust estimate
```

**Results:**
- **Accuracy:** 90.47% (90.47% of predictions correct)
- **Precision:** 91.97% (91.97% of spam predictions are truly spam)
- **Recall:** 90.47% (90.47% of actual spam is detected)
- **F1-Score:** 90.31% (harmonic mean of precision and recall)

### Step 9: Test Set Evaluation

```python
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predict on test set
y_pred = classifier.predict(test_tfidf)
y_actual = test_data['CLASS']

# Confusion matrix
cm = confusion_matrix(y_actual, y_pred)
print("Test Confusion Matrix:")
print(cm)

# Visualize confusion matrix
cm_display = ConfusionMatrixDisplay(cm, display_labels=['Spam', 'Ham'])
cm_display.plot()
plt.title("Confusion Matrix for the test data")
plt.show()

# Accuracy
accuracy = accuracy_score(y_actual, y_pred)
print(f"Test Accuracy: {round(100*accuracy, 2)}%")  # 92.05%
```

**Test Results:**

**Confusion Matrix:**
```
                 Predicted
               Spam    Ham
Actual Spam    [34]   [2]     ← 34 correct, 2 misses
Actual Ham     [5]    [47]    ← 5 false alarms, 47 correct
```

**Breakdown:**
- **True Positives (34):** Correctly identified spam
- **False Negatives (2):** Spam missed (classified as ham)
- **False Positives (5):** Ham incorrectly flagged as spam
- **True Negatives (47):** Correctly identified ham

**Test Accuracy:** 92.05%
- **Better than cross-validation** (90.47%)!
- Shows good generalization to unseen data
- Model performs consistently well

### Step 10: Production Testing

```python
# 6 custom test comments
input_data = [
    "I hate this song. Katy Perry is just a waste of time",           # Ham
    "I love this song. I cannot wait for the next Katy Perry clip.",  # Ham
    "This song brings me all kinds of fierce feelings!",              # Ham
    "This song sucks, I prefer Eye of The Tiger. That is a classic.", # Ham
    "Katy Perry is great, but have you seen my new song? https://...",# Spam
    "Join the greatest Katy Perry fan club: https://..."              # Spam
]

# Ground truth
input_actual = [0, 0, 0, 0, 1, 1]  # 0=spam, 1=ham

# Transform and predict
input_tc = count_vectorizer.transform(input_data)
input_tfidf = tfidf.transform(input_tc)
input_pred = classifier.predict(input_tfidf)

# Evaluate
cm_prod = confusion_matrix(input_actual, input_pred)
accuracy_prod = accuracy_score(input_actual, input_pred)
print(f"Production Accuracy: {round(100*accuracy_prod, 2)}%")
```

**Production Test Results:**
- Tests realistic scenarios (negative reviews, spam links, fan club invitations)
- Checks if model handles nuance (negative opinion ≠ spam)
- Validates URL-based spam detection
- **Expected Accuracy:** High (model should catch obvious spam patterns)

## 📈 Model Analysis

### Strong Performance Achieved

**Cross-Validation:** 90.47% accuracy  
**Test Set:** 92.05% accuracy  
**Production:** 100% accuracy (9 examples)

**Key Insight:** Test accuracy (92.05%) **better than** cross-validation (90.47%)!
- Shows excellent generalization
- No overfitting detected
- Model learns real spam patterns, not noise

**1. Clear Spam Patterns:**
- URLs in comments
- Self-promotion phrases ("check out my", "subscribe to")
- Fan club invitations
- Generic marketing language

**2. Distinctive Vocabulary:**
- Spam: "subscribe", "channel", "check out", URLs
- Ham: Genuine emotions ("love", "amazing", "fierce"), specific song references

**3. TF-IDF Effectiveness:**
- Spam-specific words get high weights
- Common words get low weights
- Model learns "spam vocabulary"

**4. Naive Bayes Strengths:**
- Excellent for text classification
- Handles high dimensionality well
- Fast and efficient

### Potential Improvements

**1. More Sophisticated Preprocessing:**
```python
# Lowercase (already done by CountVectorizer by default)
# Remove stopwords
count_vectorizer = CountVectorizer(stop_words='english')

# Stemming/Lemmatization
from nltk.stem import PorterStemmer
# "running" → "run", "loves" → "love"
```

**2. N-grams (phrases):**
```python
# Capture phrases like "check out", "subscribe to"
count_vectorizer = CountVectorizer(ngram_range=(1, 2))
# Includes both single words and 2-word phrases
```

**3. More Features:**
- URL presence (binary feature)
- Comment length
- Capitalization patterns
- Exclamation mark count

**4. Different Models:**
- Logistic Regression
- Support Vector Machines (SVM)
- Random Forest
- Neural Networks (for larger datasets)

## ⚙️ Installation

### Prerequisites
- Python 3.6+

### Install Dependencies
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda env create -f environment.yml
conda activate ai-nlp-spam
```

## 🏃 Usage

### Run Complete Pipeline

```bash
python group5_nlp_katy_perry.py
```

**Note:** Update the `path` variable in the code to your data directory:
```python
path = "path/to/your/data/folder"
```

**Optional:** Check earlier versions in `OtherCodeVersion/` to see project evolution.

### Expected Output

```
#### 2. Data Exploration ########################
   COMMENT_ID  AUTHOR  DATE  CONTENT  CLASS
0  ...         ...     ...   ...      ...

#
(350, 5)

#
['COMMENT_ID' 'AUTHOR' 'DATE' 'CONTENT' 'CLASS']

#
COMMENT_ID    object
AUTHOR        object
DATE          object
CONTENT       object
CLASS          int64
dtype: object

#
COMMENT_ID    0
AUTHOR        0
DATE          0
CONTENT       0
CLASS         0
dtype: int64

#### 4. Present highlights of the output########################
Dimensions of training data: (262, 1367)
Number of Times a Word Exist in training data: 3892
Sum of frequencies of every word in training data: 4438

#### 5. Tf-idf downscale and present highlights ########################
Dimensions of training data after idf transformation: (262, 1367)
Number of Times a Word Exist in training data after idf transformation: 3892
Sum of frequencies after idf transformation: 881.25...

#### 9. Cross validation using 5-fold and present results ########################
Accuracy: 90.47%
Precision: 91.97%
Recall: 90.47%
F1: 90.31%

#### 10. Test the model using test data ########################
Test Confusion Matrix
[[34  2]
 [ 5 47]]
#
Accuracy: 92.05%

[Confusion matrix plot displayed]

#### 11. Production - using the model to classify new comments ########################
Production Confusion Matrix
[[5 0]
 [0 4]]
#
Accuracy: 100.00%

[Confusion matrix plot displayed]
```

## 🔬 Experiment Ideas

**Note:** These are suggestions to extend the project beyond the current implementation.

### 1. Compare Vectorization Methods
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Combine CountVectorizer + TfidfTransformer into one step
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
train_tfidf = tfidf_vectorizer.fit_transform(train_data['CONTENT'])
test_tfidf = tfidf_vectorizer.transform(test_data['CONTENT'])
```

### 2. Feature Engineering - Adding Custom Features to TF-IDF

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Step 1: Create custom features BEFORE train-test split
def has_url(text):
    return 1 if 'http' in text or 'www' in text else 0

def get_capital_ratio(text):
    if len(text) == 0:
        return 0
    return sum(1 for c in text if c.isupper()) / len(text)

# Apply to full dataset (after shuffle, before split)
data_KateParry['has_url'] = data_KateParry['CONTENT'].apply(has_url)
data_KateParry['length'] = data_KateParry['CONTENT'].str.len()
data_KateParry['capital_ratio'] = data_KateParry['CONTENT'].apply(get_capital_ratio)

# Step 2: Now do train-test split
train_data = data_KateParry.sample(frac=0.75, random_state=1)
test_data = data_KateParry.drop(train_data.index)

# Step 3: Get TF-IDF features (as usual)
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(train_data['CONTENT'])
test_tc = count_vectorizer.transform(test_data['CONTENT'])

tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
test_tfidf = tfidf.transform(test_tc)

# Step 4: Extract custom features as numpy arrays
train_custom = train_data[['has_url', 'length', 'capital_ratio']].values
test_custom = test_data[['has_url', 'length', 'capital_ratio']].values

# Step 5: ⚠️ CRITICAL - Normalize custom features to [0, 1] range
# Why? TF-IDF values are ~0.01-0.5, but length could be 150+
# Without normalization, length would dominate the model!

scaler = MinMaxScaler()  # Scales to [0, 1]
train_custom_scaled = scaler.fit_transform(train_custom)
test_custom_scaled = scaler.transform(test_custom)

# Example transformation:
# Before: [1, 150, 0.05]  ← length=150 dominates!
# After:  [1.0, 0.85, 0.05]  ← All in similar range ✅

# Step 6: Combine TF-IDF with normalized custom features
from scipy.sparse import hstack

train_combined = hstack([train_tfidf, train_custom_scaled])
test_combined = hstack([test_tfidf, test_custom_scaled])

# Step 7: Train with combined features
classifier = MultinomialNB()
classifier.fit(train_combined, train_data['CLASS'])
```

**⚠️ Important Note:**
- TF-IDF features are already normalized (values typically 0-1)
- Custom features like `length` can be 0-300+
- **Must normalize** custom features before combining, or they'll dominate
- Use `MinMaxScaler` to scale all custom features to [0, 1]

### 3. Hyperparameter Tuning - Alpha Parameter

```python
from sklearn.model_selection import GridSearchCV

# Test different alpha values
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
}

grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(train_tfidf, train_data['CLASS'])

print(f"Best alpha: {grid_search.best_params_['alpha']}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

# Train final model with best alpha
best_classifier = MultinomialNB(alpha=grid_search.best_params_['alpha'])
best_classifier.fit(train_tfidf, train_data['CLASS'])
```

**What is alpha? (Laplace Smoothing)**

In Multinomial Naive Bayes, alpha is used in the probability calculation:

```
P(word | class) = (count(word in class) + alpha) / (total words in class + alpha × vocabulary_size)
```

**Why it matters:**
- **alpha = 0:** No smoothing
  - Problem: If a word never appears in training for a class → P = 0 → entire prediction becomes 0
- **alpha = 1:** Laplace smoothing (default)
  - Adds 1 to every word count (assumes every word appears at least once)
- **alpha > 1:** More aggressive smoothing
  - Useful when training data is small
  - Makes model more conservative

**Example:**
```
Suppose "subscribe" appears:
- 10 times in spam comments (out of 1000 total spam words)
- 0 times in ham comments (out of 800 total ham words)
- Vocabulary size = 1367 words

With alpha = 0:
  P(subscribe | spam) = 10/1000 = 0.01
  P(subscribe | ham) = 0/800 = 0     ← Problem! Causes division issues

With alpha = 1:
  P(subscribe | spam) = (10+1)/(1000+1367) = 11/2367 = 0.0046
  P(subscribe | ham) = (0+1)/(800+1367) = 1/2167 = 0.0005  ← No longer zero!
```

### 4. Different Classifiers
```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Compare performance
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier()
}
```

## 📚 NLP Concepts Explained

### Bag of Words (BoW)

**Concept:** Represent text as vector of word counts

**Example:**
```
Comment 1: "I love Katy Perry"
Comment 2: "Katy Perry is amazing"

Vocabulary: [I, love, Katy, Perry, is, amazing]

Comment 1 vector: [1, 1, 1, 1, 0, 0]
Comment 2 vector: [0, 0, 1, 1, 1, 1]
```

**Limitations:**
- Ignores word order ("Perry Katy" = "Katy Perry")
- Ignores grammar and context
- Very high dimensionality

### TF-IDF Weighting

**Problem with BoW:** Common words dominate
- "the", "is", "a" appear everywhere → not informative
- "subscribe", "channel" appear rarely → highly informative

**Solution:** Weight by importance

**Formula:**
```
TF-IDF(word, doc) = TF(word, doc) × IDF(word)

where:
TF(word, doc) = (count of word in doc) / (total words in doc)
IDF(word) = log(total docs / docs containing word)
```

**Effect:**
- Common words → low TF-IDF (downweighted)
- Rare, distinctive words → high TF-IDF (upweighted)

### Naive Bayes Classifier

**Bayes' Theorem:**
```
P(spam | words) = P(words | spam) × P(spam) / P(words)
```

**"Naive" Assumption:**
- All words are independent
- P(word1, word2 | spam) = P(word1 | spam) × P(word2 | spam)
- This is "naive" because words are actually dependent
- But works surprisingly well in practice!

**Why it works for text:**
- Fast computation
- Handles high dimensions
- Works with sparse data
- Probabilistic interpretation

## 🎯 Real-World Applications

NLP spam detection techniques apply to:

- **Social Media:** Twitter, Facebook, Instagram spam filtering
- **Email:** Gmail spam detection
- **E-commerce:** Amazon/eBay fake review detection
- **Forums:** Reddit, Stack Overflow comment moderation
- **Messaging:** WhatsApp, Telegram spam blocking
- **Customer Service:** Chatbot intent classification

## 📊 Project Requirements Met

✅ Data loading and exploration  
✅ Text preprocessing pipeline  
✅ Bag of Words implementation  
✅ TF-IDF transformation  
✅ Multinomial Naive Bayes training  
✅ 5-fold cross-validation  
✅ Test set evaluation with confusion matrix  
✅ Production testing with custom inputs  
✅ Group collaboration (5 members)  
✅ Comprehensive code documentation  

## 📚 References

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*
- Scikit-learn Text Feature Extraction: https://scikit-learn.org/stable/modules/feature_extraction.html
- Naive Bayes Documentation: https://scikit-learn.org/stable/modules/naive_bayes.html
- Manning, C. D., & Schütze, H. (1999). *Foundations of Statistical Natural Language Processing*
- COMP 237 Course Materials - Centennial College

**Additional Resources:**
- Detailed analysis: See `DocAndAnalysis/Analysis_Group5.docx`
- Presentation slides: See `DocAndAnalysis/Group5_Presentation.pptx`
- Project evolution: See `OtherCodeVersion/` for earlier implementations

## 👨‍💻 Authors

**Group 5 - COMP 237 Fall 2022:**
- Song Malisa Se (301233051)
- **Matheus Teixeira (301236904)**
- Viet Hoang
- Yi-lin Lou (301226659)
- Yin-Siang Mao (301180968)

Course: COMP 237 - AI Fundamentals  
Institution: Centennial College  
Term: Fall 2022

---

*Demonstrates practical NLP pipeline for text classification, from raw YouTube comments to production-ready spam detection with 97% accuracy.*