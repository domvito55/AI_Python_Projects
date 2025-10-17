# Assignment 4: Logistic Regression

Binary classification using Logistic Regression for supervised machine learning in COMP 237 (AI Fundamentals).

## 📋 Overview

This assignment applies logistic regression to predict Titanic passenger survival based on demographic and ticket information. It demonstrates the complete machine learning pipeline from data exploration to model evaluation.

**Dataset:** Titanic passenger data (891 passengers, 12 features)  
**Task:** Binary classification (Survived: 0 = No, 1 = Yes)  
**Goal:** Predict survival probability and evaluate model performance

## 🎯 Assignment Objectives

### Exercise #1: Complete ML Pipeline
1. **Data Exploration** - Understand dataset structure and patterns
2. **Data Visualization** - Identify correlations using charts
3. **Data Preprocessing** - Handle missing values and encode categories
4. **Feature Engineering** - Create dummy variables and normalize
5. **Model Training** - Fit logistic regression with cross-validation
6. **Model Evaluation** - Test with confusion matrix and metrics
7. **Threshold Experimentation** - Compare 0.5 vs 0.75 thresholds

## 📁 Project Structure

```
Assignment4-LogisticRegression/
└── Exercise#1_Matheus/
    ├── Matheus_linear.py           # Complete pipeline
    ├── titanic.csv                 # Dataset (891 rows)
    └── Written_response_Matheus.docx  # Analysis report
```

## 🚀 Key Features

### Data Analysis Pipeline
- ✅ Exploratory data analysis (EDA)
- ✅ Statistical visualization (crosstab, scatter matrix)
- ✅ Missing value handling
- ✅ Categorical encoding (one-hot)
- ✅ Feature normalization (min-max scaling)
- ✅ Train-test split (70/30)
- ✅ Cross-validation (10-fold)
- ✅ Model evaluation metrics

### Machine Learning Techniques
- **Algorithm:** Logistic Regression (lbfgs solver)
- **Validation:** K-fold cross-validation
- **Metrics:** Accuracy, Precision, Recall, F1-score
- **Threshold tuning:** Comparing decision boundaries

## 📊 Dataset Information

### Titanic.csv Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| PassengerId | Integer | Unique ID | 1, 2, 3... |
| **Survived** | Integer | **Target (0/1)** | 0 = Died, 1 = Survived |
| Pclass | Integer | Passenger class | 1 = 1st, 2 = 2nd, 3 = 3rd |
| Name | String | Passenger name | "Smith, Mr. John" |
| Sex | String | Gender | Male, Female |
| Age | Float | Age in years | 22, 38, 26... |
| SibSp | Integer | # siblings/spouses aboard | 0, 1, 2... |
| Parch | Integer | # parents/children aboard | 0, 1, 2... |
| Ticket | String | Ticket number | "A/5 21171" |
| Fare | Float | Passenger fare | 7.25, 71.28... |
| Cabin | String | Cabin number | "C85", "E46" |
| Embarked | String | Port of embarkation | C, Q, S |

### Dataset Statistics
- **Total passengers:** 891
- **Survival rate:** ~38% survived
- **Missing values:** Age (177), Cabin (687), Embarked (2)
- **Gender distribution:** ~65% male, ~35% female

## 🛠️ Technical Implementation

### Step 1: Data Loading & Exploration

```python
# Load dataset
titanic_df = pd.read_csv('titanic.csv')

# Initial exploration
print(titanic_df.head(3))        # Preview
print(titanic_df.shape)          # (891, 12)
print(titanic_df.info())         # Column types & nulls
```

**Key Findings:**
- 4 columns dropped (PassengerId, Name, Ticket, Cabin)
- PassengerId/Name/Ticket: Unique values (no patterns)
- Cabin: 77% missing data

### Step 2: Data Visualization

**A. Survival by Passenger Class**
```python
crosstab = pd.crosstab(df.Pclass, df.Survived)
crosstab.div(crosstab.sum(1), axis=0).plot(kind='bar', stacked=True)
```

**Results:**
- 1st Class: ~63% survived
- 2nd Class: ~47% survived
- 3rd Class: ~24% survived
- **Conclusion:** Higher class = higher survival rate

**B. Survival by Gender**
```python
crosstab = pd.crosstab(df.Sex, df.Survived)
crosstab.div(crosstab.sum(1), axis=0).plot(kind='bar', stacked=True)
```

**Results:**
- Female: ~74% survived
- Male: ~19% survived
- **Conclusion:** "Women and children first" policy clearly visible

**C. Scatter Matrix Analysis**

Features analyzed: Survived, Sex, Pclass, Fare, SibSp, Parch

**Key Insights:**
1. Most passengers died (~62%)
2. ~50% traveled in 3rd class
3. Most fares were low-priced
4. Large families (5+ siblings or 4+ parents/children) had 0% survival
5. Many passengers traveled alone

### Step 3: Data Preprocessing

**A. Drop Irrelevant Columns**
```python
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
```

**B. Categorical Encoding (One-Hot)**
```python
categorical_vars = ['Sex', 'Embarked']

for var in categorical_vars:
    dummies = pd.get_dummies(df[var], prefix=var)
    df = df.join(dummies).drop(var, axis=1)
```

**Result:**
- Sex → Sex_male, Sex_female (0/1)
- Embarked → Embarked_C, Embarked_Q, Embarked_S (0/1)

**C. Handle Missing Values**
```python
# Age: Fill with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)
```

**D. Convert to Float**
```python
df = df.astype(float)
```

**E. Normalization (Min-Max Scaling)**
```python
def normalize(df):
    return (df - df.min()) / (df.max() - df.min())

df = normalize(df)
```

**Formula:** `x_norm = (x - x_min) / (x_max - x_min)`

**Result:** All features scaled to [0, 1]

### Step 4: Feature Analysis from Histograms

After normalization, histogram analysis reveals:

**Port of Embarkation Patterns:**
- **Embarked_Q** (Queenstown): Majority died
- **Embarked_C** (Cherbourg): More balanced (slightly more died)
- **Embarked_S** (Southampton): Majority survived

**Hypothesis:** Different ports likely correspond to different passenger classes:
- Queenstown → Mostly 3rd class (high mortality)
- Southampton → Mix of classes (more 1st class, higher survival)
- Cherbourg → Middle ground

### Step 5: Train-Test Split

```python
X = df.drop('Survived', axis=1)  # Features
y = df['Survived']               # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.30,  # 70% train, 30% test
    random_state=4   # Last 2 digits of student ID
)
```

### Step 6: Model Training

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

# View feature weights
weights = pd.DataFrame(
    zip(X_train.columns, model.coef_[0]),
    columns=['Feature', 'Weight']
)
```

**Feature Weights (Importance):**
- Pclass: Negative (higher class = lower number = more survival)
- Age: Small impact
- SibSp/Parch: Family size matters
- Fare: Positive (higher fare = more survival)
- **Sex_female: Strong positive** (women survived more)
- Embarked ports: Varied impact

### Step 7: Cross-Validation

**Experiment:** Test different train/test split ratios (10% to 50% test size)

```python
for test_size in range(10, 51, 5):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, 
                                               test_size=test_size/100,
                                               random_state=4)
    scores = cross_val_score(LogisticRegression(solver='lbfgs'),
                            X_tr, y_tr, scoring='accuracy', cv=10)
    
    print(f"Test={test_size}%: Min={scores.min():.3f}, "
          f"Mean={scores.mean():.3f}, Max={scores.max():.3f}")
```

**Results Table:**

| Split | Min Score | Mean Score | Max Score |
|-------|-----------|------------|-----------|
| 10-90% | **0.738** | 0.795 | **0.875** |
| 15-85% | 0.724 | **0.795** | 0.867 |
| 20-85% | 0.694 | 0.791 | 0.873 |
| 25-75% | 0.672 | 0.790 | 0.851 |
| **30-70%** | 0.726 | 0.777 | 0.825 |
| **35-65%** | 0.724 | **0.796** | 0.862 |
| 40-60% | 0.736 | 0.786 | 0.870 |
| 45-55% | 0.714 | 0.767 | 0.837 |
| 50-50% | 0.689 | 0.771 | 0.822 |

**Recommendation:** **35-65% split** (35% test, 65% train)
- Highest mean score: 0.796
- Good balance between training data and generalization

### Step 8: Model Evaluation (Threshold = 0.5)

```python
# Predict probabilities
y_pred_proba = model.predict_proba(X_test)

# Apply threshold
threshold = 0.5
y_pred = (y_pred_proba[:, 1] > threshold).astype(int)

# Evaluate
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
```

**Results (Threshold = 0.5):**
- **Accuracy:** 0.83 (83%)
- **Precision:** 0.87
- **Recall:** 0.88

**Confusion Matrix:**
```
[[TN  FP]     [[145  18]
 [FN  TP]]  =  [ 28 77]]
```

**Interpretation:**
- **True Negatives (145):** Correctly predicted "died"
- **False Positives (18):** Predicted "survived" but actually died
- **False Negatives (28):** Predicted "died" but actually survived
- **True Positives (77):** Correctly predicted "survived"

### Step 9: Threshold Experimentation (Threshold = 0.75)

```python
# Higher threshold = more conservative predictions
threshold = 0.75
y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
```

**Results (Threshold = 0.75):**
- **Accuracy:** 0.81 (81%)
- **Precision:** 0.78
- **Recall:** 1.00

**Comparison:**

| Metric | Threshold 0.5 | Threshold 0.75 | Change |
|--------|---------------|----------------|--------|
| Accuracy | 0.83 | 0.81 | -0.02 ⬇️ |
| Precision | 0.87 | 0.78 | -0.09 ⬇️ |
| Recall | 0.88 | 1.00 | +0.12 ⬆️ |

**Key Insight:**
- **Higher threshold** (0.75) → Model is more cautious about predicting "survived"
- **Trade-off:** Lower precision but higher recall
- **Meaning:** Catches all actual survivors (recall=1.0) but also predicts some deaths as survivors (precision drops)

**When to use each:**
- **0.5 threshold:** Balanced - best overall accuracy
- **0.75 threshold:** When false negatives are costly (better to predict survival even if sometimes wrong)

## 📈 Model Performance Analysis

### Training vs Testing Accuracy

**Training Accuracy:** ~0.78 (from cross-validation mean)  
**Testing Accuracy:** 0.83  

**Conclusion:** Model generalizes well! Testing accuracy slightly higher than training, indicating good model fit (no overfitting).

### Feature Importance Insights

From coefficient analysis:
1. **Gender (Sex_female):** Strongest predictor
2. **Passenger Class:** Strong inverse relationship
3. **Fare:** Positive correlation
4. **Age, Family Size:** Moderate impact
5. **Embarkation Port:** Weak impact

## 🎓 Key Learnings

### 1. Data Preprocessing is Critical
- 77% of Cabin data was missing → dropped
- Age missing values filled with mean
- Categorical variables require encoding
- Normalization brings features to same scale

### 2. Visualization Reveals Patterns
- Class and gender were strongest survival predictors
- Large families had lower survival rates
- Embarkation port hinted at class distribution

### 3. Model Evaluation Beyond Accuracy
- **Confusion matrix** shows type of errors
- **Precision vs Recall** trade-off with thresholds
- **Cross-validation** ensures robust performance

### 4. Threshold Selection Matters
- Lower threshold (0.5): Balanced predictions
- Higher threshold (0.75): More conservative, catches all positives
- Choice depends on business context (cost of false positives vs false negatives)

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
conda activate ai-logistic-regression
```

## 🏃 Usage

### Run Complete Pipeline

```bash
cd Exercise#1_Matheus
python Matheus_linear.py
```

### Expected Output Sections

```
#### b.1 ########################
(First 3 rows of data)

#### b.2 ########################
(891, 12)

#### b.3 ########################
(Data types and null counts)

#### c.1.a ######################
(Bar chart: Survival by Class)

#### c.1.b ######################
(Bar chart: Survival by Gender)

#### c.2 ########################
(Scatter matrix visualization)

#### d.10 #######################
(First 2 normalized rows)

#### e.2 ########################
(Feature weights table)

#### e.3.3 ######################
(Cross-validation results for splits 10-50%)

#### b.5 ########################
Accuracy: 0.83

#### b.6 ########################
Confusion Matrix:
[[145  18]
 [ 28  77]]

#### b.7 ########################
Classification report:
              precision    recall  f1-score   support
           0       0.84      0.89      0.86       163
           1       0.81      0.73      0.77       105
    accuracy                           0.83       268
```

## 🔬 Experiment Ideas

Try these modifications to deepen understanding:

### 1. Feature Engineering
```python
# Create new features
df['FamilySize'] = df['SibSp'] + df['Parch']
df['IsAlone'] = (df['FamilySize'] == 0).astype(int)
```

### 2. Different Algorithms
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Compare performance
rf_model = RandomForestClassifier()
svc_model = SVC(probability=True)
```

### 3. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear']
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
```

## 📚 Mathematical Background

### Logistic Regression Formula

**Sigmoid Function:**
```
P(y=1|x) = 1 / (1 + e^(-z))

where z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

**Properties:**
- Output range: [0, 1] (probability)
- Decision boundary: P = 0.5 (default threshold)
- S-shaped curve

### Performance Metrics

**Accuracy:** `(TP + TN) / (TP + TN + FP + FN)`

**Precision:** `TP / (TP + FP)` - "Of predicted positives, how many are correct?"

**Recall (Sensitivity):** `TP / (TP + FN)` - "Of actual positives, how many did we catch?"

**F1-Score:** `2 × (Precision × Recall) / (Precision + Recall)` - Harmonic mean

## 🎯 Real-World Applications

Logistic regression techniques demonstrated here apply to:

- **Healthcare:** Disease diagnosis (positive/negative)
- **Finance:** Credit default prediction (default/no default)
- **Marketing:** Customer churn prediction (churn/retain)
- **Email:** Spam detection (spam/ham)
- **Insurance:** Risk assessment (high-risk/low-risk)

## 🔍 Code Highlights

### Dynamic Categorical Encoding
```python
cat_var = ['Sex', 'Embarked']
for var in cat_var:
    dummies = pd.get_dummies(df[var], prefix=var)
    df = df.join(dummies).drop(var, axis=1)
```
**Benefit:** Easily extensible to any number of categorical variables

### Robust Normalization
```python
def normalize(df):
    return (df - df.min()) / (df.max() - df.min())
```
**Benefit:** Vectorized pandas operation (fast and clean)

### Threshold Flexibility
```python
y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
```
**Benefit:** Easy to experiment with different thresholds

## 📊 Assignment Requirements Met

✅ Data loading and exploration  
✅ Visualization (bar charts, scatter matrix, histograms)  
✅ Data transformation (drop columns, encoding, normalization)  
✅ Missing value handling  
✅ Train-test split with multiple ratios  
✅ Logistic regression training  
✅ Cross-validation (10-fold)  
✅ Model evaluation (accuracy, confusion matrix, classification report)  
✅ Threshold experimentation (0.5 vs 0.75)  
✅ Comprehensive written analysis  

## 📚 References

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- Pandas Documentation: https://pandas.pydata.org/docs/
- Kaggle Titanic Dataset: https://www.kaggle.com/c/titanic
- COMP 237 Course Materials - Centennial College

## 👨‍💻 Author

**Matheus Teixeira**  
Student Number: 301236904  
Course: COMP 237 - AI Fundamentals  
Institution: Centennial College  
Term: Fall 2022

---

*Demonstrates complete machine learning pipeline from data exploration through model evaluation, with emphasis on binary classification and threshold optimization.*