
# Assignment 3: Linear Regression

Implementation of Linear Regression for supervised machine learning in COMP 237 (AI Fundamentals).

## 📋 Overview

This assignment explores linear regression through two exercises:
1. **Exercise #1**: Understanding noise in linear data
2. **Exercise #2**: E-commerce spending prediction with feature engineering

## 🎯 Assignment Objectives

### Exercise #1: Sampling and Noise
- Generate synthetic data from linear function
- Add Gaussian noise to simulate real-world data
- Visualize the impact of noise on data points

### Exercise #2: E-commerce Predictions
- Load and explore real e-commerce dataset
- Data preprocessing and normalization
- Feature engineering (categorical to numerical)
- Build and compare linear regression models
- Evaluate model performance using R² score

## 📁 Project Structure

```
Assignment3-LinearRegression/
├── Exercise#1_Matheus/
│   ├── Matheus_linear.py       # Noise simulation
│   ├── CodeRun_d.png          # Output: clean data plot
│   ├── CodeRun_f.png          # Output: noisy data plot
│   ├── Plot_d.png             # Scatter plot (clean)
│   └── Plot_e.png             # Scatter plot (with noise)
│
└── Exercise#2_Matheus/
    ├── Matheus_linear.py       # E-commerce regression
    ├── Ecom Expense.csv        # Dataset (2362 rows)
    ├── histograms.png          # Feature distributions
    └── ScatterMatrix.png       # Feature correlations
```

## 🚀 Features

### Exercise #1: Noise Injection
- **Linear function**: y = 12x - 4
- **Uniform sampling**: 100 points from [-1, 1]
- **Gaussian noise**: Mean=0, StdDev=1
- **Visualization**: Before/after noise comparison

### Exercise #2: E-commerce Analysis
- **Dataset**: 2362 transactions with 9 features
- **Preprocessing Pipeline**:
  - Categorical encoding (one-hot encoding)
  - Feature normalization (min-max scaling)
  - Missing value handling
  - Feature selection and engineering

## 📊 Dataset Information

### Ecom Expense.csv Columns

| Column | Type | Description |
|--------|------|-------------|
| Transaction ID | String | Unique transaction identifier |
| Age | Integer | Customer age |
| Items | Integer | Number of items purchased |
| Monthly Income | Integer | Customer's monthly income |
| Transaction Time | Float | Time spent on website |
| Record | Integer | Previous purchase count |
| Gender | String | Customer gender |
| City Tier | String | City classification (Tier 1/2/3) |
| **Total Spend** | Float | **Target variable** |

## 🛠️ Technical Implementation

### Exercise #1: Key Concepts

**1. Uniform Distribution Sampling**
```python
np.random.seed(4)  # Reproducibility
x = np.random.uniform(-1, 1, 100)  # 100 samples in [-1, 1]
```

**2. Linear Function Generation**
```python
y = 12*x - 4  # Clean theoretical values
```

**3. Gaussian Noise Addition**
```python
noise = np.random.normal(size=100)  # μ=0, σ=1
y_noisy = 12*x - 4 + noise  # Real-world simulation
```

### Exercise #2: Data Pipeline

**1. Categorical Variable Encoding**
```python
# Convert Gender: 'Male'/'Female' → Gender_Male: 0/1, Gender_Female: 0/1
cat_list = pd.get_dummies(df[column], prefix=column)
df = df.join(cat_list).drop(column, axis=1)
```

**2. Normalization Function**
```python
def normalize(dataframe):
    return (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())
```
Formula: `x_norm = (x - x_min) / (x_max - x_min)`

Result: All features scaled to [0, 1]

**3. Train-Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, 
    test_size=0.35,  # 65% train, 35% test
    random_state=4
)
```

**4. Model Training**
```python
lm = linear_model.LinearRegression()
lm.fit(X_train, y_train)
```

## 📈 Model Comparison

### Model 1: Without 'Record' Feature

**Features Used:**
- Monthly Income
- Transaction Time
- Gender (Male/Female dummy variables)
- City Tier (Tier 1/2/3 dummy variables)

**Results:**
- Coefficients: `[3.27e-01, -5.60e-03, -1.65e+13, -1.65e+13, -1.04e+12, -1.04e+12, -1.04e+12]`
- **R² Score: 0.195** (19.5% variance explained)
- **Conclusion**: Poor model - negative infinity weights for categorical variables

### Model 2: With 'Record' Feature ⭐

**Features Used:**
- Monthly Income
- Transaction Time
- **Record** (number of previous purchases)
- Gender (Male/Female dummy variables)
- City Tier (Tier 1/2/3 dummy variables)

**Results:**
- Coefficients: `[0.326, 0.017, 0.596, -0.011, 0.011, 0.005, 0.004, -0.009]`
- **R² Score: 0.918** (91.8% variance explained)
- **Conclusion**: Excellent model - all weights have reasonable magnitudes

### Key Insight 💡

**Adding just ONE feature ('Record') improved the model from 20% to 92% accuracy!**

This demonstrates the critical importance of:
- Feature selection
- Domain knowledge (previous purchases predict spending)
- Exploratory data analysis

## 📊 Visualizations Generated

### Exercise #1
1. **Clean Data Plot**: Perfect linear relationship y = 12x - 4
2. **Noisy Data Plot**: Same relationship with Gaussian noise added

### Exercise #2
1. **Histograms**: Distribution of all 11 normalized features
   - 5 binary variables (Gender, City Tier dummies)
   - Total Spend follows normal distribution
   - Other 5 continuous variables approximately uniform

2. **Scatter Matrix**: Pairwise relationships
   - Strong linear correlation: Monthly Income ↔ Total Spend
   - Hint for feature importance

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
conda activate ai-linear-regression
```

## 🏃 Usage

### Exercise #1: Noise Simulation

```bash
cd Exercise#1_Matheus
python Matheus_linear.py
```

**Output:**
- Two matplotlib plots showing data before and after noise addition

### Exercise #2: E-commerce Prediction

```bash
cd Exercise#2_Matheus
python Matheus_linear.py
```

**Expected Output:**
```
#### b.i ########################
   Transaction ID  Age  Items  Monthly Income  ...
0  T1234           32   5      45000           ...

#### b.ii #######################
(2362, 9)

#### b.iii ######################
['Transaction ID' 'Age' 'Items' 'Monthly Income' 'Transaction Time'
 'Record' 'Gender' 'City Tier' 'Total Spend']

#### b.iv #######################
Transaction ID     object
Age                 int64
...

#### b.v ########################
Transaction ID    0
Age               0
...

#### c.vii ######################
        Age     Items  Monthly Income  ...
0  0.123456  0.234567        0.456789  ...

#### d.vi #######################
Weights: [0.32566986 0.01741505 0.59634381 ...]

#### d.vii ######################
R^2:  0.9175710979534945

#### d.viii #####################
Weights: [0.32566986 0.01741505 0.59634381 ...]
R^2:  0.9175710979534945
```

## 🧪 Experiment: Feature Impact

Try removing/adding features to see impact on R²:

```python
# Test different feature combinations
features_v1 = ['Monthly Income', 'Transaction Time']
features_v2 = ['Monthly Income', 'Record']  
features_v3 = ['Age', 'Items', 'Record']

# Which gives best R²?
```

## 📝 Key Learnings

### From Exercise #1
1. **Real-world data has noise** - Never perfectly fits theoretical models
2. **Gaussian noise is common** - Many natural processes follow normal distribution
3. **Visualization is crucial** - See the impact of noise visually

### From Exercise #2
1. **Feature engineering matters** - One feature made 72% difference!
2. **Categorical encoding** - Use one-hot encoding for non-ordinal categories
3. **Normalization importance** - Brings all features to same scale
4. **R² interpretation**:
   - R² = 0.20 → Very poor model
   - R² = 0.50 → Moderate model
   - R² = 0.92 → Excellent model

### Model Evaluation Insights

**Why Model 1 Failed:**
- Missing critical feature ('Record')
- Categorical variables caused numerical instability
- Weights approaching -∞ indicate poor conditioning

**Why Model 2 Succeeded:**
- 'Record' (purchase history) is highly predictive of spending
- All features contribute meaningfully
- Weights have interpretable magnitudes

## 🔍 Code Highlights

### Robust Data Exploration
```python
# Systematic exploration pipeline
print(df.head(3))          # Preview data
print(df.shape)            # Dimensions
print(df.columns.values)   # Feature names
print(df.dtypes)           # Data types
print(df.isnull().sum())   # Missing values
```

### Dynamic Categorical Encoding
```python
# Automatically encode all categorical columns
cat_var = [name for name, dtype in df.dtypes.items() 
           if dtype.type is np.object_]

for var in cat_var:
    dummies = pd.get_dummies(df[var], prefix=var)
    df = df.join(dummies).drop(var, axis=1)
```

### Vectorized Normalization
```python
# Efficient pandas vectorization (no loops!)
def normalize(df):
    return (df - df.min()) / (df.max() - df.min())
```

## 📚 Mathematical Background

### Linear Regression Formula

**Model**: `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε`

Where:
- y = Target (Total Spend)
- β₀ = Intercept
- βᵢ = Coefficients (weights)
- xᵢ = Features (Monthly Income, Record, etc.)
- ε = Error term (noise)

### R² Score (Coefficient of Determination)

**Formula**: `R² = 1 - (SS_res / SS_tot)`

Where:
- SS_res = Residual sum of squares = Σ(yᵢ - ŷᵢ)²
- SS_tot = Total sum of squares = Σ(yᵢ - ȳ)²

**Interpretation:**
- R² = 1.0 → Perfect prediction
- R² = 0.0 → Model no better than mean
- R² < 0.0 → Model worse than predicting mean

## 📊 Statistical Analysis

### Exercise #2 Findings

**Correlation Analysis** (from scatter matrix):
- **Strong positive**: Monthly Income → Total Spend
- **Moderate positive**: Record → Total Spend  
- **Weak/None**: Age, Items, Transaction Time

**Feature Importance** (from Model 2 coefficients):
1. **Record** (0.596) - Most important!
2. **Monthly Income** (0.326) - Second most important
3. **Transaction Time** (0.017) - Minor contribution
4. Gender/City Tier (~0.005) - Minimal impact

## 🎓 Assignment Requirements Met

### Exercise #1 ✅
- [x] Uniform sampling with seed
- [x] Linear function generation
- [x] Gaussian noise addition
- [x] Scatter plots before/after
- [x] Written analysis of noise impact

### Exercise #2 ✅
- [x] Data loading and exploration
- [x] Categorical variable encoding
- [x] Feature normalization
- [x] Missing value check
- [x] Histogram visualization
- [x] Scatter matrix analysis
- [x] Train-test split (65/35)
- [x] Two model comparisons
- [x] R² evaluation
- [x] Comprehensive written analysis

## 🎯 Real-World Applications

This assignment demonstrates techniques used in:
- **E-commerce**: Customer spending prediction
- **Marketing**: Customer lifetime value (CLV) modeling
- **Finance**: Credit risk assessment
- **Sales**: Revenue forecasting
- **Healthcare**: Treatment cost estimation

## 📚 References

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/linear_model.html
- Pandas User Guide: https://pandas.pydata.org/docs/user_guide/
- COMP 237 Course Materials - Centennial College

## 👨‍💻 Author

**Matheus Teixeira**  
Student Number: 301236904  
Course: COMP 237 - AI Fundamentals  
Institution: Centennial College  
Term: Fall 2022

---

*Demonstrates practical application of linear regression, feature engineering, and model evaluation in supervised machine learning.*