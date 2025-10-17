# Assignment 5: Neural Networks

Implementation of feed-forward neural networks with backpropagation using Neurolab in COMP 237 (AI Fundamentals).

## 📋 Overview

This assignment explores neural network fundamentals through systematic experiments:
- **Target Function:** Learn the sum function: `output = x₁ + x₂` (or `x₁ + x₂ + x₃`)
- **Approach:** Compare different architectures, training sizes, and layer configurations
- **Goal:** Understand how network architecture and data size affect learning and generalization

## 🎯 Assignment Objectives

### Experimental Design
1. **Exercise 1:** Single layer (6 neurons), 10 samples - Baseline
2. **Exercise 2:** Two layers (5+3 neurons), 10 samples - Layer impact
3. **Exercise 3:** Single layer (6 neurons), 100 samples - Data size impact
4. **Exercise 4:** Two layers (5+3 neurons), 100 samples - Combined effects
5. **Exercise 5:** Three inputs - Dimensionality impact

### Learning Goals
- Understand feed-forward architecture
- Experience backpropagation training
- Observe overfitting vs generalization
- Analyze layer depth vs convergence speed
- Visualize neural network predictions in 3D

## 📁 Project Structure

```
Assignment5-NeuralNetwork/
├── Exercise_nn_Matheus.py          # Complete experiments (Exercises 1-5)
└── Written_response_Matheus.docx   # Detailed analysis with charts
```

## 🚀 Key Features

### Neural Network Experiments
- ✅ Single vs multi-layer architectures
- ✅ Small (10) vs large (100) training datasets
- ✅ 2D and 3D input spaces
- ✅ Gradient descent optimization
- ✅ 3D visualization of predictions
- ✅ Error convergence analysis
- ✅ Overfitting detection

### Technical Implementation
- **Library:** Neurolab (simple feed-forward networks)
- **Training:** Backpropagation with automatic convergence
- **Visualization:** 3D surface plots with matplotlib
- **Analysis:** Systematic comparison of architectures

## 🛠️ Technical Implementation

### Exercise 1: Baseline (Single Layer, 10 Samples)

**Setup:**
```python
np.random.seed(1)  # Reproducibility

# Generate 10 samples in [-0.6, 0.6]
set1 = np.random.uniform(-0.6, 0.6, 10).reshape(10, 1)
set2 = np.random.uniform(-0.6, 0.6, 10).reshape(10, 1)
input_data = np.concatenate((set1, set2), axis=1)

# Target: sum of inputs
output_data = set1 + set2

# Network: 2 inputs → 6 neurons → 1 output
nn = nl.net.newff([[-0.6, 0.6], [-0.6, 0.6]], [6, 1])

# Train
error_progress = nn.train(input_data, output_data, 
                          show=15, goal=0.00001)

# Test
result1 = nn.sim(np.array([0.1, 0.2]).reshape(1, 2))  # Expected: 0.3
result1b = nn.sim(np.array([0.6, 0.6]).reshape(1, 2)) # Expected: 1.2
```

**Results:**
- **Epochs:** 90
- **Average Error:** 6.74 × 10⁻⁶
- **Test Input (0.1, 0.2):** Prediction = 0.2174, Error = 0.0826
- **Test Input (0.6, 0.6):** Prediction = 1.0000, Error = 0.2000

**3D Visualization:**
- Gray plane: Original function (output = x₁ + x₂)
- Orange dots: Predictions for 10 training samples
- Blue line: 600 additional test predictions

**Key Observation:** With only 10 training samples, the network struggles to generalize to regions far from training data (like 0.6, 0.6).

### Exercise 2: Multi-Layer (Two Layers, 10 Samples)

**Setup:**
```python
# Network: 2 inputs → 5 neurons → 3 neurons → 1 output
nn = nl.net.newff([[-0.6, 0.6], [-0.6, 0.6]], [5, 3, 1])

# Training algorithm: Gradient Descent
nn.trainf = nl.train.train_gd

# Train
error_progress = nn.train(input_data, output_data, 
                          epochs=1000, show=100, goal=0.00001)
```

**Results:**
- **Epochs:** 1000 (maximum reached)
- **Average Error:** 1.45 × 10⁻²
- **Test Input (0.1, 0.2):** Prediction = 0.3295, Error = -0.0295
- **Test Input (0.6, 0.6):** Prediction = -0.2775, Error = 1.4775

**Comparison with Exercise 1:**

| Metric | Exercise 1 (1 Layer) | Exercise 2 (2 Layers) | Change |
|--------|----------------------|-----------------------|--------|
| Epochs | 90 | 1000 | +910 🐌 |
| Avg Error | 6.74 × 10⁻⁶ | 1.45 × 10⁻² | +2152x ⬆️ |
| Error (0.1, 0.2) | 0.0826 | -0.0295 | Better ✅ |
| Error (0.6, 0.6) | 0.2000 | 1.4775 | Worse ❌ |

**Key Finding:** Adding layers with insufficient training data causes **overfitting**:
- Improved predictions near training data
- Worse predictions far from training data
- Much slower convergence (1000 epochs not enough)

### Exercise 3: More Data (Single Layer, 100 Samples)

**Setup:**
```python
# Generate 100 samples instead of 10
set1 = np.random.uniform(-0.6, 0.6, 100).reshape(100, 1)
set2 = np.random.uniform(-0.6, 0.6, 100).reshape(100, 1)
# ... rest same as Exercise 1
```

**Results:**
- **Epochs:** 495
- **Average Error:** 9.24 × 10⁻³
- **Test Input (0.1, 0.2):** Prediction = 0.2980, Error = 0.0020
- **Test Input (0.6, 0.6):** Prediction = 0.9960, Error = 0.2040

**Comparison with Exercise 1:**

| Metric | Exercise 1 (10 samples) | Exercise 3 (100 samples) | Change |
|--------|-------------------------|--------------------------|--------|
| Epochs | 90 | 495 | +405 |
| Error (0.1, 0.2) | 0.0826 | 0.0020 | **98% improvement** ✅ |
| Error (0.6, 0.6) | 0.2000 | 0.2040 | Similar |

**Key Finding:** More training data **dramatically improves** predictions:
- Error near training region dropped 98%
- Data better distributed across input space
- Generalization improved

### Exercise 4: Two Layers with More Data (100 Samples) - Convergence Challenge

**Setup:**
```python
# Network: 2 inputs → 5 neurons → 3 neurons → 1 output
# Training data: 100 samples
nn = nl.net.newff([[-0.6, 0.6], [-0.6, 0.6]], [5, 3, 1])
nn.trainf = nl.train.train_gd
error_progress = nn.train(input_data, output_data, 
                          epochs=1000, show=100, goal=0.00001)
```

**Results:**
- **Epochs:** 1000 (maximum reached, did NOT converge)
- **Average Error:** 2.93 × 10⁻¹ (much worse than Exercise 3!)
- **Test Input (0.1, 0.2):** Prediction = 0.2743, Error = 0.0257
- **Test Input (0.6, 0.6):** Prediction = 0.6160, Error = 0.5840

**Comparison with Exercise 3:**

| Metric | Exercise 3 (1 Layer) ✅ | Exercise 4 (2 Layers) ❌ | Analysis |
|--------|------------------------|--------------------------|----------|
| Epochs | 495 (converged) | 1000 (not converged) | Deeper = much slower |
| Avg Error | 9.24 × 10⁻³ | 2.93 × 10⁻¹ | **32x worse!** |
| Error (0.1, 0.2) | **0.0020** | 0.0257 | 13x worse |
| Error (0.6, 0.6) | **0.2040** | 0.5840 | 3x worse |

**Critical Finding:** Adding layers **WITHOUT sufficient training time makes everything worse**:
- Error still rapidly decreasing at epoch 1000 (not converged)
- Network is too complex for the allocated training time
- **Re-run with 50,000 epochs** achieved error: 3.66 × 10⁻² (still not as good as Exercise 3!)
- **Conclusion:** For this simple problem (sum function), more layers add unnecessary complexity

**Error Progress Chart (Exercise 4):**
```
Error drops rapidly for first ~60 epochs
Then slows down significantly
By epoch 1000, still decreasing but very slowly
```

### Exercise 5: Three Inputs

#### Part 9: Single Layer, 10 Samples, 3 Inputs

**Setup:**
```python
# Generate 3 input features
set1 = np.random.uniform(-0.6, 0.6, 10).reshape(10, 1)
set2 = np.random.uniform(-0.6, 0.6, 10).reshape(10, 1)
set3 = np.random.uniform(-0.6, 0.6, 10).reshape(10, 1)
input_data = np.concatenate((set1, set2, set3), axis=1)

# Target: sum of all 3 inputs
output_data = set1 + set2 + set3

# Network: 3 inputs → 6 neurons → 1 output
nn = nl.net.newff([[-0.6, 0.6], [-0.6, 0.6], [-0.6, 0.6]], [6, 1])
```

**Results:**
- **Epochs:** 225
- **Average Error:** 1.71 × 10⁻²
- **Test Input (0.2, 0.1, 0.2):** Prediction = 1.0000, Error = -0.5000
- **Test Input (0.6, 0.6, 0.6):** Prediction = 1.0000, Error = 0.8000

**Observation:** Even with simple architecture, 3 features require more epochs (225 vs 90 for 2 features).

#### Part 11: Two Layers, 100 Samples, 3 Inputs

**Setup:**
```python
# Network: 3 inputs → 5 neurons → 3 neurons → 1 output
# Training data: 100 samples
nn = nl.net.newff([[-0.6, 0.6], [-0.6, 0.6], [-0.6, 0.6]], [5, 3, 1])
nn.trainf = nl.train.train_gd
error_progress = nn.train(input_data, output_data, 
                          epochs=1000, show=100, goal=0.00001)
```

**Results:**
- **Epochs:** 1000 (maximum)
- **Average Error:** 7.37 × 10⁻¹
- **Test Input (0.2, 0.1, 0.2):** Prediction = 0.5111, Error = -0.0111
- **Test Input (0.6, 0.6, 0.6):** Prediction = 0.9911, Error = 0.8089

**Comparison Exercise 5:**

| Metric | Part 9 (1 Layer, 10 samples) | Part 11 (2 Layers, 100 samples) |
|--------|------------------------------|----------------------------------|
| Epochs | 225 | 1000 |
| Error (0.2, 0.1, 0.2) | -0.5000 | -0.0111 (98% better) ✅ |
| Error (0.6, 0.6, 0.6) | 0.8000 | 0.8089 (similar) |

**Key Finding:** Same pattern as 2-input case:
- More data + more layers = need more epochs
- 1000 epochs insufficient (error still high)
- Network becoming too complex for available training time

## 📊 Comprehensive Analysis

### Overall Performance Ranking

**By Training Error (Average Error):**
1. 🥇 Exercise 1: 6.74 × 10⁻⁶ ← **MISLEADING! Overfitted**
2. 🥈 Exercise 3: 9.24 × 10⁻³ ← **Actually best model**
3. 🥉 Exercise 2: 1.45 × 10⁻²
4. Exercise 4: 2.93 × 10⁻¹ (worst - didn't converge)

**By Test Error (Generalization):**
1. 🥇 **Exercise 3: 0.0020** ← **TRUE WINNER!**
2. 🥈 Exercise 4: 0.0257
3. 🥉 Exercise 2: -0.0295 (but worse on far test)
4. Exercise 1: 0.0826 (overfitted)

### Why Exercise 1's Low Error is Deceptive

**Exercise 1 appears to have the best training error (6.74 × 10⁻⁶) but this is misleading:**

| Evidence of Overfitting | Exercise 1 | Exercise 3 |
|--------------------------|------------|------------|
| Training samples | 10 (very few) | 100 |
| Training error | 6.74 × 10⁻⁶ ✅ | 9.24 × 10⁻³ |
| **Test error** | **0.0826** ❌ | **0.0020** ✅ |
| Far test error (0.6, 0.6) | 0.2000 | 0.2040 |
| **Interpretation** | Memorized data | Learned pattern |

**Conclusion:** Low training error with few samples = **overfitting**, not good learning!

**Exercise 3 is the TRUE best model:**
- ✅ Good training error
- ✅ **Excellent test error** (13x better than Exercise 1)
- ✅ Good generalization
- ✅ Properly learned the sum pattern

### Impact of Training Data Size

| Configuration | 10 Samples | 100 Samples | Improvement |
|---------------|------------|-------------|-------------|
| 1 Layer, 2 Inputs | Training: 6.74×10⁻⁶ (deceptive) | Training: 9.24×10⁻³ | - |
| 1 Layer, 2 Inputs | **Test: 0.0826** | **Test: 0.0020** | **98% better!** ✅ |
| Impact | Overfitting | Good generalization | **Data size matters most** |

**Conclusion:** Training data size is **critical** for generalization. Don't trust training error alone!

### Impact of Network Depth

| Samples | 1 Layer | 2 Layers | Observation |
|---------|---------|----------|-------------|
| 10 samples | 90 epochs | 1000 epochs | Deeper = much slower |
| 10 samples | Test error: 0.0826 | Test error: -0.0295 | Slightly better near training data |
| 10 samples | - | Far test: 1.4775 | Much worse far from training! |
| 100 samples | 495 epochs, error 9.24×10⁻³ ✅ | 1000+ epochs, error 2.93×10⁻¹ ❌ | **Deeper is worse here!** |

**Conclusion:** More layers can **hurt** performance if:
1. Problem is simple (sum function doesn't need deep network)
2. Not enough training epochs
3. Network complexity exceeds problem complexity

### Impact of Input Dimensionality

| Inputs | Epochs (1 Layer) | Epochs (2 Layers) |
|--------|------------------|-------------------|
| 2 features | 90 | 1000 |
| 3 features | 225 | 1000+ |

**Conclusion:** More input features increase training time requirements.

## 🎓 Key Learnings

### 1. Training Error Can Be Deceiving! ⚠️
**Critical Insight:** Low training error ≠ Good model

**Exercise 1 Example:**
- Training error: 6.74 × 10⁻⁶ (looks amazing!)
- Test error: 0.0826 (actually poor)
- **Problem:** Network memorized 10 samples instead of learning pattern

**How to Detect:**
- Always evaluate on **separate test data**
- Compare training error vs test error
- Large gap = overfitting

**Solution:** More training data (Exercise 3 proved this!)

### 2. The Overfitting Problem
**Symptom:** Good performance on training data, poor on test data

**Causes:**
- Too few training samples (Exercise 1: only 10)
- Network too complex for data size (Exercise 2: 2 layers with 10 samples)
- Network memorizes instead of generalizing

**Solutions:**
- Increase training data size ✅ (Exercise 3: 100 samples)
- Simplify network architecture
- Use regularization techniques

### 3. More Complexity ≠ Better Results
**Surprising Finding:** For simple problems, simpler is better!

**Evidence:**
- Exercise 3 (1 layer, 100 samples): Test error = **0.0020** ✅
- Exercise 4 (2 layers, 100 samples): Test error = **0.0257** ❌
- **2 layers performed 13x worse even with same data!**

**Why:** 
- Sum function is simple (linear relationship)
- Extra layers add unnecessary complexity
- More parameters = harder to optimize
- More epochs needed to converge

**Lesson:** Match network complexity to problem complexity

### 4. Convergence Takes Time
**Symptom:** Error still decreasing when training stops

**Exercise 4 Example:**
- After 1000 epochs: Error = 2.93 × 10⁻¹ (still high)
- Error plot shows continued decrease (not flattened)
- Even at 50,000 epochs: Error = 3.66 × 10⁻² (still not as good as Exercise 3!)

**Implication:** Complex networks need exponentially more training time

### 5. Data Quality Over Architecture
**The Winner:** Exercise 3 (simple architecture + good data)

**Proof:**
| Factor | Exercise 1 | Exercise 3 | Winner |
|--------|------------|------------|--------|
| Architecture | 1 layer, 6 neurons | 1 layer, 6 neurons | Tie |
| Data | 10 samples | 100 samples | Exercise 3 |
| Test Error | 0.0826 | 0.0020 | **Exercise 3** ✅ |

**Conclusion:** **Good data + simple model > Simple data + complex model**

## 📈 Best Practices from Experiments

### Critical Lessons Learned

**1. Always Validate on Test Data**
- ❌ Don't trust training error alone (Exercise 1 trap!)
- ✅ Split data: 70-80% train, 20-30% test
- ✅ Compare training vs test error
- ⚠️ Large gap = overfitting

**2. Start Simple, Add Complexity Only If Needed**
- ✅ Begin with single layer (Exercise 3 won!)
- ✅ Use minimal neurons first
- ❌ Don't assume more layers = better (Exercise 4 was worse!)
- Only add complexity if:
  - Problem is genuinely complex
  - You have enough data
  - You have enough training time

**3. Data > Architecture**
- 🥇 Priority 1: Get more/better training data
- 🥈 Priority 2: Ensure data covers input space well
- 🥉 Priority 3: Only then consider network complexity
- **Example:** Exercise 3 (simple + 100 samples) beat Exercise 4 (complex + 100 samples)

**4. Monitor Convergence**
- ✅ Check if error still decreasing at end
- ✅ Plot error over epochs (like Exercise 4)
- ✅ If not converged, increase epochs
- ⚠️ But: if still bad after many epochs, try simpler architecture!

**5. Problem-Specific Design**
- Simple problems (like sum) → Simple networks
- Complex problems (image recognition) → Deep networks
- **Don't over-engineer for simple tasks**

### Starting a Neural Network Project

**Step 1: Understand Your Problem**
```
Is it linear? → Try linear regression first
Is it simple non-linear? → 1 hidden layer
Is it very complex? → Multiple layers
```

**Step 2: Collect Data**
```
Minimum: 10x number of parameters
Better: 100x number of parameters
Best: As much as possible with good coverage
```

**Step 3: Start Simple**
```
1. Single layer with few neurons
2. Train to convergence
3. Evaluate on test data
```

**Step 4: Iterate If Needed**
```
IF test error too high AND training error low:
    → Overfitting: Add more data
ELSE IF both errors high:
    → Underfitting: Add complexity (layers/neurons)
ELSE IF error decreasing at end:
    → Not converged: Add epochs
ELSE:
    → Good model! ✅
```

## ⚙️ Installation

### Prerequisites
- Python 3.6+
- Neurolab library (specific for neural network education)

### Install Dependencies
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda env create -f environment.yml
conda activate ai-neural-network
```

### Installing Neurolab
```bash
# Via pip
pip install neurolab

# Via conda
conda install -c conda-forge neurolab
```

## 🏃 Usage

### Run All Exercises

```bash
python Exercise_nn_Matheus.py
```

### Run Individual Exercise

Extract specific exercise code from the file and run separately:

```python
# Example: Run only Exercise 1
import numpy as np
import neurolab as nl

np.random.seed(1)
# ... (Exercise 1 code)
```

### Expected Output

Each exercise generates:
1. **Training progress:** Epoch numbers and error values
2. **Test predictions:** Results for test inputs
3. **3D plots:** Visualizations of network predictions

Example output:
```
Epoch: 15; Error: 0.00523;
Epoch: 30; Error: 0.00187;
...
Epoch: 90; Error: 0.0000067;
The goal of learning is reached

Test (0.1, 0.2): 0.2174
Test (0.6, 0.6): 1.0000
```

## 🔬 Experiment Ideas

### 1. Different Activation Functions
```python
# Try different transfer functions
nn = nl.net.newff(inputs, layers)
nn.layers[0].transf = nl.trans.TanSig()  # Hyperbolic tangent
nn.layers[1].transf = nl.trans.PureLin()  # Linear
```

### 2. Vary Network Width
```python
# Compare different neuron counts
architectures = [
    [3, 1],      # 3 neurons
    [10, 1],     # 10 neurons
    [20, 1],     # 20 neurons
]
```

### 3. Learning Rate Impact
```python
# Adjust learning rate
nn.trainf = nl.train.train_gd
nn.trainf.lr = 0.01  # Default is usually 0.01
```

### 4. Early Stopping
```python
# Monitor validation error
validation_split = 0.2
# Stop if validation error increases
```

## 📚 Mathematical Background

### Feed-Forward Neural Network

**Layer computation:**
```
z = W·x + b      (linear combination)
a = f(z)         (activation function)
```

Where:
- W = weight matrix
- x = input vector
- b = bias vector
- f = activation function (sigmoid, tanh, ReLU)

### Backpropagation Algorithm

**Error propagation:**
```
1. Forward pass: compute predictions
2. Calculate output error: δ = (y_pred - y_true)
3. Propagate backward: δₗ₋₁ = δₗ · Wₗᵀ · f'(zₗ₋₁)
4. Update weights: W = W - lr · δ · aᵀ
5. Repeat until convergence
```

### Gradient Descent

**Weight update rule:**
```
Wₙₑw = Wₒₗd - α · ∂E/∂W
```

Where:
- α = learning rate
- E = error function
- ∂E/∂W = gradient (direction of steepest ascent)

## 🎯 Real-World Applications

Neural network concepts from this assignment apply to:

- **Computer Vision:** Image classification, object detection
- **Natural Language Processing:** Sentiment analysis, translation
- **Time Series:** Stock prediction, weather forecasting
- **Control Systems:** Robotics, autonomous vehicles
- **Healthcare:** Disease diagnosis, drug discovery
- **Gaming:** AI opponents, procedural generation

## 📊 Assignment Requirements Met

✅ Exercise 1: Single layer baseline  
✅ Exercise 2: Multi-layer comparison  
✅ Exercise 3: Data size impact  
✅ Exercise 4: Error progress visualization  
✅ Exercise 5: Three-input extension  
✅ 3D visualization of all results  
✅ Comprehensive written analysis  
✅ Comparison tables and insights  

## 📚 References

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*
- Neurolab Documentation: https://pythonhosted.org/neurolab/
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*
- COMP 237 Course Materials - Centennial College

## 👨‍💻 Author

**Matheus Teixeira**  
Student Number: 301236904  
Course: COMP 237 - AI Fundamentals  
Institution: Centennial College  
Term: Fall 2022

---

*Demonstrates fundamental understanding of neural network architecture, training dynamics, and the critical importance of data size and network complexity balance.*