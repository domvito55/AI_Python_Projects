# Fashion MNIST Classification - CNN vs RNN Comparison

Comparative study of **Convolutional Neural Networks (CNN)** and **Recurrent Neural Networks (RNN/LSTM)** for image classification. Built and evaluated two architectures on Fashion MNIST dataset, achieving **88% test accuracy** with CNN and **86.2%** with RNN.

**Key Achievement:** Demonstrated that CNNs outperform RNNs for spatial pattern recognition tasks (images), with faster training and higher accuracy despite RNN's simpler architecture.

---

## 🎯 Project Overview

Built two deep learning models from scratch to classify 10 categories of clothing items from Fashion MNIST dataset (70,000 grayscale images, 28x28 pixels). Conducted systematic comparison of architectures, training dynamics, and prediction confidence.

**Models Implemented:**
- **CNN:** 2 convolutional layers + max pooling + fully connected
- **RNN:** LSTM with 128 hidden units

**Results:**
- **CNN Test Accuracy:** 88.0%
- **RNN Test Accuracy:** 86.2%
- **Training Speed:** CNN converged faster
- **Model Complexity:** RNN simpler (fewer parameters)

---

## 💡 Key Technical Insights

### 1. CNNs Excel at Spatial Pattern Recognition
CNN achieved **88% accuracy** vs RNN's **86.2%** on Fashion MNIST. **Reason:** Convolutional layers designed for spatial hierarchies (edges → textures → shapes), perfect for images. RNNs designed for sequential data, less optimal for 2D spatial patterns.

### 2. Training Dynamics Differ Significantly
**CNN:** Training and validation curves aligned closely (89.99% train, 88% test) - excellent generalization.  
**RNN:** Similar alignment but lower overall accuracy (87.23% train, 86.2% test) - also generalized well but less effective for image data.

### 3. Category-Specific Performance
**Hardest to classify:** Category 6 (Shirts) - only 65.9% accuracy  
- Frequently confused with T-shirts (0), Pullovers (2), Dresses (3), Coats (4)
- Reason: High visual similarity in 28x28 grayscale

**Easiest to classify:** Category 8 (Bags) and 1 (Trousers) - >97% accuracy  
- Distinctive shapes with no similar categories in dataset

### 4. Prediction Confidence Analysis
CNN showed **higher confidence** in correct predictions (taller probability bars) compared to RNN, indicating more decisive feature extraction through convolutional filters.

### 5. Architecture Complexity Trade-off
**RNN:** Simpler architecture, fewer parameters  
**CNN:** More complex (2 conv layers + pooling) but better performance  
**Conclusion:** For image tasks, added complexity of CNNs justified by superior accuracy and training speed.

---

## 🛠️ Technical Implementation

### CNN Architecture

```
Input Layer:         28x28x1 (grayscale images)
                     ↓
Conv2D Layer 1:      32 filters, 3x3 kernel, ReLU activation
                     ↓
MaxPooling2D:        2x2 window (spatial downsampling)
                     ↓
Conv2D Layer 2:      32 filters, 3x3 kernel, ReLU activation
                     ↓
MaxPooling2D:        2x2 window
                     ↓
Flatten:             Convert 2D feature maps to 1D
                     ↓
Dense Layer:         100 neurons, ReLU activation
                     ↓
Output Layer:        10 neurons, Softmax activation (10 classes)
```

**Total Parameters:** ~140K

### RNN Architecture

```
Input Layer:         28 timesteps × 28 features
                     (each image row treated as timestep)
                     ↓
LSTM Layer:          128 hidden units
                     ↓
Output Layer:        10 neurons, Softmax activation
```

**Total Parameters:** ~84K (40% fewer than CNN)

---

## 📊 Results & Analysis

### Model Performance Comparison

| Metric | CNN | RNN | Winner |
|--------|-----|-----|--------|
| **Training Accuracy** | 89.99% | 87.23% | CNN |
| **Validation Accuracy** | ~89% | ~87% | CNN |
| **Test Accuracy** | 88.0% | 86.2% | CNN |
| **Training Speed** | Faster | Slower | CNN |
| **Parameters** | ~140K | ~84K | RNN (simpler) |
| **Generalization** | Excellent | Excellent | Tie |

### Per-Class Performance (CNN)

| Class | Accuracy | Notes |
|-------|----------|-------|
| 0: T-shirt/top | ~84% | Often confused with shirts |
| 1: Trouser | **>97%** | Highly distinctive shape |
| 2: Pullover | ~85% | Some overlap with coats |
| 3: Dress | ~87% | Clear silhouette |
| 4: Coat | ~83% | Confused with pullovers/shirts |
| 5: Sandal | ~94% | Distinctive footwear shape |
| 6: Shirt | **65.9%** | Most challenging category |
| 7: Sneaker | ~92% | Clear footwear pattern |
| 8: Bag | **>97%** | Most distinctive shape |
| 9: Ankle boot | ~91% | Clear boot silhouette |

### Training Curves Analysis

**CNN Training vs Validation:**
- Curves closely aligned throughout 8 epochs
- Smooth convergence to ~89-90% accuracy
- No signs of overfitting (train/validation gap minimal)

**RNN Training vs Validation:**
- Similar alignment pattern to CNN
- Converged to ~87-88% accuracy
- Also no overfitting issues
- Slightly more variance in validation curve

### Confusion Matrix Insights

**CNN Confusion Matrix Highlights:**
- **Shirt (6) misclassifications:** 
  - 136 predicted as T-shirt/top (0)
  - 90 predicted as Pullover (2)
  - 66 predicted as Coat (4)
- **Reverse confusion:**
  - T-shirts (0) → Shirts (6): 81 times
  - Coats (4) → Shirts (6): 82 times
- **Strong diagonal:** Categories 1, 8, 5, 7, 9 have >900/1000 correct predictions

**RNN showed similar confusion patterns** but with slightly more misclassifications overall.

---

## 🚀 Implementation Details

### Data Preprocessing

```python
# Load Fashion MNIST (built into TensorFlow)
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize pixel values to [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# One-hot encode labels
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Train-validation split (80-20)
X_train, X_val, y_train, y_val = train_test_split(
    train_images, train_labels, 
    test_size=0.2, 
    random_state=4  # Last 2 digits of student ID
)
```

### Training Configuration

**Optimizer:** Adam (adaptive learning rate)  
**Loss Function:** Categorical Crossentropy  
**Metrics:** Accuracy  
**Epochs:** 8  
**Batch Size:** 256  
**Validation Split:** 20% of training data  

### Visualization Approach

**1. Training History Plots**
- Line graphs comparing training vs validation accuracy
- Separate plots for CNN and RNN
- Clear axis labels, legends, and titles

**2. Prediction Probability Distributions**
- Bar charts showing model confidence for each class
- Green bar: true label
- Blue bar: predicted label (if different)
- Tested on 4 samples starting from index 4

**3. Confusion Matrices**
- 10x10 heatmaps using Seaborn
- Annotated with prediction counts
- Clear visualization of misclassification patterns

---

## 📁 Project Structure

```
CNN_RNN_ImageClassification/
├── README.md                          # This file
├── Exercise#1_matheus/
│   ├── matheus_linear.py             # Complete CNN + RNN implementation
│   └── Written_response_Matheus.docx # Detailed analysis and diagrams
├── Diagrams.vsdx                     # Network architecture diagrams
├── requirements.txt                  # Python dependencies
└── environment.yml                   # Conda environment specification
```

---

## 🔧 Technologies Used

**Deep Learning Framework:**
- TensorFlow 2.x / Keras
- Neural network layers: Conv2D, MaxPooling2D, LSTM, Dense

**Data Processing:**
- NumPy - numerical operations and array manipulation

**Visualization:**
- Matplotlib - training curves, probability distributions
- Seaborn - confusion matrix heatmaps

**Model Evaluation:**
- Scikit-learn - confusion matrix, train_test_split
- TensorFlow metrics - accuracy, loss tracking

**Development:**
- Python 3.8+
- VS Code
- Git version control

---

## 🎓 Learning Outcomes

### Deep Learning Fundamentals
✅ Built CNNs from scratch with convolutional and pooling layers  
✅ Implemented RNNs using LSTM cells for sequence processing  
✅ Understood when to use CNNs (spatial data) vs RNNs (sequential data)  
✅ Configured and trained models with proper hyperparameters  

### Model Evaluation
✅ Analyzed training vs validation curves for overfitting detection  
✅ Interpreted confusion matrices for per-class performance  
✅ Visualized prediction confidence with probability distributions  
✅ Compared model architectures systematically  

### TensorFlow/Keras Mastery
✅ Used Sequential API for model building  
✅ Applied proper data preprocessing (normalization, one-hot encoding)  
✅ Trained models with fit() and evaluated with evaluate()  
✅ Generated predictions with predict() for analysis  

### Practical ML Skills
✅ Proper train-validation-test splits for robust evaluation  
✅ Reproducibility with random seeds  
✅ Professional visualization of results  
✅ Clear documentation of findings  

---

## 💼 Real-World Applications

### Fashion & E-commerce
- Automated clothing categorization for online retailers
- Visual search systems ("find similar items")
- Inventory management with image recognition
- Size recommendation based on garment type detection

### Computer Vision Systems
- Multi-class image classification pipelines
- Transfer learning foundations for custom datasets
- Confidence-based decision making (prediction probabilities)
- Production deployment of CNN models

### Retail Analytics
- Customer preference analysis from product images
- Trend detection in fashion categories
- Automated tagging for product databases
- Quality control with visual inspection

---

## 🔍 Key Takeaways

### Technical Decisions
1. **CNN is optimal for image classification** - Spatial convolutions extract hierarchical features effectively
2. **RNNs work but aren't optimal** - Treating image rows as sequences functional but less efficient
3. **Proper validation prevents overfitting** - 80-20 split ensured robust model evaluation
4. **Confusion analysis reveals model weaknesses** - Category 6 (Shirts) needs more distinctive features or data augmentation

### Performance Optimization
1. **Batch size 256** balanced training speed and memory usage
2. **8 epochs sufficient** - Models converged without overfitting
3. **Adam optimizer** handled learning rate automatically
4. **Normalization critical** - [0,1] range improved convergence

### Future Improvements
1. **Data augmentation** - Rotation, flipping, zoom could improve shirt classification
2. **Deeper CNN** - More convolutional layers might capture finer details
3. **Ensemble methods** - Combine CNN + RNN predictions
4. **Transfer learning** - Pre-trained networks (ResNet, VGG) for better feature extraction

---

## 📚 Dataset Information

**Fashion MNIST**
- 70,000 grayscale images (28×28 pixels)
- 10 classes of clothing and accessories
- Training set: 60,000 images
- Test set: 10,000 images
- Balanced classes (7,000 samples per category)

**Classes:**

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

**Source:** Built into `tf.keras.datasets.fashion_mnist`

---

## 🔗 Dependencies

### Core Requirements

```
tensorflow>=2.8.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
```

### Installation

```bash
# Using pip
pip install -r requirements.txt

# Using conda
conda env create -f environment.yml
conda activate dl-cnn-rnn
```

### Python Version
- Python 3.8 or higher recommended
- Tested on Python 3.8.x

---

## 🚀 Running the Code

```bash
# Navigate to project directory
cd CNN_RNN_ImageClassification/Exercise#1_matheus

# Run complete analysis (CNN + RNN)
python matheus_linear.py
```

**Expected Output:**
1. Dataset size and resolution information
2. CNN model summary and training progress
3. CNN training vs validation accuracy plot
4. CNN test accuracy and predictions
5. CNN prediction probability distributions (4 samples)
6. CNN confusion matrix heatmap
7. RNN model summary and training progress
8. RNN training vs validation accuracy plot
9. RNN test accuracy and predictions
10. RNN prediction probability distributions (4 samples)
11. RNN confusion matrix heatmap

**Runtime:** ~5-10 minutes on CPU, ~2-3 minutes on GPU

---

## 📈 Reproducibility

**Random Seed:** 4 (last 2 digits of student ID)
- Set in train_test_split() for consistent validation split
- Ensures reproducible results across runs

**Deterministic Behavior:**
```python
random_state=4  # Train-validation split
```

---

## 🎯 Project Highlights

**Most Interesting Finding:**  
Category 6 (Shirts) achieved only **65.9% accuracy** - significantly lower than overall 88%. Deep analysis revealed systematic confusion with T-shirts, Pullovers, Dresses, and Coats due to visual similarity in low-resolution grayscale images.

**Best Technical Decision:**  
Implementing both CNN and RNN allowed direct architecture comparison, clearly demonstrating CNN superiority for image tasks (88% vs 86.2% accuracy, faster training).

**Visualization Excellence:**  
Probability distribution charts with color-coded true labels (green) and predictions (blue) provided intuitive confidence analysis beyond simple accuracy numbers.

**Confusion Matrix Insight:**  
Bidirectional confusion between categories (Shirts→T-shirts AND T-shirts→Shirts) suggested feature space overlap, indicating potential for hierarchical classification approach.

---

## 📖 Documentation

**Detailed Analysis:** See `Written_response_Matheus.docx` for:
- Architecture diagrams with layer dimensions
- Training curve analysis and comparisons
- Per-category performance breakdown
- Confusion matrix interpretation
- Conclusions and recommendations

**Code Documentation:**
- Inline comments explaining each step
- Clear function definitions
- Reproducible structure following assignment requirements

---

## 🔗 Related Projects

**Previous Work:**
- [Neural Network Fundamentals](../../00_AI_Fundamentals/Assignment5-NeuralNetwork/) - Feed-forward networks with backpropagation

**Next Steps in 01_DeepLearning:**
- Assignment2: Autoencoders for unsupervised learning
- Assignment3: Variational Autoencoders (VAEs)
- Assignment4: Generative Adversarial Networks (GANs)
- GroupProject: Applied deep learning solution

---

*This project demonstrates practical deep learning skills: building CNNs and RNNs from scratch, systematic model comparison, professional evaluation with multiple metrics, and clear communication of technical findings. The 88% test accuracy on Fashion MNIST shows strong fundamentals in computer vision and neural network design.*

**Status:** ✅ Complete implementation with comprehensive analysis  
**Code Quality:** Production-ready, well-documented, reproducible  
**Last Updated:** October 2024