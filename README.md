# 🤖 AI & Machine Learning Projects

Academic projects from **Advanced Diploma in Software Engineering - Artificial Intelligence** at Centennial College (2022-2024)

**Graduate:** Matheus Teixeira | **GPA:** 4.45/4.5 (High Honors) | **Ex-NASA Intern** 🚀

---

## 📚 Project Categories

### [00_AI_Fundamentals](./00_AI_Fundamentals)
Core AI algorithms and classical machine learning implementations

**Topics covered:**
- 🎯 **Agents** - Simple reflex agents, environment simulation, OOP design
- 🔍 **Search Algorithms** - BFS, DFS, UCS, Greedy Best-First, A* with heuristics
- 📈 **Linear Regression** - Data preprocessing, normalization, model training
- 📊 **Logistic Regression** - Binary classification, confusion matrix analysis
- 🧠 **Neural Networks** - Feedforward networks, backpropagation, gradient descent
- 📝 **NLP Group Project** - Text classification using Naive Bayes and TF-IDF
- 🎓 **Final Exam** - Comprehensive NLP implementation

### [01_DeepLearning](./01_DeepLearning)
Advanced deep learning architectures and implementations

**Topics covered:**
- 🖼️ **CNN & RNN** - Image classification with Convolutional and Recurrent Networks
- 🔄 **Autoencoders** - Dimensionality reduction and feature learning
- 🎨 **Variational Autoencoders (VAE)** - Generative models for image synthesis
- 🎭 **Convolutional GANs** - Generative Adversarial Networks for image generation
- 🚗 **Group Project** - Car price prediction using deep learning

---

## 🛠️ Technologies & Tools

**Core:**
- Python 3.9+
- NumPy (numerical computing)
- Pandas (data manipulation)
- Matplotlib (visualization)

**Machine Learning:**
- Scikit-learn (models, preprocessing, metrics, cross-validation)
- Neurolab (neural network experiments)

**Deep Learning:**
- TensorFlow / Keras (CNN, RNN, LSTM implementations)
- Seaborn (model performance visualization)

**Natural Language Processing:**
- NLTK (tokenization, stopwords, text preprocessing)
- CountVectorizer & TfidfTransformer (feature extraction)
- MultinomialNB (text classification)

**Development Environment:**
- Conda package manager
- Spyder / VS Code
- Git/GitHub

---

## 📂 Repository Structure
```
AI_Python_Projects/
├── README.md
├── 00_AI_Fundamentals/
│   ├── Assignment1-Agents/
│   │   ├── requirements.txt
│   │   └── Matheus_agent.py
│   ├── Assignment2-Search/
│   │   ├── requirements.txt
│   │   └── Exercise#1_Matheus/
│   ├── Assignment3-LinearRegression/
│   │   ├── requirements.txt
│   │   └── Exercise#2_Matheus/
│   ├── Assignment4-LogisticRegression/
│   │   ├── requirements.txt
│   │   └── Exercise#1_Matheus/
│   ├── Assignment5-NeuralNetwork/
│   │   ├── requirements.txt
│   │   └── Exercise_nn_Matheus.py
│   ├── GroupProject-NPL/
│   │   ├── requirements.txt
│   │   └── GroupProject.py
│   └── Final/
│       ├── requirements.txt
│       └── Final.py
└── 01_DeepLearning/
    ├── Assignment1_CNN-RNN/
    │   ├── requirements.txt
    │   └── Exercise#1_matheus/
    ├── Assignment2_Autoencoder/
    │   ├── requirements.txt
    │   └── Exercise#1_matheus/
    ├── Assignment3_VariationalAutoencoder/
    │   ├── requirements.txt
    │   └── Exercise1_Matheus/
    ├── Assignment4_ConvolutionalGAN/
    │   ├── requirements.txt
    │   └── Exercise1_Matheus/
    └── GroupProject/
        ├── requirements.txt
        └── Submission/
```

---

## 🎯 Key Learning Outcomes

✅ Implemented AI algorithms including agents, search, and machine learning from scratch  
✅ Built and trained deep neural networks (CNNs, RNNs, GANs, VAEs)  
✅ Applied ML/DL to real-world datasets (Fashion MNIST, Titanic, E-commerce, YouTube comments)  
✅ Worked with modern deep learning frameworks (TensorFlow/Keras)  
✅ Developed end-to-end ML pipelines with data preprocessing and model evaluation  
✅ Collaborated on group projects with version control and agile practices  
✅ Analyzed model performance using confusion matrices, accuracy metrics, and visualization

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **pip** or **conda** package manager

### Installation

Each assignment folder contains its own `requirements.txt` with specific dependencies.

**Using pip:**
```bash
# Navigate to any assignment folder
cd 00_AI_Fundamentals/Assignment3-LinearRegression

# Install dependencies
pip install -r requirements.txt

# Run the project
python Exercise#2_Matheus/Matheus_linear.py
```

**Using conda (recommended):**
```bash
# Create a new environment
conda create -n ai_projects python=3.9
conda activate ai_projects

# Navigate to assignment and install
cd 00_AI_Fundamentals/Assignment3-LinearRegression
pip install -r requirements.txt

# Run the project
python Exercise#2_Matheus/Matheus_linear.py
```

**Note:** Some projects require data files (CSV, images) which are included in their respective folders.

---

## 📊 Project Highlights

### AI Fundamentals Showcase

**🎯 Simple Reflex Agent (Assignment 1)**
- Environment-based decision making with custom Agent class
- OOP design with Agent, Environment, and Thing base classes
- Percept-action mapping for autonomous behavior

**🔍 Graph Search Algorithms (Assignment 2)**
- BFS implementation with shortest path finding
- Comparison of UCS, Greedy, and A* performance
- Custom heuristics for 8-puzzle problem solving

**📈 Linear Regression Pipeline (Assignment 3)**
- Complete data preprocessing (normalization, dummy variables)
- Model training with 80/20 train-test split
- Feature importance analysis and R² evaluation

**📊 Logistic Regression & Classification (Assignment 4)**
- Titanic survival prediction with 80%+ accuracy
- Confusion matrix analysis and threshold optimization
- Cross-validation with multiple train-test splits

**🧠 Neural Network Experiments (Assignment 5)**
- Feedforward networks with different architectures (single vs multi-layer)
- Gradient descent backpropagation implementation
- Effect of training data size on model performance (10 vs 100 samples)

**📝 NLP Text Classification (Group Project)**
- YouTube comment spam detection using Naive Bayes
- TF-IDF feature extraction and text preprocessing
- Achieved 95%+ accuracy on spam classification

### Deep Learning Showcase

**🖼️ CNN Image Classification (Assignment 1)**
- Fashion MNIST classification with 88%+ test accuracy
- Custom CNN: 2 Conv layers (32 filters) + MaxPooling + Dense(100)
- Confusion matrix analysis and probability distribution visualization

**📱 RNN Sequential Processing (Assignment 1)**
- LSTM network with 128 hidden units for image classification
- Treats image rows as sequential time steps
- Performance comparison between CNN and RNN approaches

**🔄 Autoencoders (Assignment 2)**
- Dimensionality reduction and feature learning
- Image reconstruction quality analysis

**🎨 VAE Generative Models (Assignment 3)**
- Variational autoencoder for image synthesis
- Latent space exploration

**🎭 Convolutional GAN (Assignment 4)**
- Generated synthetic fashion images
- Adversarial training dynamics analysis

---

## 📫 Contact

**Matheus Teixeira**  
- 🌐 [LinkedIn](https://linkedin.com/in/mathteixeira)
- 📧 mathteixeira55@gmail.com
- 💼 [Portfolio Website](http://studentweb.cencol.ca/mferre39/)
- 🐙 [GitHub](https://github.com/domvito55)

**Available for:**
- Remote software engineering positions
- AI/ML consulting projects
- Collaborative research opportunities

---

## 🎓 Academic Context

These projects were completed as part of the **Advanced Diploma in Software Engineering - Artificial Intelligence** program at Centennial College, Toronto, Canada (2022-2024).

**Program Highlights:**
- 3-year intensive program with focus on practical AI/ML applications
- 8-month Co-op experience at TD Bank
- Multiple research projects with industry partners
- **Final GPA:** 4.45/4.5 (High Honors)

---

## 📝 License

Educational projects completed at Centennial College (2022-2024). Code available for learning purposes and portfolio demonstration.

---

*Last Updated: October 2025*