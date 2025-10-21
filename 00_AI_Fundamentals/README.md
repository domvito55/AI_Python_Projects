# AI & Machine Learning Projects - Fundamentals

Collection of **6 production-quality implementations** demonstrating core AI/ML capabilities: autonomous agents, search algorithms, supervised learning, neural networks, and natural language processing.

**Portfolio Highlights:**
- 🤖 Agent-based systems with environment simulation
- 🔍 Graph search algorithms (BFS, UCS, Greedy, A*)
- 📊 Predictive models (R² 0.918 regression, 83% classification accuracy)
- 🧠 Neural network architecture experiments with overfitting analysis
- 💬 NLP spam detection system (92% accuracy on production data)

---

## 💡 Technical Skills Demonstrated

### Algorithm Implementation
✅ **Search Algorithms** - BFS, UCS, Greedy, A* with complexity analysis  
✅ **Machine Learning** - Linear regression, logistic regression, neural networks  
✅ **NLP Techniques** - Text vectorization, TF-IDF, Naive Bayes classification  
✅ **Optimization** - Gradient descent, backpropagation, heuristic search  

### Data Engineering
✅ **Preprocessing** - Missing value handling, normalization, categorical encoding  
✅ **Feature Engineering** - Feature selection and creation for improved model performance  
✅ **Model Evaluation** - Cross-validation, confusion matrices, multiple metrics  
✅ **Visualization** - 3D plots, error analysis, performance comparisons  

### Software Engineering
✅ **Clean Code** - Modular, documented, reusable implementations  
✅ **Version Control** - Professional Git workflow  
✅ **Testing & Validation** - Separate test sets, production testing  
✅ **Reproducibility** - Requirements files, environment configs, random seeds  

---

## 🛠️ Technologies & Tools

**Core Stack:**
- **Python 3.11**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib** - Visualization
- **Scikit-learn** - ML algorithms and evaluation

**Specialized:**
- **Neurolab** - Neural network implementation
- **Graphviz/Pydot** - Search tree visualization
- **Collections** - Advanced data structures

**Development:**
- **VS Code** - Primary editor
- **Git/GitHub** - Version control
- **Conda/pip** - Dependency management

---

## 🚀 Real-World Applications

### AI & Autonomous Systems
- Robotics control and autonomous navigation
- Game AI (NPC behavior, pathfinding)
- Resource optimization and scheduling

### Business Intelligence
- Customer behavior prediction and analytics
- Fraud detection systems
- Financial forecasting models
- Marketing campaign optimization

### Content & Communication
- Spam and content moderation (92% accuracy achieved)
- Sentiment analysis
- Chatbot development
- Document classification and routing

---

## 🎯 Projects Overview

Six production-quality implementations organized by increasing complexity:

### 1️⃣ Simple Reflex Agent - Environment Simulation
**[View Project →](Assignment1-Agents/)**

Autonomous agent (BlindDog) navigating a park environment using percept-action mapping. Demonstrates object-oriented design, state management, and agent architecture fundamentals.

**Key Features:**
- Environment simulation with multiple entity types
- Percept-based decision making (eat, drink, bark, move)
- Termination condition handling
- 18-step simulation with resource management

**Tech:** Python, OOP patterns, collections

---

### 2️⃣ Graph Search Algorithms - Pathfinding & Navigation
**[View Project →](Assignment2-Search/)**

High-performance implementation of 4 search algorithms with visual tree exploration and complexity analysis. Solves social network pathfinding and campus navigation problems.

**Algorithms Implemented:**
- **Breadth-First Search (BFS)** - Shortest path in social networks
- **Uniform Cost Search (UCS)** - Cost-optimal navigation
- **Greedy Best-First** - Heuristic-guided search
- **A* Search** - Optimal informed search

**Key Features:**
- Dynamic graph loading from any dataset
- Real-time visualization with Graphviz
- Heuristic optimization for informed search
- Performance comparison across algorithms

**Tech:** Python, Graphviz, Pydot, data structures

---

### 3️⃣ E-commerce Spending Prediction - Linear Regression
**[View Project →](Assignment3-LinearRegression/)**

Predictive model for customer spending using feature engineering and normalization. Demonstrates dramatic impact of feature selection on model performance.

**Key Results:**
- **R² Score: 0.918** (91.8% variance explained)
- Feature engineering improved model from 20% → 92% accuracy
- 2362 transactions analyzed

**Key Features:**
- Categorical variable encoding (one-hot)
- Feature normalization (min-max scaling)
- Model comparison (with/without key features)
- Data visualization (histograms, scatter matrices)

**Tech:** Python, Scikit-learn, Pandas, Matplotlib

---

### 4️⃣ Titanic Survival Prediction - Logistic Regression
**[View Project →](Assignment4-LogisticRegression/)**

Binary classification system predicting passenger survival with 83% accuracy. Includes cross-validation, confusion matrix analysis, and threshold tuning experiments.

**Key Results:**
- **Accuracy: 83%** (0.5 threshold), 81% (0.75 threshold)
- **Precision: 87%** / **Recall: 88%**
- 10-fold cross-validation: 90.47% average accuracy

**Key Features:**
- Complete data preprocessing pipeline
- Cross-validation with multiple metrics
- Threshold experimentation (precision vs recall trade-offs)
- Confusion matrix visualization

**Tech:** Python, Scikit-learn, Pandas, Matplotlib

---

### 5️⃣ Neural Network Training - Architecture & Data Analysis
**[View Project →](Assignment5-NeuralNetwork/)**

Systematic experiments comparing network architectures and data sizes. Demonstrates overfitting detection, convergence analysis, and 3D prediction visualization.

**Experiments:**
- 1-layer vs 2-layer architectures
- 10 vs 100 training samples
- 2D and 3D input spaces
- Error convergence analysis

**Key Findings:**
- Simple architecture + good data > Complex architecture + limited data
- Training error can be deceiving (overfitting detection critical)
- Data quality matters more than network depth for simple problems

**Key Features:**
- Feed-forward networks with backpropagation
- 3D surface plot visualizations
- Systematic overfitting analysis
- Gradient descent optimization

**Tech:** Python, Neurolab, NumPy, Matplotlib

---

### 6️⃣ YouTube Spam Detection - Natural Language Processing
**[View Project →](GroupProject-NLP/)**

Production-ready spam classifier using Bag of Words, TF-IDF, and Naive Bayes. Achieved 92% accuracy on real YouTube comments dataset.

**Key Results:**
- **Cross-Validation: 90.47%** accuracy
- **Test Set: 92.05%** accuracy  
- **Production: 100%** accuracy (9 custom test cases)

**Key Features:**
- Text preprocessing pipeline (tokenization, normalization)
- TF-IDF feature weighting
- Multinomial Naive Bayes classification
- Confusion matrix analysis
- Production testing with real-world examples

**Tech:** Python, Scikit-learn, Pandas, NLP techniques

---

## 📁 Repository Structure

```
00_AI_Fundamentals/
├── Assignment1-Agents/              # Agent-based system
│   ├── README.md                    # Comprehensive documentation
│   ├── Matheus_agent.py            # Implementation
│   ├── requirements.txt            # Python dependencies
│   └── environment.yml             # Conda environment
│
├── Assignment2-Search/              # Search algorithms
│   ├── README.md
│   ├── Exercise#1_Matheus/         # BFS implementation
│   ├── Exercise#2_Matheus/         # UCS, Greedy, A*
│   └── [requirements + environment]
│
├── Assignment3-LinearRegression/    # E-commerce prediction
│   ├── README.md
│   ├── Exercise#1_Matheus/         # Noise simulation
│   ├── Exercise#2_Matheus/         # Main analysis
│   └── [requirements + environment]
│
├── Assignment4-LogisticRegression/  # Titanic classification
│   ├── README.md
│   ├── Exercise#1_Matheus/         # Complete pipeline
│   ├── titanic.csv                 # Dataset
│   └── [requirements + environment]
│
├── Assignment5-NeuralNetwork/       # Neural network experiments
│   ├── README.md
│   ├── Exercise_nn_Matheus.py      # All 5 experiments
│   └── [requirements + environment]
│
├── GroupProject-NLP/                # Spam detection
│   ├── README.md
│   ├── GroupProject.py             # Production implementation
│   ├── Youtube02-KatyPerry.csv     # Dataset (350 comments)
│   └── [requirements + environment]
│
└── README.md                        # This file
```

---

## 🎯 Project Highlights

### Most Technically Complex
**🏆 Neural Network Experiments** - Five systematic experiments comparing architectures and data sizes, with 3D visualizations and deep analysis of overfitting vs generalization.

### Best Real-World Impact
**🏆 YouTube Spam Detection** - Production-ready classifier achieving 92% accuracy. Complete ML pipeline from raw text to deployed system.

### Most Algorithmic Depth
**🏆 Search Algorithms** - Four different search strategies implemented and compared with visual tree exploration and complexity analysis.

### Best Feature Engineering
**🏆 Linear Regression** - Demonstrated dramatic impact of feature selection: adding one feature improved model from 20% → 92% R².

### Most Comprehensive Evaluation
**🏆 Logistic Regression** - Multiple evaluation methods: 10-fold cross-validation, confusion matrices, threshold tuning, precision-recall analysis.

---

## 💡 Key Technical Insights

### 1. Data Quality > Model Complexity
**Neural Networks project:** Simple architecture with 100 training samples (test error: 0.0020) outperformed complex 2-layer network with same data (test error: 0.0257). **Takeaway:** Focus on data before adding architectural complexity.

### 2. Feature Selection is Critical
**Linear Regression:** One additional feature ('Record' - purchase history) improved R² from 0.195 to 0.918. **Takeaway:** Domain knowledge and feature engineering can matter more than algorithm choice.

### 3. Evaluation Beyond Accuracy
**Logistic Regression:** Threshold tuning creates precision-recall trade-offs. 0.5 threshold: 83% accuracy. 0.75 threshold: 81% accuracy but 100% recall. **Takeaway:** Business context determines optimal metrics.

### 4. Algorithm-Problem Matching
**Search Algorithms:** A* optimal for pathfinding, Greedy fast but risky, BFS simple for unweighted graphs, UCS for cost-optimization. **Takeaway:** No "best" algorithm - only best for specific problem.

### 5. Overfitting Detection
**Neural Networks experiment:** Training error 6.74×10⁻⁶ looked excellent, but test error 0.0826 revealed overfitting. **Takeaway:** Always validate on separate test data.

---

## 🎓 Academic Context

Completed as part of **COMP 237: Artificial Intelligence** at Centennial College during the Advanced Diploma in Software Engineering - AI program (2022-2024, GPA 4.45/4.5 - High Honors).

For advanced deep learning projects (CNNs, RNNs, GANs, VAEs), see [01_DeepLearning →](../01_DeepLearning/)

---

## 🔗 Navigation

**Explore Projects:**
- [Agent System →](Assignment1-Agents/) - OOP, simulation
- [Search Algorithms →](Assignment2-Search/) - BFS, UCS, Greedy, A*
- [Linear Regression →](Assignment3-LinearRegression/) - Feature engineering
- [Logistic Regression →](Assignment4-LogisticRegression/) - Classification
- [Neural Networks →](Assignment5-NeuralNetwork/) - Architecture experiments
- [NLP Spam Detection →](GroupProject-NLP/) - Text classification

**Portfolio:**
- [Repository Overview ←](../README.md) - All projects and live demos

---

*Comprehensive documentation available for all 6 projects. Each includes detailed README, complete implementation, and real-world applicability.*

**Last Updated:** October 2024