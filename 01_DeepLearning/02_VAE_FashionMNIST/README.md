# Variational Autoencoder (VAE) - Fashion MNIST Generation

A complete implementation of a Variational Autoencoder (VAE) for learning compressed latent representations and generating new fashion item images. This project demonstrates deep learning concepts including probabilistic modeling, reparameterization trick, and generative neural networks.

## 🎯 Project Overview

**Objective:** Build a VAE that learns a 2-dimensional latent space representation of Fashion MNIST images, enabling both reconstruction and generation of new fashion items.

**Key Achievement:** Successfully trained a VAE with β-weighting that produces well-separated class clusters in latent space while maintaining generation quality.

## 🏗️ Architecture

### Encoder Network
- **Input:** 28×28×1 grayscale images
- **Convolutional layers:** 4 Conv2D layers (32→64→64→64 filters)
- **Latent space:** 2 dimensions (mean and log-variance)
- **Output:** Probabilistic latent representation z ~ N(μ, σ²)

### Decoder Network
- **Input:** 2D latent vector
- **Architecture:** Dense → Reshape → Conv2DTranspose → Conv2D
- **Output:** Reconstructed 28×28×1 image

### Loss Function
```
Total Loss = Reconstruction Loss + β * KL Divergence Loss
           = MSE(x, x_reconstructed) + 0.001 * KL(q(z|x) || N(0,1))
```

## 🔬 Technical Implementation

### Reparameterization Trick
Implemented custom `SampleLayer` to enable backpropagation through stochastic sampling:

```python
z = μ + exp(0.5 * log(σ²)) * ε,  where ε ~ N(0,1)
```

This technique allows gradients to flow through the sampling operation during training.

### β-VAE Implementation
Applied β-weighting (β=0.001) to balance reconstruction quality with latent space regularization, preventing mode collapse while maintaining meaningful representations.

## 📊 Results

### Latent Space Statistics
```
Dimension z[0]: min=-3.66, max=2.76, mean=-0.09
Dimension z[1]: min=-3.46, max=3.25, mean=-0.09

Target distribution: N(0,1) ✓
```

**Note:** Results may vary slightly between runs due to stochastic training (weight initialization, data shuffling, random sampling). All runs converge to similar latent space distributions centered around N(0,1).

### Key Findings
- **Well-separated clusters:** Each fashion category occupies distinct regions
- **Smooth transitions:** Interpolation between classes produces realistic intermediate samples
- **Controlled generation:** Sampling from N(0,1) generates diverse, realistic fashion items

### Visualizations
- **Latent Space Plot:** Clear separation of 10 fashion classes in 2D space
- **Generated Samples:** 15×15 grid showing smooth interpolation across latent space
- **Edge sharpness:** High-quality samples at cluster centers, smooth blending at boundaries

## 🛠️ Technical Stack

**Framework:** TensorFlow/Keras  
**Dataset:** Fashion MNIST (60k train, 10k test)  
**Training:** 10 epochs, batch size 512, Adam optimizer (lr=0.001)  
**Environment:** Python 3.10, TensorFlow 2.10.0

### Key Libraries
```python
tensorflow==2.10.0
tensorflow-probability==0.18.0
numpy==1.23.5
matplotlib==3.6.2
```

## 🎓 Concepts Demonstrated

1. **Probabilistic Deep Learning**
   - Variational inference
   - KL divergence regularization
   - Reparameterization trick for gradient flow

2. **Generative Modeling**
   - Learning compressed representations
   - Sampling from learned distributions
   - Image generation through latent space manipulation

3. **Custom Keras Layers**
   - Implementing stochastic sampling layers
   - Managing loss computation in complex architectures

4. **β-VAE Technique**
   - Balancing reconstruction vs. regularization
   - Preventing posterior collapse
   - Controlling disentanglement in latent space

## 📈 Training Performance

- **Training time:** ~30-35 minutes (10 epochs on CPU)
- **Final loss:** 0.0356 (reconstruction + weighted KL)
- **Latent space:** Successfully regularized to approximate N(0,1)
- **Convergence:** Stable training, loss decreases from 0.0799 → 0.0356

## 🎯 Applications

This VAE implementation demonstrates techniques applicable to:
- **Data compression:** Reducing high-dimensional images to 2D representations
- **Anomaly detection:** Identifying out-of-distribution samples
- **Data augmentation:** Generating synthetic training samples
- **Feature learning:** Extracting meaningful low-dimensional features

## 📝 Project Structure

```
02_VAE_FashionMNIST/
├── Exercise1_Matheus/
│   └── Matheus_lab3.py        # Main implementation
├── Diagrams.vsdx              # Architecture diagrams
├── environment.yml            # Conda environment
├── Lab 3 Assignment_FALL2024.pdf  # Assignment specifications
├── Matheus_lab3.html          # code execution and outputs export
├── README.md                  # This file
└── requirements.txt           # Pip dependencies
```

## 🚀 Running the Code

```bash
# Create environment
conda env create -f environment.yml
conda activate vae-assignment

# Run training and generation
python Exercise1_Matheus/Matheus_lab3.py
```

### Reproducibility
Results vary slightly between runs due to:
- Random weight initialization
- Stochastic gradient descent
- Random sampling in latent space

To ensure reproducibility, add seeds at the beginning of the script:
```python
import random, numpy as np, tensorflow as tf
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

Note: Even with seeds, minor variations may occur on different hardware.

## 🔍 Key Insights

### Why β=0.001?

The choice of β=0.001 is not arbitrary but based on **balancing the magnitude of two competing losses**:

**Order of Magnitude Analysis:**
```
Reconstruction Loss (MSE) ≈ 0.04-0.08  (typical for normalized images)
KL Loss (unweighted)      ≈ 10-50      (typical for VAEs)

If β=1.0 (standard VAE):
  Total Loss = 0.04 + 50 = 50.04
  → KL loss dominates → posterior collapse (all inputs map to same point)

If β=0.001 (β-VAE):
  Total Loss = 0.04 + 0.05 = 0.09
  → Balanced contributions → stable learning ✓
```

**Rule of Thumb for Choosing β:**
```
β ≈ (target_proportion × MSE) / expected_KL
β ≈ (0.1 × 0.04) / 10 ≈ 0.0004 to 0.001

Target: KL loss contributes ~10-20% of total loss
```

**Typical Values in Literature:**
- β = 1.0: Standard VAE (Kingma & Welling, 2013) - often collapses in low dimensions
- β = 0.0001-0.01: β-VAE (Higgins et al., 2017) - better disentanglement
- β = 0 → ∞: Cyclical annealing - gradual warm-up during training

**Symptoms of Incorrect β:**
- **β too high:** Posterior collapse (z → 0 for all inputs, blurry reconstructions)
- **β too low:** Unregularized latent space (mean ≠ 0, std ≫ 1, poor generation)
- **β optimal:** mean ≈ 0, std ≈ 1, sharp reconstructions, smooth generation ✓

### Latent Space Topology
The "fan" shape observed in the latent space visualization is expected behavior:
- **Natural manifold:** Data lies on a curved 2D surface in latent space
- **Class boundaries:** Sharp transitions between fashion categories
- **Interpolation quality:** Smooth blending in transition regions

### Generation Quality
- **Cluster centers:** Sharp, realistic images (trained regions)
- **Boundary regions:** Smooth blends, more abstract (interpolated regions)
- **Overall:** Demonstrates successful learning of continuous latent representation

---

**Course:** COMP 263 - Deep Learning  
**Institution:** Centennial College  
**Semester:** Fall 2024  
**Grade:** High Honors (GPA: 4.45/4.5)