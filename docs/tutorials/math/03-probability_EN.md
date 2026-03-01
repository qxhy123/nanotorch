# Chapter 3: Probability Theory

Probability theory is the foundation for understanding **machine learning uncertainty** and **statistical learning theory**. From Softmax outputs in classification problems to generative models, from Dropout regularization to Bayesian neural networks, probability theory is everywhere. This chapter systematically introduces the probability theory knowledge needed for deep learning.

---

## Chapter Structure

For ease of learning and in-depth understanding, this chapter is divided into four sub-chapters:

### [3.1 Probability Basics and Conditional Probability](03a-probability-basics-conditional_EN.md)

**Content Overview**:
- Random experiments and sample spaces
- Events and event operations
- Definition and basic properties of probability
- Conditional probability and multiplication rule
- Independence
- Law of total probability
- Bayes' theorem

**Core Concepts**:
| Concept | Formula | Application in Deep Learning |
|---------|---------|----------------------------|
| Conditional probability | $P(A\|B) = P(A \cap B)/P(B)$ | Bayesian inference, conditional generation |
| Independence | $P(A \cap B) = P(A)P(B)$ | Dropout, data assumptions |
| Bayes' theorem | $P(B\|A) = P(A\|B)P(B)/P(A)$ | Model updating, posterior inference |

**[Start Learning →](03a-probability-basics-conditional_EN.md)**

---

### [3.2 Random Variables and Common Distributions](03b-random-variables-distributions_EN.md)

**Content Overview**:
- Definition of random variables (discrete/continuous)
- Probability mass function (PMF) and probability density function (PDF)
- Cumulative distribution function (CDF)
- Discrete distributions: Bernoulli, Binomial, Poisson, Categorical
- Continuous distributions: Uniform, Normal, Exponential, Laplace, Beta, Gamma
- Relationships between distributions

**Core Distributions**:
| Distribution | Formula/Parameters | Deep Learning Application |
|--------------|-------------------|--------------------------|
| Bernoulli | $P(X=1)=p$ | Dropout, binary classification |
| Normal distribution | $\mathcal{N}(\mu, \sigma^2)$ | Weight initialization, VAE |
| Categorical distribution | $\text{Cat}(p_1, \ldots, p_K)$ | Multi-class, language models |
| Beta | $\text{Beta}(\alpha, \beta)$ | Bayesian inference |

**[Start Learning →](03b-random-variables-distributions_EN.md)**

---

### [3.3 Multivariate Random Variables and Numerical Characteristics](03c-multivariate-random-variables_EN.md)

**Content Overview**:
- Joint distribution, marginal distribution, conditional distribution
- Independence and equivalent conditions
- Properties of expectation and moments
- Covariance and correlation coefficient
- Covariance matrix
- Higher moments: skewness and kurtosis

**Core Concepts**:
| Concept | Formula | Deep Learning Application |
|---------|---------|--------------------------|
| Covariance | $\text{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$ | Feature correlation analysis |
| Correlation coefficient | $\rho = \text{Cov}(X,Y)/(\sigma_X \sigma_Y)$ | Feature selection |
| Covariance matrix | $\mathbf{\Sigma} = \mathbb{E}[(\mathbf{X}-\boldsymbol{\mu})(\mathbf{X}-\boldsymbol{\mu})^\top]$ | PCA, BatchNorm |

**[Start Learning →](03c-multivariate-random-variables_EN.md)**

---

### [3.4 Limit Theorems and Information Theory](03d-limit-theorems-information-theory_EN.md)

**Content Overview**:
- Law of large numbers (weak and strong)
- Central limit theorem
- Monte Carlo methods
- Entropy, joint entropy, conditional entropy
- Cross-entropy
- KL divergence
- Mutual information
- Relationship between information theory and machine learning

**Core Concepts**:
| Concept | Formula | Deep Learning Application |
|---------|---------|--------------------------|
| Law of large numbers | $\bar{X}_n \xrightarrow{P} \mu$ | Batch statistics estimation |
| Central limit theorem | $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2)$ | Weight initialization |
| Cross-entropy | $H(P,Q) = -\sum p \log q$ | Classification loss function |
| KL divergence | $D_{KL} = \sum p \log(p/q)$ | VAE regularization |

**[Start Learning →](03d-limit-theorems-information-theory_EN.md)**

---

## Learning Path

```
┌─────────────────────────────────────────────────────────────┐
│                   Chapter 3: Probability Theory              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  3.1 Probability Basics → 3.2 Random Variables → 3.3 Multi- │
│  & Conditional         & Distributions     variate & Numer- │
│                                          ical Characteristics│
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Sample   │   │ PMF/PDF  │   │ Joint    │   │ Law of   │ │
│  │ Space    │   │ CDF      │   │ Dist     │   │ Large    │ │
│  │ Event    │   │ Discrete │   │ Marginal │   │ Numbers  │ │
│  │ Operations│   │ Continuous│   │ Covariance│  │ Central  │ │
│  │ Cond Prob│   │ Distributions│  │ Correlation│  │ Limit    │ │
│  │ Bayes    │   │          │   │          │   │ Entropy  │ │
│  └──────────┘   └──────────┘   └──────────┘   │ KL Div   │ │
│                                                └──────────┘ │
│  Applications: Softmax, Dropout, Initialization, VAE, Loss   │
└─────────────────────────────────────────────────────────────┘
```

---

## Why is Probability Theory Important for Deep Learning?

### 1. Uncertainty in Data

```
Real-world data → Contains noise → Probabilistic models
     ↓
Measurement errors, labeling noise, missing data → Need probabilistic modeling
```

### 2. Probabilistic Interpretation of Models

| Model Component | Probabilistic Interpretation |
|-----------------|------------------------------|
| Softmax output | Categorical distribution $P(y\|x)$ |
| Sigmoid output | Bernoulli distribution parameter $p$ |
| MSE loss | Negative log-likelihood under Gaussian noise assumption |
| Cross Entropy | Maximum likelihood estimation |
| Dropout | Sampling from Bayesian approximation |

### 3. Core of Generative Models

- **VAE**: Learning latent distribution of data
- **GAN**: Generated distribution approximates real distribution
- **Diffusion**: Progressive denoising probabilistic process

### 4. Uncertainty Quantification

```python
# Example: Prediction uncertainty
prediction = model(x)  # Point estimate
prediction_dist = bayesian_model(x)  # Distribution estimate
# Can answer: "How confident is the model?"
```

---

## Core Formula Quick Reference

### Probability Basics

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$

$$
P(B|A) = \frac{P(A|B)P(B)}{P(A)} \quad \text{(Bayes' theorem)}
$$

### Expectation and Variance

$$
\mathbb{E}[X] = \sum_x x \cdot p(x) \quad \text{(discrete)}
$$

$$
\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

### Normal Distribution

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

### Cross-Entropy Loss

$$
L = -\sum_{i=1}^K y_i \log \hat{y}_i
$$

### KL Divergence

$$
D_{KL}(P \| Q) = \sum_x p(x) \log \frac{p(x)}{q(x)}
$$

---

## Python Code Examples

### Softmax and Cross-Entropy

```python
import numpy as np

def softmax(logits):
    """Convert logits to probability distribution"""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

def cross_entropy_loss(probs, target):
    """Cross-entropy loss"""
    return -np.log(probs[target])

# Example
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Probability distribution: {probs}")  # [0.659, 0.242, 0.099]
print(f"Loss for predicting class 0: {cross_entropy_loss(probs, 0):.4f}")
```

### Normal Distribution Initialization

```python
import numpy as np

def he_init(fan_in, fan_out):
    """He initialization (for ReLU)"""
    std = np.sqrt(2 / fan_in)
    return np.random.normal(0, std, (fan_out, fan_in))

weights = he_init(784, 256)
print(f"Weight shape: {weights.shape}")
print(f"Theoretical standard deviation: {np.sqrt(2/784):.4f}")
```

---

## Study Recommendations

1. **Understand basic concepts first**: Conditional probability and independence are the foundation for everything that follows
2. **Focus on the normal distribution**: It's the most important distribution, appearing at every stage of deep learning
3. **Understand cross-entropy**: This is the most commonly used loss function for classification problems
4. **Hands-on implementation**: Implement sampling and visualization of various distributions using NumPy
5. **Connect to practice**: Think about specific applications of each concept in deep learning

---

## Further Reading

- [Chapter 1: Linear Algebra](01-linear-algebra_EN.md) - Basic operations of deep learning
- [Chapter 2: Calculus](02-calculus_EN.md) - Gradient descent and optimization
- [Chapter 4: Mathematical Statistics](04-statistics_EN.md) - Parameter estimation and hypothesis testing

---

**Back to**: [Mathematical Foundations Directory](../math-fundamentals_EN.md)
