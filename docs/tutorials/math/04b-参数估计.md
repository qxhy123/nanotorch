# 第四章（b）：参数估计

参数估计是统计推断的核心任务之一，即根据样本数据推断总体分布中的未知参数。在深度学习中，最大似然估计是设计损失函数的理论基础，理解参数估计对于理解为什么交叉熵损失有效至关重要。

---

## 目录

1. [参数估计概述](#参数估计概述)
2. [点估计与评价标准](#点估计与评价标准)
3. [矩估计法](#矩估计法)
4. [最大似然估计](#最大似然估计)
5. [MLE 的性质与计算](#mle-的性质与计算)
6. [贝叶斯估计](#贝叶斯估计)
7. [MLE 与损失函数的关系](#mle-与损失函数的关系)
8. [在深度学习中的应用](#在深度学习中的应用)
9. [小结](#小结)

---

## 参数估计概述

### 🎯 生活类比：盲人摸象后的猜测

想象一个盲人摸到大象的一部分，然后猜测整头大象的样子：
- **摸到象腿**："大象像柱子"
- **摸到象耳**："大象像扇子"
- **摸到象鼻**："大象像蛇"

**参数估计就是**：用**有限的样本**（摸到的一部分）猜测**总体的参数**（完整的大象）。

**更具体的例子**：
- 你想知道全校学生的平均身高（参数θ）
- 但不能测量所有人（总体太大）
- 你随机抽取100人测量（样本）
- 用这100人的平均身高估计全校平均身高（估计量θ̂）

### 📖 关键概念

| 概念 | 类比 | 统计学术语 |
|------|------|-----------|
| 真实情况 | 全校学生平均身高170cm | 参数 θ（未知） |
| 我们的猜测 | 100人样本平均169cm | 估计量 θ̂ |
| 估计方法 | 求平均值 | 点估计 |

### 参数估计问题

**问题描述**：已知总体分布的形式，但含有未知参数 $\theta$，需要根据样本 $X_1, \ldots, X_n$ 来估计 $\theta$。

**示例**：
- 估计正态分布 $\mathcal{N}(\mu, \sigma^2)$ 的均值 $\mu$ 和方差 $\sigma^2$
- 估计 Bernoulli 分布的成功概率 $p$
- 估计 Poisson 分布的参数 $\lambda$

### 估计的类型

| 类型 | 目标 | 结果 |
|------|------|------|
| **点估计** | 用单个值估计参数 | $\hat{\theta}$ |
| **区间估计** | 给出参数的可能范围 | $(L, U)$ |

### 符号约定

| 符号 | 含义 |
|------|------|
| $\theta$ | 真实参数（未知、固定） |
| $\hat{\theta}$ | 参数的估计值（已知、随机） |
| $\hat{\theta}_{MLE}$ | 最大似然估计 |
| $\hat{\theta}_{MM}$ | 矩估计 |

---

## 点估计与评价标准

### 🎯 生活类比：评价一个"猜测者"的好坏

想象三个人都在猜全校平均身高：
- **A说**：170cm（真实是170cm）
- **B说**：175cm
- **C说**：160cm

怎么评价谁猜得更好？

**评价标准**：

1. **无偏性**：多次猜测的平均值是否接近真实值？
   - A的猜测平均值=170 → 无偏
   - B的猜测平均值=175 → 有偏（高估5cm）

2. **有效性**：谁的猜测波动更小？
   - A的猜测范围：168-172（波动小）
   - B的猜测范围：160-190（波动大）
   - A更有效

3. **一致性**：信息越多，猜测越准？
   - 问10个人 vs 问1000个人
   - 1000人的估计应该更准确

### 📝 手把手判断无偏性

**例子**：用样本均值估计总体均值

```
假设真实均值 μ = 100
多次抽样：

第1次抽样：样本均值 = 102
第2次抽样：样本均值 = 98
第3次抽样：样本均值 = 101
第4次抽样：样本均值 = 99
第5次抽样：样本均值 = 100

平均估计 = (102+98+101+99+100)/5 = 100 = μ

结论：样本均值是总体均值的"无偏估计"
```

### 点估计的定义

**点估计**：构造一个统计量 $\hat{\theta} = g(X_1, \ldots, X_n)$ 作为参数 $\theta$ 的估计。

- **估计量**（Estimator）：$\hat{\theta}$ 作为随机变量
- **估计值**（Estimate）：$\hat{\theta}$ 的具体取值

### 评价标准

#### 1. 无偏性 (Unbiasedness)

$$
\mathbb{E}[\hat{\theta}] = \theta
$$

**偏差**（Bias）：

$$
\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta
$$

无偏估计：$\text{Bias}(\hat{\theta}) = 0$

#### 2. 有效性 (Efficiency)

在无偏估计中，**方差最小**的估计量最有效。

$$
\text{Var}(\hat{\theta}_1) < \text{Var}(\hat{\theta}_2) \Rightarrow \hat{\theta}_1 \text{ 更有效}
$$

**Cramér-Rao 下界**：无偏估计方差的理论下界：

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{n \cdot I(\theta)}
$$

其中 $I(\theta)$ 是 **Fisher 信息量**：

$$
I(\theta) = \mathbb{E}\left[\left(\frac{\partial \ln f(X; \theta)}{\partial \theta}\right)^2\right]
$$

#### 3. 一致性 (Consistency)

当样本量 $n \to \infty$ 时，估计量收敛到真实参数：

$$
\hat{\theta}_n \xrightarrow{P} \theta
$$

#### 4. 均方误差 (Mean Squared Error)

$$
\text{MSE}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \theta)^2] = \text{Var}(\hat{\theta}) + \text{Bias}(\hat{\theta})^2
$$

**偏差-方差权衡**：有时略微有偏的估计量可能有更小的 MSE。

```python
import numpy as np
import matplotlib.pyplot as plt

# 演示无偏性和有效性的权衡
np.random.seed(42)
n_simulations = 10000
n = 10
true_sigma2 = 4

# 两种方差估计量
unbiased_vars = []
biased_vars = []

for _ in range(n_simulations):
    sample = np.random.normal(0, np.sqrt(true_sigma2), n)
    unbiased_vars.append(np.var(sample, ddof=1))  # n-1
    biased_vars.append(np.var(sample, ddof=0))    # n

unbiased_vars = np.array(unbiased_vars)
biased_vars = np.array(biased_vars)

print("方差估计量的比较:")
print("="*50)
print(f"真实方差: σ² = {true_sigma2}")
print()
print("无偏估计 (分母 n-1):")
print(f"  期望: {unbiased_vars.mean():.4f} (偏差: {unbiased_vars.mean() - true_sigma2:.4f})")
print(f"  方差: {unbiased_vars.var():.4f}")
print(f"  MSE: {np.mean((unbiased_vars - true_sigma2)**2):.4f}")
print()
print("有偏估计 (分母 n):")
print(f"  期望: {biased_vars.mean():.4f} (偏差: {biased_vars.mean() - true_sigma2:.4f})")
print(f"  方差: {biased_vars.var():.4f}")
print(f"  MSE: {np.mean((biased_vars - true_sigma2)**2):.4f}")

# 可视化
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.hist(unbiased_vars, bins=50, alpha=0.7, label=f'无偏估计 (MSE={np.mean((unbiased_vars - true_sigma2)**2):.3f})')
ax.hist(biased_vars, bins=50, alpha=0.7, label=f'有偏估计 (MSE={np.mean((biased_vars - true_sigma2)**2):.3f})')
ax.axvline(true_sigma2, color='red', linestyle='--', linewidth=2, label=f'真实值 σ²={true_sigma2}')
ax.set_xlabel('估计值')
ax.set_ylabel('频数')
ax.set_title('无偏估计 vs 有偏估计 (n=10)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bias_variance_tradeoff.png', dpi=100)
print("\n图像已保存: bias_variance_tradeoff.png")
```

---

## 矩估计法

### 基本思想

用**样本矩**估计**总体矩**。

### 理论基础

**大数定律**：样本矩依概率收敛于总体矩。

$$
A_k = \frac{1}{n}\sum_{i=1}^n X_i^k \xrightarrow{P} \mathbb{E}[X^k] = \mu_k
$$

### 矩估计的步骤

1. 计算总体矩 $\mu_k(\theta_1, \ldots, \theta_m)$，表示为参数的函数
2. 用样本矩 $A_k$ 替代总体矩
3. 解方程组，得到参数估计

### 示例：正态分布

设 $X \sim \mathcal{N}(\mu, \sigma^2)$，用矩估计法估计 $\mu$ 和 $\sigma^2$。

**步骤 1**：计算总体矩

$$
\mu_1 = \mathbb{E}[X] = \mu
$$

$$
\mu_2 = \mathbb{E}[X^2] = \text{Var}(X) + (\mathbb{E}[X])^2 = \sigma^2 + \mu^2
$$

**步骤 2**：建立方程组

$$
\begin{cases}
A_1 = \mu \\
A_2 = \sigma^2 + \mu^2
\end{cases}
$$

**步骤 3**：求解

$$
\hat{\mu}_{MM} = \bar{X}
$$

$$
\hat{\sigma}^2_{MM} = A_2 - \bar{X}^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

**注意**：矩估计的 $\hat{\sigma}^2$ 是**有偏的**（分母是 $n$ 而不是 $n-1$）。

```python
import numpy as np

# 矩估计示例
np.random.seed(42)
true_mu = 5
true_sigma2 = 4
n = 100

sample = np.random.normal(true_mu, np.sqrt(true_sigma2), n)

# 矩估计
mu_mm = sample.mean()
sigma2_mm = np.mean(sample**2) - sample.mean()**2

# 无偏估计（MLE + 修正）
mu_mle = sample.mean()
sigma2_mle = np.mean((sample - sample.mean())**2)
sigma2_unbiased = sample.var(ddof=1)

print("正态分布参数估计:")
print("="*50)
print(f"真实参数: μ = {true_mu}, σ² = {true_sigma2}")
print()
print(f"矩估计: μ̂ = {mu_mm:.4f}, σ̂² = {sigma2_mm:.4f}")
print(f"MLE:    μ̂ = {mu_mle:.4f}, σ̂² = {sigma2_mle:.4f}")
print(f"无偏:   μ̂ = {mu_mle:.4f}, σ̂² = {sigma2_unbiased:.4f}")
print()
print(f"σ² 的估计偏差:")
print(f"  矩估计/MLE 偏差: {sigma2_mm - true_sigma2:.4f}")
print(f"  无偏估计 偏差: {sigma2_unbiased - true_sigma2:.4f}")

# 泊松分布的矩估计
print("\n" + "="*50)
print("泊松分布参数估计:")
true_lambda = 3
n = 100
sample_poisson = np.random.poisson(true_lambda, n)

# 矩估计：μ₁ = λ
lambda_mm = sample_poisson.mean()
print(f"真实参数: λ = {true_lambda}")
print(f"矩估计: λ̂ = {lambda_mm:.4f}")
```

---

## 最大似然估计

### 似然函数

#### 定义

给定观测数据 $x_1, \ldots, x_n$，**似然函数**是参数 $\theta$ 的函数：

$$
L(\theta) = L(\theta; x_1, \ldots, x_n) = \prod_{i=1}^n f(x_i; \theta)
$$

其中 $f(x; \theta)$ 是概率密度函数（连续）或概率质量函数（离散）。

#### 似然 vs 概率

| 概率 | 似然 |
|------|------|
| 参数固定，数据变化 | 数据固定，参数变化 |
| $f(x | \theta)$ | $L(\theta | x)$ |

### 对数似然函数

$$
\ell(\theta) = \ln L(\theta) = \sum_{i=1}^n \ln f(x_i; \theta)
$$

**优点**：
1. 将乘法转化为加法，计算更简单
2. 避免数值下溢
3. 对数函数单调，不影响最大化

### 最大似然估计 (MLE)

**定义**：选择使似然函数最大的参数值：

$$
\hat{\theta}_{MLE} = \arg\max_\theta L(\theta) = \arg\max_\theta \ell(\theta)
$$

**求解方法**：通常通过求解似然方程：

$$
\frac{\partial \ell(\theta)}{\partial \theta} = 0
$$

### 示例：正态分布参数估计

设 $X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2)$，求 $\mu$ 和 $\sigma^2$ 的 MLE。

**似然函数**：

$$
L(\mu, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
$$

**对数似然**：

$$
\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2
$$

**对 $\mu$ 求偏导**：

$$
\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0
$$

$$
\Rightarrow \hat{\mu}_{MLE} = \bar{x} = \frac{1}{n}\sum_{i=1}^n x_i
$$

**对 $\sigma^2$ 求偏导**：

$$
\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^n (x_i - \mu)^2 = 0
$$

$$
\Rightarrow \hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2
$$

**结论**：
- $\hat{\mu}_{MLE}$ 是无偏的
- $\hat{\sigma}^2_{MLE}$ 是**有偏的**（偏差为 $-\sigma^2/n$）

```python
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# MLE 示例：正态分布
np.random.seed(42)
true_mu = 3
true_sigma = 2
n = 100

sample = np.random.normal(true_mu, true_sigma, n)

# 解析解
mu_mle = sample.mean()
sigma2_mle = np.mean((sample - mu_mle)**2)

print("正态分布 MLE:")
print("="*50)
print(f"真实参数: μ = {true_mu}, σ = {true_sigma}")
print(f"MLE: μ̂ = {mu_mle:.4f}, σ̂ = {np.sqrt(sigma2_mle):.4f}")

# 数值优化验证
def neg_log_likelihood(params, data):
    """负对数似然（用于最小化）"""
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    n = len(data)
    return n/2 * np.log(2*np.pi) + n * np.log(sigma) + np.sum((data - mu)**2) / (2*sigma**2)

# 数值优化
result = minimize(neg_log_likelihood, [0, 0], args=(sample,), method='BFGS')
mu_num, sigma_num = result.x[0], np.exp(result.x[1])

print(f"\n数值优化: μ̂ = {mu_num:.4f}, σ̂ = {sigma_num:.4f}")

# 可视化似然函数
mu_range = np.linspace(mu_mle - 2, mu_mle + 2, 100)
sigma_range = np.linspace(0.5, 4, 100)
MU, SIGMA = np.meshgrid(mu_range, sigma_range)

# 计算对数似然
def log_likelihood(mu, sigma, data):
    n = len(data)
    return -n/2 * np.log(2*np.pi) - n * np.log(sigma) - np.sum((data - mu)**2) / (2*sigma**2)

LL = np.zeros_like(MU)
for i in range(MU.shape[0]):
    for j in range(MU.shape[1]):
        LL[i, j] = log_likelihood(MU[i, j], SIGMA[i, j], sample)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
contour = ax.contour(MU, SIGMA, LL, levels=20, cmap='viridis')
ax.scatter([mu_mle], [np.sqrt(sigma2_mle)], color='red', s=100, marker='*', label=f'MLE: ({mu_mle:.2f}, {np.sqrt(sigma2_mle):.2f})')
ax.scatter([true_mu], [true_sigma], color='blue', s=100, marker='o', label=f'True: ({true_mu}, {true_sigma})')
ax.set_xlabel('μ')
ax.set_ylabel('σ')
ax.set_title('对数似然函数等高线图')
ax.legend()
ax.grid(True, alpha=0.3)
plt.colorbar(contour, ax=ax, label='Log-Likelihood')

plt.tight_layout()
plt.savefig('mle_likelihood.png', dpi=100)
print("\n图像已保存: mle_likelihood.png")
```

---

## MLE 的性质与计算

### 渐进性质

在大样本情况下，MLE 具有以下**优良性质**：

#### 1. 一致性

$$
\hat{\theta}_{MLE} \xrightarrow{P} \theta \quad \text{当 } n \to \infty
$$

#### 2. 渐进正态性

$$
\sqrt{n}(\hat{\theta}_{MLE} - \theta) \xrightarrow{d} \mathcal{N}(0, I(\theta)^{-1})
$$

其中 $I(\theta)$ 是 Fisher 信息量。

#### 3. 渐进有效性

MLE 达到 **Cramér-Rao 下界**，即方差最小的无偏估计。

$$
\text{Var}(\hat{\theta}_{MLE}) \approx \frac{1}{n \cdot I(\theta)}
$$

### Fisher 信息量

**定义**（单个观测）：

$$
I(\theta) = \mathbb{E}\left[\left(\frac{\partial \ln f(X; \theta)}{\partial \theta}\right)^2\right] = -\mathbb{E}\left[\frac{\partial^2 \ln f(X; \theta)}{\partial \theta^2}\right]
$$

**样本的 Fisher 信息量**：$n \cdot I(\theta)$

### 计算技巧

#### 1. 对数似然的性质

$$
\frac{\partial \ell}{\partial \theta} = \sum_{i=1}^n \frac{\partial \ln f(x_i; \theta)}{\partial \theta}
$$

#### 2. 似然比检验

似然比统计量：

$$
\Lambda = 2[\ell(\hat{\theta}) - \ell(\theta_0)] \xrightarrow{d} \chi^2(k)
$$

其中 $k$ 是参数个数。

```python
import numpy as np
from scipy import stats

# 演示 MLE 的渐进正态性
np.random.seed(42)
true_p = 0.3  # Bernoulli 参数

sample_sizes = [30, 100, 500, 1000]
n_simulations = 10000

print("MLE 的渐进正态性验证 (Bernoulli 分布):")
print("="*60)
print(f"真实参数: p = {true_p}")
print()

for n in sample_sizes:
    mle_estimates = []
    
    for _ in range(n_simulations):
        sample = np.random.binomial(1, true_p, n)
        p_mle = sample.mean()
        mle_estimates.append(p_mle)
    
    mle_estimates = np.array(mle_estimates)
    
    # 标准化
    # Fisher 信息量: I(p) = 1/(p(1-p))
    fisher_info = 1 / (true_p * (1 - true_p))
    standardized = np.sqrt(n * fisher_info) * (mle_estimates - true_p)
    
    print(f"样本量 n = {n}:")
    print(f"  MLE 均值: {mle_estimates.mean():.4f} (真实值: {true_p})")
    print(f"  MLE 标准差: {mle_estimates.std():.4f} (理论: {np.sqrt(true_p*(1-true_p)/n):.4f})")
    print(f"  标准化后的均值: {standardized.mean():.4f}")
    print(f"  标准化后的标准差: {standardized.std():.4f}")
    print()
```

---

## 贝叶斯估计

### 贝叶斯框架

在贝叶斯框架下，参数 $\theta$ 被视为**随机变量**，有自己的分布。

**贝叶斯公式**：

$$
P(\theta | \text{data}) = \frac{P(\text{data} | \theta) \cdot P(\theta)}{P(\text{data})} \propto P(\text{data} | \theta) \cdot P(\theta)
$$

| 术语 | 含义 |
|------|------|
| **先验分布** $P(\theta)$ | 观察数据前对参数的信念 |
| **似然** $P(\text{data} | \theta)$ | 数据在给定参数下的概率 |
| **后验分布** $P(\theta | \text{data})$ | 观察数据后对参数的更新信念 |
| **证据** $P(\text{data})$ | 归一化常数 |

### 贝叶斯估计量

常见的贝叶斯点估计：

#### 1. 后验均值

$$
\hat{\theta}_{Bayes} = \mathbb{E}[\theta | \text{data}] = \int \theta \cdot P(\theta | \text{data}) d\theta
$$

#### 2. 后验中位数

$$
\hat{\theta}_{Bayes}: P(\theta \leq \hat{\theta} | \text{data}) = 0.5
$$

#### 3. 最大后验估计 (MAP)

$$
\hat{\theta}_{MAP} = \arg\max_\theta P(\theta | \text{data}) = \arg\max_\theta [P(\text{data} | \theta) \cdot P(\theta)]
$$

### 共轭先验

若先验分布和后验分布属于同一分布族，则称该先验为**共轭先验**。

| 似然 | 共轭先验 | 后验 |
|------|----------|------|
| Bernoulli | Beta | Beta |
| Poisson | Gamma | Gamma |
| 正态（均值） | 正态 | 正态 |

### 示例：Bernoulli 分布的贝叶斯估计

设 $X_1, \ldots, X_n \sim \text{Bernoulli}(p)$，取 $p$ 的先验为 $\text{Beta}(\alpha, \beta)$。

**后验分布**：

$$
p | \text{data} \sim \text{Beta}\left(\alpha + \sum_{i=1}^n x_i, \beta + n - \sum_{i=1}^n x_i\right)
$$

**后验均值**：

$$
\hat{p}_{Bayes} = \frac{\alpha + \sum x_i}{\alpha + \beta + n}
$$

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 贝叶斯估计示例：Bernoulli 分布
np.random.seed(42)
true_p = 0.3
n = 50
data = np.random.binomial(1, true_p, n)

# 先验参数 (Beta 分布)
alpha_prior = 1  # 相当于均匀分布
beta_prior = 1

# 后验参数
sum_x = data.sum()
alpha_post = alpha_prior + sum_x
beta_post = beta_prior + n - sum_x

# 各种估计
p_mle = data.mean()
p_bayes = alpha_post / (alpha_post + beta_post)  # 后验均值

print("Bernoulli 参数估计:")
print("="*50)
print(f"真实参数: p = {true_p}")
print(f"数据: n = {n}, 成功次数 = {sum_x}")
print()
print(f"MLE: p̂ = {p_mle:.4f}")
print(f"Bayes (后验均值): p̂ = {p_bayes:.4f}")
print(f"MAP (后验众数): p̂ = {(alpha_post - 1)/(alpha_post + beta_post - 2):.4f}")
print()

# 可视化
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

x = np.linspace(0, 1, 1000)
prior_pdf = stats.beta.pdf(x, alpha_prior, beta_prior)
posterior_pdf = stats.beta.pdf(x, alpha_post, beta_post)

ax.plot(x, prior_pdf, 'b-', linewidth=2, label=f'先验: Beta({alpha_prior}, {beta_prior})')
ax.plot(x, posterior_pdf, 'r-', linewidth=2, label=f'后验: Beta({alpha_post}, {beta_post})')
ax.axvline(true_p, color='green', linestyle='--', linewidth=2, label=f'真实值 p={true_p}')
ax.axvline(p_mle, color='purple', linestyle=':', linewidth=2, label=f'MLE p̂={p_mle:.3f}')
ax.axvline(p_bayes, color='orange', linestyle=':', linewidth=2, label=f'Bayes p̂={p_bayes:.3f}')

ax.set_xlabel('p')
ax.set_ylabel('密度')
ax.set_title('贝叶斯更新：先验 → 后验')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bayesian_estimation.png', dpi=100)
print("图像已保存: bayesian_estimation.png")
```

---

## MLE 与损失函数的关系

### 关键定理

**最小化损失函数 = 最大化似然估计**

| 概率模型 | 负对数似然 = 损失函数 |
|----------|----------------------|
| $Y \sim \mathcal{N}(f_\theta(X), \sigma^2)$ | MSE Loss |
| $Y \sim \text{Bernoulli}(\sigma(f_\theta(X)))$ | Binary Cross-Entropy |
| $Y \sim \text{Categorical}(\text{Softmax}(f_\theta(X)))$ | Categorical Cross-Entropy |

### 推导：MSE Loss

假设 $Y | X \sim \mathcal{N}(f_\theta(X), \sigma^2)$：

$$
\ell(\theta) = \sum_{i=1}^n \ln f(y_i | x_i; \theta) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - f_\theta(x_i))^2
$$

最大化 $\ell(\theta)$ 等价于最小化：

$$
\sum_{i=1}^n (y_i - f_\theta(x_i))^2 = \text{MSE Loss}
$$

### 推导：交叉熵损失

假设 $Y | X \sim \text{Categorical}(\text{Softmax}(f_\theta(X)))$：

$$
\ell(\theta) = \sum_{i=1}^n \ln P(y_i | x_i; \theta) = \sum_{i=1}^n \ln \text{Softmax}(f_\theta(x_i))_{y_i}
$$

负对数似然 = 交叉熵损失。

```python
import numpy as np

# 验证 MLE 与损失函数的关系

# 1. MSE Loss 对应高斯噪声假设
print("MLE 与 MSE Loss 的关系:")
print("="*50)

np.random.seed(42)
n = 100
true_slope = 2
true_intercept = 1

X = np.random.randn(n)
Y = true_slope * X + true_intercept + np.random.randn(n) * 0.5  # 高斯噪声

# 最小二乘解 = MLE
X_design = np.column_stack([np.ones(n), X])
theta_mle = np.linalg.lstsq(X_design, Y, rcond=None)[0]

print(f"真实参数: slope = {true_slope}, intercept = {true_intercept}")
print(f"MLE/LSE:  slope = {theta_mle[1]:.4f}, intercept = {theta_mle[0]:.4f}")

# 2. 交叉熵损失对应类别分布
print("\n" + "="*50)
print("MLE 与交叉熵损失的关系:")
print("="*50)

# 假设真实类别分布
true_logits = np.array([1.0, 2.0, 0.5])
true_probs = np.exp(true_logits) / np.exp(true_logits).sum()
print(f"真实类别概率: {true_probs}")

# 抽样
n_samples = 1000
labels = np.random.choice(3, n_samples, p=true_probs)

# MLE 估计 logits
# 对于充分统计量（类别计数），MLE 直接由频数决定
counts = np.bincount(labels, minlength=3)
mle_probs = counts / n_samples

# 避免 log(0)，加小量
eps = 1e-10
mle_logits = np.log(mle_probs + eps)
mle_logits = mle_logits - mle_logits.mean()  # 归一化

print(f"MLE 估计的概率: {mle_probs}")
print(f"交叉熵损失（使用真实概率）: {-np.sum(true_probs * np.log(true_probs + eps)):.4f}")
```

---

## 在深度学习中的应用

### 1. 损失函数设计

| 任务 | 概率模型 | 损失函数 |
|------|----------|----------|
| 回归 | $Y \sim \mathcal{N}(f(X), \sigma^2)$ | MSE |
| 二分类 | $Y \sim \text{Bernoulli}(\sigma(f(X)))$ | BCE |
| 多分类 | $Y \sim \text{Categorical}(\text{Softmax}(f(X)))$ | CE |

### 2. 正则化的贝叶斯解释

**L2 正则化** = 高斯先验

$$
\text{Loss}_{reg} = \text{Loss} + \lambda \|\theta\|^2 = -\ell(\theta) - \ln P(\theta)
$$

其中 $P(\theta) \propto \exp(-\lambda \|\theta\|^2)$ 是高斯先验。

**L1 正则化** = 拉普拉斯先验

$$
P(\theta) \propto \exp(-\lambda \|\theta\|_1)
$$

### 3. 参数初始化

理解参数的先验分布有助于选择合适的初始化策略。

```python
import numpy as np

# 正则化的贝叶斯解释
print("正则化的贝叶斯解释:")
print("="*50)

# L2 正则化 = 高斯先验 N(0, 1/(2λ))
lambda_l2 = 0.01
sigma_prior = 1 / np.sqrt(2 * lambda_l2)
print(f"L2 正则化 (λ={lambda_l2})")
print(f"  等价于高斯先验: θ ~ N(0, {sigma_prior**2:.2f})")

# L1 正则化 = 拉普拉斯先验
lambda_l1 = 0.01
b_prior = 1 / lambda_l1
print(f"\nL1 正则化 (λ={lambda_l1})")
print(f"  等价于拉普拉斯先验: θ ~ Laplace(0, {b_prior:.2f})")

# 可视化
import matplotlib.pyplot as plt

theta = np.linspace(-3, 3, 1000)

gaussian_prior = np.exp(-lambda_l2 * theta**2) / np.sqrt(np.pi / lambda_l2)
laplace_prior = np.exp(-lambda_l1 * np.abs(theta)) / (2 / lambda_l1)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(theta, gaussian_prior, label=f'高斯先验 (L2, λ={lambda_l2})')
ax.plot(theta, laplace_prior, label=f'拉普拉斯先验 (L1, λ={lambda_l1})')
ax.set_xlabel('θ')
ax.set_ylabel('概率密度')
ax.set_title('正则化对应的先验分布')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regularization_prior.png', dpi=100)
print("\n图像已保存: regularization_prior.png")
```

---

## 小结

本章介绍了参数估计的核心方法，特别是最大似然估计及其与深度学习损失函数的关系。

### 方法比较

| 方法 | 思想 | 优点 | 缺点 |
|------|------|------|------|
| 矩估计 | 样本矩 ≈ 总体矩 | 简单、直观 | 可能不唯一、效率低 |
| MLE | 最大化似然 | 渐进有效、一致 | 可能无解析解 |
| 贝叶斯 | 结合先验信息 | 不确定性量化 | 需要选择先验 |

### 核心公式

| 公式 | 应用 |
|------|------|
| $L(\theta) = \prod f(x_i; \theta)$ | 似然函数 |
| $\ell(\theta) = \sum \ln f(x_i; \theta)$ | 对数似然 |
| $\text{MSE} = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | 回归损失（高斯 MLE） |
| $\text{CE} = -\sum y_i \ln \hat{y}_i$ | 分类损失（类别 MLE） |

### 关键要点

1. **MLE 是最常用的点估计方法**
2. **最小化损失 = 最大化似然**
3. **正则化 = 对数先验**
4. **贝叶斯方法提供完整的不确定性量化**

---

**上一节**：[第四章（a）：统计量与抽样分布](04a-统计量与抽样分布.md)

**下一节**：[第四章（c）：假设检验](04c-假设检验.md) - 学习假设检验、p 值、置信区间等概念。

**返回**：[数学基础教程目录](../math-fundamentals.md)
