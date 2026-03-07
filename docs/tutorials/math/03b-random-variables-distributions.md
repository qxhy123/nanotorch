# 第三章（b）：随机变量与常见分布

随机变量是概率论的核心概念，它将随机试验的结果数量化，使得我们可以用数学工具分析随机现象。本章将深入讲解离散和连续随机变量，以及在深度学习中广泛使用的各种概率分布。

---

## 目录

1. [随机变量的定义](#随机变量的定义)
2. [离散随机变量](#离散随机变量)
3. [连续随机变量](#连续随机变量)
4. [累积分布函数](#累积分布函数)
5. [离散概率分布](#离散概率分布)
   - [伯努利分布](#伯努利分布-bernoulli)
   - [二项分布](#二项分布-binomial)
   - [泊松分布](#泊松分布-poisson)
   - [类别分布](#类别分布-categorical)
6. [连续概率分布](#连续概率分布)
   - [均匀分布](#均匀分布-uniform)
   - [正态分布](#正态分布-normalgaussian)
   - [指数分布](#指数分布-exponential)
   - [拉普拉斯分布](#拉普拉斯分布-laplace)
   - [Beta 分布](#beta-分布)
   - [Gamma 分布](#gamma-分布)
7. [分布之间的关系](#分布之间的关系)
8. [在深度学习中的应用](#在深度学习中的应用)
9. [小结](#小结)

---

## 随机变量的定义

### 🎯 生活类比：把不确定的事变成数字

随机变量就是**把随机事件的结果变成数字**：

| 随机事件 | 原始结果 | 随机变量（数字化） |
|----------|----------|-------------------|
| 抛硬币 | 正面/反面 | 1/0 |
| 天气 | 晴/阴/雨 | 0/1/2 |
| 考试 | 及格/不及格 | 1/0 |
| 明天温度 | 20-30度 | 25.3（具体数值） |

**随机变量的本质**：用数字来描述不确定的结果，这样我们就能用数学来分析它了！

### 📖 为什么需要随机变量？

- **原始问题**："明天会下雨吗？" → 很难用数学分析
- **变成随机变量**："设X=1表示下雨，X=0表示不下雨" → 可以计算概率、期望、方差

**把"文字描述"变成"数字游戏"**。

### 基本定义

**随机变量**是定义在样本空间 $\Omega$ 上的实值函数：

$$
X: \Omega \to \mathbb{R}
$$

**通俗翻译**：随机变量就是一条规则，把每个可能的结果对应到一个数字。

即对每个样本点 $\omega \in \Omega$，都有唯一的实数 $X(\omega)$ 与之对应。

### 直观理解

随机变量是将**随机现象的结果**映射为**数值**的规则：

| 随机试验 | 样本空间 $\Omega$ | 随机变量 $X$ |
|----------|------------------|--------------|
| 掷骰子 | $\{1,2,3,4,5,6\}$ | 点数本身 |
| 抛硬币 | $\{正面, 反面\}$ | 正面=1, 反面=0 |
| 测量身高 | $(0, \infty)$ | 身高值（cm） |
| 图像分类 | 所有图像 | 类别标签（0-9） |

### 随机变量的类型

| 类型 | 取值特点 | 示例 |
|------|----------|------|
| **离散型** | 有限或可数无限个值 | 骰子点数、单词数 |
| **连续型** | 连续区间内的值 | 身高、温度、权重 |

```python
import numpy as np
import matplotlib.pyplot as plt

# 离散随机变量示例：掷骰子
def dice_random_variable(outcome):
    """将骰子结果映射为数值"""
    return outcome

# 连续随机变量示例：测量误差
def measurement_error():
    """测量误差服从正态分布"""
    return np.random.normal(0, 1)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 离散型
outcomes = [1, 2, 3, 4, 5, 6]
axes[0].bar(outcomes, [1/6]*6, alpha=0.7)
axes[0].set_title('离散随机变量：掷骰子')
axes[0].set_xlabel('X (点数)')
axes[0].set_ylabel('P(X)')

# 连续型
x = np.linspace(-4, 4, 100)
pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)
axes[1].plot(x, pdf, 'b-', linewidth=2)
axes[1].fill_between(x, pdf, alpha=0.3)
axes[1].set_title('连续随机变量：测量误差')
axes[1].set_xlabel('X (误差)')
axes[1].set_ylabel('f(x)')

plt.tight_layout()
plt.savefig('random_variables.png', dpi=100)
print("图像已保存")
```

---

## 离散随机变量

### 定义

离散随机变量的取值为**有限个**或**可数无限个**值 $x_1, x_2, x_3, \ldots$

### 概率质量函数 (PMF)

**概率质量函数**描述离散随机变量取各个值的概率：

$$
p(x) = P(X = x)
$$

**性质**：
1. **非负性**：$p(x) \geq 0$ 对所有 $x$
2. **归一化**：$\displaystyle\sum_x p(x) = 1$
3. **概率计算**：$P(X \in A) = \displaystyle\sum_{x \in A} p(x)$

### 期望（均值）

#### 🎯 生活类比：赌博游戏的"公平价格"

假设一个游戏：50%赢10元，50%赢30元。这个游戏的"平均收益"是多少？

$$E[X] = 0.5 \times 10 + 0.5 \times 30 = 5 + 15 = 20 \text{元}$$

**期望 = 长期平均值**。如果你玩1000次，平均每次收益约20元。

| 场景 | 期望的含义 |
|------|-----------|
| 股票投资 | 预期收益率 |
| 考试成绩 | 平均分 |
| 商品销量 | 平均日销量 |
| 游戏伤害 | 平均伤害值 |

#### 📝 手把手计算

掷一个均匀骰子，求期望。

| 结果 $x$ | 概率 $p(x)$ | $x \times p(x)$ |
|----------|-------------|-----------------|
| 1 | 1/6 | 1/6 |
| 2 | 1/6 | 2/6 |
| 3 | 1/6 | 3/6 |
| 4 | 1/6 | 4/6 |
| 5 | 1/6 | 5/6 |
| 6 | 1/6 | 6/6 |
| **总和** | 1 | **21/6 = 3.5** |

**期望 = 3.5**

**注意**：期望不一定是可能的结果（你永远掷不出3.5点）！

离散随机变量的期望：

$$
\mathbb{E}[X] = \sum_x x \cdot p(x)
$$

**通俗翻译**：期望 = 每个结果 × 它的概率，然后全部加起来。

**期望的性质**：
- **线性**：$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$
- **常数**：$\mathbb{E}[c] = c$
- **函数期望**：$\mathbb{E}[g(X)] = \sum_x g(x) \cdot p(x)$

### 方差

#### 🎯 生活类比：股票 vs 存款

| 投资方式 | 期望收益 | 方差（风险） |
|----------|----------|-------------|
| 银行存款 | 3% | 很小（稳定） |
| 股票 | 10% | 很大（波动） |

**方差 = 波动程度/风险大小**

- 高方差：结果很不确定（可能大赚也可能大亏）
- 低方差：结果比较稳定（总是在期望附近）

#### 📝 手把手计算

掷骰子的方差（期望=3.5）：

| 结果 $x$ | $(x - 3.5)^2$ | 概率 | 贡献 |
|----------|---------------|------|------|
| 1 | 6.25 | 1/6 | 1.04 |
| 2 | 2.25 | 1/6 | 0.375 |
| 3 | 0.25 | 1/6 | 0.042 |
| 4 | 0.25 | 1/6 | 0.042 |
| 5 | 2.25 | 1/6 | 0.375 |
| 6 | 6.25 | 1/6 | 1.04 |
| **方差** | | | **2.92** |

**标准差 = $\sqrt{2.92} \approx 1.71$**

$$
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

**通俗翻译**：方差 = 每个结果与期望的差的平方，再取平均。

**方差的性质**：
- $\text{Var}(X) \geq 0$
- $\text{Var}(c) = 0$
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$

```python
import numpy as np

# 离散随机变量示例：不均匀骰子
# PMF: P(1)=0.1, P(2)=0.1, P(3)=0.2, P(4)=0.2, P(5)=0.2, P(6)=0.2

x_values = np.array([1, 2, 3, 4, 5, 6])
pmf = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])

# 验证归一化
print(f"概率之和: {pmf.sum():.2f}")

# 计算期望
expected_value = np.sum(x_values * pmf)
print(f"期望 E[X] = {expected_value:.2f}")

# 计算方差
expected_x_squared = np.sum(x_values**2 * pmf)
variance = expected_x_squared - expected_value**2
print(f"方差 Var(X) = {variance:.2f}")

# 标准差
std_dev = np.sqrt(variance)
print(f"标准差 σ = {std_dev:.2f}")

# 模拟验证
n_samples = 100000
samples = np.random.choice(x_values, size=n_samples, p=pmf)
print(f"\n模拟结果:")
print(f"样本均值: {samples.mean():.4f}")
print(f"样本方差: {samples.var():.4f}")
```

---

## 连续随机变量

### 定义

连续随机变量的取值充满一个或多个**连续区间**。

### 概率密度函数 (PDF)

**概率密度函数** $f(x)$ 描述连续随机变量的分布：

$$
P(a \leq X \leq b) = \int_a^b f(x) \, dx
$$

**性质**：
1. **非负性**：$f(x) \geq 0$ 对所有 $x$
2. **归一化**：$\displaystyle\int_{-\infty}^{+\infty} f(x) \, dx = 1$
3. **单点概率为零**：$P(X = x) = 0$（对任意单点 $x$）

### ⚠️ 重要注意

$f(x)$ **本身不是概率**！它是"概率密度"，可以大于1。

- 对于连续变量，概率是 PDF 曲线下的**面积**
- $f(x) = 2$ 在 $[0, 0.5]$ 上是合法的（积分为1）

### 期望与方差

**期望**：

$$
\mathbb{E}[X] = \int_{-\infty}^{+\infty} x \cdot f(x) \, dx
$$

**方差**：

$$
\text{Var}(X) = \int_{-\infty}^{+\infty} (x - \mu)^2 f(x) \, dx = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

```python
import numpy as np
from scipy import integrate

# 连续随机变量示例：自定义 PDF
def custom_pdf(x):
    """三角形分布: f(x) = 2x for x in [0, 1]"""
    if 0 <= x <= 1:
        return 2 * x
    return 0

# 向量化
custom_pdf_vec = np.vectorize(custom_pdf)

# 验证归一化
integral, _ = integrate.quad(custom_pdf, -np.inf, np.inf)
print(f"PDF 积分 = {integral:.4f}")

# 计算期望
def x_times_pdf(x):
    return x * custom_pdf(x)

expected, _ = integrate.quad(x_times_pdf, 0, 1)
print(f"期望 E[X] = {expected:.4f}")

# 计算方差
def x_squared_times_pdf(x):
    return x**2 * custom_pdf(x)

expected_x2, _ = integrate.quad(x_squared_times_pdf, 0, 1)
variance = expected_x2 - expected**2
print(f"方差 Var(X) = {variance:.4f}")

# 可视化
x = np.linspace(-0.2, 1.2, 200)
pdf_values = custom_pdf_vec(x)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(x, pdf_values, 'b-', linewidth=2, label='PDF: f(x) = 2x')
plt.fill_between(x[x >= 0], pdf_values[x >= 0], alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('连续随机变量的 PDF')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('continuous_pdf.png', dpi=100)
print("\nPDF 图像已保存")
```

---

## 累积分布函数

### 定义

**累积分布函数**（CDF）描述随机变量小于等于某个值的概率：

$$
F(x) = P(X \leq x)
$$

### 离散情况

$$
F(x) = \sum_{x_i \leq x} p(x_i)
$$

### 连续情况

$$
F(x) = \int_{-\infty}^x f(t) \, dt
$$

### CDF 的性质

1. **范围**：$0 \leq F(x) \leq 1$
2. **边界**：$\lim_{x \to -\infty} F(x) = 0$，$\lim_{x \to +\infty} F(x) = 1$
3. **单调性**：单调不减（$x_1 < x_2 \Rightarrow F(x_1) \leq F(x_2)$）
4. **右连续**：$\lim_{t \to x^+} F(t) = F(x)$

### PDF 与 CDF 的关系

$$
F(x) = \int_{-\infty}^x f(t) \, dt \quad \Longleftrightarrow \quad f(x) = F'(x) = \frac{dF}{dx}
$$

### 概率计算

对于连续随机变量：

$$
P(a < X \leq b) = F(b) - F(a)
$$

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 标准正态分布的 CDF
x = np.linspace(-4, 4, 200)
pdf = stats.norm.pdf(x)  # 概率密度函数
cdf = stats.norm.cdf(x)  # 累积分布函数

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# PDF
axes[0].plot(x, pdf, 'b-', linewidth=2)
axes[0].fill_between(x, pdf, alpha=0.3)
axes[0].set_title('PDF: f(x)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].grid(True, alpha=0.3)

# CDF
axes[1].plot(x, cdf, 'r-', linewidth=2)
axes[1].set_title('CDF: F(x) = P(X ≤ x)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('F(x)')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.05, 1.05)

# CDF 和 PDF 的关系
axes[2].plot(x, pdf, 'b-', linewidth=2, label='PDF')
x_shade = x[(x >= -1) & (x <= 1)]
axes[2].fill_between(x_shade, stats.norm.pdf(x_shade), alpha=0.3, color='blue')
axes[2].axvline(x=-1, color='r', linestyle='--', label=f'F(-1)={stats.norm.cdf(-1):.3f}')
axes[2].axvline(x=1, color='g', linestyle='--', label=f'F(1)={stats.norm.cdf(1):.3f}')
axes[2].set_title('P(-1 ≤ X ≤ 1) = F(1) - F(-1)')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pdf_cdf.png', dpi=100)
print("PDF 和 CDF 图像已保存")

# 使用 CDF 计算概率
print(f"\n使用 CDF 计算概率:")
print(f"P(X ≤ 0) = F(0) = {stats.norm.cdf(0):.4f}")
print(f"P(X > 1) = 1 - F(1) = {1 - stats.norm.cdf(1):.4f}")
print(f"P(-1 < X ≤ 1) = F(1) - F(-1) = {stats.norm.cdf(1) - stats.norm.cdf(-1):.4f}")
```

---

## 离散概率分布

### 伯努利分布 (Bernoulli)

#### 定义

**伯努利分布**描述单次二元试验（成功/失败）的结果。

$$
X \sim \text{Bernoulli}(p)
$$

#### PMF

$$
P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}
$$

或等价地：
- $P(X = 1) = p$（成功）
- $P(X = 0) = 1 - p$（失败）

#### 期望与方差

$$
\mathbb{E}[X] = p
$$

$$
\text{Var}(X) = p(1 - p)
$$

#### 深度学习应用

- **二元分类**：输出层的 Sigmoid + 阈值
- **Dropout**：神经元保留的伯努利采样

```python
import numpy as np
from scipy import stats

# 伯努利分布
p = 0.7  # 成功概率

# PMF
print("伯努利分布 PMF:")
print(f"P(X=0) = {1 - p:.2f}")
print(f"P(X=1) = {p:.2f}")

# 使用 scipy
bernoulli = stats.bernoulli(p)
print(f"\n理论期望: {bernoulli.mean():.2f}")
print(f"理论方差: {bernoulli.var():.4f}")

# 模拟
samples = bernoulli.rvs(size=10000)
print(f"\n模拟期望: {samples.mean():.4f}")
print(f"模拟方差: {samples.var():.4f}")

# 深度学习示例：Dropout mask
def bernoulli_dropout(shape, p=0.5):
    """生成 Dropout mask"""
    return (np.random.random(shape) < p).astype(float) / p

mask = bernoulli_dropout((5, 5), p=0.5)
print(f"\nDropout mask (p=0.5):")
print(mask)
```

---

### 二项分布 (Binomial)

#### 定义

**二项分布**描述 $n$ 次独立伯努利试验中成功的次数。

$$
X \sim \text{Binomial}(n, p)
$$

#### PMF

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, 2, \ldots, n
$$

其中 $\displaystyle\binom{n}{k} = \frac{n!}{k!(n-k)!}$ 是二项式系数。

#### 期望与方差

$$
\mathbb{E}[X] = np
$$

$$
\text{Var}(X) = np(1-p)
$$

#### 与伯努利分布的关系

若 $X_1, X_2, \ldots, X_n \stackrel{\text{iid}}{\sim} \text{Bernoulli}(p)$，则：

$$
\sum_{i=1}^n X_i \sim \text{Binomial}(n, p)
$$

#### 深度学习应用

- **集成学习**：多次独立预测的成功次数
- **数据增强**：随机变换的成功/失败计数

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 二项分布参数
n, p = 20, 0.3

# PMF
k_values = np.arange(0, n + 1)
pmf = stats.binom.pmf(k_values, n, p)

print(f"二项分布 B({n}, {p}):")
print(f"理论期望: {n * p:.2f}")
print(f"理论方差: {n * p * (1-p):.2f}")

# P(X = 6)
print(f"\nP(X = 6) = {stats.binom.pmf(6, n, p):.4f}")
# P(X <= 10)
print(f"P(X ≤ 10) = {stats.binom.cdf(10, n, p):.4f}")

# 可视化 PMF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(k_values, pmf, alpha=0.7, color='steelblue')
axes[0].axvline(x=n*p, color='r', linestyle='--', label=f'期望 = {n*p:.1f}')
axes[0].set_xlabel('k (成功次数)')
axes[0].set_ylabel('P(X = k)')
axes[0].set_title(f'二项分布 PMF: B({n}, {p})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 不同参数的对比
for p_val in [0.2, 0.5, 0.8]:
    axes[1].plot(k_values, stats.binom.pmf(k_values, n, p_val), 
                 'o-', label=f'p = {p_val}', alpha=0.7)
axes[1].set_xlabel('k')
axes[1].set_ylabel('P(X = k)')
axes[1].set_title(f'不同 p 值的二项分布 (n={n})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('binomial.png', dpi=100)
print("\n二项分布图像已保存")

# 模拟
samples = np.random.binomial(n, p, size=10000)
print(f"\n模拟结果:")
print(f"样本均值: {samples.mean():.4f}")
print(f"样本方差: {samples.var():.4f}")
```

---

### 泊松分布 (Poisson)

#### 定义

**泊松分布**描述单位时间/空间内**稀有事件**发生的次数。

$$
X \sim \text{Poisson}(\lambda)
$$

#### PMF

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots
$$

#### 期望与方差

$$
\mathbb{E}[X] = \text{Var}(X) = \lambda
$$

**特点**：期望等于方差！

#### 泊松定理

当 $n \to \infty$，$p \to 0$，$np = \lambda$（常数）时：

$$
\text{Binomial}(n, p) \approx \text{Poisson}(\lambda)
$$

#### 深度学习应用

- **稀疏编码**：神经元激活的稀疏性建模
- **计数数据**：文本中词频、推荐系统中的点击数

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 泊松分布
lambdas = [1, 4, 10]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# PMF 对比
k = np.arange(0, 25)
for lam in lambdas:
    pmf = stats.poisson.pmf(k, lam)
    axes[0].plot(k, pmf, 'o-', label=f'λ = {lam}', alpha=0.7)
    
axes[0].set_xlabel('k (事件次数)')
axes[0].set_ylabel('P(X = k)')
axes[0].set_title('泊松分布 PMF')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 二项分布近似泊松分布
n, p = 1000, 0.01
lambda_approx = n * p  # = 10

k = np.arange(0, 25)
binom_pmf = stats.binom.pmf(k, n, p)
poisson_pmf = stats.poisson.pmf(k, lambda_approx)

axes[1].plot(k, binom_pmf, 'o-', label=f'Binom({n}, {p})', alpha=0.7)
axes[1].plot(k, poisson_pmf, 's--', label=f'Poisson({lambda_approx})', alpha=0.7)
axes[1].set_xlabel('k')
axes[1].set_ylabel('P(X = k)')
axes[1].set_title('二项分布 → 泊松分布近似')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('poisson.png', dpi=100)
print("泊松分布图像已保存")

# 应用示例：神经元稀疏激活
print("\n神经元稀疏激活模拟:")
lam = 3  # 平均激活 3 个神经元
n_neurons = 100
n_samples = 10

activations = np.random.poisson(lam, (n_samples, n_neurons))
print(f"激活矩阵形状: {activations.shape}")
print(f"平均激活数: {activations.sum(axis=1).mean():.2f}")
print(f"激活稀疏度: {(activations == 0).mean():.2%}")
```

---

### 类别分布 (Categorical)

#### 定义

**类别分布**（也叫 Multinoulli）描述从 $K$ 个类别中选择一个的概率。

$$
X \sim \text{Categorical}(p_1, p_2, \ldots, p_K)
$$

#### PMF

$$
P(X = i) = p_i, \quad \sum_{i=1}^K p_i = 1
$$

#### 表示方式

通常用 **one-hot 编码**表示：

$$
\mathbf{x} \in \{0, 1\}^K, \quad \|\mathbf{x}\|_1 = 1
$$

#### 与 Softmax 的关系

Softmax 输出就是类别分布的概率向量：

$$
P(y = i | \mathbf{z}) = \text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

#### 深度学习应用

- **多分类问题**：图像分类、文本分类
- **语言模型**：下一个词预测（词汇表大小 = K）

```python
import numpy as np

# 类别分布
probabilities = np.array([0.1, 0.2, 0.3, 0.15, 0.25])  # 5 个类别
K = len(probabilities)

print(f"类别数: {K}")
print(f"概率分布: {probabilities}")
print(f"概率之和: {probabilities.sum():.2f}")

# 采样
n_samples = 1000
samples = np.random.choice(K, size=n_samples, p=probabilities)

# 统计频率
counts = np.bincount(samples, minlength=K)
print(f"\n采样频率: {counts / n_samples}")

# One-hot 编码
def one_hot(indices, K):
    """将索引转换为 one-hot 编码"""
    n = len(indices)
    one_hot_matrix = np.zeros((n, K))
    one_hot_matrix[np.arange(n), indices] = 1
    return one_hot_matrix

one_hot_samples = one_hot(samples[:5], K)
print(f"\n前 5 个样本的 one-hot 编码:")
print(one_hot_samples)

# Softmax 生成类别分布
def softmax(logits):
    """将 logits 转换为概率分布"""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

logits = np.array([1.0, 2.0, 3.0, 0.5, 1.5])
probs = softmax(logits)
print(f"\nLogits: {logits}")
print(f"Softmax 概率: {probs}")
print(f"概率之和: {probs.sum():.6f}")
```

---

## 连续概率分布

### 均匀分布 (Uniform)

#### PDF

$$
f(x) = \begin{cases} \frac{1}{b-a} & x \in [a, b] \\ 0 & \text{otherwise} \end{cases}
$$

记作 $X \sim \text{Uniform}(a, b)$ 或 $X \sim U(a, b)$。

#### 期望与方差

$$
\mathbb{E}[X] = \frac{a + b}{2}
$$

$$
\text{Var}(X) = \frac{(b - a)^2}{12}
$$

#### 深度学习应用

- **参数初始化**：Xavier 初始化使用均匀分布
- **数据增强**：随机裁剪位置
- **超参数搜索**：均匀采样搜索空间

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 均匀分布
a, b = 0, 10

# PDF 和 CDF
x = np.linspace(-2, 12, 200)
pdf = stats.uniform.pdf(x, loc=a, scale=b-a)
cdf = stats.uniform.cdf(x, loc=a, scale=b-a)

print(f"均匀分布 U({a}, {b}):")
print(f"期望: {(a + b) / 2:.2f}")
print(f"方差: {(b - a)**2 / 12:.2f}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x, pdf, 'b-', linewidth=2)
axes[0].fill_between(x[(x >= a) & (x <= b)], pdf[(x >= a) & (x <= b)], alpha=0.3)
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title(f'均匀分布 PDF: U({a}, {b})')
axes[0].grid(True, alpha=0.3)

axes[1].plot(x, cdf, 'r-', linewidth=2)
axes[1].set_xlabel('x')
axes[1].set_ylabel('F(x)')
axes[1].set_title(f'均匀分布 CDF: U({a}, {b})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('uniform.png', dpi=100)
print("\n均匀分布图像已保存")

# 深度学习应用：Xavier 初始化
def xavier_uniform_init(fan_in, fan_out):
    """Xavier 均匀初始化"""
    bound = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-bound, bound, (fan_out, fan_in))

# 示例：初始化 Linear 层权重
fan_in, fan_out = 784, 256
weights = xavier_uniform_init(fan_in, fan_out)
print(f"\nXavier 初始化权重:")
print(f"形状: {weights.shape}")
print(f"理论边界: ±{np.sqrt(6 / (fan_in + fan_out)):.4f}")
print(f"实际范围: [{weights.min():.4f}, {weights.max():.4f}]")
print(f"实际均值: {weights.mean():.4f}")
print(f"实际方差: {weights.var():.6f}")
```

---

### 正态分布 (Normal/Gaussian)

#### PDF

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

记作 $X \sim \mathcal{N}(\mu, \sigma^2)$。

#### 期望与方差

$$
\mathbb{E}[X] = \mu, \quad \text{Var}(X) = \sigma^2
$$

#### 标准正态分布

当 $\mu = 0$，$\sigma = 1$ 时，称为**标准正态分布**，记作 $Z \sim \mathcal{N}(0, 1)$：

$$
\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}
$$

### 正态分布归一化常数的推导

**问题**：证明 $\displaystyle\int_{-\infty}^{\infty} e^{-x^2/2} dx = \sqrt{2\pi}$

**证明（使用极坐标变换）**：

**第一步**：设 $I = \displaystyle\int_{-\infty}^{\infty} e^{-x^2/2} dx$，考虑 $I^2$。

$$I^2 = \left(\int_{-\infty}^{\infty} e^{-x^2/2} dx\right) \left(\int_{-\infty}^{\infty} e^{-y^2/2} dy\right) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} e^{-(x^2+y^2)/2} dx\, dy$$

**第二步**：转换为极坐标。

令 $x = r\cos\theta$，$y = r\sin\theta$，则 $x^2 + y^2 = r^2$，$dx\, dy = r\, dr\, d\theta$。

积分区域：$r \in [0, \infty)$，$\theta \in [0, 2\pi)$。

$$I^2 = \int_{0}^{2\pi} \int_{0}^{\infty} e^{-r^2/2} \cdot r\, dr\, d\theta$$

**第三步**：分离变量并计算。

$$I^2 = \left(\int_{0}^{2\pi} d\theta\right) \left(\int_{0}^{\infty} r e^{-r^2/2} dr\right) = 2\pi \cdot \int_{0}^{\infty} r e^{-r^2/2} dr$$

**第四步**：计算内层积分（换元法）。

令 $u = r^2/2$，则 $du = r\, dr$：

$$\int_{0}^{\infty} r e^{-r^2/2} dr = \int_{0}^{\infty} e^{-u} du = \left[-e^{-u}\right]_{0}^{\infty} = 0 - (-1) = 1$$

**第五步**：得出结果。

$$I^2 = 2\pi \cdot 1 = 2\pi$$

$$I = \sqrt{2\pi}$$

因此，正态分布的归一化常数为 $\dfrac{1}{\sqrt{2\pi}}$：

$$\boxed{\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}} e^{-x^2/2} dx = 1}$$

**更一般的情况**：

对于 $\mathcal{N}(\mu, \sigma^2)$，通过换元 $z = \dfrac{x-\mu}{\sigma}$：

$$\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} dx = \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}} e^{-z^2/2} dz = 1$$

#### 标准化

任意正态分布可标准化：

$$
Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)
$$

#### 经验法则（68-95-99.7 规则）

- $P(\mu - \sigma \leq X \leq \mu + \sigma) \approx 68.27\%$
- $P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) \approx 95.45\%$
- $P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) \approx 99.73\%$

#### 深度学习应用

- **权重初始化**：He 初始化、截断正态
- **噪声注入**：数据增强、变分推断
- **VAE**：潜在空间分布

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 正态分布
mu, sigma = 0, 1
x = np.linspace(-4, 4, 200)
pdf = stats.norm.pdf(x, mu, sigma)
cdf = stats.norm.cdf(x, mu, sigma)

print(f"正态分布 N({mu}, {sigma**2}):")
print(f"68.27% 区间: [{mu-sigma:.2f}, {mu+sigma:.2f}]")
print(f"95.45% 区间: [{mu-2*sigma:.2f}, {mu+2*sigma:.2f}]")
print(f"99.73% 区间: [{mu-3*sigma:.2f}, {mu+3*sigma:.2f}]")

# 验证
print(f"\n验证:")
print(f"P(-1 ≤ X ≤ 1) = {stats.norm.cdf(1) - stats.norm.cdf(-1):.4f}")
print(f"P(-2 ≤ X ≤ 2) = {stats.norm.cdf(2) - stats.norm.cdf(-2):.4f}")
print(f"P(-3 ≤ X ≤ 3) = {stats.norm.cdf(3) - stats.norm.cdf(-3):.4f}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 不同参数的正态分布
for mu_val, sigma_val in [(0, 1), (0, 2), (2, 1)]:
    pdf = stats.norm.pdf(x, mu_val, sigma_val)
    label = f'μ={mu_val}, σ={sigma_val}'
    axes[0].plot(x, pdf, label=label, linewidth=2)

axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title('正态分布 PDF')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 68-95-99.7 规则可视化
x_fill = np.linspace(-3, 3, 100)
pdf_standard = stats.norm.pdf(x_fill, 0, 1)

axes[1].plot(x_fill, pdf_standard, 'b-', linewidth=2)
axes[1].fill_between(x_fill[(x_fill >= -1) & (x_fill <= 1)], 
                     pdf_standard[(x_fill >= -1) & (x_fill <= 1)], 
                     alpha=0.3, label='68.27%')
axes[1].fill_between(x_fill[(x_fill >= -2) & (x_fill <= 2)], 
                     pdf_standard[(x_fill >= -2) & (x_fill <= 2)], 
                     alpha=0.2, label='95.45%')
axes[1].fill_between(x_fill, pdf_standard, alpha=0.1, label='99.73%')
axes[1].set_xlabel('x')
axes[1].set_ylabel('f(x)')
axes[1].set_title('68-95-99.7 规则')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('normal.png', dpi=100)
print("\n正态分布图像已保存")

# He 初始化
def he_normal_init(fan_in, fan_out):
    """He 正态初始化（用于 ReLU 激活）"""
    std = np.sqrt(2 / fan_in)
    return np.random.normal(0, std, (fan_out, fan_in))

weights = he_normal_init(784, 256)
print(f"\nHe 初始化权重:")
print(f"形状: {weights.shape}")
print(f"理论标准差: {np.sqrt(2/784):.4f}")
print(f"实际标准差: {weights.std():.4f}")
```

---

### 指数分布 (Exponential)

#### PDF

$$
f(x) = \lambda e^{-\lambda x}, \quad x \geq 0
$$

记作 $X \sim \text{Exp}(\lambda)$。

#### 期望与方差

$$
\mathbb{E}[X] = \frac{1}{\lambda}
$$

$$
\text{Var}(X) = \frac{1}{\lambda^2}
$$

#### 无记忆性（关键性质）

$$
P(X > s + t | X > s) = P(X > t)
$$

"已经等待了 s 时间，再等 t 时间的概率，与从头等 t 时间相同"。

#### 与泊松分布的关系

如果事件以速率 $\lambda$ 泊松到达，则**等待时间**服从 $\text{Exp}(\lambda)$。

#### 深度学习应用

- **Dropout 时间**：随机失活的间隔
- **正则化**：L2 正则化可视为参数的指数先验

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 指数分布
lambdas = [0.5, 1, 2]
x = np.linspace(0, 5, 200)

plt.figure(figsize=(10, 4))

for lam in lambdas:
    pdf = stats.expon.pdf(x, scale=1/lam)  # scipy 用 β=1/λ 作为 scale
    plt.plot(x, pdf, label=f'λ = {lam} (E[X] = {1/lam:.2f})', linewidth=2)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('指数分布 PDF')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('exponential.png', dpi=100)
print("指数分布图像已保存")

# 验证无记忆性
lam = 1
print(f"\n验证无记忆性 (λ = {lam}):")
s, t = 2, 1

# P(X > s + t | X > s) = P(X > s + t) / P(X > s)
p_s_plus_t = stats.expon.sf(s + t, scale=1/lam)
p_s = stats.expon.sf(s, scale=1/lam)
conditional = p_s_plus_t / p_s

# P(X > t)
p_t = stats.expon.sf(t, scale=1/lam)

print(f"P(X > {s+t} | X > {s}) = {conditional:.4f}")
print(f"P(X > {t}) = {p_t:.4f}")
print(f"相等? {np.isclose(conditional, p_t)}")

# 模拟验证无记忆性
samples = np.random.exponential(1/lam, 100000)
# 条件: X > s
filtered = samples[samples > s]
conditional_sim = (filtered > s + t).mean()
print(f"\n模拟 P(X > {s+t} | X > {s}) = {conditional_sim:.4f}")
```

---

### 拉普拉斯分布 (Laplace)

#### PDF

$$
f(x) = \frac{1}{2b} \exp\left(-\frac{|x - \mu|}{b}\right)
$$

记作 $X \sim \text{Laplace}(\mu, b)$。

#### 期望与方差

$$
\mathbb{E}[X] = \mu
$$

$$
\text{Var}(X) = 2b^2
$$

#### 与正态分布的区别

- 正态分布：$(x - \mu)^2$ → 平滑的峰值
- 拉普拉斯分布：$|x - \mu|$ → 尖锐的峰值，更厚的尾部

#### 深度学习应用

- **L1 正则化**：拉普拉斯先验导致 L1 惩罚
- **稀疏建模**：比高斯更倾向于产生稀疏解
- **异常检测**：厚尾特性对异常值更鲁棒

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 对比拉普拉斯和正态分布
x = np.linspace(-5, 5, 200)

# 拉普拉斯分布 (μ=0, b=1)
laplace_pdf = stats.laplace.pdf(x, loc=0, scale=1)

# 正态分布 (μ=0, σ=1)
# 使得方差相同: Var(Laplace) = 2b² = 2, Var(Normal) = σ² = 2 → σ = √2
normal_pdf = stats.norm.pdf(x, loc=0, scale=np.sqrt(1))

plt.figure(figsize=(10, 5))
plt.plot(x, laplace_pdf, 'b-', linewidth=2, label='Laplace(0, 1)')
plt.plot(x, normal_pdf, 'r-', linewidth=2, label='Normal(0, 1)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('拉普拉斯分布 vs 正态分布')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('laplace_vs_normal.png', dpi=100)
print("拉普拉斯与正态分布对比图像已保存")

# L1 正则化 = 拉普拉斯先验的 MAP
print("\nL1 正则化的贝叶斯解释:")
print("先验: θ ~ Laplace(0, b)")
print("对数先验: log p(θ) = -|θ|/b + const")
print("MAP: min -log p(y|x,θ) - log p(θ)")
print("    = min Loss + λ|θ|  (L1 正则化)")
```

---

### Beta 分布

#### PDF

$$
f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad x \in [0, 1]
$$

其中 $B(\alpha, \beta) = \dfrac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ 是 Beta 函数。

#### 期望与方差

$$
\mathbb{E}[X] = \frac{\alpha}{\alpha + \beta}
$$

$$
\text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}
$$

#### 形状特性

| 参数 | 形状 |
|------|------|
| $\alpha = \beta = 1$ | 均匀分布 |
| $\alpha = \beta > 1$ | 对称，中心峰值 |
| $\alpha = \beta < 1$ | 对称，U 形 |
| $\alpha > \beta$ | 右偏 |
| $\alpha < \beta$ | 左偏 |

#### 深度学习应用

- **贝叶斯推断**：伯努利/二项分布的**共轭先验**
- **超参数调优**：学习率、Dropout 率的不确定性建模
- **强化学习**：Bandit 问题中的 Thompson Sampling

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Beta 分布的不同形状
params = [
    (1, 1, 'Uniform'),
    (2, 2, 'Symmetric peak'),
    (0.5, 0.5, 'U-shape'),
    (2, 5, 'Left-skewed'),
    (5, 2, 'Right-skewed'),
]

x = np.linspace(0, 1, 200)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for i, (alpha, beta, name) in enumerate(params):
    pdf = stats.beta.pdf(x, alpha, beta)
    axes[i].plot(x, pdf, 'b-', linewidth=2)
    axes[i].fill_between(x, pdf, alpha=0.3)
    axes[i].set_title(f'Beta({alpha}, {beta}) - {name}\nE[X] = {alpha/(alpha+beta):.2f}')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('f(x)')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim(0, None)

# 删除多余的子图
axes[5].axis('off')

plt.tight_layout()
plt.savefig('beta.png', dpi=100)
print("Beta 分布图像已保存")

# 贝叶斯更新示例：投币实验
print("\n贝叶斯更新: 投币实验")
print("="*50)

# 先验: Beta(1, 1) = 均匀分布
alpha_prior, beta_prior = 1, 1

# 观察数据: 7 正面, 3 反面
heads, tails = 7, 3

# 后验: Beta(α + heads, β + tails)
alpha_post = alpha_prior + heads
beta_post = beta_prior + tails

print(f"先验: Beta({alpha_prior}, {beta_prior})")
print(f"数据: {heads} 正面, {tails} 反面")
print(f"后验: Beta({alpha_post}, {beta_post})")
print(f"后验均值 (硬币公平性估计): {alpha_post/(alpha_post+beta_post):.4f}")

# 可视化更新过程
x = np.linspace(0, 1, 200)
prior_pdf = stats.beta.pdf(x, alpha_prior, beta_prior)
posterior_pdf = stats.beta.pdf(x, alpha_post, beta_post)

plt.figure(figsize=(10, 5))
plt.plot(x, prior_pdf, 'b--', linewidth=2, label='Prior: Beta(1,1)')
plt.plot(x, posterior_pdf, 'r-', linewidth=2, label=f'Posterior: Beta({alpha_post},{beta_post})')
plt.axvline(x=0.5, color='gray', linestyle=':', label='Fair coin (p=0.5)')
plt.axvline(x=heads/(heads+tails), color='green', linestyle=':', label=f'MLE (p={heads/(heads+tails):.2f})')
plt.xlabel('Probability of Heads (p)')
plt.ylabel('Density')
plt.title('Bayesian Updating: Coin Flip')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('bayesian_update.png', dpi=100)
print("\n贝叶斯更新图像已保存")
```

---

### Gamma 分布

#### PDF

$$
f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0
$$

记作 $X \sim \text{Gamma}(\alpha, \beta)$。

- $\alpha$（形状参数）：控制分布形状
- $\beta$（率参数）：控制衰减速度

#### 期望与方差

$$
\mathbb{E}[X] = \frac{\alpha}{\beta}
$$

$$
\text{Var}(X) = \frac{\alpha}{\beta^2}
$$

#### 特殊情况

- $\alpha = 1$：指数分布 $\text{Exp}(\beta)$
- $\alpha = n/2, \beta = 1/2$：卡方分布 $\chi^2_n$
- $\alpha$ 为整数：$n$ 个独立指数分布之和

#### 深度学习应用

- **等待时间建模**：多个泊松事件的总等待时间
- **共轭先验**：泊松分布的共轭先验
- **变分推断**：参数的先验分布

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Gamma 分布
x = np.linspace(0, 20, 200)

# 不同形状参数
params = [(1, 0.5), (2, 0.5), (3, 0.5), (5, 1), (9, 2)]

plt.figure(figsize=(10, 5))

for alpha, beta in params:
    pdf = stats.gamma.pdf(x, a=alpha, scale=1/beta)
    plt.plot(x, pdf, linewidth=2, label=f'α={alpha}, β={beta} (E[X]={alpha/beta:.1f})')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gamma 分布')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('gamma.png', dpi=100)
print("Gamma 分布图像已保存")
```

---

## 分布之间的关系

### 重要关系图

```
                    伯努利分布 (Bernoulli)
                         │
          n 次独立求和   │
                         ↓
                    二项分布 (Binomial)
                         │
    n→∞, p→0, np=λ      │
                         ↓
                    泊松分布 (Poisson)
                         
────────────────────────────────────────────────

         指数分布 (Exponential)
              │
  n 个独立求和│
              ↓
         Gamma 分布
              │
   α=n/2, β=1/2
              ↓
         卡方分布 (Chi-squared)

────────────────────────────────────────────────

         均匀分布 Uniform(0,1)
              │
     逆变换采样
              ↓
      任意分布

────────────────────────────────────────────────

    Beta(α,β) 是二项分布的共轭先验
    Gamma(α,β) 是泊松分布的共轭先验
```

### 正态分布的中心地位

根据**中心极限定理**，大量独立随机变量之和趋于正态分布：

$$
\frac{1}{\sqrt{n}} \sum_{i=1}^n X_i \xrightarrow{d} \mathcal{N}(0, \sigma^2)
$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 演示中心极限定理
n_samples = 10000

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# 原始分布：均匀分布
original_dist = np.random.uniform(0, 1, n_samples)

# 不同样本量的和的分布
for i, n in enumerate([1, 2, 5, 10, 30, 100]):
    row, col = i // 3, i % 3
    
    # 采样 n 个均匀分布的和
    sums = np.sum(np.random.uniform(0, 1, (n_samples, n)), axis=1)
    # 标准化
    standardized = (sums - n * 0.5) / np.sqrt(n / 12)
    
    # 直方图
    axes[row, col].hist(standardized, bins=50, density=True, alpha=0.7)
    
    # 叠加正态分布
    x = np.linspace(-4, 4, 100)
    axes[row, col].plot(x, stats.norm.pdf(x), 'r-', linewidth=2)
    
    axes[row, col].set_title(f'n = {n}')
    axes[row, col].set_xlim(-4, 4)
    axes[row, col].grid(True, alpha=0.3)

plt.suptitle('中心极限定理演示：均匀分布的和趋于正态分布', fontsize=14)
plt.tight_layout()
plt.savefig('clt.png', dpi=100)
print("中心极限定理演示图像已保存")
```

---

## 在深度学习中的应用

### 1. 权重初始化

```python
import numpy as np

def initialize_weights(shape, method='xavier', activation='relu'):
    """
    权重初始化方法
    
    Parameters:
    -----------
    shape : tuple
        权重矩阵形状
    method : str
        'xavier' 或 'he'
    activation : str
        激活函数类型 ('relu', 'tanh', 'sigmoid')
    """
    fan_in = shape[1] if len(shape) == 2 else shape[0]
    fan_out = shape[0] if len(shape) == 2 else shape[1]
    
    if method == 'xavier':
        # Xavier/Glorot 初始化
        # 适用于 tanh, sigmoid
        std = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, std, shape)
    
    elif method == 'he':
        # He/Kaiming 初始化
        # 适用于 ReLU 及其变体
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, shape)
    
    elif method == 'lecun':
        # LeCun 初始化
        # 适用于 SELU
        std = np.sqrt(1 / fan_in)
        return np.random.normal(0, std, shape)

# 示例
print("权重初始化示例:")
print("="*50)

# Xavier 初始化 (用于 tanh)
w_xavier = initialize_weights((256, 784), method='xavier')
print(f"Xavier 初始化: {w_xavier.shape}")
print(f"  理论方差: {2 / (784 + 256):.6f}")
print(f"  实际方差: {w_xavier.var():.6f}")

# He 初始化 (用于 ReLU)
w_he = initialize_weights((256, 784), method='he')
print(f"\nHe 初始化: {w_he.shape}")
print(f"  理论方差: {2 / 784:.6f}")
print(f"  实际方差: {w_he.var():.6f}")
```

### 2. Dropout 实现

```python
import numpy as np

class Dropout:
    """
    Dropout 正则化
    
    训练时以概率 p 随机将神经元置零，
    测试时保留所有神经元但缩放输出。
    """
    
    def __init__(self, p=0.5):
        """
        Parameters:
        -----------
        p : float
            保留概率 (不是丢弃概率!)
        """
        self.p = p
        self.training = True
        self.mask = None
    
    def forward(self, x):
        if not self.training:
            return x
        
        # 伯努利采样
        self.mask = (np.random.random(x.shape) < self.p).astype(x.dtype)
        # 缩放以保持期望
        return x * self.mask / self.p
    
    def backward(self, grad_output):
        if not self.training:
            return grad_output
        return grad_output * self.mask / self.p

# 示例
dropout = Dropout(p=0.5)
x = np.random.randn(1000, 100)

# 训练模式
dropout.training = True
x_train = dropout.forward(x)
print(f"训练模式:")
print(f"  原始均值: {x.mean():.4f}")
print(f"  Dropout 后均值: {x_train.mean():.4f}")
print(f"  零元素比例: {(x_train == 0).mean():.2%}")

# 测试模式
dropout.training = False
x_test = dropout.forward(x)
print(f"\n测试模式:")
print(f"  输出均值: {x_test.mean():.4f}")
print(f"  零元素比例: {(x_test == 0).mean():.2%}")
```

### 3. 交叉熵损失与类别分布

```python
import numpy as np

def cross_entropy_loss(predictions, targets):
    """
    交叉熵损失
    
    Parameters:
    -----------
    predictions : array, shape (N, K)
        Softmax 输出 (概率分布)
    targets : array, shape (N,) or (N, K)
        类别索引或 one-hot 编码
    """
    N = predictions.shape[0]
    
    # 数值稳定性
    eps = 1e-15
    predictions = np.clip(predictions, eps, 1 - eps)
    
    if targets.ndim == 1:
        # 类别索引
        return -np.mean(np.log(predictions[np.arange(N), targets]))
    else:
        # one-hot 编码
        return -np.mean(np.sum(targets * np.log(predictions), axis=1))

# Softmax 函数
def softmax(logits):
    """数值稳定的 Softmax"""
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# 示例：3 分类问题
logits = np.array([
    [2.0, 1.0, 0.1],  # 预测类别 0
    [0.1, 2.0, 1.0],  # 预测类别 1
    [0.5, 0.5, 2.0],  # 预测类别 2
])

targets = np.array([0, 1, 2])  # 真实类别

probs = softmax(logits)
loss = cross_entropy_loss(probs, targets)

print("交叉熵损失示例:")
print(f"Logits:\n{logits}")
print(f"\nSoftmax 概率:\n{probs}")
print(f"\n真实类别: {targets}")
print(f"交叉熵损失: {loss:.4f}")

# 与负对数似然的关系
print("\n负对数似然解释:")
for i, (p, t) in enumerate(zip(probs, targets)):
    print(f"  样本 {i}: -log(p[{t}]) = {-np.log(p[t]):.4f}")
```

### 4. 变分自编码器 (VAE) 中的重参数化

```python
import numpy as np

def reparameterize(mu, log_var):
    """
    VAE 重参数化技巧
    
    从 N(μ, σ²) 采样:
    z = μ + σ * ε, where ε ~ N(0, 1)
    
    这使得采样操作可微，从而可以通过反向传播训练。
    """
    # 从标准正态分布采样
    epsilon = np.random.standard_normal(mu.shape)
    
    # 重参数化
    std = np.exp(0.5 * log_var)  # σ = exp(0.5 * log(σ²))
    z = mu + std * epsilon
    
    return z, epsilon

def vae_loss(x_reconstructed, x, mu, log_var, beta=1.0):
    """
    VAE 损失函数
    
    Loss = Reconstruction Loss + β * KL Divergence
    
    KL(N(μ,σ²) || N(0,1)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    """
    # 重构损失 (假设高斯分布)
    recon_loss = np.mean((x_reconstructed - x) ** 2)
    
    # KL 散度
    kl_loss = -0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var))
    
    return recon_loss + beta * kl_loss

# 示例
np.random.seed(42)

# 编码器输出
mu = np.array([0.0, 1.0, -0.5])
log_var = np.array([0.0, 0.5, 1.0])  # σ² = [1, 1.65, 2.72]

# 重参数化采样
z, epsilon = reparameterize(mu, log_var)

print("VAE 重参数化示例:")
print(f"μ = {mu}")
print(f"log(σ²) = {log_var}")
print(f"σ = {np.exp(0.5 * log_var)}")
print(f"ε = {epsilon}")
print(f"z = μ + σ*ε = {z}")

# 多次采样
n_samples = 10000
z_samples = np.zeros((n_samples, 3))
for i in range(n_samples):
    z_samples[i], _ = reparameterize(mu, log_var)

print(f"\n采样统计 ({n_samples} 次):")
print(f"样本均值: {z_samples.mean(axis=0)}")
print(f"理论均值: {mu}")
print(f"样本方差: {z_samples.var(axis=0)}")
print(f"理论方差: {np.exp(log_var)}")
```

---

## 小结

本章介绍了随机变量和常见概率分布，这些是理解深度学习中不确定性和随机性的基础。

### 核心概念对照表

| 概念 | 离散型 | 连续型 |
|------|--------|--------|
| 描述函数 | PMF: $p(x)$ | PDF: $f(x)$ |
| 归一化 | $\sum_x p(x) = 1$ | $\int f(x)dx = 1$ |
| 概率计算 | $P(X = x) = p(x)$ | $P(a \le X \le b) = \int_a^b f(x)dx$ |
| 期望 | $\sum x \cdot p(x)$ | $\int x \cdot f(x)dx$ |
| 方差 | $\mathbb{E}[X^2] - (\mathbb{E}[X])^2$ | $\mathbb{E}[X^2] - (\mathbb{E}[X])^2$ |

### 常见分布总结

| 分布 | 类型 | 参数 | 期望 | 方差 | 深度学习应用 |
|------|------|------|------|------|--------------|
| Bernoulli | 离散 | $p$ | $p$ | $p(1-p)$ | Dropout, 二分类 |
| Binomial | 离散 | $n, p$ | $np$ | $np(1-p)$ | 集成学习 |
| Poisson | 离散 | $\lambda$ | $\lambda$ | $\lambda$ | 稀疏编码 |
| Categorical | 离散 | $\mathbf{p}$ | - | - | 多分类 |
| Uniform | 连续 | $a, b$ | $(a+b)/2$ | $(b-a)^2/12$ | 初始化 |
| Normal | 连续 | $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ | 初始化, VAE |
| Exponential | 连续 | $\lambda$ | $1/\lambda$ | $1/\lambda^2$ | 等待时间 |
| Laplace | 连续 | $\mu, b$ | $\mu$ | $2b^2$ | L1 正则化 |
| Beta | 连续 | $\alpha, \beta$ | $\alpha/(\alpha+\beta)$ | - | 贝叶斯推断 |

### 关键要点

1. **随机变量**：将随机现象数量化的函数
2. **PMF vs PDF**：离散用质量函数，连续用密度函数
3. **正态分布**：最重要，中心极限定理保证其普遍性
4. **共轭先验**：Beta-二项，Gamma-泊松，简化贝叶斯推断
5. **重参数化**：使随机采样可微，VAE 的核心技术

---

**上一节**：[第三章（a）：概率基础与条件概率](03a-概率基础与条件概率.md)

**下一节**：[第三章（c）：多维随机变量与数字特征](03c-多维随机变量与数字特征.md) - 学习联合分布、边缘分布、协方差等概念。

**返回**：[数学基础教程目录](../math-fundamentals.md)
