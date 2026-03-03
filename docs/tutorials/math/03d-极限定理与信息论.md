# 第三章（d）：极限定理与信息论

极限定理（大数定律和中心极限定理）是概率论的基石，解释了为什么深度学习中的许多技术（如批量归一化、随机梯度下降）能够有效工作。信息论则提供了量化不确定性和信息量的数学框架，交叉熵损失函数和 KL 散度都是信息论的核心概念。

---

## 🎯 生活类比：极限定理就是"平均下来会怎样"

### 大数定律 = "人多力量大，平均稳定"

想象你在估算全校学生的平均身高：
- 问 1 个人：可能很高或很矮，不准确
- 问 10 个人：好一点，但还是可能有偏差
- 问 1000 个人：非常接近真实平均身高了！
- 问全校：几乎等于真实平均身高

```
样本量:    1人      10人     100人    1000人
           ↓         ↓        ↓        ↓
估计值:  185cm   172cm    168cm    170.1cm
           ↑         ↑        ↑        ↑
        很不稳定   慢慢稳定  更稳定   非常接近真值(170cm)
```

**大数定律告诉我们**：样本越多，平均值越稳定，最终收敛到真实期望。

### 中心极限定理 = "为什么钟形曲线到处都是"

不管原始分布长什么样，**大量独立随机变量的和**会趋向于**正态分布**（钟形曲线）。

```
掷1个骰子: 每面概率相等 ━━━━━━━━ (均匀分布)

掷100个骰子的总和: ╭───╮
                   ╱     ╲
                  ╱       ╲
                 ╱         ╲
                ───────────── (正态分布！)
```

**这就是为什么**：身高、考试成绩、测量误差都近似正态分布——因为它们都是很多小因素的叠加。

### 信息熵 = "不确定性的度量"

**熵 = 惊喜程度的平均值**

| 事件 | 概率 | 惊喜度 |
|------|------|--------|
| 明天太阳升起 | 99.99% | 很小（意料之中） |
| 中彩票 | 0.01% | 很大（意外！） |
| 抛硬币正面 | 50% | 中等 |

```
熵高 ─────────────────────────────────→ 熵低
    │                                        │
很不确定（抛硬币）                     很确定（太阳升起）
    │                                        │
信息量大（结果告诉你很多）             信息量小（结果不稀奇）
```

### 📖 交叉熵 = "用错误的分布编码的平均位数"

**场景**：你要猜测一个硬币是正面还是反面
- 真实情况：硬币有90%概率正面
- 你的猜测：你认为50%概率正面

用你的猜测来编码，你会浪费很多比特！

**交叉熵损失**：衡量你的预测分布和真实分布之间的差距。

---

## 目录

1. [大数定律](#大数定律)
2. [中心极限定理](#中心极限定理)
3. [蒙特卡洛方法](#蒙特卡洛方法)
4. [信息论基础](#信息论基础)
   - [熵](#熵-entropy)
   - [联合熵与条件熵](#联合熵与条件熵)
   - [交叉熵](#交叉熵-cross-entropy)
5. [KL 散度](#kl-散度-kullback-leibler-divergence)
6. [互信息](#互信息-mutual-information)
7. [信息论与机器学习的关系](#信息论与机器学习的关系)
8. [在深度学习中的应用](#在深度学习中的应用)
9. [小结](#小结)

---

## 大数定律

### 弱大数定律 (WLLN)

设 $X_1, X_2, \ldots, X_n$ 是**独立同分布 (i.i.d.)** 的随机变量，$\mathbb{E}[X_i] = \mu$，则**样本均值**依概率收敛于期望：

$$
\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{P} \mu
$$

即对任意 $\epsilon > 0$：

$$
\lim_{n \to \infty} P(|\bar{X}_n - \mu| < \epsilon) = 1
$$

### 强大数定律 (SLLN)

样本均值**几乎必然**收敛于期望：

$$
P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1
$$

### 直观理解

当样本量足够大时，样本均值会"接近"真实期望。这解释了为什么：
- **大量数据训练**能学到真实分布
- **大批量训练**梯度估计更稳定

### 大数定律的条件

1. 样本独立同分布
2. 期望存在且有限

### 弱大数定律的证明（使用切比雪夫不等式）

**定理**：设 $X_1, X_2, \ldots, X_n$ 是 i.i.d. 随机变量，$\mathbb{E}[X_i] = \mu$，$\text{Var}(X_i) = \sigma^2 < \infty$，则对任意 $\epsilon > 0$：

$$
\lim_{n \to \infty} P(|\bar{X}_n - \mu| \geq \epsilon) = 0
$$

**证明**：

**Step 1：计算样本均值的期望和方差**

$$
\mathbb{E}[\bar{X}_n] = \mathbb{E}\left[\frac{1}{n}\sum_{i=1}^n X_i\right] = \frac{1}{n}\sum_{i=1}^n \mathbb{E}[X_i] = \frac{n\mu}{n} = \mu
$$

$$
\text{Var}(\bar{X}_n) = \text{Var}\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}
$$

**Step 2：应用切比雪夫不等式**

切比雪夫不等式：对于任意随机变量 $Y$ 和 $\epsilon > 0$：

$$
P(|Y - \mathbb{E}[Y]| \geq \epsilon) \leq \frac{\text{Var}(Y)}{\epsilon^2}
$$

对 $Y = \bar{X}_n$ 应用：

$$
P(|\bar{X}_n - \mu| \geq \epsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\epsilon^2} = \frac{\sigma^2}{n\epsilon^2}
$$

**Step 3：取极限**

$$
\lim_{n \to \infty} P(|\bar{X}_n - \mu| \geq \epsilon) \leq \lim_{n \to \infty} \frac{\sigma^2}{n\epsilon^2} = 0
$$

由于概率非负，得 $\boxed{\lim_{n \to \infty} P(|\bar{X}_n - \mu| \geq \epsilon) = 0}$，即 $\bar{X}_n \xrightarrow{P} \mu$。 $\square$

**收敛速率**：误差概率 $P(|\bar{X}_n - \mu| \geq \epsilon) = O(1/n)$，即依 $1/n$ 的速率收敛。

```python
import numpy as np
import matplotlib.pyplot as plt

# 大数定律演示
np.random.seed(42)

# 设置
true_mean = 3.5  # 骰子的期望
max_samples = 10000

# 掷骰子实验
dice_rolls = np.random.randint(1, 7, max_samples)

# 计算累积均值
cumulative_mean = np.cumsum(dice_rolls) / np.arange(1, max_samples + 1)

# 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(np.arange(1, max_samples + 1), cumulative_mean, 'b-', alpha=0.7, label='样本均值')
plt.axhline(y=true_mean, color='r', linestyle='--', linewidth=2, label=f'真实期望 μ={true_mean}')
plt.xlabel('样本数 n')
plt.ylabel('样本均值')
plt.title('大数定律演示：掷骰子')
plt.legend()
plt.grid(True, alpha=0.3)

# 放大前 500 次
plt.subplot(1, 2, 2)
plt.plot(np.arange(1, 501), cumulative_mean[:500], 'b-', alpha=0.7)
plt.axhline(y=true_mean, color='r', linestyle='--', linewidth=2)
plt.xlabel('样本数 n')
plt.ylabel('样本均值')
plt.title('大数定律演示（前 500 次）')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('law_of_large_numbers.png', dpi=100)
print("大数定律演示图像已保存")

# 收敛速度分析
print("\n收敛速度分析:")
print("="*50)
checkpoints = [10, 50, 100, 500, 1000, 5000, 10000]
for n in checkpoints:
    error = abs(cumulative_mean[n-1] - true_mean)
    print(f"n = {n:5d}: 样本均值 = {cumulative_mean[n-1]:.4f}, 误差 = {error:.4f}")
```

---

## 中心极限定理

### 标准形式

设 $X_1, X_2, \ldots, X_n$ 是 i.i.d. 随机变量，$\mathbb{E}[X_i] = \mu$，$\text{Var}(X_i) = \sigma^2$，则：

$$
\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)
$$

等价地：

$$
\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)
$$

### 实用形式

对于大 $n$：

$$
\bar{X}_n \approx \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)
$$

$$
\sum_{i=1}^n X_i \approx \mathcal{N}(n\mu, n\sigma^2)
$$

### 中心极限定理的意义

1. **正态分布的普遍性**：无论原始分布是什么，样本均值都趋于正态
2. **统计推断的基础**：置信区间、假设检验
3. **神经网络的权重**：大量输入的加权和近似正态分布

### 应用条件

- 样本量 $n$ 足够大（通常 $n \geq 30$）
- 原始分布方差有限

### 中心极限定理的证明概要（特征函数法）

**定理**：设 $X_1, X_2, \ldots, X_n$ 是 i.i.d. 随机变量，$\mathbb{E}[X_i] = \mu$，$\text{Var}(X_i) = \sigma^2$，则：

$$
\frac{\sum_{i=1}^n (X_i - \mu)}{\sigma\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)
$$

**证明概要**：

**Step 1：标准化**

令 $Y_i = \frac{X_i - \mu}{\sigma}$，则 $\mathbb{E}[Y_i] = 0$，$\text{Var}(Y_i) = 1$。

定义 $S_n = \frac{1}{\sqrt{n}}\sum_{i=1}^n Y_i$，我们要证明 $S_n \xrightarrow{d} \mathcal{N}(0, 1)$。

**Step 2：特征函数的定义**

随机变量 $X$ 的特征函数定义为：

$$
\varphi_X(t) = \mathbb{E}[e^{itX}]
$$

**关键性质**：若 $\varphi_{S_n}(t) \to \varphi_Z(t)$ 对所有 $t$ 成立，则 $S_n \xrightarrow{d} Z$（Lévy 连续性定理）。

**Step 3：计算 $S_n$ 的特征函数**

由于 $Y_i$ 独立：

$$
\varphi_{S_n}(t) = \mathbb{E}\left[\exp\left(it \cdot \frac{1}{\sqrt{n}}\sum_{i=1}^n Y_i\right)\right] = \left[\varphi_{Y}\left(\frac{t}{\sqrt{n}}\right)\right]^n
$$

**Step 4：泰勒展开 $\varphi_Y$**

由于 $\mathbb{E}[Y] = 0$，$\mathbb{E}[Y^2] = 1$，特征函数在 $t=0$ 处展开：

$$
\varphi_Y(u) = 1 + iu\mathbb{E}[Y] + \frac{(iu)^2}{2}\mathbb{E}[Y^2] + o(u^2) = 1 - \frac{u^2}{2} + o(u^2)
$$

因此：

$$
\varphi_Y\left(\frac{t}{\sqrt{n}}\right) = 1 - \frac{t^2}{2n} + o\left(\frac{1}{n}\right)
$$

**Step 5：取极限**

$$
\varphi_{S_n}(t) = \left[1 - \frac{t^2}{2n} + o\left(\frac{1}{n}\right)\right]^n \xrightarrow{n \to \infty} e^{-t^2/2}
$$

这里用到了 $\lim_{n \to \infty} (1 + \frac{a}{n})^n = e^a$。

**Step 6：得出结论**

$e^{-t^2/2}$ 正是标准正态分布 $\mathcal{N}(0, 1)$ 的特征函数。

由 Lévy 连续性定理：

$$
\boxed{S_n = \frac{\sum_{i=1}^n (X_i - \mu)}{\sigma\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)}
$$

**收敛速率**：Berry-Esseen 定理给出收敛速率 $O(1/\sqrt{n})$。 $\square$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 中心极限定理演示
np.random.seed(42)

# 设置
n_samples = 10000  # 重复采样次数
sample_sizes = [1, 5, 10, 30, 100, 500]  # 不同样本量

# 原始分布：均匀分布 (0, 1)
# E[X] = 0.5, Var(X) = 1/12

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, n in enumerate(sample_sizes):
    row, col = idx // 3, idx % 3
    
    # 生成 n_samples 个样本均值
    # 每个样本均值是 n 个均匀分布的平均
    samples = np.random.uniform(0, 1, (n_samples, n))
    sample_means = samples.mean(axis=1)
    
    # 标准化
    true_mean = 0.5
    true_std = np.sqrt(1/12 / n)
    standardized = (sample_means - true_mean) / true_std
    
    # 绘制直方图
    axes[row, col].hist(standardized, bins=50, density=True, alpha=0.7, label='标准化样本均值')
    
    # 叠加标准正态分布
    x = np.linspace(-4, 4, 100)
    axes[row, col].plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='N(0,1)')
    
    axes[row, col].set_title(f'n = {n}')
    axes[row, col].set_xlabel('标准化值')
    axes[row, col].set_ylabel('密度')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)
    axes[row, col].set_xlim(-4, 4)

plt.suptitle('中心极限定理：均匀分布的样本均值趋于正态分布', fontsize=14)
plt.tight_layout()
plt.savefig('central_limit_theorem.png', dpi=100)
print("中心极限定理演示图像已保存")

# 定量分析
print("\n中心极限定理定量分析:")
print("="*60)
print(f"{'样本量 n':>10} | {'样本均值':>10} | {'理论均值':>10} | {'样本方差':>10} | {'理论方差':>10}")
print("-"*60)

for n in [10, 30, 100, 500]:
    samples = np.random.uniform(0, 1, (10000, n))
    sample_means = samples.mean(axis=1)
    
    theoretical_mean = 0.5
    theoretical_var = 1/12 / n
    
    print(f"{n:>10} | {sample_means.mean():>10.4f} | {theoretical_mean:>10.4f} | {sample_means.var():>10.6f} | {theoretical_var:>10.6f}")
```

---

## 蒙特卡洛方法

### 基本思想

利用**随机采样**来估计数值结果：

$$
\mathbb{E}[g(X)] \approx \frac{1}{n} \sum_{i=1}^n g(x_i)
$$

其中 $x_1, x_2, \ldots, x_n$ 是从分布中采样的独立样本。

### 蒙特卡洛积分

计算积分 $\int_a^b f(x) dx$：

$$
\int_a^b f(x) dx = (b-a) \mathbb{E}[f(X)] \approx \frac{b-a}{n} \sum_{i=1}^n f(x_i)
$$

其中 $X \sim \text{Uniform}(a, b)$。

### 误差分析

蒙特卡洛估计的标准误差：

$$
\text{SE} = \frac{\sigma}{\sqrt{n}}
$$

收敛速度：$O(n^{-1/2})$（与维度无关）

```python
import numpy as np

# 蒙特卡洛积分示例：计算 π
np.random.seed(42)

def monte_carlo_pi(n_samples):
    """使用蒙特卡洛方法估计 π"""
    # 在单位正方形内随机采样
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    
    # 计算落在单位圆内的比例
    inside = (x**2 + y**2) <= 1
    pi_estimate = 4 * inside.mean()
    
    return pi_estimate, inside

# 不同样本量的估计
sample_sizes = [100, 1000, 10000, 100000, 1000000]

print("蒙特卡洛估计 π:")
print("="*50)
print(f"{'样本量':>10} | {'估计值':>10} | {'误差':>10}")
print("-"*50)

for n in sample_sizes:
    pi_est, _ = monte_carlo_pi(n)
    error = abs(pi_est - np.pi)
    print(f"{n:>10} | {pi_est:>10.6f} | {error:>10.6f}")

# 可视化
n_vis = 5000
pi_est, inside = monte_carlo_pi(n_vis)
x = np.random.uniform(-1, 1, n_vis)
y = np.random.uniform(-1, 1, n_vis)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.scatter(x[inside], y[inside], c='blue', alpha=0.5, s=1, label='圆内')
plt.scatter(x[~inside], y[~inside], c='red', alpha=0.5, s=1, label='圆外')
circle = plt.Circle((0, 0), 1, fill=False, color='green', linewidth=2)
plt.gca().add_patch(circle)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.gca().set_aspect('equal')
plt.title(f'蒙特卡洛估计 π ≈ {pi_est:.4f} (n={n_vis})')
plt.legend()
plt.savefig('monte_carlo_pi.png', dpi=100)
print("\n蒙特卡洛 π 图像已保存")

# 蒙特卡洛积分示例：计算 E[sin(X)] where X ~ N(0,1)
print("\n蒙特卡洛积分:")
print("="*50)

# 真实值（数值积分）
from scipy import integrate
true_value, _ = integrate.quad(lambda x: np.sin(x) * stats.norm.pdf(x), -10, 10)
print(f"真实值 E[sin(X)] = {true_value:.6f}")

# 蒙特卡洛估计
for n in [100, 1000, 10000, 100000]:
    samples = np.random.normal(0, 1, n)
    estimate = np.sin(samples).mean()
    error = abs(estimate - true_value)
    print(f"n = {n:>6}: 估计 = {estimate:.6f}, 误差 = {error:.6f}")
```

---

## 信息论基础

### 熵 (Entropy)

#### 🎯 生活类比：猜谜游戏

想象你在玩一个"二十个问题"的猜谜游戏：

| 情况 | 你需要问几个问题？ | 熵的大小 |
|------|-------------------|----------|
| 猜1-100中的一个数 | 最多7个问题（$\log_2(100) \approx 6.6$） | 高 |
| 猜硬币正反面 | 只需1个问题 | 中 |
| 猜明天太阳是否升起 | 0个问题（确定性事件） | 0 |

**熵的本质**：熵就是"平均需要问多少个是/否问题，才能确定结果"。

#### 📖 为什么用 $-\log p$ 表示信息量？

假设一个事件发生的概率是 $p$：
- 如果 $p = 1$（必然发生），信息量 = $-\log(1) = 0$（不惊喜）
- 如果 $p = 0.5$（抛硬币），信息量 = $-\log(0.5) = 1$ 比特
- 如果 $p = 0.01$（罕见事件），信息量 = $-\log(0.01) \approx 6.6$ 比特（很惊喜！）

**直观理解**：越不可能发生的事情，一旦发生了，带来的"信息量"越大。

#### 定义

**熵**衡量随机变量的**不确定性**或**信息量**。

**离散熵**：

$$
H(X) = -\sum_{x} p(x) \log p(x) = \mathbb{E}[-\log p(X)]
$$

**通俗翻译**：熵 = 所有可能结果的"信息量"的加权平均（用概率作为权重）。

#### 📝 手把手计算示例

假设一个袋子里有4个球：🔴🔴🔵🟢（2红1蓝1绿）

**Step 1：写出概率分布**
- P(红) = 2/4 = 0.5
- P(蓝) = 1/4 = 0.25
- P(绿) = 1/4 = 0.25

**Step 2：计算每个结果的信息量**（使用 $\log_2$）
- 信息量(红) = $-\log_2(0.5) = 1$ 比特
- 信息量(蓝) = $-\log_2(0.25) = 2$ 比特
- 信息量(绿) = $-\log_2(0.25) = 2$ 比特

**Step 3：加权平均得到熵**
$$H = 0.5 \times 1 + 0.25 \times 2 + 0.25 \times 2 = 0.5 + 0.5 + 0.5 = 1.5 \text{ 比特}$$

**解释**：平均来说，你需要问 1.5 个是/否问题来确定摸到什么颜色的球。

**微分熵**（连续）：

$$
H(X) = -\int f(x) \log f(x) \, dx
$$

#### 熵的单位

| 对数底 | 单位 |
|--------|------|
| 2 | 比特 (bit) |
| $e$ | 奈特 (nat) |
| 10 | 哈特 (hart) |

在机器学习中通常使用 nat（自然对数）。

#### 熵的性质

1. **非负性**：$H(X) \geq 0$（离散情况）
2. **等概率最大**：对于有 $n$ 个取值的离散变量，$H(X) \leq \log n$，等号在 $p(x) = 1/n$ 时成立
3. **确定性变量熵为0**：若 $X$ 恒等于某值，则 $H(X) = 0$
4. **信息量**：$-\log p(x)$ 是观察到事件 $x$ 的"惊喜程度"

#### 熵的直观理解

- **高熵**：分布均匀，不确定性大，"惊喜"多
- **低熵**：分布集中，不确定性小，结果可预测

```python
import numpy as np
import matplotlib.pyplot as plt

def entropy(p):
    """计算离散分布的熵"""
    # 过滤零概率
    p = p[p > 0]
    return -np.sum(p * np.log(p))

# 示例：不同分布的熵
print("熵的计算示例:")
print("="*50)

# 1. 均匀分布（最大熵）
p_uniform = np.array([0.25, 0.25, 0.25, 0.25])
print(f"均匀分布 {p_uniform}: H = {entropy(p_uniform):.4f} (最大熵 = {np.log(4):.4f})")

# 2. 偏斜分布
p_skewed = np.array([0.7, 0.1, 0.1, 0.1])
print(f"偏斜分布 {p_skewed}: H = {entropy(p_skewed):.4f}")

# 3. 确定性分布（最小熵）
p_deterministic = np.array([1.0, 0.0, 0.0, 0.0])
print(f"确定性分布 {p_deterministic}: H = {entropy(p_deterministic):.4f}")

# 4. 二元分布：H(p) = -p*log(p) - (1-p)*log(1-p)
print("\n二元熵函数 H(p) = -p*log(p) - (1-p)*log(1-p):")
p_values = np.linspace(0.01, 0.99, 100)
binary_entropy = lambda p: -p * np.log(p) - (1-p) * np.log(1-p)
H_values = [binary_entropy(p) for p in p_values]

plt.figure(figsize=(10, 5))
plt.plot(p_values, H_values, 'b-', linewidth=2)
plt.axvline(x=0.5, color='r', linestyle='--', label='p=0.5 (最大熵)')
plt.xlabel('p')
plt.ylabel('H(p)')
plt.title('二元熵函数 H(p) = -p log p - (1-p) log(1-p)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('binary_entropy.png', dpi=100)
print("二元熵函数图像已保存")

print(f"最大熵在 p=0.5: H(0.5) = {binary_entropy(0.5):.4f} = log(2)")
```

---

### 联合熵与条件熵

#### 联合熵

$$
H(X, Y) = -\sum_{x,y} p(x, y) \log p(x, y)
$$

#### 条件熵

$$
H(Y|X) = -\sum_{x,y} p(x, y) \log p(y|x) = \mathbb{E}_{X}[-\log p(Y|X)]
$$

#### 链式法则

$$
H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
$$

#### 直观理解

- $H(Y|X)$：已知 $X$ 后，$Y$ 的剩余不确定性
- 若 $X$ 和 $Y$ 独立：$H(Y|X) = H(Y)$

```python
import numpy as np

# 联合熵和条件熵示例
# 定义联合分布 P(X, Y)
joint = np.array([
    [0.1, 0.1, 0.05],  # X=0
    [0.15, 0.2, 0.1],  # X=1
    [0.1, 0.15, 0.05]  # X=2
])

# 边缘分布
p_X = joint.sum(axis=1)
p_Y = joint.sum(axis=0)

# 计算各种熵
def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

H_X = entropy(p_X)
H_Y = entropy(p_Y)
H_XY = entropy(joint.flatten())

# 条件熵
H_Y_given_X = H_XY - H_X
H_X_given_Y = H_XY - H_Y

print("联合熵与条件熵:")
print("="*50)
print(f"H(X) = {H_X:.4f}")
print(f"H(Y) = {H_Y:.4f}")
print(f"H(X,Y) = {H_XY:.4f}")
print(f"H(Y|X) = H(X,Y) - H(X) = {H_Y_given_X:.4f}")
print(f"H(X|Y) = H(X,Y) - H(Y) = {H_X_given_Y:.4f}")

# 验证链式法则
print(f"\n验证链式法则:")
print(f"H(X,Y) = H(X) + H(Y|X) = {H_X + H_Y_given_X:.4f}")
print(f"H(X,Y) = H(Y) + H(X|Y) = {H_Y + H_X_given_Y:.4f}")
```

---

### 交叉熵 (Cross Entropy)

#### 🎯 生活类比：用错误的密码表发报文

想象你是一个发报员：
- **真实情况**：消息按概率分布 $P$ 出现（比如英语中 'e' 最常见）
- **你的编码表**：你按分布 $Q$ 来设计编码长度（比如你误以为 'z' 最常见）

**交叉熵** = 用错误的编码表 $Q$ 来编码真实消息 $P$，平均每个字符需要多少比特。

| 编码表准确程度 | 交叉熵大小 | 实际效果 |
|---------------|-----------|---------|
| $Q = P$（完美） | $H(P)$（最小） | 编码效率最高 |
| $Q \neq P$（有偏差） | $> H(P)$ | 浪费比特 |

#### 📖 为什么分类问题用交叉熵？

假设图片是一只猫，模型预测概率：
- 好模型：[猫:0.9, 狗:0.05, 鸟:0.05] → 交叉熵 = $-\log(0.9) = 0.105$
- 差模型：[猫:0.3, 狗:0.4, 鸟:0.3] → 交叉熵 = $-\log(0.3) = 1.204$

**交叉熵越小，预测越准确！**

#### 定义

**交叉熵**衡量用分布 $Q$ 来编码分布 $P$ 所需的平均比特数：

$$
H(P, Q) = -\sum_x p(x) \log q(x) = \mathbb{E}_{x \sim P}[-\log q(x)]
$$

**通俗翻译**：用"错误的概率 $Q$"来猜测"真实的结果 $P$"，平均会多"惊讶"多少。

#### 与熵的关系

$$
H(P, Q) = H(P) + D_{KL}(P \| Q)
$$

**解释**：交叉熵 = 理想的最小编码长度（熵）+ 因编码表错误而浪费的比特（KL散度）。

#### 交叉熵损失

在分类问题中，真实分布 $P$ 是 one-hot 编码，预测分布是 $Q$：

$$
L = H(P, Q) = -\sum_{i=1}^K y_i \log \hat{y}_i
$$

对于 one-hot 标签，简化为：

$$
L = -\log \hat{y}_{\text{true}}
$$

```python
import numpy as np

def cross_entropy(p, q):
    """计算交叉熵 H(P, Q)"""
    # 过滤零概率（避免 log(0)）
    mask = p > 0
    return -np.sum(p[mask] * np.log(q[mask]))

# 示例
print("交叉熵示例:")
print("="*50)

# 真实分布
P = np.array([1, 0, 0, 0])  # one-hot: 类别0是真实标签

# 不同的预测分布
Q1 = np.array([0.9, 0.05, 0.03, 0.02])  # 好的预测
Q2 = np.array([0.5, 0.3, 0.15, 0.05])   # 中等预测
Q3 = np.array([0.1, 0.3, 0.4, 0.2])     # 差的预测

print(f"真实分布 P: {P}")
print(f"\n预测 Q1: {Q1}")
print(f"  交叉熵 H(P, Q1) = {cross_entropy(P, Q1):.4f}")
print(f"  -log(0.9) = {-np.log(0.9):.4f}")

print(f"\n预测 Q2: {Q2}")
print(f"  交叉熵 H(P, Q2) = {cross_entropy(P, Q2):.4f}")
print(f"  -log(0.5) = {-np.log(0.5):.4f}")

print(f"\n预测 Q3: {Q3}")
print(f"  交叉熵 H(P, Q3) = {cross_entropy(P, Q3):.4f}")
print(f"  -log(0.1) = {-np.log(0.1):.4f}")

print("\n结论: 预测越准确，交叉熵越小")
```

---

## KL 散度 (Kullback-Leibler Divergence)

### 🎯 生活类比：两套键盘的效率差异

想象你在用键盘打字：
- **真实情况**：你按概率分布 $P$ 输入字符（比如英文中 'e' 很常见）
- **你的假设**：你按分布 $Q$ 设计了键盘布局（误以为 'z' 很常见）

**KL 散度** = 因为用错误的键盘布局，你平均每个字符多花多少时间。

| 真实分布 vs 假设分布 | KL散度 | 含义 |
|---------------------|--------|------|
| $P = Q$（完全一样） | 0 | 完美，没有浪费 |
| $P \neq Q$（有差异） | $> 0$ | 有浪费，差异越大浪费越多 |

**KL 散度衡量"用错误假设造成的额外代价"**。

### 📝 手把手计算示例

假设真实分布 $P$ 和假设分布 $Q$：

| 事件 | P（真实） | Q（假设） | $\frac{P}{Q}$ | $P \log\frac{P}{Q}$ |
|------|----------|----------|---------------|---------------------|
| A | 0.5 | 0.3 | 1.67 | $0.5 \times \log(1.67) = 0.27$ |
| B | 0.3 | 0.4 | 0.75 | $0.3 \times \log(0.75) = -0.07$ |
| C | 0.2 | 0.3 | 0.67 | $0.2 \times \log(0.67) = -0.08$ |

**KL散度** = $0.27 + (-0.07) + (-0.08) = 0.12$

**解释**：用 $Q$ 代替 $P$，平均每个事件浪费 0.12 奈特的信息。

### 定义

**KL 散度**（相对熵）衡量两个分布之间的"距离"：

$$
D_{KL}(P \| Q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(P, Q) - H(P)
$$

**通俗翻译**：$D_{KL}(P\|Q)$ = 假设 $Q$ 而实际是 $P$ 时，你"多惊讶"的程度。

连续情况：

$$
D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx
$$

### 性质

1. **非负性**：$D_{KL}(P \| Q) \geq 0$
2. **零值**：$D_{KL}(P \| Q) = 0$ 当且仅当 $P = Q$
3. **不对称性**：$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$
4. **不是真正的距离**：不满足三角不等式

**KL 散度非负性的证明**（使用 Jensen 不等式）：

**第一步**：写出 KL 散度的定义。

$$D_{KL}(P \| Q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

**第二步**：转换为负的形式。

$$= -\sum_x p(x) \log \frac{q(x)}{p(x)}$$

**第三步**：识别这是一个期望。

$$= -\mathbb{E}_{x \sim P}\left[\log \frac{q(x)}{p(x)}\right]$$

**第四步**：应用 Jensen 不等式。

由于 $\log$ 是**凹函数**（concave），Jensen 不等式给出：

$$\mathbb{E}[\log f(x)] \leq \log \mathbb{E}[f(x)]$$

因此：

$$-\mathbb{E}\left[\log \frac{q(x)}{p(x)}\right] \geq -\log \mathbb{E}\left[\frac{q(x)}{p(x)}\right]$$

**第五步**：计算期望。

$$\mathbb{E}\left[\frac{q(x)}{p(x)}\right] = \sum_x p(x) \cdot \frac{q(x)}{p(x)} = \sum_x q(x) = 1$$

**第六步**：得出结论。

$$D_{KL}(P \| Q) \geq -\log(1) = 0$$

$$\boxed{D_{KL}(P \| Q) \geq 0}$$

**等号成立条件**：当且仅当 $\frac{q(x)}{p(x)}$ 对所有 $x$ 为常数，即 $p(x) = q(x)$。

**为什么 KL 散度不是对称的**：

考虑简单的二元分布：
- $P = (1, 0)$（确定性在第一个状态）
- $Q = (0.5, 0.5)$（均匀分布）

$$D_{KL}(P \| Q) = 1 \cdot \log\frac{1}{0.5} + 0 \cdot \log\frac{0}{0.5} = \log 2$$

但 $\log\frac{0}{0.5}$ 是未定义的（$-\infty$），所以 $D_{KL}(Q \| P) = +\infty$

### 前向 KL vs 反向 KL

| 类型 | 公式 | 行为 |
|------|------|------|
| 前向 KL | $D_{KL}(P \| Q)$ | Q 覆盖 P 的所有支撑 |
| 反向 KL | $D_{KL}(Q \| P)$ | Q 聚焦于 P 的高概率区域 |

### 深度学习应用

- **VAE**：KL 散度作为正则化项
- **知识蒸馏**：学生网络学习教师网络的分布
- **生成模型**：最小化生成分布与真实分布的 KL 散度

```python
import numpy as np
import matplotlib.pyplot as plt

def kl_divergence(p, q):
    """计算 KL 散度 D_KL(P || Q)"""
    # 过滤零概率
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

# 示例：两个高斯分布之间的 KL 散度
print("KL 散度示例:")
print("="*50)

# 离散分布示例
P = np.array([0.3, 0.4, 0.2, 0.1])
Q = np.array([0.25, 0.25, 0.25, 0.25])

kl_pq = kl_divergence(P, Q)
kl_qp = kl_divergence(Q, P)

print(f"P = {P}")
print(f"Q = {Q}")
print(f"D_KL(P || Q) = {kl_pq:.4f}")
print(f"D_KL(Q || P) = {kl_qp:.4f}")
print(f"不对称: D_KL(P||Q) ≠ D_KL(Q||P)")

# 高斯分布的 KL 散度（有解析解）
print("\n高斯分布的 KL 散度:")
print("-"*50)

def kl_gaussian(mu1, sigma1, mu2, sigma2):
    """
    计算 N(mu1, sigma1²) 和 N(mu2, sigma2²) 之间的 KL 散度
    D_KL(N1 || N2) = log(sigma2/sigma1) + (sigma1² + (mu1-mu2)²) / (2*sigma2²) - 1/2
    """
    return np.log(sigma2/sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5

# 示例
mu1, sigma1 = 0, 1
mu2, sigma2 = 1, 2

kl_n1_n2 = kl_gaussian(mu1, sigma1, mu2, sigma2)
kl_n2_n1 = kl_gaussian(mu2, sigma2, mu1, sigma1)

print(f"N1 = N({mu1}, {sigma1}²)")
print(f"N2 = N({mu2}, {sigma2}²)")
print(f"D_KL(N1 || N2) = {kl_n1_n2:.4f}")
print(f"D_KL(N2 || N1) = {kl_n2_n1:.4f}")

# 可视化
x = np.linspace(-5, 7, 200)
from scipy.stats import norm

plt.figure(figsize=(10, 5))
plt.plot(x, norm.pdf(x, mu1, sigma1), 'b-', linewidth=2, label=f'N1: N({mu1}, {sigma1}²)')
plt.plot(x, norm.pdf(x, mu2, sigma2), 'r-', linewidth=2, label=f'N2: N({mu2}, {sigma2}²)')
plt.xlabel('x')
plt.ylabel('密度')
plt.title(f'KL 散度: D_KL(N1||N2) = {kl_n1_n2:.4f}, D_KL(N2||N1) = {kl_n2_n1:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('kl_divergence.png', dpi=100)
print("\nKL 散度可视化图像已保存")
```

---

## 互信息 (Mutual Information)

### 定义

**互信息**衡量两个随机变量之间的**相互依赖程度**：

$$
I(X; Y) = \sum_{x,y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}
$$

### 等价形式

$$
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
$$

$$
I(X; Y) = H(X) + H(Y) - H(X, Y)
$$

$$
I(X; Y) = D_{KL}(P(X,Y) \| P(X)P(Y))
$$

### 性质

1. **非负性**：$I(X; Y) \geq 0$
2. **对称性**：$I(X; Y) = I(Y; X)$
3. **独立性**：$X$ 和 $Y$ 独立 $\Leftrightarrow$ $I(X; Y) = 0$
4. **自信息**：$I(X; X) = H(X)$

### 互信息的直观理解

- $I(X; Y)$：知道 $Y$ 后，关于 $X$ 的不确定性减少了多少
- 也是知道 $X$ 后，关于 $Y$ 的不确定性减少了多少

### 信息图 (Information Diagram)

```
        H(X)              H(Y)
    ┌───────────┐     ┌───────────┐
    │    ╭──────┴─────╶──────╮    │
    │    │   I(X;Y)    │    │
    │    ╰──────┬─────╶──────╯    │
    │ H(X|Y)    │    H(Y|X) │
    └───────────┴───────────┘
              H(X,Y)
```

```python
import numpy as np

def mutual_information(joint):
    """计算互信息 I(X; Y)"""
    # 边缘分布
    p_X = joint.sum(axis=1, keepdims=True)
    p_Y = joint.sum(axis=0, keepdims=True)
    
    # 独立时的联合分布
    p_independent = p_X * p_Y
    
    # I(X;Y) = D_KL(P(X,Y) || P(X)P(Y))
    # 过滤零概率
    mask = (joint > 0) & (p_independent > 0)
    mi = np.sum(joint[mask] * np.log(joint[mask] / p_independent[mask]))
    
    return mi

# 示例
print("互信息示例:")
print("="*50)

# 完全依赖（Y 完全由 X 决定）
joint_dependent = np.array([
    [0.25, 0, 0, 0],
    [0, 0.25, 0, 0],
    [0, 0, 0.25, 0],
    [0, 0, 0, 0.25]
])

# 独立
joint_independent = np.outer([0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25])

# 部分依赖
joint_partial = np.array([
    [0.2, 0.05, 0, 0],
    [0.05, 0.2, 0, 0],
    [0, 0, 0.2, 0.05],
    [0, 0, 0.05, 0.2]
])

mi_dependent = mutual_information(joint_dependent)
mi_independent = mutual_information(joint_independent)
mi_partial = mutual_information(joint_partial)

print(f"完全依赖: I(X;Y) = {mi_dependent:.4f} (最大依赖)")
print(f"独立: I(X;Y) = {mi_independent:.4f} (独立)")
print(f"部分依赖: I(X;Y) = {mi_partial:.4f}")

# 验证 I(X;Y) = H(X) - H(X|Y)
def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

p_X = joint_partial.sum(axis=1)
p_Y = joint_partial.sum(axis=0)
H_X = entropy(p_X)
H_XY = entropy(joint_partial.flatten())
H_X_given_Y = H_XY - H_Y

print(f"\n验证 I(X;Y) = H(X) - H(X|Y):")
print(f"H(X) = {H_X:.4f}")
print(f"H(X|Y) = {H_X_given_Y:.4f}")
print(f"H(X) - H(X|Y) = {H_X - H_X_given_Y:.4f}")
print(f"I(X;Y) = {mi_partial:.4f}")
```

---

## 信息论与机器学习的关系

### 关键联系

| 信息论概念 | 机器学习应用 |
|------------|--------------|
| 熵 | 决策树分裂准则 |
| 交叉熵 | 分类损失函数 |
| KL 散度 | VAE 正则化、知识蒸馏 |
| 互信息 | 特征选择、表征学习 |
| 信息增益 | 决策树、主动学习 |

### 交叉熵损失 = 最大似然估计

最小化交叉熵损失等价于最大化似然：

$$
\min_\theta H(P_{\text{data}}, P_\theta) \Leftrightarrow \max_\theta \mathbb{E}_{x \sim P_{\text{data}}}[\log P_\theta(x)]
$$

### 决策树的信息增益

选择使信息增益最大的特征进行分裂：

$$
IG(Y|X) = H(Y) - H(Y|X)
$$

信息增益 = 互信息 $I(X; Y)$

```python
import numpy as np

# 决策树信息增益示例
def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def information_gain(y, x):
    """
    计算特征 x 对目标 y 的信息增益
    IG(Y|X) = H(Y) - H(Y|X)
    """
    # H(Y)
    p_y = np.bincount(y) / len(y)
    H_y = entropy(p_y)
    
    # H(Y|X) = sum_x P(X=x) * H(Y|X=x)
    unique_x = np.unique(x)
    H_y_given_x = 0
    for val in unique_x:
        mask = x == val
        p_x = mask.mean()  # P(X=x)
        y_subset = y[mask]
        p_y_given_x = np.bincount(y_subset) / len(y_subset)
        H_y_given_x += p_x * entropy(p_y_given_x)
    
    return H_y - H_y_given_x

# 示例：是否打网球
# 特征：天气 (0=晴, 1=阴, 2=雨)
# 目标：是否打球 (0=否, 1=是)

weather = np.array([0, 0, 1, 2, 2, 2, 1, 0, 0, 2, 0, 1, 1, 2])
play = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])

print("决策树信息增益示例:")
print("="*50)
print(f"天气对打球的信息增益: {information_gain(play, weather):.4f}")

# 更多特征
temp = np.array([0, 0, 0, 1, 2, 2, 2, 1, 2, 1, 1, 1, 0, 1])  # 0=热, 1=温, 2=凉
humidity = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0])  # 0=高, 1=正常
windy = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1])  # 0=无风, 1=有风

print(f"温度对打球的信息增益: {information_gain(play, temp):.4f}")
print(f"湿度对打球的信息增益: {information_gain(play, humidity):.4f}")
print(f"风力对打球的信息增益: {information_gain(play, windy):.4f}")
```

---

## 在深度学习中的应用

### 1. 交叉熵损失函数

```python
import numpy as np

def softmax(logits):
    """数值稳定的 Softmax"""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def cross_entropy_loss(logits, targets):
    """
    交叉熵损失
    
    Parameters:
    -----------
    logits : array, shape (batch_size, num_classes)
        未归一化的预测分数
    targets : array, shape (batch_size,)
        类别索引
    """
    batch_size = logits.shape[0]
    
    # Softmax
    probs = softmax(logits)
    
    # 数值稳定性
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    
    # 交叉熵
    log_probs = np.log(probs[np.arange(batch_size), targets])
    loss = -np.mean(log_probs)
    
    return loss

# 示例
logits = np.array([
    [2.0, 1.0, 0.1],
    [0.1, 3.0, 0.5],
    [0.5, 0.5, 2.0]
])
targets = np.array([0, 1, 2])

loss = cross_entropy_loss(logits, targets)
print(f"交叉熵损失: {loss:.4f}")

# 与 MLE 的关系
print("\n交叉熵损失与 MLE 的关系:")
print("最小化交叉熵 = 最大化对数似然")
print("L = -1/N * sum(log p(y_i|x_i))")
print("= -1/N * sum(log softmax(z_i)[y_i])")
```

### 2. VAE 的 KL 散度正则化

```python
import numpy as np

def vae_kl_loss(mu, log_var):
    """
    VAE 中的 KL 散度损失
    
    KL(N(μ, σ²) || N(0, 1)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    """
    return -0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var))

def vae_loss(x_reconstructed, x, mu, log_var, beta=1.0):
    """
    β-VAE 损失函数
    
    L = Reconstruction Loss + β * KL Divergence
    """
    # 重构损失（假设高斯分布）
    recon_loss = np.mean((x_reconstructed - x) ** 2)
    
    # KL 散度
    kl_loss = vae_kl_loss(mu, log_var)
    
    return recon_loss + beta * kl_loss

# 示例
np.random.seed(42)
batch_size = 32
latent_dim = 10

mu = np.random.randn(batch_size, latent_dim)
log_var = np.random.randn(batch_size, latent_dim) * 0.1
x = np.random.randn(batch_size, 784)
x_recon = x + np.random.randn(batch_size, 784) * 0.1

loss = vae_loss(x_recon, x, mu, log_var)
print(f"VAE 损失: {loss:.4f}")
print(f"  KL 散度: {vae_kl_loss(mu, log_var):.4f}")
```

### 3. 知识蒸馏

```python
import numpy as np

def distillation_loss(teacher_logits, student_logits, temperature=2.0):
    """
    知识蒸馏损失
    
    使用软标签（温度 T）传递知识
    """
    # 软化概率分布
    teacher_probs = softmax(teacher_logits / temperature)
    student_probs = softmax(student_logits / temperature)
    
    # KL 散度
    # D_KL(teacher || student)
    mask = teacher_probs > 0
    kl = np.sum(teacher_probs[mask] * np.log(teacher_probs[mask] / student_probs[mask]))
    
    # 缩放（因为温度影响了熵）
    return kl * (temperature ** 2)

# 示例
np.random.seed(42)
teacher_logits = np.array([[2.0, 1.0, 0.1, 0.5, 0.3]])
student_logits = np.array([[1.5, 0.8, 0.2, 0.4, 0.2]])

print("知识蒸馏示例:")
print("="*50)
print(f"温度 T=1.0: 损失 = {distillation_loss(teacher_logits, student_logits, 1.0):.4f}")
print(f"温度 T=2.0: 损失 = {distillation_loss(teacher_logits, student_logits, 2.0):.4f}")
print(f"温度 T=4.0: 损失 = {distillation_loss(teacher_logits, student_logits, 4.0):.4f}")
print("\n较高温度使分布更平滑，传递更多'暗知识'")
```

### 4. Batch Normalization 的统计基础

```python
import numpy as np

# 大数定律在 Batch Normalization 中的应用
np.random.seed(42)

# 真实分布参数
true_mean = 5.0
true_var = 4.0

# 不同批次大小
batch_sizes = [8, 32, 128, 512, 2048]
n_trials = 1000

print("Batch Normalization 统计量估计:")
print("="*60)
print(f"{'批次大小':>10} | {'均值误差':>15} | {'方差误差':>15}")
print("-"*60)

for batch_size in batch_sizes:
    # 多次采样的均值和方差估计
    mean_estimates = []
    var_estimates = []
    
    for _ in range(n_trials):
        batch = np.random.normal(true_mean, np.sqrt(true_var), batch_size)
        mean_estimates.append(batch.mean())
        var_estimates.append(batch.var(ddof=1))  # 无偏估计
    
    mean_error = np.abs(np.mean(mean_estimates) - true_mean)
    var_error = np.abs(np.mean(var_estimates) - true_var)
    
    print(f"{batch_size:>10} | {mean_error:>15.6f} | {var_error:>15.6f}")

print("\n结论: 批次越大，统计量估计越准确（大数定律）")
```

### 5. Diffusion 模型的数学基础

Diffusion 模型是当前最先进的生成模型之一，其数学基础涉及随机过程、随机微分方程和分数匹配。

#### 前向扩散过程

**定义**：逐步向数据添加高斯噪声，直到变成纯噪声。

$$
\mathbf{x}_t = \sqrt{1-\beta_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}_{t-1}, \quad \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

其中 $\beta_t$ 是噪声调度（noise schedule），通常 $\beta_1 < \beta_2 < \cdots < \beta_T$。

**重参数化**：设 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$，则可以直接从 $\mathbf{x}_0$ 采样任意时刻的 $\mathbf{x}_t$：

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

**推导**：

**第一步**：展开前两步。

$$\mathbf{x}_1 = \sqrt{\alpha_1}\mathbf{x}_0 + \sqrt{1-\alpha_1}\boldsymbol{\epsilon}_0$$

$$\mathbf{x}_2 = \sqrt{\alpha_2}\mathbf{x}_1 + \sqrt{1-\alpha_2}\boldsymbol{\epsilon}_1$$

$$= \sqrt{\alpha_2}\left(\sqrt{\alpha_1}\mathbf{x}_0 + \sqrt{1-\alpha_1}\boldsymbol{\epsilon}_0\right) + \sqrt{1-\alpha_2}\boldsymbol{\epsilon}_1$$

$$= \sqrt{\alpha_1\alpha_2}\mathbf{x}_0 + \sqrt{\alpha_2(1-\alpha_1)}\boldsymbol{\epsilon}_0 + \sqrt{1-\alpha_2}\boldsymbol{\epsilon}_1$$

**第二步**：合并独立高斯噪声。

两个独立高斯 $\mathcal{N}(0, \sigma_1^2)$ 和 $\mathcal{N}(0, \sigma_2^2)$ 的加权和：

$$\sigma_1\epsilon_1 + \sigma_2\epsilon_2 \sim \mathcal{N}(0, \sigma_1^2 + \sigma_2^2)$$

因此：
$$\sqrt{\alpha_2(1-\alpha_1)}\boldsymbol{\epsilon}_0 + \sqrt{1-\alpha_2}\boldsymbol{\epsilon}_1 \sim \mathcal{N}\left(\mathbf{0}, (1-\alpha_1)\alpha_2 + (1-\alpha_2)\mathbf{I}\right)$$

$$= \mathcal{N}\left(\mathbf{0}, (1 - \alpha_1\alpha_2)\mathbf{I}\right) = \mathcal{N}\left(\mathbf{0}, (1 - \bar{\alpha}_2)\mathbf{I}\right)$$

**第三步**：归纳得到一般形式。

$$\boxed{\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}$$

**直观理解**：
- 当 $t=0$ 时，$\bar{\alpha}_0=1$，$\mathbf{x}_0$ 是原始数据
- 当 $t=T$ 时，$\bar{\alpha}_T\approx 0$，$\mathbf{x}_T\approx\boldsymbol{\epsilon}$ 是纯噪声

#### 逆向扩散过程

**目标**：学习逆向过程 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$，从噪声重建数据。

**后验分布**（已知 $\mathbf{x}_0$ 时）：

$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\tilde{\boldsymbol{\mu}}_t, \tilde{\beta}_t\mathbf{I})
$$

其中：

$$
\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t
$$

$$
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
$$

**后验均值公式的推导**：

使用贝叶斯公式和高斯分布的乘积性质。

**第一步**：写出条件分布。

$$q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \propto q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0) \cdot q(\mathbf{x}_{t-1}|\mathbf{x}_0)$$

**第二步**：展开两项。

$$q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$$

$$q(\mathbf{x}_{t-1}|\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0, (1-\bar{\alpha}_{t-1})\mathbf{I})$$

**第三步**：利用高斯乘积公式。

两个高斯的乘积仍是高斯，其均值和方差可通过配方得到：

$$\boxed{\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t}$$

#### 训练目标：分数匹配

**关键洞察**：神经网络预测噪声 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$。

**损失函数**（简化版）：

$$
\mathcal{L}_{simple} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]
$$

其中 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$。

**为什么预测噪声等价于学习分数函数**：

分数函数定义为 $\nabla_{\mathbf{x}}\log p(\mathbf{x})$。

**第一步**：写出 $\mathbf{x}_t$ 的分布。

$$p(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

$$\log p(\mathbf{x}_t|\mathbf{x}_0) = -\frac{1}{2(1-\bar{\alpha}_t)}\|\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0\|^2 + C$$

**第二步**：计算分数。

$$\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t|\mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{1-\bar{\alpha}_t} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar{\alpha}_t}}$$

**第三步**：建立等价关系。

$$\boxed{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) = -\sqrt{1-\bar{\alpha}_t} \cdot \nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)}$$

预测噪声 = 预测分数函数（带缩放）

```python
import numpy as np

class SimpleDiffusion:
    """简化的 Diffusion 模型实现"""
    
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        
        # 线性噪声调度
        self.betas = np.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
        # 预计算
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1 - self.alphas_cumprod)
        
    def q_sample(self, x_0, t, noise=None):
        """前向扩散：从 x_0 采样 x_t"""
        if noise is None:
            noise = np.random.randn(*x_0.shape)
        
        return (
            self.sqrt_alphas_cumprod[t] * x_0 +
            self.sqrt_one_minus_alphas_cumprod[t] * noise
        )
    
    def predict_start_from_noise(self, x_t, t, noise_pred):
        """从预测的噪声重建 x_0"""
        return (
            x_t - self.sqrt_one_minus_alphas_cumprod[t] * noise_pred
        ) / self.sqrt_alphas_cumprod[t]
    
    def p_mean_variance(self, model_pred, x_t, t):
        """计算逆向过程的均值和方差"""
        # 预测 x_0
        x_0_pred = self.predict_start_from_noise(x_t, t, model_pred)
        
        # 计算均值（使用后验公式）
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod[t-1] if t > 0 else 1.0
        
        mean = (
            np.sqrt(alpha_cumprod_prev) * (1 - alpha_t) / (1 - alpha_cumprod_t) * x_0_pred +
            np.sqrt(alpha_t) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * x_t
        )
        
        # 方差
        var = (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * self.betas[t]
        
        return mean, var
    
    def p_sample(self, model_pred, x_t, t):
        """单步逆向采样"""
        mean, var = self.p_mean_variance(model_pred, x_t, t)
        
        if t == 0:
            return mean
        
        noise = np.random.randn(*x_t.shape)
        return mean + np.sqrt(var) * noise
    
    def training_loss(self, x_0, noise_pred, noise_true):
        """计算训练损失"""
        return np.mean((noise_pred - noise_true) ** 2)

# 示例
diffusion = SimpleDiffusion(timesteps=1000)

# 模拟数据
x_0 = np.random.randn(32, 3, 32, 32)  # 一批 32x32 RGB 图像
t = 500  # 中间时刻

# 前向扩散
noise = np.random.randn(*x_0.shape)
x_t = diffusion.q_sample(x_0, t, noise)

# 假设模型预测（实际中是神经网络）
noise_pred = noise + np.random.randn(*x_0.shape) * 0.1  # 加点噪声模拟不完美预测

# 计算损失
loss = diffusion.training_loss(x_0, noise_pred, noise)
print(f"训练损失: {loss:.4f}")

# 重建 x_0
x_0_reconstructed = diffusion.predict_start_from_noise(x_t, t, noise_pred)
reconstruction_error = np.mean((x_0 - x_0_reconstructed) ** 2)
print(f"重建误差: {reconstruction_error:.4f}")
```

#### 与其他生成模型的联系

| 模型 | 核心思想 | 损失函数 |
|------|---------|---------|
| VAE | 变分推断 + 重参数化 | $\mathcal{L} = \text{Recon} + \text{KL}$ |
| GAN | 对抗训练 | Min-Max 博弈 |
| Diffusion | 逐步去噪 | $\mathbb{E}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2]$ |
| Flow | 可逆变换 | $\log p(\mathbf{x}) = \log p(\mathbf{z}) + \log\|\det \mathbf{J}\|$ |

### 6. GAN 的博弈论基础

生成对抗网络（GAN）的数学基础是博弈论中的极小极大博弈（Min-Max Game）。

#### 基本框架

**生成器** $G$：从噪声 $\mathbf{z}$ 生成假样本 $G(\mathbf{z})$

**判别器** $D$：区分真实样本 $\mathbf{x}$ 和假样本 $G(\mathbf{z})$

**目标函数**：

$$
\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{data}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]
$$

#### 理论分析：最优判别器

**定理**：对于固定的生成器 $G$，最优判别器为：

$$
D^*(\mathbf{x}) = \frac{p_{data}(\mathbf{x})}{p_{data}(\mathbf{x}) + p_g(\mathbf{x})}
$$

**证明**：

**第一步**：展开目标函数。

$$V(D, G) = \int_{\mathbf{x}} p_{data}(\mathbf{x})\log D(\mathbf{x})\,d\mathbf{x} + \int_{\mathbf{z}} p_z(\mathbf{z})\log(1-D(G(\mathbf{z})))\,d\mathbf{z}$$

**第二步**：变量替换。

设 $\mathbf{x} = G(\mathbf{z})$，则：

$$\int_{\mathbf{z}} p_z(\mathbf{z})\log(1-D(G(\mathbf{z})))\,d\mathbf{z} = \int_{\mathbf{x}} p_g(\mathbf{x})\log(1-D(\mathbf{x}))\,d\mathbf{x}$$

**第三步**：合并积分。

$$V(D, G) = \int_{\mathbf{x}} \left[p_{data}(\mathbf{x})\log D(\mathbf{x}) + p_g(\mathbf{x})\log(1-D(\mathbf{x}))\right]\,d\mathbf{x}$$

**第四步**：对每点 $\mathbf{x}$ 最大化被积函数。

$$f(D) = a\log D + b\log(1-D), \quad a = p_{data}(\mathbf{x}), \quad b = p_g(\mathbf{x})$$

求导并令其为零：

$$\frac{df}{dD} = \frac{a}{D} - \frac{b}{1-D} = 0$$

$$a(1-D) = bD$$

$$D^* = \frac{a}{a+b} = \frac{p_{data}(\mathbf{x})}{p_{data}(\mathbf{x}) + p_g(\mathbf{x})}$$

$$\boxed{D^*(\mathbf{x}) = \frac{p_{data}(\mathbf{x})}{p_{data}(\mathbf{x}) + p_g(\mathbf{x})}}$$

#### 理论分析：最优生成器

**定理**：当 $p_g = p_{data}$ 时，生成器达到最优，此时 $V(D^*, G) = -\log 4$。

**证明**：

**第一步**：将最优判别器代入目标函数。

$$V(D^*, G) = \mathbb{E}_{\mathbf{x} \sim p_{data}}\left[\log \frac{p_{data}(\mathbf{x})}{p_{data}(\mathbf{x}) + p_g(\mathbf{x})}\right] + \mathbb{E}_{\mathbf{x} \sim p_g}\left[\log \frac{p_g(\mathbf{x})}{p_{data}(\mathbf{x}) + p_g(\mathbf{x})}\right]$$

**第二步**：与 KL 散度联系。

注意到：

$$V(D^*, G) = -\log 4 + D_{KL}(p_{data}\|p_{data}+p_g) + D_{KL}(p_g\|p_{data}+p_g)$$

**第三步**：引入 Jensen-Shannon 散度（JSD）。

$$JSD(P\|Q) = \frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M), \quad M = \frac{P+Q}{2}$$

因此：

$$V(D^*, G) = -\log 4 + 2 \cdot JSD(p_{data}\|p_g)$$

**第四步**：最优情况。

JSD 非负，且当 $p_g = p_{data}$ 时为 0：

$$\boxed{\min_G V(D^*, G) = -\log 4 \approx -1.386}$$

此时 $D^*(\mathbf{x}) = 0.5$ 对所有 $\mathbf{x}$ 成立。

```python
import numpy as np

class SimpleGAN:
    """简化的 GAN 实现（线性生成器和判别器）"""
    
    def __init__(self, data_dim, noise_dim):
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        
        # 初始化参数
        self.G_W = np.random.randn(noise_dim, data_dim) * 0.01
        self.G_b = np.zeros(data_dim)
        self.D_W = np.random.randn(data_dim, 1) * 0.01
        self.D_b = np.zeros(1)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def generate(self, z):
        """生成器前向传播"""
        return z @ self.G_W + self.G_b
    
    def discriminate(self, x):
        """判别器前向传播"""
        return self.sigmoid(x @ self.D_W + self.D_b)
    
    def train_discriminator(self, real_data, fake_data, lr=0.01):
        """训练判别器"""
        batch_size = real_data.shape[0]
        
        # 前向传播
        d_real = self.discriminate(real_data)
        d_fake = self.discriminate(fake_data)
        
        # 判别器损失: -[log(D(x)) + log(1-D(G(z)))]
        d_loss = -(np.log(d_real + 1e-8).mean() + np.log(1 - d_fake + 1e-8).mean())
        
        # 反向传播（简化版）
        d_real_grad = (d_real - 1) / batch_size
        d_fake_grad = d_fake / batch_size
        
        # 更新判别器
        dW = real_data.T @ d_real_grad + fake_data.T @ d_fake_grad
        db = d_real_grad.sum() + d_fake_grad.sum()
        
        self.D_W -= lr * dW
        self.D_b -= lr * db
        
        return d_loss
    
    def train_generator(self, fake_data, lr=0.01):
        """训练生成器"""
        batch_size = fake_data.shape[0]
        
        # 前向传播
        d_fake = self.discriminate(fake_data)
        
        # 生成器损失: -log(D(G(z))) (非饱和版本)
        g_loss = -np.log(d_fake + 1e-8).mean()
        
        # 反向传播
        d_fake_grad = (d_fake - 1) / batch_size
        
        # 通过判别器反向传播到生成器
        dW = d_fake_grad @ self.D_W.T
        db = d_fake_grad.sum(axis=0)
        
        # 更新生成器
        gW_grad = np.random.randn(self.noise_dim, batch_size) @ dW / batch_size
        gb_grad = db
        
        self.G_W -= lr * gW_grad
        self.G_b -= lr * gb_grad
        
        return g_loss

# 演示
np.random.seed(42)
data_dim = 2
noise_dim = 10

gan = SimpleGAN(data_dim, noise_dim)

# 真实数据分布：二维高斯
real_mean = np.array([1, 1])
real_cov = np.array([[0.5, 0.2], [0.2, 0.5]])

print("GAN 训练演示:")
print("="*50)

for epoch in range(1000):
    # 采样真实数据
    real_data = np.random.multivariate_normal(real_mean, real_cov, 32)
    
    # 采样噪声并生成假数据
    z = np.random.randn(32, noise_dim)
    fake_data = gan.generate(z)
    
    # 训练判别器
    d_loss = gan.train_discriminator(real_data, fake_data)
    
    # 重新生成假数据（因为判别器已更新）
    z = np.random.randn(32, noise_dim)
    fake_data = gan.generate(z)
    
    # 训练生成器
    g_loss = gan.train_generator(fake_data)
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")

print("\n理论最优损失: -log(4) ≈ -1.386 (对应 D_loss ≈ 0.693)")
```

#### GAN 训练的挑战

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 模式崩溃 | 生成器只产生少数模式 | Minibatch discrimination, WGAN |
| 训练不稳定 | 判别器过强 | 渐进式训练, 标签平滑 |
| 梯度消失 | 判别器太准 | 非饱和损失, WGAN-GP |

#### Wasserstein GAN (WGAN)

使用 Wasserstein 距离替代 JS 散度：

$$
W(P_r, P_g) = \inf_{\gamma \in \Pi(P_r, P_g)} \mathbb{E}_{(x,y)\sim\gamma}[\|x-y\|]
$$

**Kantorovich-Rubinstein 对偶**：

$$
W(P_r, P_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x\sim P_r}[f(x)] - \mathbb{E}_{x\sim P_g}[f(x)]
$$

其中 $\|f\|_L \leq 1$ 表示 $f$ 是 1-Lipschitz 函数。

**WGAN 损失**：

$$
\min_G \max_{D \in \mathcal{D}} \mathbb{E}_{x\sim P_r}[D(x)] - \mathbb{E}_{z\sim p_z}[D(G(z))]
$$

WGAN 提供了更有意义的梯度，训练更稳定。

---

## 小结

本章介绍了概率论的两大支柱——极限定理和信息论，它们在深度学习中有着广泛的应用。

### 极限定理总结

| 定理 | 内容 | 应用 |
|------|------|------|
| 大数定律 | $\bar{X}_n \xrightarrow{P} \mu$ | 批统计量估计、SGD 收敛 |
| 中心极限定理 | $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2)$ | 置信区间、权重初始化 |

### 信息论总结

| 概念 | 公式 | 含义 | 应用 |
|------|------|------|------|
| 熵 | $H(X) = -\sum p \log p$ | 不确定性 | 决策树 |
| 交叉熵 | $H(P,Q) = -\sum p \log q$ | 编码效率 | 分类损失 |
| KL 散度 | $D_{KL} = \sum p \log(p/q)$ | 分布距离 | VAE 正则化 |
| 互信息 | $I(X;Y) = H(X) - H(X\|Y)$ | 相互依赖 | 特征选择 |

### 核心要点

1. **大数定律**：样本量足够大时，样本统计量趋于总体参数
2. **中心极限定理**：样本均值近似正态分布
3. **熵**：衡量不确定性，等概率时最大
4. **交叉熵**：机器学习最常用的损失函数
5. **KL 散度**：衡量分布差异，非对称
6. **互信息**：衡量变量间的依赖关系

### 深度学习中的核心公式


交叉熵损失: $L = -log p(y_true)$

VAE 损失:   $L = Reconstruction + β·KL(q(z|x) \|\| p(z))$

知识蒸馏:   $L = KL(teacher || student) · T²$

信息增益:   $IG = H(Y) - H(Y\|X) = I(X; Y)$


---

**上一节**：[第三章（c）：多维随机变量与数字特征](03c-多维随机变量与数字特征.md)

**下一章**：[第四章：数理统计](04-statistics.md) - 学习统计推断、参数估计、假设检验等概念。

**返回**：[数学基础教程目录](../math-fundamentals.md)
