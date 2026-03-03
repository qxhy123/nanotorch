# 第七章（c）：级数与求和

级数是数列求和的自然延伸，在深度学习中有着广泛的应用。从正则化项的分析到 RNN 梯度传播，从位置编码的设计到注意力机制的理论基础，级数的概念无处不在。本节将系统介绍级数的基本概念、收敛判别法和求和技巧。

---

## 目录

1. [级数的基本概念](#级数的基本概念)
2. [正项级数的判别法](#正项级数的判别法)
3. [交错级数](#交错级数)
4. [幂级数](#幂级数)
5. [常见级数的求和](#常见级数的求和)
6. [在深度学习中的应用](#在深度学习中的应用)

---

## 级数的基本概念

### 🎯 生活类比：存钱罐记账

想象你每天往存钱罐存钱，想知道最后总共有多少：
- 第1天存：1元
- 第2天存：0.5元
- 第3天存：0.25元
- ...

**级数就是把所有项加起来！**

```
数列：1, 0.5, 0.25, 0.125, ...
级数：1 + 0.5 + 0.25 + 0.125 + ... = ?
```

**关键问题**：无限加下去，总和是无穷大还是一个具体的数？

### 📖 收敛 vs 发散

**类比**：一个水桶接水
- **收敛**：水桶有底，水位会稳定在某个高度
- **发散**：水桶没底，水一直流，永远装不满

**级数的两种命运**：
- **收敛**：加到某一点后，总和趋于稳定值
- **发散**：总和越来越大，趋向无穷大

### 📝 手把手例子

**例1：几何级数** $\frac{1}{2} + \frac{1}{4} + \frac{1}{8} + \ldots$

```
部分和：
S₁ = 0.5
S₂ = 0.5 + 0.25 = 0.75
S₃ = 0.75 + 0.125 = 0.875
S₄ = 0.875 + 0.0625 = 0.9375
S₅ = 0.9375 + 0.03125 = 0.96875
...
越接近 1！

结论：级数收敛，和 = 1
```

**例2：调和级数** $1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \ldots$

```
部分和：
S₁ = 1
S₂ = 1.5
S₄ = 2.083
S₁₀ = 2.929
S₁₀₀ = 5.187
S₁₀₀₀ = 7.485
...
增长越来越慢，但是一直在增长！

结论：级数发散！（趋向无穷大）
```

### 定义

**级数**是将数列的项依次相加：

$$
\sum_{n=1}^{\infty} a_n = a_1 + a_2 + a_3 + \cdots
$$

**部分和**：
$$
S_n = \sum_{k=1}^{n} a_k = a_1 + a_2 + \cdots + a_n
$$

### 收敛与发散

- **收敛**：若部分和序列 $\{S_n\}$ 收敛于 $S$，则称级数收敛，$S$ 称为级数的和
  $$
  \sum_{n=1}^{\infty} a_n = S = \lim_{n \to \infty} S_n
  $$

- **发散**：若 $\{S_n\}$ 发散，则称级数发散

### 收敛的必要条件

**定理**：若 $\sum a_n$ 收敛，则 $\lim_{n \to \infty} a_n = 0$。

**注意**：这是必要条件而非充分条件！

例：调和级数 $\sum \frac{1}{n}$ 发散，虽然 $\frac{1}{n} \to 0$。

```python
import numpy as np
import matplotlib.pyplot as plt

def partial_sum(sequence, n_terms):
    """计算部分和"""
    return np.sum(sequence[:n_terms])

def visualize_series(sequence_func, name, n_max=50):
    """可视化级数部分和"""
    terms = np.array([sequence_func(n) for n in range(1, n_max + 1)])
    partial_sums = np.cumsum(terms)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 级数项
    axes[0].stem(range(1, n_max + 1), terms, basefmt=' ')
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('$a_n$')
    axes[0].set_title(f'级数项: $a_n = {name}$')
    axes[0].grid(True, alpha=0.3)
    
    # 部分和
    axes[1].plot(range(1, n_max + 1), partial_sums, 'b-', linewidth=2)
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('$S_n$')
    axes[1].set_title(f'部分和: $S_n = \\sum_{{k=1}}^n a_k$')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"前10项部分和: {partial_sums[:10]}")
    print(f"最后5项部分和: {partial_sums[-5:]}")

# 示例：几何级数（收敛）
visualize_series(lambda n: 0.5**n, "(0.5)^n")

# 示例：调和级数（发散）
visualize_series(lambda n: 1/n, "1/n")
```

---

## 正项级数的判别法

对于正项级数 $\sum a_n$（$a_n \geq 0$），有专门的收敛判别法。

### 比较判别法

**定理**：设 $0 \leq a_n \leq b_n$，则：
- 若 $\sum b_n$ 收敛，则 $\sum a_n$ 收敛
- 若 $\sum a_n$ 发散，则 $\sum b_n$ 发散

**极限形式**：若 $\lim_{n \to \infty} \frac{a_n}{b_n} = c$（$0 < c < \infty$），则两级数同敛散。

### 比值判别法（D'Alembert）

**定理**：设 $\lim_{n \to \infty} \frac{a_{n+1}}{a_n} = r$，则：
- $r < 1$：级数收敛
- $r > 1$：级数发散
- $r = 1$：无法判断

### 根值判别法（Cauchy）

**定理**：设 $\lim_{n \to \infty} \sqrt[n]{a_n} = r$，则：
- $r < 1$：级数收敛
- $r > 1$：级数发散
- $r = 1$：无法判断

### 积分判别法

**定理**：设 $f(x)$ 在 $[1, +\infty)$ 上非负、连续、单调递减，$a_n = f(n)$，则：
$$
\sum_{n=1}^{\infty} a_n \text{ 收敛} \Leftrightarrow \int_1^{\infty} f(x)\,dx \text{ 收敛}
$$

### 收敛判别法的证明

#### 比值判别法（D'Alembert）的证明

**定理**：设 $\lim_{n \to \infty} \frac{a_{n+1}}{a_n} = r$，则当 $r < 1$ 时级数收敛，$r > 1$ 时级数发散。

**证明（$r < 1$ 情形）**：

**Step 1：选择比较几何级数**

设 $r < 1$，取 $\epsilon > 0$ 使得 $r + \epsilon < 1$。由极限定义，存在 $N$，当 $n \geq N$ 时：

$$
\frac{a_{n+1}}{a_n} < r + \epsilon \quad \Rightarrow \quad a_{n+1} < (r + \epsilon) a_n
$$

**Step 2：建立不等式**

递推得：

$$
a_{N+k} < (r + \epsilon)^k a_N
$$

**Step 3：与几何级数比较**

因此：

$$
\sum_{k=N}^{\infty} a_k = \sum_{k=0}^{\infty} a_{N+k} < a_N \sum_{k=0}^{\infty} (r + \epsilon)^k
$$

由于 $r + \epsilon < 1$，右边几何级数收敛，故原级数收敛。

**$r > 1$ 情形**：类似可证 $a_n$ 不趋于零，故发散。 $\square$

#### 根值判别法（Cauchy）的证明

**定理**：设 $\lim_{n \to \infty} \sqrt[n]{a_n} = r$，则当 $r < 1$ 时级数收敛，$r > 1$ 时级数发散。

**证明**：

**Step 1**：由极限定义，对任意 $\epsilon > 0$，存在 $N$，当 $n \geq N$ 时：

$$
|\sqrt[n]{a_n} - r| < \epsilon \quad \Rightarrow \quad a_n < (r + \epsilon)^n
$$

**Step 2**：若 $r < 1$，取 $\epsilon$ 使得 $r + \epsilon < 1$，则：

$$
\sum_{n=N}^{\infty} a_n < \sum_{n=N}^{\infty} (r + \epsilon)^n
$$

右边是收敛的几何级数，故原级数收敛。

**Step 3**：若 $r > 1$，则 $a_n > 1$ 对无穷多个 $n$ 成立，故 $\lim a_n \neq 0$，级数发散。 $\square$

#### 积分判别法的证明

**定理**：设 $f(x)$ 在 $[1, +\infty)$ 上非负、连续、单调递减，$a_n = f(n)$，则级数与积分同敛散。

**证明**：

**Step 1：建立积分与和的关系**

由于 $f$ 单调递减，对 $k \leq x \leq k+1$，有 $f(k+1) \leq f(x) \leq f(k)$。

在 $[k, k+1]$ 上积分：

$$
f(k+1) \leq \int_k^{k+1} f(x)\,dx \leq f(k)
$$

**Step 2：求和得到不等式**

从 $k = 1$ 到 $n$ 求和：

$$
\sum_{k=2}^{n+1} f(k) \leq \int_1^{n+1} f(x)\,dx \leq \sum_{k=1}^{n} f(k)
$$

**Step 3：分析敛散性**

令 $S_n = \sum_{k=1}^{n} f(k)$，$I_n = \int_1^{n} f(x)\,dx$，则：

$$
S_{n+1} - f(1) \leq I_{n+1} \leq S_n
$$

- 若 $I_n$ 收敛，则 $S_n$ 有上界（由右不等式），故 $S_n$ 收敛。
- 若 $S_n$ 收敛，则 $I_n$ 有上界（由左不等式），故 $I_n$ 收敛。

因此级数与积分同敛散。 $\square$

```python
def test_convergence_ratio(a_func, n_max=100):
    """比值判别法测试"""
    ratios = []
    for n in range(1, n_max):
        a_n = a_func(n)
        a_next = a_func(n+1)
        if a_n > 1e-10:
            ratios.append(a_next / a_n)
    
    if ratios:
        limit = np.mean(ratios[-10:])  # 用最后几项的平均估计极限
        print(f"比值判别法: lim(a_{{n+1}}/a_n) ≈ {limit:.4f}")
        if limit < 1:
            print(f"  → 收敛 (因为 {limit:.4f} < 1)")
        elif limit > 1:
            print(f"  → 发散 (因为 {limit:.4f} > 1)")
        else:
            print(f"  → 无法判断 (因为 {limit:.4f} = 1)")
    
    return ratios

def test_convergence_root(a_func, n_max=100):
    """根值判别法测试"""
    roots = []
    for n in range(1, n_max):
        a_n = a_func(n)
        if a_n > 0:
            roots.append(a_n ** (1/n))
    
    if roots:
        limit = np.mean(roots[-10:])
        print(f"根值判别法: lim(a_n^(1/n)) ≈ {limit:.4f}")
        if limit < 1:
            print(f"  → 收敛 (因为 {limit:.4f} < 1)")
        elif limit > 1:
            print(f"  → 发散 (因为 {limit:.4f} > 1)")
        else:
            print(f"  → 无法判断 (因为 {limit:.4f} = 1)")
    
    return roots

# 测试几何级数
print("=== 几何级数 Σ(0.5)^n ===")
test_convergence_ratio(lambda n: 0.5**n)

# 测试调和级数
print("\n=== 调和级数 Σ1/n ===")
test_convergence_ratio(lambda n: 1/n)

# 测试 p-级数 (p=2)
print("\n=== p-级数 Σ1/n² (p=2) ===")
test_convergence_ratio(lambda n: 1/n**2)
```

### 常见正项级数

| 级数 | 收敛条件 | 和（若收敛） |
|------|----------|--------------|
| 几何级数 $\sum r^n$ | $\|r\| < 1$ | $\frac{1}{1-r}$ |
| p-级数 $\sum \frac{1}{n^p}$ | $p > 1$ | 无闭式解 |
| 调和级数 $\sum \frac{1}{n}$ | 发散 | - |

---

## 交错级数

### 定义

**交错级数**是正负项交替出现的级数：

$$
\sum_{n=1}^{\infty} (-1)^{n-1} a_n = a_1 - a_2 + a_3 - a_4 + \cdots
$$

### Leibniz 判别法

**定理**：若 $\{a_n\}$ 单调递减且 $\lim_{n \to \infty} a_n = 0$，则交错级数 $\sum (-1)^{n-1} a_n$ 收敛。

**误差估计**：若用 $S_n$ 近似级数和 $S$，则误差：
$$
|S - S_n| \leq a_{n+1}
$$

### Leibniz 判别法的证明

**定理**：若 $\{a_n\}$ 单调递减趋于零，则交错级数 $\sum (-1)^{n-1} a_n$ 收敛。

**证明**：

**Step 1：分析偶数次部分和**

设 $S_n = \sum_{k=1}^{n} (-1)^{k-1} a_k$，考虑偶数次部分和 $S_{2m}$：

$$
S_{2m} = (a_1 - a_2) + (a_3 - a_4) + \cdots + (a_{2m-1} - a_{2m})
$$

由于 $a_n$ 递减，每个括号内 $a_{2k-1} - a_{2k} \geq 0$，故 $S_{2m}$ 单调递增。

**Step 2：重新分组**

另一方面：

$$
S_{2m} = a_1 - (a_2 - a_3) - (a_4 - a_5) - \cdots - (a_{2m-2} - a_{2m-1}) - a_{2m}
$$

每个括号内非负，故 $S_{2m} \leq a_1$。

**Step 3：偶数次部分和的极限**

$\{S_{2m}\}$ 单调递增且有上界 $a_1$，故存在极限 $S = \lim_{m \to \infty} S_{2m}$。

**Step 4：分析奇数次部分和**

对于奇数次部分和：

$$
S_{2m+1} = S_{2m} + a_{2m+1}
$$

由 $\lim a_n = 0$，得：

$$
\lim_{m \to \infty} S_{2m+1} = \lim_{m \to \infty} S_{2m} + \lim_{m \to \infty} a_{2m+1} = S + 0 = S
$$

**Step 5：得出收敛性**

偶数次和奇数次部分和都趋于 $S$，故 $\lim_{n \to \infty} S_n = S$，级数收敛。 $\square$

**误差估计的证明**：

由上述证明，$S_{2m} \leq S \leq S_{2m+1}$，故：

$$
|S - S_{2m}| \leq S_{2m+1} - S_{2m} = a_{2m+1}
$$

同理 $|S - S_{2m+1}| \leq a_{2m+2}$，一般地 $|S - S_n| \leq a_{n+1}$。 $\square$

```python
def alternating_series_sum(a_func, n_terms, true_sum=None):
    """交错级数求和"""
    terms = [((-1)**(n-1)) * a_func(n) for n in range(1, n_terms + 1)]
    partial_sums = np.cumsum(terms)
    
    print(f"交错级数前{n_terms}项部分和: {partial_sums[-1]:.6f}")
    
    if true_sum is not None:
        error = abs(partial_sums[-1] - true_sum)
        next_term = a_func(n_terms + 1)
        print(f"实际误差: {error:.6f}")
        print(f"Leibniz误差上界: {next_term:.6f}")
        print(f"误差 <= 下一项: {error <= next_term}")
    
    return partial_sums

# 交错调和级数: 1 - 1/2 + 1/3 - 1/4 + ... = ln(2)
print("=== 交错调和级数 ===")
print("理论值: ln(2) ≈", np.log(2))
alternating_series_sum(lambda n: 1/n, 100, np.log(2))

# 可视化
n_max = 50
terms = [((-1)**(n-1)) / n for n in range(1, n_max + 1)]
partial_sums = np.cumsum(terms)

plt.figure(figsize=(10, 5))
plt.plot(range(1, n_max + 1), partial_sums, 'b-', label='部分和 $S_n$')
plt.axhline(y=np.log(2), color='r', linestyle='--', label=f'$\\ln(2) = {np.log(2):.4f}$')
plt.xlabel('n')
plt.ylabel('$S_n$')
plt.title('交错调和级数: $\\sum (-1)^{{n-1}}/n = \\ln(2)$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 绝对收敛与条件收敛

- **绝对收敛**：$\sum |a_n|$ 收敛 $\Rightarrow$ $\sum a_n$ 收敛
- **条件收敛**：$\sum a_n$ 收敛但 $\sum |a_n|$ 发散

**性质**：
- 绝对收敛级数可以任意重排，和不变
- 条件收敛级数重排后可能改变和（Riemann 重排定理）

---

## 幂级数

### 定义

**幂级数**是形如：
$$
\sum_{n=0}^{\infty} c_n (x-a)^n = c_0 + c_1(x-a) + c_2(x-a)^2 + \cdots
$$

的级数。当 $a=0$ 时：
$$
\sum_{n=0}^{\infty} c_n x^n = c_0 + c_1x + c_2x^2 + \cdots
$$

### 收敛半径

**定理**：幂级数的收敛域是一个区间 $(-R, R)$，其中 $R$ 称为收敛半径：

$$
R = \frac{1}{\limsup_{n \to \infty} \sqrt[n]{|c_n|}}
$$

或用比值法：
$$
R = \lim_{n \to \infty} \left| \frac{c_n}{c_{n+1}} \right|
$$

- 当 $|x| < R$：级数绝对收敛
- 当 $|x| > R$：级数发散
- 当 $|x| = R$：需要单独判断

### 常见幂级数展开

| 函数 | 幂级数展开 | 收敛域 |
|------|-----------|--------|
| $e^x$ | $\sum_{n=0}^{\infty} \frac{x^n}{n!}$ | $(-\infty, +\infty)$ |
| $\sin x$ | $\sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!}$ | $(-\infty, +\infty)$ |
| $\cos x$ | $\sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!}$ | $(-\infty, +\infty)$ |
| $\frac{1}{1-x}$ | $\sum_{n=0}^{\infty} x^n$ | $(-1, 1)$ |
| $\ln(1+x)$ | $\sum_{n=1}^{\infty} \frac{(-1)^{n-1} x^n}{n}$ | $(-1, 1]$ |
| $(1+x)^\alpha$ | $\sum_{n=0}^{\infty} \binom{\alpha}{n} x^n$ | $(-1, 1)$ |

```python
def power_series_approximation(x, series_func, name, true_func, max_terms=20):
    """幂级数近似可视化"""
    terms = [series_func(n) * (x**n) for n in range(max_terms)]
    partial_sums = np.cumsum(terms)
    
    true_value = true_func(x)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(max_terms), partial_sums, 'b-o', label='部分和')
    plt.axhline(y=true_value, color='r', linestyle='--', 
               label=f'真实值 = {true_value:.6f}')
    plt.xlabel('项数')
    plt.ylabel('近似值')
    plt.title(f'幂级数近似: {name} 在 x={x}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"真实值: {true_value:.10f}")
    print(f"20项近似: {partial_sums[-1]:.10f}")
    print(f"误差: {abs(partial_sums[-1] - true_value):.2e}")

# e^x 的幂级数
power_series_approximation(
    x=1, 
    series_func=lambda n: 1/np.math.factorial(n),
    name='$e^x$',
    true_func=np.exp
)

# sin(x) 的幂级数
power_series_approximation(
    x=np.pi/4, 
    series_func=lambda n: ((-1)**n) / np.math.factorial(2*n + 1) if n >= 0 else 0,
    name='$\\sin(x)$',
    true_func=np.sin
)
```

---

## 常见级数的求和

### 1. 几何级数

$$
\sum_{n=0}^{\infty} r^n = \frac{1}{1-r}, \quad |r| < 1
$$

有限和：
$$
\sum_{n=0}^{N-1} r^n = \frac{1-r^N}{1-r}
$$

### 2. 等差级数

$$
\sum_{n=1}^{N} n = \frac{N(N+1)}{2}
$$

$$
\sum_{n=1}^{N} n^2 = \frac{N(N+1)(2N+1)}{6}
$$

$$
\sum_{n=1}^{N} n^3 = \left[\frac{N(N+1)}{2}\right]^2
$$

### 3. 调和级数的近似

调和级数部分和：
$$
H_N = \sum_{n=1}^{N} \frac{1}{n} \approx \ln N + \gamma + \frac{1}{2N}
$$

其中 $\gamma \approx 0.5772$ 是欧拉-马歇罗尼常数。

### 4. 常见无穷级数

| 级数 | 和 |
|------|-----|
| $\sum_{n=1}^{\infty} \frac{1}{n^2}$ | $\frac{\pi^2}{6}$ |
| $\sum_{n=1}^{\infty} \frac{1}{n^4}$ | $\frac{\pi^4}{90}$ |
| $\sum_{n=0}^{\infty} \frac{1}{n!}$ | $e$ |
| $\sum_{n=0}^{\infty} \frac{(-1)^n}{2n+1}$ | $\frac{\pi}{4}$ |

```python
def verify_famous_sums():
    """验证著名级数和"""
    n_terms = 10000
    
    # Σ1/n² = π²/6
    sum_1_n2 = np.sum(1 / np.arange(1, n_terms+1)**2)
    true_val = np.pi**2 / 6
    print(f"Σ1/n² ≈ {sum_1_n2:.10f}")
    print(f"π²/6 = {true_val:.10f}")
    print(f"误差: {abs(sum_1_n2 - true_val):.2e}\n")
    
    # Σ1/n! = e
    sum_1_fact = np.sum(1 / np.array([np.math.factorial(n) for n in range(20)]))
    true_val = np.e
    print(f"Σ1/n! ≈ {sum_1_fact:.10f}")
    print(f"e = {true_val:.10f}")
    print(f"误差: {abs(sum_1_fact - true_val):.2e}\n")
    
    # Σ(-1)^n/(2n+1) = π/4 (Leibniz 公式)
    n_leibniz = 100000
    terms = ((-1)**np.arange(n_leibniz)) / (2*np.arange(n_leibniz) + 1)
    sum_leibniz = np.sum(terms)
    true_val = np.pi / 4
    print(f"Σ(-1)^n/(2n+1) ≈ {sum_leibniz:.10f} (n={n_leibniz})")
    print(f"π/4 = {true_val:.10f}")
    print(f"误差: {abs(sum_leibniz - true_val):.2e}")

verify_famous_sums()
```

---

## 在深度学习中的应用

### 1. RNN 梯度传播的级数分析

RNN 梯度涉及矩阵乘积的无穷级数：

$$
\frac{\partial h_T}{\partial h_0} = \sum_{k=0}^{T-1} \prod_{t=T-k}^{T-1} W_h
$$

当 $W_h$ 的谱半径 $\rho(W_h) < 1$ 时，梯度收敛（不会爆炸）。

```python
def rnn_gradient_analysis():
    """RNN 梯度的级数分析"""
    eigenvalues = [0.5, 0.9, 1.0, 1.1]
    T_max = 100
    
    plt.figure(figsize=(12, 5))
    
    for i, eig in enumerate(eigenvalues):
        gradients = [eig ** t for t in range(T_max)]
        plt.subplot(1, 2, 1)
        plt.semilogy(gradients, label=f'$\\lambda = {eig}$')
        
        plt.subplot(1, 2, 2)
        cumsum = np.cumsum(gradients)
        plt.plot(cumsum, label=f'$\\lambda = {eig}$')
    
    plt.subplot(1, 2, 1)
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Gradient (log scale)')
    plt.title('RNN 梯度传播: $\\lambda^t$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Gradient')
    plt.title('梯度累积: $\\sum \\lambda^t$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

rnn_gradient_analysis()
```

### 2. 正则化的级数形式

L2 正则化（权重衰减）的梯度涉及级数：

$$
W_t = W_0(1-\lambda)^t + \text{(gradient contributions)}
$$

没有梯度时，权重按几何级数衰减：
$$
W_t = W_0(1-\lambda)^t \to 0
$$

### 3. 注意力机制的 Softmax

Softmax 权重可以看作概率分布，满足：
$$
\sum_{i=1}^{n} \text{softmax}(z)_i = 1
$$

这是有限级数和的形式。

### 4. 泰勒展开在优化中的应用

牛顿法使用泰勒级数二阶近似：

$$
f(x + \Delta x) \approx f(x) + \nabla f(x)^\top \Delta x + \frac{1}{2} \Delta x^\top H \Delta x
$$

```python
def taylor_optimization_demo():
    """泰勒展开优化演示"""
    # 目标函数: f(x) = x^4 - 2x^2 + 1
    f = lambda x: x**4 - 2*x**2 + 1
    df = lambda x: 4*x**3 - 4*x
    d2f = lambda x: 12*x**2 - 4
    
    x = np.linspace(-2, 2, 200)
    
    # 在 x = 1.5 处的泰勒展开
    x0 = 1.5
    taylor_order1 = f(x0) + df(x0) * (x - x0)
    taylor_order2 = f(x0) + df(x0) * (x - x0) + 0.5 * d2f(x0) * (x - x0)**2
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, f(x), 'b-', linewidth=2, label='原函数 $f(x)$')
    plt.plot(x, taylor_order1, 'g--', linewidth=2, label='一阶泰勒')
    plt.plot(x, taylor_order2, 'r--', linewidth=2, label='二阶泰勒')
    plt.axvline(x=x0, color='k', linestyle=':', alpha=0.5)
    plt.scatter([x0], [f(x0)], color='k', s=100, zorder=5)
    plt.xlim(-2, 2)
    plt.ylim(-2, 6)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'泰勒级数近似 (在 $x_0 = {x0}$ 处)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

taylor_optimization_demo()
```

---

## 小结

本节介绍了级数的核心概念：

| 概念 | 定义/方法 | 应用 |
|------|----------|------|
| 级数收敛 | $\lim S_n$ 存在 | 判断无穷求和 |
| 比值判别法 | $\lim a_{n+1}/a_n < 1$ | 判断收敛 |
| 根值判别法 | $\lim \sqrt[n]{a_n} < 1$ | 判断收敛 |
| 交错级数 | Leibniz 判别法 | 条件收敛 |
| 幂级数 | 收敛半径 $R$ | 泰勒展开 |
| 几何级数 | $\sum r^n = \frac{1}{1-r}$ | 学习率分析 |

**关键要点**：
- 收敛的必要条件是通项趋于零（但不充分）
- 正项级数有多种收敛判别法
- 幂级数在收敛半径内绝对收敛
- 泰勒级数是函数近似的重要工具

---

**下一节**：[数列在深度学习中的应用](07d-数列在深度学习中的应用.md) - 学习数列与级数在深度学习中的具体应用。

**返回**：[第七章：数列与级数](07-sequences-series.md) | [数学基础教程目录](../math-fundamentals.md)
