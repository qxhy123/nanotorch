# 第四章（c）：假设检验

假设检验是统计推断的另一大核心任务，用于根据样本数据判断关于总体的某个假设是否成立。在深度学习中，假设检验用于模型比较、A/B 测试和特征选择等场景。本章将介绍假设检验的基本概念、常见检验方法及其在深度学习中的应用。

---

## 🎯 生活类比：法庭审判

假设检验就像一场**法庭审判**：

```
┌─────────────────────────────────────────────────────────────┐
│                        法庭审判                              │
├─────────────────────────────────────────────────────────────┤
│  被告人：新药/新模型/新方法                                  │
│  原假设 H₀：被告人无罪（新药无效/无差异）                     │
│  备择假设 H₁：被告人有罪（新药有效/有差异）                   │
│                                                             │
│  证据：样本数据                                              │
│  判决标准：显著性水平 α = 0.05（5%的冤案率）                  │
│                                                             │
│  判决结果：                                                  │
│  • p值 < 0.05 → 证据足够强 → 拒绝H₀（认定有罪）              │
│  • p值 ≥ 0.05 → 证据不足 → 不拒绝H₀（无罪释放）              │
└─────────────────────────────────────────────────────────────┘
```

### 两类错误 = 两种司法失误

| 错误类型 | 法庭类比 | 统计含义 | 后果 |
|---------|---------|---------|------|
| **第一类错误 (α)** | 冤枉好人 | 本来无效却认为有效 | 把没用的药当成神药 |
| **第二类错误 (β)** | 放走坏人 | 本来有效却认为无效 | 错失好药 |

**关键原则**：在法庭上，我们宁可放走坏人，也不要冤枉好人。所以α（冤案率）通常控制在5%或1%。

### 📖 通俗翻译

| 统计术语 | 通俗翻译 |
|---------|---------|
| 原假设 H₀ | "默认立场"：假设什么都没发生 |
| 备择假设 H₁ | "起诉书"：声称有事情发生了 |
| p 值 | "证据强度"：如果 H₀ 是真的，观察到当前数据的概率 |
| 显著性水平 α | "判决门槛"：证据多强才能定罪 |
| 拒绝 H₀ | "罪名成立"：有足够证据支持 H₁ |
| 不拒绝 H₀ | "证据不足"：不代表 H₀ 一定正确 |

---

## 目录

1. [假设检验的基本概念](#假设检验的基本概念)
2. [两类错误与显著性水平](#两类错误与显著性水平)
3. [p 值与决策规则](#p-值与决策规则)
4. [常见假设检验](#常见假设检验)
5. [置信区间](#置信区间)
6. [假设检验与置信区间的关系](#假设检验与置信区间的关系)
7. [在深度学习中的应用](#在深度学习中的应用)
8. [小结](#小结)

---

## 假设检验的基本概念

### 假设检验问题

**问题**：根据样本数据，判断关于总体的某个假设是否成立。

**基本思想**：在假设成立的前提下，计算观察到当前样本（或更极端情况）的概率。如果这个概率很小，则有理由拒绝该假设。

### 原假设与备择假设

**原假设 (Null Hypothesis)** $H_0$：待检验的假设，通常表示"无差异"或"无效果"。

**备择假设 (Alternative Hypothesis)** $H_1$（或 $H_A$）：与原假设对立的假设。

### 假设的类型

| 类型 | 原假设 | 备择假设 | 含义 |
|------|--------|----------|------|
| 双侧检验 | $H_0: \theta = \theta_0$ | $H_1: \theta \neq \theta_0$ | 参数是否等于某值 |
| 左侧检验 | $H_0: \theta \geq \theta_0$ | $H_1: \theta < \theta_0$ | 参数是否小于某值 |
| 右侧检验 | $H_0: \theta \leq \theta_0$ | $H_1: \theta > \theta_0$ | 参数是否大于某值 |

### 检验统计量

**检验统计量**：用于判断的统计量，在 $H_0$ 成立时其分布已知。

$$
T = T(X_1, \ldots, X_n)
$$

### 拒绝域

**拒绝域 (Critical Region)**：检验统计量的取值范围，当统计量落在此范围内时拒绝 $H_0$。

**临界值**：拒绝域的边界值。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 假设检验的直观理解
# 示例：检验硬币是否公平

# 假设: H0: p = 0.5 (硬币公平)
#      H1: p ≠ 0.5 (硬币不公平)

# 观察数据：抛 100 次硬币，正面 65 次
n = 100
heads = 65
p_hat = heads / n

print("硬币公平性检验:")
print("="*50)
print(f"观察数据: n={n}, 正面={heads}, 比例={p_hat:.2f}")
print()

# 理论上，如果硬币公平，正面次数 ~ Binomial(100, 0.5)
# 均值 = 50, 标准差 = sqrt(100*0.5*0.5) = 5

# 标准化
z_stat = (heads - 50) / 5

# 双侧 p 值
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"检验统计量 z = {z_stat:.2f}")
print(f"p 值 = {p_value:.4f}")
print(f"结论 (α=0.05): {'拒绝 H0' if p_value < 0.05 else '不拒绝 H0'}")
print()

# 可视化
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

x = np.linspace(30, 70, 200)
# 正态近似
pdf = stats.norm.pdf(x, 50, 5)

ax.plot(x, pdf, 'b-', linewidth=2, label='H0 下的分布: N(50, 25)')
ax.fill_between(x[x >= 60], pdf[x >= 60], alpha=0.3, color='red', label='右侧拒绝域')
ax.fill_between(x[x <= 40], pdf[x <= 40], alpha=0.3, color='red', label='左侧拒绝域')
ax.axvline(heads, color='green', linestyle='--', linewidth=2, label=f'观察值: {heads}')
ax.axvline(60, color='red', linestyle=':', linewidth=1.5)
ax.axvline(40, color='red', linestyle=':', linewidth=1.5)

ax.set_xlabel('正面次数')
ax.set_ylabel('概率密度')
ax.set_title('双侧假设检验：硬币是否公平')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hypothesis_test_intuition.png', dpi=100)
print("图像已保存: hypothesis_test_intuition.png")
```

---

## 两类错误与显著性水平

### 两类错误

|  | $H_0$ 为真 | $H_0$ 为假 |
|--|-----------|-----------|
| **拒绝 $H_0$** | **第一类错误 (Type I)** | 正确 |
| **不拒绝 $H_0$** | 正确 | **第二类错误 (Type II)** |

**第一类错误**（弃真）：$H_0$ 实际上为真，但被拒绝。

$$
\alpha = P(\text{拒绝 } H_0 | H_0 \text{ 为真})
$$

**第二类错误**（取伪）：$H_0$ 实际上为假，但未被拒绝。

$$
\beta = P(\text{不拒绝 } H_0 | H_0 \text{ 为假})
$$

### 显著性水平

**显著性水平** $\alpha$：犯第一类错误的最大允许概率。

常用值：$\alpha = 0.05$, $\alpha = 0.01$, $\alpha = 0.1$

### 功效 (Power)

**检验功效**：正确拒绝错误的原假设的概率。

$$
\text{Power} = 1 - \beta = P(\text{拒绝 } H_0 | H_1 \text{ 为真})
$$

### 两类错误的权衡

- 降低 $\alpha$ 会增加 $\beta$（在其他条件不变时）
- 同时降低两类错误的唯一方法是增加样本量

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 两类错误的可视化
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# H0: μ = 0
# H1: μ = 2
mu0, mu1 = 0, 2
sigma = 1
n = 25  # 样本量
se = sigma / np.sqrt(n)  # 标准误

x = np.linspace(-2, 5, 200)

# H0 下的分布
y0 = stats.norm.pdf(x, mu0, se)
# H1 下的分布
y1 = stats.norm.pdf(x, mu1, se)

ax.plot(x, y0, 'b-', linewidth=2, label=f'H0: N({mu0}, {se**2:.2f})')
ax.plot(x, y1, 'r-', linewidth=2, label=f'H1: N({mu1}, {se**2:.2f})')

# 临界值（右侧检验，α = 0.05）
alpha = 0.05
critical_value = stats.norm.ppf(1 - alpha, mu0, se)

# 填充第一类错误区域
x_alpha = x[x >= critical_value]
ax.fill_between(x_alpha, stats.norm.pdf(x_alpha, mu0, se), alpha=0.3, color='blue', label=f'第一类错误 α = {alpha}')

# 填充第二类错误区域
x_beta = x[x < critical_value]
ax.fill_between(x_beta, stats.norm.pdf(x_beta, mu1, se), alpha=0.3, color='red', label=f'第二类错误 β')

# 临界值线
ax.axvline(critical_value, color='green', linestyle='--', linewidth=2, label=f'临界值 = {critical_value:.2f}')

# 计算 β
beta = stats.norm.cdf(critical_value, mu1, se)
power = 1 - beta

ax.set_xlabel('样本均值')
ax.set_ylabel('概率密度')
ax.set_title(f'两类错误可视化 (α={alpha}, β={beta:.4f}, Power={power:.4f})')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('type_errors.png', dpi=100)
print("图像已保存: type_errors.png")

print(f"\n两类错误:")
print(f"显著性水平 α = {alpha}")
print(f"第二类错误 β = {beta:.4f}")
print(f"检验功效 = {power:.4f}")
```

---

## p 值与决策规则

### p 值的定义

**p 值**：在 $H_0$ 成立的条件下，观察到比当前结果**更极端**的概率。

$$
p\text{-value} = P(T \geq t_{obs} | H_0)
$$

### p 值的解释

| p 值范围 | 解释 |
|----------|------|
| $p < 0.01$ | 非常强的证据拒绝 $H_0$ |
| $0.01 \leq p < 0.05$ | 较强的证据拒绝 $H_0$ |
| $0.05 \leq p < 0.1$ | 弱证据拒绝 $H_0$ |
| $p \geq 0.1$ | 没有足够证据拒绝 $H_0$ |

### 决策规则

**使用 p 值**：
- 若 $p < \alpha$：拒绝 $H_0$
- 若 $p \geq \alpha$：不拒绝 $H_0$

**使用临界值**：
- 若检验统计量落在拒绝域内：拒绝 $H_0$
- 否则：不拒绝 $H_0$

### p 值计算

**双侧检验**：

$$
p = 2 \cdot P(Z \geq |z_{obs}|)
$$

**单侧检验（右侧）**：

$$
p = P(Z \geq z_{obs})
$$

**单侧检验（左侧）**：

$$
p = P(Z \leq z_{obs})
$$

```python
import numpy as np
from scipy import stats

# p 值计算示例
np.random.seed(42)

# 单样本 t 检验
# H0: μ = 100
# H1: μ ≠ 100

mu0 = 100
sample = np.random.normal(102, 15, 50)  # 真实 μ = 102
n = len(sample)

# 计算检验统计量
sample_mean = sample.mean()
sample_std = sample.std(ddof=1)
se = sample_std / np.sqrt(n)

t_stat = (sample_mean - mu0) / se
df = n - 1

# p 值（双侧）
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

print("单样本 t 检验:")
print("="*50)
print(f"H0: μ = {mu0}")
print(f"样本均值: {sample_mean:.2f}")
print(f"样本标准差: {sample_std:.2f}")
print(f"标准误: {se:.2f}")
print(f"t 统计量: {t_stat:.4f}")
print(f"自由度: {df}")
print(f"p 值: {p_value:.4f}")
print()
print(f"结论 (α=0.05): {'拒绝 H0' if p_value < 0.05 else '不拒绝 H0'}")

# 使用 scipy 验证
t_stat_scipy, p_value_scipy = stats.ttest_1samp(sample, mu0)
print(f"\nscipy 验证:")
print(f"t 统计量: {t_stat_scipy:.4f}")
print(f"p 值: {p_value_scipy:.4f}")
```

---

## 常见假设检验

### 单样本 t 检验

**问题**：检验样本均值是否等于某个值。

**假设**：
- $H_0: \mu = \mu_0$
- $H_1: \mu \neq \mu_0$

**检验统计量**：

$$
t = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t(n-1)
$$

### 📝 手把手计算：检验能量饮料是否真的有效

**场景**：一款能量饮料声称能提高反应速度。标准反应时间是 100 毫秒。我们测试了 5 个人：

| 受试者 | 喝饮料后反应时间(毫秒) |
|-------|---------------------|
| 小明 | 95 |
| 小红 | 98 |
| 小华 | 92 |
| 小李 | 97 |
| 小王 | 93 |

**第1步：计算样本均值**
$$
\bar{X} = \frac{95 + 98 + 92 + 97 + 93}{5} = \frac{475}{5} = 95 \text{ 毫秒}
$$

**第2步：计算样本标准差**
$$
S = \sqrt{\frac{(95-95)^2 + (98-95)^2 + (92-95)^2 + (97-95)^2 + (93-95)^2}{5-1}}
$$
$$
= \sqrt{\frac{0 + 9 + 9 + 4 + 4}{4}} = \sqrt{\frac{26}{4}} = \sqrt{6.5} ≈ 2.55 \text{ 毫秒}
$$

**第3步：计算 t 统计量**
$$
t = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} = \frac{95 - 100}{2.55 / \sqrt{5}} = \frac{-5}{1.14} ≈ -4.39
$$

**第4步：查表得 p 值**
- 自由度 df = n - 1 = 4
- 查 t 分布表，|t| = 4.39 对应 p < 0.01

**第5步：做出结论**

| 判断标准 | 本例结果 | 结论 |
|---------|---------|------|
| p < 0.05？ | p ≈ 0.01 < 0.05 ✓ | **拒绝 H₀** |

**结论**：能量饮料显著提高了反应速度！（从 100ms 降到 95ms，p ≈ 0.01）

```
可视化理解：

       H₀ 分布（饮料无效，均值=100）
              ↓
    92  94  96  98  100 102 104 106
    │   │   │   │   │   │   │   │
    └───┴───┴───┴───┴───┴───┴───┘
              ↑
        我们的样本均值=95
        落在拒绝域内！
```

### 双样本 t 检验

**问题**：检验两个独立样本的均值是否相等。

**假设**：
- $H_0: \mu_1 = \mu_2$
- $H_1: \mu_1 \neq \mu_2$

**检验统计量**（方差相等时）：

$$
t = \frac{\bar{X}_1 - \bar{X}_2}{S_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}
$$

其中 $S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}$ 是**合并方差**。

### 配对 t 检验

**问题**：检验配对样本的差值均值是否为零。

**检验统计量**：

$$
t = \frac{\bar{D}}{S_D / \sqrt{n}}
$$

其中 $D_i = X_i - Y_i$。

### χ² 检验

**问题**：检验分类变量的独立性。

**卡方统计量**：

$$
\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

其中 $O_{ij}$ 是观察频数，$E_{ij}$ 是期望频数。

### F 检验（方差比较）

**问题**：检验两个总体方差是否相等。

**检验统计量**：

$$
F = \frac{S_1^2}{S_2^2} \sim F(n_1-1, n_2-1)
$$

```python
import numpy as np
from scipy import stats

# 各种假设检验示例

print("="*60)
print("1. 双样本 t 检验（独立样本）")
print("="*60)

np.random.seed(42)
group1 = np.random.normal(100, 15, 50)
group2 = np.random.normal(105, 15, 50)

t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"组1均值: {group1.mean():.2f}")
print(f"组2均值: {group2.mean():.2f}")
print(f"t 统计量: {t_stat:.4f}")
print(f"p 值: {p_value:.4f}")
print(f"结论: {'显著差异' if p_value < 0.05 else '无显著差异'}")

print("\n" + "="*60)
print("2. 配对 t 检验")
print("="*60)

# 同一批受试者的前后测量
before = np.random.normal(100, 15, 30)
after = before + np.random.normal(5, 5, 30)  # 平均提升 5

t_stat, p_value = stats.ttest_rel(before, after)
print(f"前测均值: {before.mean():.2f}")
print(f"后测均值: {after.mean():.2f}")
print(f"平均变化: {(after - before).mean():.2f}")
print(f"t 统计量: {t_stat:.4f}")
print(f"p 值: {p_value:.4f}")
print(f"结论: {'显著变化' if p_value < 0.05 else '无显著变化'}")

print("\n" + "="*60)
print("3. χ² 检验（独立性检验）")
print("="*60)

# 列联表
observed = np.array([[50, 30], [20, 40]])
chi2, p_value, dof, expected = stats.chi2_contingency(observed)

print(f"观察频数:\n{observed}")
print(f"期望频数:\n{expected}")
print(f"χ² 统计量: {chi2:.4f}")
print(f"自由度: {dof}")
print(f"p 值: {p_value:.4f}")
print(f"结论: {'相关' if p_value < 0.05 else '独立'}")
```

---

## 置信区间

### 定义

参数 $\theta$ 的**置信水平** $1-\alpha$ 的**置信区间**是随机区间 $(L, U)$，使得：

$$
P(L \leq \theta \leq U) = 1 - \alpha
$$

### 解释

置信区间**不是**"参数有 $1-\alpha$ 的概率落在区间内"。

正确解释：如果重复抽样多次，约 $(1-\alpha) \times 100\%$ 的置信区间会包含真实参数。

### 正态总体均值的置信区间

#### σ 已知

$$
\left(\bar{X} - z_{\alpha/2} \frac{\sigma}{\sqrt{n}}, \bar{X} + z_{\alpha/2} \frac{\sigma}{\sqrt{n}}\right)
$$

#### σ 未知

$$
\left(\bar{X} - t_{\alpha/2}(n-1) \frac{S}{\sqrt{n}}, \bar{X} + t_{\alpha/2}(n-1) \frac{S}{\sqrt{n}}\right)
$$

### 比例的置信区间

$$
\left(\hat{p} - z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}, \hat{p} + z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}\right)
$$

### 方差的置信区间

$$
\left(\frac{(n-1)S^2}{\chi^2_{\alpha/2}(n-1)}, \frac{(n-1)S^2}{\chi^2_{1-\alpha/2}(n-1)}\right)
$$

```python
import numpy as np
from scipy import stats

# 置信区间计算示例
np.random.seed(42)

# 生成数据
sample = np.random.normal(100, 15, 50)
n = len(sample)
sample_mean = sample.mean()
sample_std = sample.std(ddof=1)

print("置信区间计算:")
print("="*50)
print(f"样本容量: n = {n}")
print(f"样本均值: {sample_mean:.2f}")
print(f"样本标准差: {sample_std:.2f}")
print()

# 1. 均值的置信区间（σ 未知）
confidence = 0.95
se = sample_std / np.sqrt(n)
ci_mean = stats.t.interval(confidence, n-1, loc=sample_mean, scale=se)
print(f"均值 {confidence*100:.0f}% 置信区间: ({ci_mean[0]:.2f}, {ci_mean[1]:.2f})")

# 2. 方差的置信区间
alpha = 1 - confidence
chi2_upper = stats.chi2.ppf(1 - alpha/2, n-1)
chi2_lower = stats.chi2.ppf(alpha/2, n-1)
ci_var = ((n-1) * sample_std**2 / chi2_upper, (n-1) * sample_std**2 / chi2_lower)
print(f"方差 {confidence*100:.0f}% 置信区间: ({ci_var[0]:.2f}, {ci_var[1]:.2f})")
print(f"标准差 {confidence*100:.0f}% 置信区间: ({np.sqrt(ci_var[0]):.2f}, {np.sqrt(ci_var[1]):.2f})")

# 3. 比例的置信区间
# 示例：100 次试验中 35 次成功
n_trials = 100
n_success = 35
p_hat = n_success / n_trials
se_prop = np.sqrt(p_hat * (1 - p_hat) / n_trials)
ci_prop = stats.norm.interval(confidence, loc=p_hat, scale=se_prop)
print(f"\n比例 {confidence*100:.0f}% 置信区间: ({ci_prop[0]:.4f}, {ci_prop[1]:.4f})")

# 可视化多次抽样的置信区间
print("\n" + "="*50)
print("置信区间的频率解释（重复抽样）")

n_simulations = 100
true_mu = 100
true_sigma = 15
sample_size = 30

contains_true = 0
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

for i in range(n_simulations):
    sample_i = np.random.normal(true_mu, true_sigma, sample_size)
    mean_i = sample_i.mean()
    std_i = sample_i.std(ddof=1)
    se_i = std_i / np.sqrt(sample_size)
    ci_i = stats.t.interval(0.95, sample_size-1, loc=mean_i, scale=se_i)
    
    contains = (ci_i[0] <= true_mu <= ci_i[1])
    if contains:
        contains_true += 1
        color = 'blue'
    else:
        color = 'red'
    
    ax.plot([ci_i[0], ci_i[1]], [i, i], color=color, linewidth=0.5)
    ax.plot(mean_i, i, 'o', color=color, markersize=2)

ax.axvline(true_mu, color='green', linestyle='--', linewidth=2, label=f'真实均值 = {true_mu}')
ax.set_xlabel('值')
ax.set_ylabel('样本编号')
ax.set_title(f'置信区间的频率解释\n{n_simulations}次抽样中 {contains_true} 次包含真实值 ({contains_true/n_simulations*100:.1f}%)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('confidence_intervals.png', dpi=100)
print(f"\n{n_simulations} 次抽样中，{contains_true} 次置信区间包含真实值")
print(f"比例: {contains_true/n_simulations*100:.1f}% (理论值: 95%)")
print("图像已保存: confidence_intervals.png")
```

---

## 假设检验与置信区间的关系

### 对偶关系

**双侧假设检验**与**置信区间**存在**对偶关系**：

$$
\text{拒绝 } H_0: \theta = \theta_0 \iff \theta_0 \notin \text{置信区间}
$$

### 示例

对于均值检验 $H_0: \mu = \mu_0$：

- 如果 $\mu_0$ 落在 $(1-\alpha)$ 置信区间外，则在显著性水平 $\alpha$ 下拒绝 $H_0$
- 如果 $\mu_0$ 落在 $(1-\alpha)$ 置信区间内，则在显著性水平 $\alpha$ 下不拒绝 $H_0$

```python
import numpy as np
from scipy import stats

# 假设检验与置信区间的对偶关系
np.random.seed(42)
sample = np.random.normal(102, 15, 50)
n = len(sample)

# 方法1：假设检验
mu0 = 100
t_stat, p_value = stats.ttest_1samp(sample, mu0)

# 方法2：置信区间
sample_mean = sample.mean()
sample_std = sample.std(ddof=1)
se = sample_std / np.sqrt(n)
ci = stats.t.interval(0.95, n-1, loc=sample_mean, scale=se)

print("假设检验与置信区间的对偶关系:")
print("="*50)
print(f"H0: μ = {mu0}")
print(f"样本均值: {sample_mean:.2f}")
print(f"95% 置信区间: ({ci[0]:.2f}, {ci[1]:.2f})")
print()
print("方法1 - 假设检验:")
print(f"  t 统计量: {t_stat:.4f}")
print(f"  p 值: {p_value:.4f}")
print(f"  结论 (α=0.05): {'拒绝 H0' if p_value < 0.05 else '不拒绝 H0'}")
print()
print("方法2 - 置信区间:")
print(f"  {mu0} 是否在置信区间内? {ci[0] <= mu0 <= ci[1]}")
print(f"  结论: {'不拒绝 H0' if ci[0] <= mu0 <= ci[1] else '拒绝 H0'}")
print()
print("两种方法结论一致!")
```

---

## 在深度学习中的应用

### 1. 模型比较

使用**配对 t 检验**比较两个模型在多个数据集或多次运行上的性能差异。

```python
import numpy as np
from scipy import stats

# 模型比较示例
np.random.seed(42)
n_folds = 10

# 模型 A 和 B 在 K 折交叉验证上的准确率
model_a_acc = np.array([0.85, 0.87, 0.84, 0.86, 0.88, 0.85, 0.87, 0.86, 0.85, 0.87])
model_b_acc = np.array([0.82, 0.84, 0.81, 0.83, 0.85, 0.82, 0.84, 0.83, 0.82, 0.84])

# 配对 t 检验
t_stat, p_value = stats.ttest_rel(model_a_acc, model_b_acc)

print("模型比较（配对 t 检验）:")
print("="*50)
print(f"模型 A 平均准确率: {model_a_acc.mean():.4f} ± {model_a_acc.std():.4f}")
print(f"模型 B 平均准确率: {model_b_acc.mean():.4f} ± {model_b_acc.std():.4f}")
print(f"平均差异: {(model_a_acc - model_b_acc).mean():.4f}")
print()
print(f"t 统计量: {t_stat:.4f}")
print(f"p 值: {p_value:.6f}")
print(f"结论: {'模型 A 显著优于 B' if p_value < 0.05 else '无显著差异'} (α=0.05)")

# 效应量 (Cohen's d)
diff = model_a_acc - model_b_acc
cohens_d = diff.mean() / diff.std()
print(f"\n效应量 (Cohen's d): {cohens_d:.4f}")
print(f"效应大小: {'大' if abs(cohens_d) > 0.8 else '中' if abs(cohens_d) > 0.5 else '小'}")
```

### 2. A/B 测试

```python
import numpy as np
from scipy import stats

# A/B 测试示例
np.random.seed(42)

# 对照组 A 和实验组 B
n_A = 1000
n_B = 1000

# 点击率
clicks_A = 100  # 10% 点击率
clicks_B = 130  # 13% 点击率

p_A = clicks_A / n_A
p_B = clicks_B / n_B

# 双样本比例检验
# 合并比例
p_pooled = (clicks_A + clicks_B) / (n_A + n_B)
se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_A + 1/n_B))

z_stat = (p_B - p_A) / se_pooled
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print("A/B 测试:")
print("="*50)
print(f"对照组 A: {clicks_A}/{n_A} = {p_A:.2%}")
print(f"实验组 B: {clicks_B}/{n_B} = {p_B:.2%}")
print(f"差异: {(p_B - p_A):.2%}")
print()
print(f"z 统计量: {z_stat:.4f}")
print(f"p 值: {p_value:.4f}")
print(f"结论: {'实验组显著优于对照组' if p_value < 0.05 else '无显著差异'} (α=0.05)")

# 置信区间
diff = p_B - p_A
se_diff = np.sqrt(p_A*(1-p_A)/n_A + p_B*(1-p_B)/n_B)
ci_diff = stats.norm.interval(0.95, loc=diff, scale=se_diff)
print(f"\n差异的 95% 置信区间: ({ci_diff[0]:.4f}, {ci_diff[1]:.4f})")
```

### 3. 特征选择

使用统计检验筛选重要特征。

```python
import numpy as np
from scipy import stats

# 特征选择示例
np.random.seed(42)
n_samples = 100
n_features = 10

# 生成特征数据
X = np.random.randn(n_samples, n_features)
# 目标变量（与部分特征相关）
y = 0.5 * X[:, 0] + 0.3 * X[:, 2] + np.random.randn(n_samples) * 0.5

# 对每个特征进行 t 检验（与目标的相关性）
print("特征选择（相关性检验）:")
print("="*50)
print(f"{'特征':<10} {'相关系数':<12} {'p 值':<12} {'是否保留'}")
print("-"*50)

selected_features = []
for i in range(n_features):
    # Pearson 相关性检验
    corr, p_value = stats.pearsonr(X[:, i], y)
    keep = p_value < 0.05
    if keep:
        selected_features.append(i)
    print(f"特征 {i:<5} {corr:>10.4f} {p_value:>10.4f} {'是' if keep else '否'}")

print()
print(f"选择的特征: {selected_features}")
print(f"(真实相关特征: 0, 2)")
```

---

## 小结

本章介绍了假设检验的基本概念和方法，这些是进行科学实验和模型比较的理论基础。

### 核心概念对照表

| 概念 | 定义 | 应用 |
|------|------|------|
| 第一类错误 | $P(\text{拒绝 } H_0 | H_0 \text{ 为真})$ | 显著性水平 |
| 第二类错误 | $P(\text{不拒绝 } H_0 | H_1 \text{ 为真})$ | 检验功效 |
| p 值 | 观察到更极端结果的概率 | 决策依据 |
| 置信区间 | 参数的可靠范围 | 参数估计 |

### 常用检验方法

| 检验 | 适用场景 | 统计量分布 |
|------|----------|------------|
| 单样本 t 检验 | 均值检验 | $t(n-1)$ |
| 双样本 t 检验 | 两组均值比较 | $t(n_1+n_2-2)$ |
| 配对 t 检验 | 配对样本比较 | $t(n-1)$ |
| χ² 检验 | 分类变量独立性 | $\chi^2$ |
| F 检验 | 方差比较 | $F$ |

### 关键要点

1. **假设检验**：基于样本推断总体的统计方法
2. **两类错误**：需要权衡
3. **p 值**：不是假设为真的概率
4. **置信区间**：与假设检验对偶
5. **深度学习应用**：模型比较、A/B 测试、特征选择

---

**上一节**：[第四章（b）：参数估计](04b-参数估计.md)

**下一节**：[第四章（d）：回归分析与贝叶斯统计](04d-回归分析与贝叶斯统计.md) - 学习线性回归、逻辑回归和贝叶斯方法在深度学习中的应用。

**返回**：[数学基础教程目录](../math-fundamentals.md)
