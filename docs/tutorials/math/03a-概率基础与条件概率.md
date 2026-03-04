# 第三章（a）：概率基础与条件概率

概率论是理解**机器学习不确定性**和**统计学习理论**的基础。从分类问题的 Softmax 输出到生成模型，概率论无处不在。本章将系统介绍概率论的基本概念、条件概率和贝叶斯定理。

---

## 🎯 生活类比：概率就是"有多大概率发生"

### 概率 = 打赌的胜算

想象你在和朋友打赌明天会不会下雨：
- 气象预报说"降水概率 70%"
- 意思是：如果同样的天气条件重复100天，大约70天会下雨

```
概率为 0    ────────────────────────→    概率为 1
   │                                              │
绝对不会发生                              一定会发生

0.1 = 10%: 很少见（中彩票）
0.5 = 50%: 一半一半（抛硬币）
0.9 = 90%: 很可能发生（明天太阳升起）
```

### 条件概率 = "已知信息后的概率"

**例子**：你看到一个人戴眼镜，他更可能是程序员吗？

$$P(\text{程序员} | \text{戴眼镜})$$

读作："在戴眼镜的条件下，是程序员的概率"

### 贝叶斯定理 = "根据新证据更新判断"

```
┌─────────────────────────────────────────────────────────┐
│  医生诊断的例子                                          │
├─────────────────────────────────────────────────────────┤
│  先验概率 P(生病): 总体人群中1%的人有这种病               │
│  似然 P(阳性|生病): 生病的人95%检测阳性                  │
│  似然 P(阳性|没病): 没病的人5%也会检测阳性（假阳性）       │
│                                                         │
│  问题：检测阳性后，真正生病的概率是多少？                  │
│  答案：P(生病|阳性) ≈ 16%  （远低于直觉！）               │
│                                                         │
│  教训：稀有疾病的阳性结果，大部分是假阳性                 │
└─────────────────────────────────────────────────────────┘
```

### 📖 通俗翻译

| 概率术语 | 通俗翻译 |
|---------|---------|
| 样本空间 $\Omega$ | 所有可能的结果 |
| 事件 A | 我们关心的某些结果 |
| $P(A)$ | A 发生的可能性（0到1之间） |
| $P(A \cap B)$ | A 和 B 同时发生 |
| $P(A \cup B)$ | A 或 B 至少一个发生 |
| $P(A \| B)$ | 已知 B 发生后，A 的概率 |
| 独立 | 一件事发生不影响另一件事 |

---

## 目录

1. [随机试验与样本空间](#随机试验与样本空间)
2. [事件与事件运算](#事件与事件运算)
3. [概率的定义](#概率的定义)
4. [概率的基本性质](#概率的基本性质)
5. [条件概率](#条件概率)
6. [乘法公式](#乘法公式)
7. [独立性](#独立性)
8. [全概率公式](#全概率公式)
9. [贝叶斯定理](#贝叶斯定理)
10. [在深度学习中的应用](#在深度学习中的应用)
11. [小结](#小结)

---

## 随机试验与样本空间

### 随机试验

满足以下三个条件的试验称为**随机试验**：
1. 可以在相同条件下重复进行
2. 所有可能结果事先已知
3. 每次试验的结果事先不确定

**示例**：
- 掷一枚骰子
- 抛一枚硬币
- 测量某人的身高
- 观察神经网络的输出

### 样本空间

**定义**：随机试验所有可能结果的集合，记作 $\Omega$ 或 $S$。

**示例**：
- 掷骰子：$\Omega = \{1, 2, 3, 4, 5, 6\}$
- 抛硬币：$\Omega = \{正面, 反面\}$
- 测量身高：$\Omega = (0$, $+\infty)$
- 二分类输出：$\Omega = \{0, 1\}$

### 样本点

样本空间中的每个元素称为**样本点**或**基本事件**，记作 $\omega$。

```python
import numpy as np

# 样本空间示例
# 掷骰子
omega_dice = {1, 2, 3, 4, 5, 6}
print(f"掷骰子样本空间: {omega_dice}")

# 抛两次硬币
omega_two_coins = {(h1, h2) for h1 in ['H', 'T'] for h2 in ['H', 'T']}
print(f"抛两次硬币样本空间: {omega_two_coins}")

# 连续样本空间（身高测量）
# 用区间表示
height_range = (0, 300)  # 单位：厘米
print(f"身高测量样本空间: (0, 300) 厘米")
```

---

## 事件与事件运算

### 事件的定义

**事件**是样本空间的子集，即某些可能结果的集合。事件发生当且仅当试验结果属于该子集。

### 事件的分类

| 类型 | 定义 | 示例 |
|------|------|------|
| 必然事件 | $\Omega$（样本空间本身） | 掷骰子点数 $\leq 6$ |
| 不可能事件 | $\emptyset$（空集） | 掷骰子点数 $> 6$ |
| 基本事件 | 单个样本点 | 掷骰子点数为 3 |

### 事件的关系与运算

**包含**：$A \subset B$，事件 A 发生蕴含 B 发生

**并（和）**：$A \cup B$ 或 $A + B$，A 或 B 至少一个发生

**交（积）**：$A \cap B$ 或 $AB$，A 和 B 同时发生

**补（对立）**：$A^c$ 或 $\bar{A}$，A 不发生

**差**：$A - B = A \cap B^c$，A 发生但 B 不发生

**互斥（不相容）**：$A \cap B = \emptyset$，A 和 B 不能同时发生

### 事件运算的性质

**交换律**：

$$
A \cup B = B \cup A, \quad A \cap B = B \cap A
$$

**结合律**：

$$
(A \cup B) \cup C = A \cup (B \cup C)
$$

$$
(A \cap B) \cap C = A \cap (B \cap C)
$$

**分配律**：

$$
A \cap (B \cup C) = (A \cap B) \cup (A \cap C)
$$

$$
A \cup (B \cap C) = (A \cup B) \cap (A \cup C)
$$

**德摩根律（De Morgan's Laws）**：

$$
\overline{A \cup B} = \bar{A} \cap \bar{B}
$$

$$
\overline{A \cap B} = \bar{A} \cup \bar{B}
$$

```python
# 事件运算示例
omega = {1, 2, 3, 4, 5, 6}
A = {1, 2, 3, 4}  # 点数 ≤ 4
B = {3, 4, 5, 6}  # 点数 ≥ 3

# 并
union = A | B
print(f"A ∪ B = {union}")  # {1, 2, 3, 4, 5, 6}

# 交
intersection = A & B
print(f"A ∩ B = {intersection}")  # {3, 4}

# 补
complement_A = omega - A
print(f"A^c = {complement_A}")  # {5, 6}

# 差
difference = A - B
print(f"A - B = {difference}")  # {1, 2}

# 德摩根律验证
de_morgan_1 = (A | B) == ((omega - A) & (omega - B))  # 注意：这个是错的演示
de_morgan_1 = omega - (A | B)  # {}
de_morgan_2 = (omega - A) & (omega - B)  # {}
print(f"德摩根律: Ω - (A∪B) = (Ω-A) ∩ (Ω-B): {de_morgan_1 == de_morgan_2}")
```

---

## 概率的定义

### 🎯 生活类比：天气预报

"明天降雨概率 70%" 是什么意思？

- **不是**：明天 70% 的时间在下雨
- **不是**：明天 70% 的地区会下雨
- **而是**：如果这样的天气重复100次，大约70次会下雨

**概率 = 长期来看，某事件发生的频率**

| 概率值 | 含义 | 生活例子 |
|--------|------|----------|
| 0 | 不可能 | 太阳从西边升起 |
| 0.5 | 一半一半 | 抛硬币正面朝上 |
| 1 | 必然发生 | 人终有一死 |

### 📝 手把手计算：掷骰子

**问题**：掷一个骰子，点数大于4的概率是多少？

**Step 1**：确定所有可能结果（样本空间）
$$\Omega = \{1, 2, 3, 4, 5, 6\}$$

**Step 2**：找出满足条件的结果
$$A = \{点数 > 4\} = \{5, 6\}$$

**Step 3**：计算概率
$$P(A) = \frac{|A|}{|\Omega|} = \frac{2}{6} = \frac{1}{3} \approx 33.3\%$$

**通俗理解**：6个结果中有2个满足条件，所以概率是 2/6。

### 古典概型

当样本空间有限且每个基本事件等可能发生时：

$$
P(A) = \frac{A \text{中基本事件数}}{\Omega \text{中基本事件总数}} = \frac{|A|}{|\Omega|}
$$

**示例**：掷骰子，求点数为偶数的概率。

$$
A = \{2, 4, 6\}, \quad |\Omega| = 6
$$

$$
P(A) = \frac{3}{6} = \frac{1}{2}
$$

### 几何概型

当样本空间是连续区域时：

$$
P(A) = \frac{A \text{的度量}}{\Omega \text{的度量}}
$$

（度量可以是长度、面积、体积等）

### 公理化定义（Kolmogorov 公理）

概率 $P$ 是定义在事件集上的函数，满足以下三条公理：

**公理1（非负性）**：

$$
P(A) \geq 0, \quad \forall A
$$

**公理2（规范性）**：

$$
P(\Omega) = 1
$$

**公理3（可列可加性）**：

若 $A_1, A_2, \ldots$ 两两互斥，则：

$$
P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)
$$

### 频率学派 vs 贝叶斯学派

| 学派 | 概率解释 | 特点 |
|------|----------|------|
| 频率学派 | 长期频率 | 客观，基于重复试验 |
| 贝叶斯学派 | 信念程度 | 主观，可更新 |

```python
import numpy as np

# 古典概型示例：抽奖
# 100张奖券，10张中奖
n_total = 100
n_win = 10

p_win = n_win / n_total
print(f"中奖概率: {p_win:.2%}")

# 模拟验证
n_trials = 100000
results = np.random.choice([0, 1], size=n_trials, p=[1-p_win, p_win])
frequency = results.mean()
print(f"模拟频率: {frequency:.4f}")
print(f"理论概率: {p_win:.4f}")
```

---

## 概率的基本性质

由公理可推导出以下性质：

### 基本性质

**空集概率为零**：

$$
P(\emptyset) = 0
$$

**对立事件**：

$$
P(A^c) = 1 - P(A)
$$

**加法公式**：

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

**单调性**：

$$
A \subset B \Rightarrow P(A) \leq P(B)
$$

**有界性**：

$$
0 \leq P(A) \leq 1
$$

### 推广的加法公式

**三个事件**：

$$
P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)
$$

**容斥原理**（n个事件）：

$$
P\left(\bigcup_{i=1}^n A_i\right) = \sum_{i} P(A_i) - \sum_{i<j} P(A_i \cap A_j) + \sum_{i<j<k} P(A_i \cap A_j \cap A_k) - \cdots
$$

```python
# 加法公式验证
# P(A∪B) = P(A) + P(B) - P(A∩B)
omega = {1, 2, 3, 4, 5, 6}
A = {1, 2, 3, 4}  # P(A) = 4/6
B = {3, 4, 5, 6}  # P(B) = 4/6

p_A = len(A) / len(omega)
p_B = len(B) / len(omega)
p_AB = len(A & B) / len(omega)
p_union = len(A | B) / len(omega)

# 验证加法公式
p_union_formula = p_A + p_B - p_AB
print(f"P(A∪B) 直接计算: {p_union:.4f}")
print(f"P(A∪B) 公式计算: {p_union_formula:.4f}")
print(f"验证: {abs(p_union - p_union_formula) < 1e-10}")

# 对立事件
p_complement_A = 1 - p_A
p_complement_A_direct = len(omega - A) / len(omega)
print(f"\nP(A^c) = 1 - P(A): {p_complement_A:.4f}")
print(f"P(A^c) 直接计算: {p_complement_A_direct:.4f}")
```

---

## 条件概率

### 🎯 生活类比：缩小搜索范围

想象你在图书馆找一本书：

| 情况 | 搜索范围 | 找到的概率 |
|------|----------|-----------|
| 整个图书馆 | 100万本书 | 很低 |
| 只在历史区 | 5万本书 | 高一些 |
| 只在"中国历史"架 | 1000本书 | 更高 |
| 只在"明朝"那一层 | 50本书 | 最高 |

**条件概率 = 在已知部分信息后，更新你的预测**

"在历史区找到《明朝那些事儿》的概率" 远大于 "在整个图书馆找到它的概率"

### 📝 手把手计算：抽球游戏

**问题**：袋子里有3红球、2白球。先抽一个红球，不放回。再抽一个是红球的概率？

**Step 1**：理解"条件"的含义
- 条件：第一个球是红色
- 这意味着：袋子里剩下的球变了！

**Step 2**：分析新的情况
- 原本：🔴🔴🔴 ⚪⚪（3红2白）
- 抽走红球后：🔴🔴 ⚪⚪（2红2白）

**Step 3**：在新的条件下计算
$$P(\text{第二个红} | \text{第一个红}) = \frac{2}{4} = \frac{1}{2} = 50\%$$

**对比**：如果没有条件，第二个红球的概率是 $\frac{3}{5} = 60\%$

**结论**：知道第一个是红球后，第二个红球的概率**降低了**（因为红球少了一个）。

### 定义

在事件 B 发生的条件下，事件 A 发生的**条件概率**：

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0
$$

**通俗翻译**：$P(A|B)$ = 在 B 已经发生的那些情况中，A 也发生的比例。

### 几何理解

条件概率可以理解为在**缩小后的样本空间**（B）中，A 发生的概率。

### 性质

1. **一般不等于反向条件概率**：$P(A|B) \neq P(B|A)$（通常）

2. **乘法公式**：

$$
P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)
$$

3. **条件概率仍是概率**：满足概率的所有公理

$$
P(A|B) \geq 0
$$

$$
P(\Omega|B) = 1
$$

$$
P(A_1 \cup A_2|B) = P(A_1|B) + P(A_2|B) \quad \text{（若 } A_1, A_2 \text{ 互斥）}
$$

### 示例

袋中有5个球：3红2白。不放回地取2个球，求第一个是红球条件下第二个也是红球的概率。

设 $A$ = "第一个是红球"，$B$ = "第二个是红球"

$$
P(B|A) = \frac{2}{4} = \frac{1}{2}
$$

因为第一个已取走红球，剩余4个球中有2个红球。

```python
import numpy as np

# 条件概率示例
# 袋中有5球：3红2白
# 不放回取2球

# 方法1：直接计算
# P(第二个红 | 第一个红) = 2/4 = 0.5
p_second_red_given_first_red = 2 / 4
print(f"条件概率（直接）: {p_second_red_given_first_red:.4f}")

# 方法2：用定义
# P(第一个红) = 3/5
# P(两个都红) = 3/5 * 2/4 = 6/20 = 3/10
p_first_red = 3 / 5
p_both_red = 3 / 5 * 2 / 4
p_second_given_first = p_both_red / p_first_red
print(f"条件概率（定义）: {p_second_given_first:.4f}")

# 模拟验证
def simulate_draws(n_trials=100000):
    count_first_red = 0
    count_both_red = 0
    
    for _ in range(n_trials):
        balls = ['R', 'R', 'R', 'W', 'W']
        np.random.shuffle(balls)
        first = balls[0]
        second = balls[1]
        
        if first == 'R':
            count_first_red += 1
            if second == 'R':
                count_both_red += 1
    
    return count_both_red / count_first_red

simulated_prob = simulate_draws()
print(f"模拟结果: {simulated_prob:.4f}")
```

---

## 乘法公式

### 两事件

$$
P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)
$$

### 多事件（链式法则）

$$
P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \cdot P(A_2|A_1) \cdot P(A_3|A_1 \cap A_2) \cdots P(A_n|A_1 \cap \cdots \cap A_{n-1})
$$

### 应用示例

袋中有5球：3红2白。不放回地取3球，求依次为红、红、白的概率。

$$
P(\text{红}_1 \cap \text{红}_2 \cap \text{白}_3) = \frac{3}{5} \times \frac{2}{4} \times \frac{2}{3} = \frac{12}{60} = \frac{1}{5}
$$

```python
# 乘法公式示例
# 袋中有5球：3红2白，不放回取3球

# P(红1, 红2, 白3)
p_r1 = 3 / 5
p_r2_given_r1 = 2 / 4  # 取走红后剩2红2白
p_w3_given_r1r2 = 2 / 3  # 取走红红后剩1红2白

p_rrw = p_r1 * p_r2_given_r1 * p_w3_given_r1r2
print(f"P(红红白) = {p_rrw:.4f}")

# 验证：所有可能的排列
from itertools import permutations

balls = ['R', 'R', 'R', 'W', 'W']
all_sequences = list(permutations(balls, 3))
rrw_sequences = [seq for seq in all_sequences if seq == ('R', 'R', 'W')]

# 注意：permutations会重复计算相同的球，需要用组合方式
# 更简单的方法：直接计算组合数
from math import comb
total = comb(5, 3)  # 从5个选3个的所有组合
favorable = comb(3, 2) * comb(2, 1)  # 从3红选2个，从2白选1个
print(f"组合方法验证: {favorable / total:.4f}")
```

---

## 独立性

### 定义

事件 A 和 B **相互独立**，当且仅当：

$$
P(A \cap B) = P(A) \cdot P(B)
$$

等价条件：

$$
P(A|B) = P(A) \quad \text{（B的发生不影响A的概率）}
$$

### 独立 vs 互斥

| 概念 | 条件 | 含义 |
|------|------|------|
| 独立 | $P(A \cap B) = P(A)P(B)$ | 一个事件发生不影响另一个 |
| 互斥 | $A \cap B = \emptyset$ | 两个事件不能同时发生 |

**重要区别**：
- 若 $P(A) > 0$ 且 $P(B) > 0$，则独立和互斥**不能同时成立**
- 互斥意味着：一个发生则另一个必不发生（强相关）
- 独立意味着：一个发生对另一个无影响（无相关）

### 多个事件的独立性

$A_1, A_2, \ldots, A_n$ **相互独立**，当且仅当对任意子集 $I \subseteq \{1, 2, \ldots, n\}$：

$$
P\left(\bigcap_{i \in I} A_i\right) \prod_{i \in I} P(A_i)
$$

**注意**：两两独立 $\neq$ 相互独立

### 条件独立

在 C 发生的条件下，A 和 B 条件独立：

$$
P(A \cap B | C) = P(A|C) \cdot P(B|C)
$$

```python
import numpy as np

# 独立性验证示例
# 掷两个骰子，A=第一个为6，B=第二个为6

# P(A) = P(B) = 1/6
p_A = 1 / 6
p_B = 1 / 6

# P(A∩B) = 1/36
p_AB = 1 / 36

# 验证独立性
is_independent = abs(p_AB - p_A * p_B) < 1e-10
print(f"P(A) × P(B) = {p_A * p_B:.6f}")
print(f"P(A∩B) = {p_AB:.6f}")
print(f"独立? {is_independent}")

# 独立 vs 互斥
# 若 A 和 B 互斥且 P(A) > 0, P(B) > 0
# 则 P(A∩B) = 0 ≠ P(A)P(B) > 0
# 所以互斥 ⇒ 不独立

# 示例：掷一个骰子
# A = 点数为1, B = 点数为2
# 互斥（不能同时发生）
# 但不独立：P(A)P(B) = 1/36 ≠ 0 = P(A∩B)
p_A = 1 / 6
p_B = 1 / 6
p_AB_mutual = 0
print(f"\n互斥事件:")
print(f"P(A) × P(B) = {p_A * p_B:.6f}")
print(f"P(A∩B) = {p_AB_mutual:.6f}")
print(f"独立? {abs(p_AB_mutual - p_A * p_B) < 1e-10}")

# 模拟验证独立性
def test_independence(n_trials=100000):
    # 掷两个骰子
    dice1 = np.random.randint(1, 7, n_trials)
    dice2 = np.random.randint(1, 7, n_trials)
    
    # A: 第一个是6, B: 第二个是6
    count_A = np.sum(dice1 == 6)
    count_B = np.sum(dice2 == 6)
    count_AB = np.sum((dice1 == 6) & (dice2 == 6))
    
    p_A_sim = count_A / n_trials
    p_B_sim = count_B / n_trials
    p_AB_sim = count_AB / n_trials
    
    print(f"\n模拟验证独立性:")
    print(f"P(A) ≈ {p_A_sim:.4f}")
    print(f"P(B) ≈ {p_B_sim:.4f}")
    print(f"P(A∩B) ≈ {p_AB_sim:.4f}")
    print(f"P(A)P(B) ≈ {p_A_sim * p_B_sim:.4f}")

test_independence()
```

---

## 全概率公式

### 划分

设 $B_1, B_2, \ldots, B_n$ 是样本空间的一个**划分**，满足：
1. $B_i \cap B_j = \emptyset$（$i \neq j$）—— 两两互斥
2. $\bigcup_{i=1}^n B_i = \Omega$ —— 覆盖整个样本空间

### 全概率公式

对于任意事件 A 和划分 $B_1, \ldots, B_n$：

$$
P(A) = \sum_{i=1}^n P(A|B_i) \cdot P(B_i)
$$

### 直观理解

将事件 A 分解为互斥的几种情况，分别计算概率再求和。

### 应用示例

工厂有三个车间生产同一产品：
- 一车间生产50%，次品率2%
- 二车间生产30%，次品率3%
- 三车间生产20%，次品率4%

求：任取一件产品是次品的概率。

**解**：

设 $A$ = "产品是次品"，$B_i$ = "产品由车间i生产"

$$
P(A) = P(A|B_1)P(B_1) + P(A|B_2)P(B_2) + P(A|B_3)P(B_3)
$$

$$
= 0.02 \times 0.5 + 0.03 \times 0.3 + 0.04 \times 0.2 = 0.027
$$

```python
# 全概率公式示例
# 三个车间生产产品

p_B1, p_B2, p_B3 = 0.5, 0.3, 0.2  # 各车间产量占比
p_A_given_B1 = 0.02  # 一车间次品率
p_A_given_B2 = 0.03  # 二车间次品率
p_A_given_B3 = 0.04  # 三车间次品率

# 全概率公式
p_A = (p_A_given_B1 * p_B1 + 
       p_A_given_B2 * p_B2 + 
       p_A_given_B3 * p_B3)

print(f"次品概率 P(A) = {p_A:.4f} = {p_A:.2%}")

# 模拟验证
def simulate_production(n_trials=100000):
    # 随机选择车间
    workshops = np.random.choice([1, 2, 3], size=n_trials, p=[p_B1, p_B2, p_B3])
    
    # 各车间的次品概率
    defect_probs = {1: p_A_given_B1, 2: p_A_given_B2, 3: p_A_given_B3}
    
    # 模拟次品
    is_defective = np.array([
        np.random.random() < defect_probs[w] for w in workshops
    ])
    
    return is_defective.mean()

simulated_prob = simulate_production()
print(f"模拟次品率: {simulated_prob:.4f}")
```

---

## 贝叶斯定理

### 贝叶斯公式

$$
P(B_i|A) = \frac{P(A|B_i) \cdot P(B_i)}{P(A)} = \frac{P(A|B_i) \cdot P(B_i)}{\sum_{j=1}^n P(A|B_j) \cdot P(B_j)}
$$

**贝叶斯公式的推导**：

**第一步**：从条件概率的定义出发。

条件概率定义为：

$$P(B|A) = \frac{P(A \cap B)}{P(A)}$$

**第二步**：利用乘法公式展开分子。

$$P(A \cap B_i) = P(A|B_i) \cdot P(B_i)$$

因此：

$$P(B_i|A) = \frac{P(A|B_i) \cdot P(B_i)}{P(A)}$$

**第三步**：使用全概率公式展开分母 $P(A)$。

$$P(A) = \sum_{j=1}^n P(A|B_j) \cdot P(B_j)$$

**第四步**：综合得到贝叶斯公式。

$$\boxed{P(B_i|A) = \frac{P(A|B_i) \cdot P(B_i)}{\sum_{j=1}^n P(A|B_j) \cdot P(B_j)}}$$

**直观理解**：
- $P(B_i)$ 是**先验概率**：在观察到 $A$ 之前，$B_i$ 发生的概率
- $P(B_i|A)$ 是**后验概率**：观察到 $A$ 之后，更新后的 $B_i$ 发生概率
- 贝叶斯公式描述了**如何根据新证据更新信念**

### 贝叶斯解释

$$
\text{后验概率} = \frac{\text{似然} \times \text{先验概率}}{\text{证据（归一化常数）}}
$$

| 术语 | 符号 | 含义 |
|------|------|------|
| 先验概率 | $P(B_i)$ | 观察数据前的信念 |
| 似然 | $P(A|B_i)$ | 假设成立时观察到数据的概率 |
| 证据 | $P(A)$ | 观察到数据的总概率 |
| 后验概率 | $P(B_i|A)$ | 观察数据后的更新信念 |

### 贝叶斯推断的本质

贝叶斯定理描述了**如何根据新证据更新信念**：

$$
\text{后验} \propto \text{似然} \times \text{先验}
$$

### 应用示例：垃圾邮件分类

设：
- $S$：邮件是垃圾邮件
- $W$：邮件包含词"免费"

已知：
- $P(S) = 0.3$（先验：30%邮件是垃圾邮件）
- $P(W|S) = 0.8$（似然：80%垃圾邮件含"免费"）
- $P(W|\bar{S}) = 0.1$（10%正常邮件含"免费"）

求：$P(S|W)$（含"免费"的邮件是垃圾邮件的概率）

**解**：

**步骤1**：计算证据 $P(W)$

$$
P(W) = P(W|S)P(S) + P(W|\bar{S})P(\bar{S}) = 0.8 \times 0.3 + 0.1 \times 0.7 = 0.31
$$

**步骤2**：应用贝叶斯公式

$$
P(S|W) = \frac{P(W|S)P(S)}{P(W)} = \frac{0.8 \times 0.3}{0.31} \approx 0.774
$$

**解释**：看到"免费"后，垃圾邮件的概率从30%更新到77.4%。

```python
import numpy as np

# 贝叶斯定理示例：垃圾邮件分类

# 先验概率
p_spam = 0.3
p_not_spam = 1 - p_spam

# 似然
p_free_given_spam = 0.8
p_free_given_not_spam = 0.1

# 计算证据（全概率公式）
p_free = (p_free_given_spam * p_spam + 
          p_free_given_not_spam * p_not_spam)

print(f"证据 P(免费): {p_free:.4f}")

# 贝叶斯公式
p_spam_given_free = (p_free_given_spam * p_spam) / p_free

print(f"\n先验 P(垃圾邮件): {p_spam:.2%}")
print(f"后验 P(垃圾邮件|免费): {p_spam_given_free:.2%}")
print(f"信念更新: {p_spam:.2%} → {p_spam_given_free:.2%}")

# 模拟验证
def simulate_email(n_trials=100000):
    # 生成邮件类型
    is_spam = np.random.random(n_trials) < p_spam
    
    # 根据类型决定是否包含"免费"
    contains_free = np.array([
        np.random.random() < (p_free_given_spam if spam else p_free_given_not_spam)
        for spam in is_spam
    ])
    
    # 计算 P(垃圾邮件 | 包含免费)
    free_emails = contains_free == True
    spam_given_free = is_spam[free_emails].mean()
    
    return spam_given_free

simulated = simulate_email()
print(f"\n模拟 P(垃圾邮件|免费): {simulated:.2%}")
```

---

## 在深度学习中的应用

### Softmax 与类别概率

Softmax 将网络输出转换为概率分布：

$$
P(y = i|x) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

```python
import numpy as np

def softmax(logits):
    """将 logits 转换为概率分布"""
    exp_logits = np.exp(logits - np.max(logits))  # 数值稳定
    return exp_logits / np.sum(exp_logits)

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Logits: {logits}")
print(f"概率分布: {probs}")
print(f"概率之和: {probs.sum():.6f}")
```

### Dropout 的概率解释

训练时每个神经元以概率 $p$ 保留：

$$
\tilde{h}_i = \frac{h_i \cdot m_i}{p}, \quad m_i \sim \text{Bernoulli}(p)
$$

期望保持不变：$\mathbb{E}[\tilde{h}_i] = h_i$

```python
def dropout(x, p=0.5, training=True):
    """Dropout 实现"""
    if not training:
        return x
    
    mask = (np.random.random(x.shape) > p).astype(float)
    return x * mask / p

# 示例
x = np.random.randn(1000)
x_dropped = dropout(x, p=0.5)

print(f"原始均值: {x.mean():.4f}")
print(f"Dropout后均值: {x_dropped.mean():.4f}")  # 应该接近
```

### 贝叶斯神经网络

参数的后验分布：

$$
P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}
$$

预测分布（对参数积分）：

$$
P(y|x,D) = \int P(y|x,\theta) P(\theta|D) d\theta
$$

---

## 小结

本章介绍了概率论的基础概念：

| 概念 | 定义/公式 | 应用 |
|------|----------|------|
| 条件概率 | $P(A|B) = P(A \cap B)/P(B)$ | 贝叶斯推断 |
| 独立性 | $P(A \cap B) = P(A)P(B)$ | Dropout、数据假设 |
| 乘法公式 | $P(AB) = P(A|B)P(B)$ | 链式计算 |
| 全概率公式 | $P(A) = \sum P(A|B_i)P(B_i)$ | 分解复杂事件 |
| 贝叶斯定理 | $P(B|A) = P(A|B)P(B)/P(A)$ | 信念更新 |

### 关键概念

1. **条件概率**：在已知信息下更新概率
2. **独立性**：事件之间无相互影响
3. **贝叶斯推断**：先验 + 证据 → 后验
4. **全概率公式**：将复杂问题分解为简单子问题

---

**下一节**：[第三章（b）：随机变量与常见分布](03b-随机变量与常见分布.md) - 学习离散和连续随机变量及其常见分布。

**返回**：[数学基础教程目录](../math-fundamentals.md)
