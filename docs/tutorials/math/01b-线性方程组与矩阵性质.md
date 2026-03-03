# 第一章（b）：线性方程组与矩阵性质

线性方程组是线性代数的核心问题之一，而行列式和矩阵的秩是理解矩阵性质的关键工具。本章将系统介绍这些概念及其在深度学习中的应用。

---

## 目录

1. [线性方程组](#线性方程组)
2. [行列式](#行列式)
3. [矩阵的秩](#矩阵的秩)
4. [线性相关与线性无关](#线性相关与线性无关)
5. [矩阵的四个基本子空间](#矩阵的四个基本子空间)
6. [在深度学习中的应用](#在深度学习中的应用)
7. [小结](#小结)

---

## 线性方程组

### 🎯 生活类比：超市购物账单

想象你去超市买了苹果和橘子，但小票丢了。你只记得：
- **第一次**：买了2个苹果和3个橘子，共花了17元
- **第二次**：买了1个苹果和2个橘子，共花了10元

**问题**：苹果和橘子各多少钱一个？

设苹果价格为 $x_1$，橘子价格为 $x_2$，列出方程组：
$$
\begin{cases}
2x_1 + 3x_2 = 17 \\
x_1 + 2x_2 = 10
\end{cases}
$$

这就是一个**线性方程组**！我们要做的，就是找出满足所有方程的未知数 $x_1$ 和 $x_2$。

### 基本形式

**线性方程组**是由若干线性方程组成的系统，其矩阵形式为：

$$
\mathbf{Ax} = \mathbf{b}
$$

其中：
- $\mathbf{A} \in \mathbb{R}^{m \times n}$：系数矩阵
- $\mathbf{x} \in \mathbb{R}^n$：未知数向量
- $\mathbf{b} \in \mathbb{R}^m$：常数项向量

**展开形式**：

$$
\begin{cases}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\
\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m
\end{cases}
$$

### 增广矩阵

将系数矩阵和常数项合并：

$$
[\mathbf{A}|\mathbf{b}] = \begin{bmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} & | & b_1 \\
a_{21} & a_{22} & \cdots & a_{2n} & | & b_2 \\
\vdots & \vdots & \ddots & \vdots & | & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn} & | & b_m
\end{bmatrix}
$$

### 解的判定（Rouché-Capelli 定理）

线性方程组 $\mathbf{Ax} = \mathbf{b}$ 的解取决于系数矩阵 $\mathbf{A}$ 和增广矩阵 $[\mathbf{A}|\mathbf{b}]$ 的秩：

| 秩的关系 | 解的情况 |
|----------|----------|
| $\text{rank}(\mathbf{A}) = \text{rank}([\mathbf{A}\|\mathbf{b}]) = n$ | 唯一解 |
| $\text{rank}(\mathbf{A}) = \text{rank}([\mathbf{A}\|\mathbf{b}]) < n$ | 无穷多解 |
| $\text{rank}(\mathbf{A}) < \text{rank}([\mathbf{A}\|\mathbf{b}])$ | 无解 |

其中 $n$ 是未知数的个数。

### 特殊情况：方阵系统

对于 $n \times n$ 方阵系统，判定简化为：

| 条件 | 解的情况 |
|------|----------|
| $\det(\mathbf{A}) \neq 0$ | 唯一解 $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$ |
| $\det(\mathbf{A}) = 0$ 且 $\mathbf{b} \in \text{range}(\mathbf{A})$ | 无穷多解 |
| $\det(\mathbf{A}) = 0$ 且 $\mathbf{b} \notin \text{range}(\mathbf{A})$ | 无解 |

### 齐次线性方程组

当 $\mathbf{b} = \mathbf{0}$ 时，$\mathbf{Ax} = \mathbf{0}$ 称为齐次线性方程组。

**性质**：
- 齐次方程组**总是有解**（至少 $\mathbf{x} = \mathbf{0}$ 是零解）
- 有非零解当且仅当 $\det(\mathbf{A}) = 0$（或 $\text{rank}(\mathbf{A}) < n$）
- 解的集合构成一个**线性子空间**

### 求解方法

#### 1. 直接求解（方阵，可逆）

```python
import numpy as np

A = np.array([[3, 1],
              [1, 2]], dtype=float)
b = np.array([9, 8], dtype=float)

# 方法1：numpy.linalg.solve（推荐）
x = np.linalg.solve(A, b)
print(f"解: {x}")  # [2., 3.]

# 验证
print(f"A @ x = {A @ x}")  # [9., 8.]
```

#### 2. 矩阵求逆（不推荐）

```python
# 方法2：求逆矩阵（数值不稳定，效率低）
x_inv = np.linalg.inv(A) @ b
print(f"逆矩阵法解: {x_inv}")  # [2., 3.]
```

**为什么不推荐**：
- 数值不稳定（对条件数敏感）
- 计算复杂度高（$O(n^3)$）
- 很多情况下矩阵不可逆

#### 3. 最小二乘法（超定系统）

当方程个数多于未知数（$m > n$），通常无精确解，使用最小二乘法：

$$
\min_{\mathbf{x}} \|\mathbf{Ax} - \mathbf{b}\|_2^2
$$

正规方程：

$$
\mathbf{A}^\top\mathbf{Ax} = \mathbf{A}^\top\mathbf{b}
$$

**正规方程的推导**：

目标是最小化残差平方和：

$$f(\mathbf{x}) = \|\mathbf{Ax} - \mathbf{b}\|_2^2 = (\mathbf{Ax} - \mathbf{b})^\top(\mathbf{Ax} - \mathbf{b})$$

**第一步**：展开目标函数：

$$f(\mathbf{x}) = \mathbf{x}^\top\mathbf{A}^\top\mathbf{Ax} - \mathbf{x}^\top\mathbf{A}^\top\mathbf{b} - \mathbf{b}^\top\mathbf{Ax} + \mathbf{b}^\top\mathbf{b}$$

由于 $\mathbf{b}^\top\mathbf{Ax}$ 是标量，等于其转置 $\mathbf{x}^\top\mathbf{A}^\top\mathbf{b}$，所以：

$$f(\mathbf{x}) = \mathbf{x}^\top\mathbf{A}^\top\mathbf{Ax} - 2\mathbf{x}^\top\mathbf{A}^\top\mathbf{b} + \mathbf{b}^\top\mathbf{b}$$

**第二步**：对 $\mathbf{x}$ 求梯度并令其为零：

利用矩阵求导公式 $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^\top\mathbf{M}\mathbf{x}) = 2\mathbf{Mx}$（$\mathbf{M}$ 对称）和 $\frac{\partial}{\partial \mathbf{x}}(\mathbf{c}^\top\mathbf{x}) = \mathbf{c}$：

$$\nabla f(\mathbf{x}) = 2\mathbf{A}^\top\mathbf{Ax} - 2\mathbf{A}^\top\mathbf{b} = \mathbf{0}$$

**第三步**：整理得到正规方程：

$$\mathbf{A}^\top\mathbf{Ax} = \mathbf{A}^\top\mathbf{b}$$

**第四步**：解正规方程：

如果 $\mathbf{A}^\top\mathbf{A}$ 可逆（即 $\mathbf{A}$ 列满秩），则：

$$\boxed{\mathbf{x}^* = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}}$$

这就是最小二乘解。

**注**：$(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top$ 称为 $\mathbf{A}$ 的**左伪逆**，记作 $\mathbf{A}^+$。

```python
# 超定系统（方程多于未知数）
A_over = np.array([[1, 1],
                   [1, 2],
                   [1, 3]], dtype=float)
b_over = np.array([1, 2, 2], dtype=float)

# 最小二乘解
x_lstsq, residuals, rank, s = np.linalg.lstsq(A_over, b_over, rcond=None)
print(f"最小二乘解: {x_lstsq}")
print(f"残差: {residuals}")
print(f"秩: {rank}")

# 使用正规方程验证
x_normal = np.linalg.solve(A_over.T @ A_over, A_over.T @ b_over)
print(f"正规方程解: {x_normal}")
```

#### 4. 欠定系统（$m < n$）

当未知数多于方程时，有无穷多解，可以使用最小范数解：

$$
\mathbf{x}^* = \mathbf{A}^\top(\mathbf{AA}^\top)^{-1}\mathbf{b}
$$

```python
# 欠定系统（未知数多于方程）
A_under = np.array([[1, 2, 3],
                    [4, 5, 6]], dtype=float)
b_under = np.array([6, 15], dtype=float)

# 使用伪逆求最小范数解
A_pinv = np.linalg.pinv(A_under)
x_min_norm = A_pinv @ b_under
print(f"最小范数解: {x_min_norm}")
print(f"解的范数: {np.linalg.norm(x_min_norm)}")
```

### 高斯消元法

高斯消元法是求解线性方程组的经典方法：

1. **前向消元**：将矩阵化为上三角形式
2. **回代**：从最后一行开始求解

```python
def gaussian_elimination(A, b):
    """高斯消元法求解 Ax = b"""
    n = len(b)
    # 构造增广矩阵
    Ab = np.hstack([A, b.reshape(-1, 1)])
    
    # 前向消元
    for i in range(n):
        # 找主元
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        Ab[[i, max_row]] = Ab[[max_row, i]]
        
        # 消元
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
    
    # 回代
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]
    
    return x

A = np.array([[3, 1, 2],
              [1, 2, 1],
              [2, 1, 3]], dtype=float)
b = np.array([9, 8, 11], dtype=float)

x = gaussian_elimination(A, b)
print(f"高斯消元法解: {x}")
print(f"验证: A @ x = {A @ x}")
```

### LU 分解求解

LU 分解将矩阵分解为下三角和上三角矩阵的乘积：

$$
\mathbf{A} = \mathbf{LU}
$$

求解 $\mathbf{Ax} = \mathbf{b}$ 变为：
1. 求解 $\mathbf{Ly} = \mathbf{b}$（前向替换）
2. 求解 $\mathbf{Ux} = \mathbf{y}$（回代）

```python
from scipy.linalg import lu_factor, lu_solve

A = np.array([[3, 1, 2],
              [1, 2, 1],
              [2, 1, 3]], dtype=float)
b = np.array([9, 8, 11], dtype=float)

# LU 分解
lu, piv = lu_factor(A)
x = lu_solve((lu, piv), b)
print(f"LU 分解解: {x}")
```

---

## 行列式

### 🎯 生活类比：橡皮泥的缩放倍数

想象你有一块正方形的橡皮泥（面积=1），用一个矩阵变换来"拉扯"它：
- 有的变换会让橡皮泥**变大**（行列式>1）
- 有的变换会让橡皮泥**变小**（行列式<1）
- 有的变换会把橡皮泥**压扁成一条线**（行列式=0）

**行列式就是"面积（或体积）缩放了多少倍"！**

- 行列式=2：面积变成原来的2倍
- 行列式=-1：面积不变，但翻转了（如镜像）
- 行列式=0：被压扁了！这就是"奇异矩阵"

### 📝 手把手计算：2×2 行列式

对于矩阵 $\begin{bmatrix} a & b \\ c & d \end{bmatrix}$，行列式 = $ad - bc$

**例子**：计算 $\det\begin{bmatrix} 3 & 2 \\ 1 & 4 \end{bmatrix}$

```
步骤：
① 主对角线相乘：3 × 4 = 12
② 副对角线相乘：2 × 1 = 2
③ 相减：12 - 2 = 10

答案：行列式 = 10
```

**验证**：行列式≠0，说明这个矩阵可以"逆转"，有唯一解！

### 定义

$n$ 阶方阵 $\mathbf{A}$ 的**行列式** $\det(\mathbf{A})$ 或 $|\mathbf{A}|$ 是一个标量，反映矩阵的"体积缩放因子"和可逆性。

### 低阶行列式

**2×2 矩阵**：

$$
\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc
$$

**3×3 矩阵**（Sarrus 规则）：

$$
\det\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} = aei + bfg + cdh - ceg - bdi - afh
$$

### 通用定义（按行/列展开）

按第 $i$ 行展开（Laplace 展开）：

$$
\det(\mathbf{A}) = \sum_{j=1}^n (-1)^{i+j} A_{ij} M_{ij}
$$

其中：
- $M_{ij}$ 是**余子式**：去掉第 $i$ 行第 $j$ 列后的子矩阵的行列式
- $C_{ij} = (-1)^{i+j} M_{ij}$ 是**代数余子式**

### 行列式的性质

| 性质 | 公式 |
|------|------|
| 单位矩阵 | $\det(\mathbf{I}) = 1$ |
| 转置不变 | $\det(\mathbf{A}^\top) = \det(\mathbf{A})$ |
| 乘积 | $\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})$ |
| 逆矩阵 | $\det(\mathbf{A}^{-1}) = \frac{1}{\det(\mathbf{A})}$ |
| 标量乘法 | $\det(\alpha\mathbf{A}) = \alpha^n \det(\mathbf{A})$（$n \times n$ 矩阵） |
| 交换两行 | 行列式变号 |
| 两行相同 | 行列式为零 |
| 某行为零 | 行列式为零 |
| 行的线性组合 | 不改变行列式 |

**行列式乘法性质 $\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})$ 的证明**：

**方法一：利用特征值**

**第一步**：设 $\lambda_1, \ldots, \lambda_n$ 是 $\mathbf{A}$ 的特征值，$\mu_1, \ldots, \mu_n$ 是 $\mathbf{B}$ 的特征值。

已知 $\det(\mathbf{A}) = \prod_{i=1}^n \lambda_i$，$\det(\mathbf{B}) = \prod_{i=1}^n \mu_i$。

**第二步**：考虑 $\mathbf{AB}$ 的特征值。

如果 $\mathbf{v}$ 是 $\mathbf{B}$ 的特征向量，$\mathbf{Bv} = \mu \mathbf{v}$，则：

$$\mathbf{ABv} = \mathbf{A}(\mu\mathbf{v}) = \mu(\mathbf{Av})$$

但 $\mathbf{Av}$ 不一定是 $\mathbf{A}$ 的特征向量。我们换一种方法。

**第三步**：利用对角化（假设可对角化）。

设 $\mathbf{A} = \mathbf{P}\mathbf{D}_A\mathbf{P}^{-1}$，$\mathbf{B} = \mathbf{Q}\mathbf{D}_B\mathbf{Q}^{-1}$，其中 $\mathbf{D}_A = \text{diag}(\lambda_1, \ldots, \lambda_n)$，$\mathbf{D}_B = \text{diag}(\mu_1, \ldots, \mu_n)$。

**第四步**：计算 $\det(\mathbf{AB})$：

$$\det(\mathbf{AB}) = \det(\mathbf{P}\mathbf{D}_A\mathbf{P}^{-1}\mathbf{Q}\mathbf{D}_B\mathbf{Q}^{-1})$$

利用 $\det(\mathbf{XY}) = \det(\mathbf{X})\det(\mathbf{Y})$（这是我们正在证明的）会循环论证。改用初等变换法。

**方法二：利用初等矩阵**

**第一步**：任何矩阵 $\mathbf{A}$ 可以通过初等行变换化为上三角矩阵 $\mathbf{U}$：

$$\mathbf{A} = \mathbf{E}_k \cdots \mathbf{E}_2 \mathbf{E}_1 \mathbf{U}$$

其中 $\mathbf{E}_i$ 是初等矩阵。

**第二步**：对于初等矩阵，直接验证 $\det(\mathbf{E}_1\mathbf{E}_2) = \det(\mathbf{E}_1)\det(\mathbf{E}_2)$。

- 交换两行：$\det = -1$
- 某行乘以 $c$：$\det = c$
- 某行加上另一行的倍数：$\det = 1$

三种情况下乘法性质都成立。

**第三步**：对于上三角矩阵 $\mathbf{U}$，$\det(\mathbf{U}) = \prod_{i} U_{ii}$。

同理 $\mathbf{B} = \mathbf{F}_l \cdots \mathbf{F}_1 \mathbf{V}$（$\mathbf{V}$ 上三角）。

**第四步**：因此 $\det(\mathbf{A})\det(\mathbf{B}) = \det(\mathbf{U})\det(\mathbf{V}) \cdot (\text{初等矩阵行列式之积})$

$\det(\mathbf{AB}) = \det(\mathbf{E}_k \cdots \mathbf{F}_l \cdots \mathbf{U}\mathbf{B}) = \ldots = \det(\mathbf{U})\det(\mathbf{V}) \cdot (\text{相同因子})$

$$\boxed{\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})}$$

**逆矩阵行列式 $\det(\mathbf{A}^{-1}) = \frac{1}{\det(\mathbf{A})}$ 的推导**：

由 $\mathbf{AA}^{-1} = \mathbf{I}$，两边取行列式：

$$\det(\mathbf{AA}^{-1}) = \det(\mathbf{I})$$

$$\det(\mathbf{A})\det(\mathbf{A}^{-1}) = 1$$

$$\boxed{\det(\mathbf{A}^{-1}) = \frac{1}{\det(\mathbf{A})}}$$

### 特殊矩阵的行列式

**对角矩阵**：

$$
\det(\text{diag}(d_1, \ldots, d_n)) = \prod_{i=1}^n d_i
$$

**三角矩阵**：

$$
\det\begin{bmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\
0 & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & a_{nn}
\end{bmatrix} = \prod_{i=1}^n a_{ii}
$$

**正交矩阵**：

$$
\det(\mathbf{Q}) = \pm 1
$$

**分块矩阵**（$\mathbf{A}$ 可逆）：

$$
\det\begin{bmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{C} & \mathbf{D} \end{bmatrix} = \det(\mathbf{A})\det(\mathbf{D} - \mathbf{CA}^{-1}\mathbf{B})
$$

### 几何意义

行列式的绝对值表示线性变换对体积的缩放倍数：

| $\det(\mathbf{A})$ 的值 | 几何意义 |
|------------------------|----------|
| $\det(\mathbf{A}) > 0$ | 保持方向（正体积） |
| $\det(\mathbf{A}) < 0$ | 反转方向（负体积） |
| $\det(\mathbf{A}) = 0$ | 降维（体积为零，不可逆） |
| $\det(\mathbf{A}) = 1$ | 保持体积（如旋转） |

**2D 示例**：行列式表示由列向量张成的平行四边形面积。

**3D 示例**：行列式表示由列向量张成的平行六面体体积。

### Python 实现

```python
import numpy as np

# 2x2 矩阵的行列式
A_2x2 = np.array([[1, 2],
                  [3, 4]])
det_2x2 = np.linalg.det(A_2x2)
print(f"det(A_2x2) = {det_2x2}")  # -2.0

# 解析验证: 1*4 - 2*3 = -2

# 3x3 矩阵的行列式
A_3x3 = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 10]])
det_3x3 = np.linalg.det(A_3x3)
print(f"det(A_3x3) = {det_3x3}")  # -3.0

# 对角矩阵的行列式
D = np.diag([1, 2, 3, 4])
det_D = np.linalg.det(D)
print(f"det(diag) = {det_D}")  # 24 = 1*2*3*4

# 性质验证
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# det(AB) = det(A) * det(B)
det_AB = np.linalg.det(A @ B)
det_A_det_B = np.linalg.det(A) * np.linalg.det(B)
print(f"det(AB) = {det_AB:.2f}")
print(f"det(A) * det(B) = {det_A_det_B:.2f}")
print(f"相等? {np.allclose(det_AB, det_A_det_B)}")  # True

# det(A^T) = det(A)
det_A = np.linalg.det(A)
det_At = np.linalg.det(A.T)
print(f"det(A) = {det_A}, det(A^T) = {det_At}")
print(f"相等? {np.allclose(det_A, det_At)}")  # True
```

### 行列式与可逆性

```python
def is_invertible(A, tol=1e-10):
    """判断矩阵是否可逆"""
    det = np.linalg.det(A)
    return abs(det) > tol

# 可逆矩阵
A_inv = np.array([[1, 2], [3, 4]])
print(f"A 可逆? {is_invertible(A_inv)}")  # True (det = -2)

# 不可逆矩阵（奇异矩阵）
A_sing = np.array([[1, 2], [2, 4]])
print(f"奇异矩阵可逆? {is_invertible(A_sing)}")  # False (det = 0)

# 验证：第二行是第一行的2倍
```

---

## 矩阵的秩

### 🎯 生活类比：真正有效的方程数

想象你在做数学作业，有5道方程题，但是：
- 第1题和第2题其实是一样的（只是数字乘了2倍）
- 第3题可以用第1题和第2题组合出来
- 只有第4题和第5题是真正独立的

**秩 = 真正独立的方程个数 = 实际有用的信息量**

**具体例子**：
$$
\begin{cases}
x + y = 5 \quad \text{(独立)} \\
2x + 2y = 10 \quad \text{(冗余，是第1个方程×2)} \\
3x + 3y = 15 \quad \text{(冗余，是第1个方程×3)}
\end{cases}
$$

这个方程组的秩=1，因为只有1个真正独立的方程！

### 📝 手把手判断：计算矩阵的秩

矩阵：
$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 1 & 1 \end{bmatrix}
$$

**方法：化简为行阶梯形**

```
原矩阵：
[1  2  3]
[2  4  6]  ← 第2行 = 2×第1行，消掉！
[1  1  1]

消元后：
[1  2  3]
[0  0  0]  ← 全是0
[0 -1 -2]

再整理：
[1  2  3]
[1  1  1]  ← 交换行
[0  0  0]

继续消元：
[1  2  3]
[0 -1 -2]  ← 非零行！
[0  0  0]
```

**非零行数 = 2，所以秩 = 2**

### 定义

矩阵 $\mathbf{A}$ 的**秩** $\text{rank}(\mathbf{A})$ 是其行（或列）向量组的最大线性无关向量个数。

### 等价定义

矩阵的秩有以下等价定义：

1. **线性无关的行向量最大数目**（行秩）
2. **线性无关的列向量最大数目**（列秩）
3. **非零奇异值的个数**
4. **最高阶非零子式的阶数**
5. **矩阵像空间的维度**：$\text{rank}(\mathbf{A}) = \dim(\text{range}(\mathbf{A}))$

### 性质

| 性质 | 公式 |
|------|------|
| 范围 | $0 \leq \text{rank}(\mathbf{A}_{m \times n}) \leq \min(m, n)$ |
| 转置不变 | $\text{rank}(\mathbf{A}) = \text{rank}(\mathbf{A}^\top)$ |
| 乘积 | $\text{rank}(\mathbf{AB}) \leq \min(\text{rank}(\mathbf{A}), \text{rank}(\mathbf{B}))$ |
| 加法 | $\text{rank}(\mathbf{A} + \mathbf{B}) \leq \text{rank}(\mathbf{A}) + \text{rank}(\mathbf{B})$ |
| 与逆矩阵 | $\text{rank}(\mathbf{A}^{-1}\mathbf{A}) = \text{rank}(\mathbf{A})$ |

### 满秩

对于 $m \times n$ 矩阵 $\mathbf{A}$：

| 类型 | 定义 | 条件 |
|------|------|------|
| **满秩** | $\text{rank}(\mathbf{A}) = \min(m, n)$ | 秩达到最大可能值 |
| **列满秩** | $\text{rank}(\mathbf{A}) = n$ | 列向量线性无关 |
| **行满秩** | $\text{rank}(\mathbf{A}) = m$ | 行向量线性无关 |
| **满秩方阵** | $\text{rank}(\mathbf{A}) = n = m$ | 可逆矩阵 |

### 秩与行列式的关系

对于 $n \times n$ 方阵：

$$
\text{rank}(\mathbf{A}) = n \iff \det(\mathbf{A}) \neq 0 \iff \mathbf{A} \text{ 可逆}
$$

### 秩的计算

**方法1**：高斯消元法化为行阶梯形，非零行的数目就是秩。

**方法2**：计算非零奇异值的个数。

```python
import numpy as np

# 计算矩阵的秩
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
rank_A = np.linalg.matrix_rank(A)
print(f"rank(A) = {rank_A}")  # 2（第三行是前两行的线性组合）

# 验证：第三行 = 2*第二行 - 第一行
print(f"第三行: {A[2]}")
print(f"2*第二行 - 第一行: {2*A[1] - A[0]}")  # [7, 8, 9]

# 满秩矩阵
B = np.array([[1, 0],
              [0, 1]])
rank_B = np.linalg.matrix_rank(B)
print(f"rank(B) = {rank_B}")  # 2（满秩）

# 通过奇异值计算秩
U, S, Vt = np.linalg.svd(A)
print(f"奇异值: {S}")
print(f"非零奇异值个数: {np.sum(S > 1e-10)}")  # 2

# 秩与行列式的关系
C = np.array([[1, 2], [3, 4]])
print(f"rank(C) = {np.linalg.matrix_rank(C)}")  # 2
print(f"det(C) = {np.linalg.det(C)}")  # -2（非零，满秩）

D = np.array([[1, 2], [2, 4]])
print(f"rank(D) = {np.linalg.matrix_rank(D)}")  # 1
print(f"det(D) = {np.linalg.det(D)}")  # 0（零，不满秩）
```

### 秩分解

任何 $m \times n$ 矩阵 $\mathbf{A}$ 可以分解为：

$$
\mathbf{A} = \mathbf{CR}
$$

其中 $\mathbf{C} \in \mathbb{R}^{m \times r}$ 是列满秩矩阵，$\mathbf{R} \in \mathbb{R}^{r \times n}$ 是行满秩矩阵，$r = \text{rank}(\mathbf{A})$。

---

## 线性相关与线性无关

### 定义

**线性组合**：向量 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ 的线性组合是：

$$
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k
$$

**线性相关**：如果存在不全为零的标量 $c_1, c_2, \ldots, c_k$ 使得：

$$
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0}
$$

则称这些向量**线性相关**。

**线性无关**：如果只有当 $c_1 = c_2 = \cdots = c_k = 0$ 时上式才成立，则称这些向量**线性无关**。

### 判定方法

1. **定义法**：检查齐次方程组是否有非零解
2. **行列式法**：对于 $n$ 个 $n$ 维向量，构成矩阵后行列式非零则线性无关
3. **秩法**：向量构成的矩阵的秩等于向量个数则线性无关

### 性质

- 如果向量组中有一个零向量，则线性相关
- 如果向量组中有两个相同的向量，则线性相关
- 如果向量组中有一个向量是其他向量的线性组合，则线性相关
- $n$ 维空间中最多有 $n$ 个线性无关的向量

```python
import numpy as np

def check_linear_independence(vectors):
    """检查向量组是否线性无关"""
    # 构建矩阵（向量作为列）
    A = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(A)
    n_vectors = len(vectors)
    
    return rank == n_vectors, rank

# 线性无关的向量
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])

indep, rank = check_linear_independence([v1, v2, v3])
print(f"单位向量线性无关? {indep}, 秩 = {rank}")  # True, 3

# 线性相关的向量
v4 = np.array([1, 2, 3])
v5 = np.array([2, 4, 6])  # v5 = 2 * v4
v6 = np.array([3, 6, 9])  # v6 = 3 * v4

indep, rank = check_linear_independence([v4, v5, v6])
print(f"相关向量线性无关? {indep}, 秩 = {rank}")  # False, 1

# 使用行列式判定（方阵情况）
A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])
det_A = np.linalg.det(A)
print(f"det(A) = {det_A:.2f}")  # 非零意味着线性无关
```

---

## 矩阵的四个基本子空间

对于 $\mathbf{A} \in \mathbb{R}^{m \times n}$，存在四个重要的子空间：

### 1. 列空间（像空间）

$$
\text{Col}(\mathbf{A}) = \text{range}(\mathbf{A}) = \{\mathbf{Ax} : \mathbf{x} \in \mathbb{R}^n\} \subseteq \mathbb{R}^m
$$

- 由矩阵的列向量张成的空间
- 维度：$\dim(\text{Col}(\mathbf{A})) = \text{rank}(\mathbf{A})$

### 2. 行空间

$$
\text{Row}(\mathbf{A}) = \text{Col}(\mathbf{A}^\top) \subseteq \mathbb{R}^n
$$

- 由矩阵的行向量张成的空间
- 维度：$\dim(\text{Row}(\mathbf{A})) = \text{rank}(\mathbf{A})$

### 3. 零空间（核）

$$
\text{Null}(\mathbf{A}) = \{\mathbf{x} : \mathbf{Ax} = \mathbf{0}\} \subseteq \mathbb{R}^n
$$

- 齐次方程 $\mathbf{Ax} = \mathbf{0}$ 的解空间
- 维度：$\dim(\text{Null}(\mathbf{A})) = n - \text{rank}(\mathbf{A})$（零化度）

### 4. 左零空间

$$
\text{Null}(\mathbf{A}^\top) = \{\mathbf{y} : \mathbf{A}^\top\mathbf{y} = \mathbf{0}\} \subseteq \mathbb{R}^m
$$

- 维度：$\dim(\text{Null}(\mathbf{A}^\top)) = m - \text{rank}(\mathbf{A})$

### 基本定理（秩-零化度定理）

$$
\text{rank}(\mathbf{A}) + \dim(\text{Null}(\mathbf{A})) = n
$$

$$
\text{rank}(\mathbf{A}) + \dim(\text{Null}(\mathbf{A}^\top)) = m
$$

**秩-零化度定理的证明**：

设 $\mathbf{A} \in \mathbb{R}^{m \times n}$，$\text{rank}(\mathbf{A}) = r$。

**第一步**：通过初等变换，$\mathbf{A}$ 可以化为行阶梯形：

$$\mathbf{A} \sim \begin{bmatrix} \mathbf{I}_r & \mathbf{F} \\ \mathbf{0} & \mathbf{0} \end{bmatrix}$$

其中 $\mathbf{I}_r$ 是 $r \times r$ 单位矩阵，$\mathbf{F}$ 是 $r \times (n-r)$ 矩阵。

**第二步**：列空间的维度。

行阶梯形的前 $r$ 列是线性无关的（主元列），所以：

$$\dim(\text{Col}(\mathbf{A})) = r = \text{rank}(\mathbf{A})$$

**第三步**：零空间的维度。

求解 $\mathbf{Ax} = \mathbf{0}$ 等价于求解行阶梯形的齐次方程组。

主元变量有 $r$ 个，自由变量有 $n - r$ 个。

每个自由变量对应零空间的一个基向量，所以：

$$\dim(\text{Null}(\mathbf{A})) = n - r$$

**第四步**：综合结果：

$$\text{rank}(\mathbf{A}) + \dim(\text{Null}(\mathbf{A})) = r + (n - r) = n$$

$$\boxed{\text{rank}(\mathbf{A}) + \dim(\text{Null}(\mathbf{A})) = n}$$

**直观理解**：
- $\text{rank}(\mathbf{A})$ = 信息保留的维度
- $\dim(\text{Null}(\mathbf{A}))$ = 信息丢失的维度
- 两者之和等于输入空间的维度 $n$

### 正交关系

- $\text{Row}(\mathbf{A}) \perp \text{Null}(\mathbf{A})$（$\mathbb{R}^n$ 中的正交补）
- $\text{Col}(\mathbf{A}) \perp \text{Null}(\mathbf{A}^\top)$（$\mathbb{R}^m$ 中的正交补）

```python
import numpy as np

def compute_null_space(A, tol=1e-10):
    """计算矩阵的零空间"""
    U, S, Vt = np.linalg.svd(A)
    # 找到小于阈值的奇异值索引
    null_mask = S < tol
    # 零空间是 V^T 对应的行
    null_space = Vt[len(S):, :].T if len(S) < A.shape[1] else Vt[null_mask, :].T
    if null_space.size == 0:
        # 没有零奇异值，零空间只有零向量
        return np.zeros((A.shape[1], 0))
    return null_space

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

# 计算秩
rank = np.linalg.matrix_rank(A)
print(f"rank(A) = {rank}")  # 2

# 验证秩-零化度定理
n = A.shape[1]
nullity = n - rank
print(f"零化度 = {nullity}")  # 1

# 零空间
null_space = compute_null_space(A)
print(f"零空间基向量:\n{null_space}")

# 验证：A @ null_vector ≈ 0
if null_space.size > 0:
    print(f"A @ null_space = {A @ null_space[:, 0]}")
```

---

## 在深度学习中的应用

### 权重矩阵的可逆性

在深度网络中，权重矩阵通常不是方阵，即使行列式无定义，秩仍然重要：

- **满秩权重**：信息能够完整传递
- **低秩权重**：可能表示冗余或过参数化

```python
import numpy as np

# 深度网络中的权重
d_in, d_out = 784, 128
W = np.random.randn(d_in, d_out)

rank_W = np.linalg.matrix_rank(W)
print(f"权重矩阵秩: {rank_W}")
print(f"最大可能秩: {min(d_in, d_out)}")
print(f"满秩? {rank_W == min(d_in, d_out)}")

# 低秩初始化（可能导致问题）
W_low_rank = np.random.randn(d_in, 10) @ np.random.randn(10, d_out)
rank_low = np.linalg.matrix_rank(W_low_rank)
print(f"低秩权重秩: {rank_low}")  # 最多 10
```

### 过参数化与欠参数化

```python
def analyze_model_capacity(n_samples, n_features, n_parameters):
    """分析模型容量"""
    if n_parameters >= n_samples:
        return "过参数化（可能过拟合）"
    elif n_parameters == n_features:
        return "恰好确定"
    else:
        return "欠参数化（可能欠拟合）"

# 示例：MNIST 分类
n_samples = 60000  # 训练样本数
n_features = 784   # 特征数

# 单层网络：784 -> 10
n_params_single = 784 * 10 + 10
print(f"单层网络参数: {n_params_single}")
print(analyze_model_capacity(n_samples, n_features, n_params_single))

# 两层网络：784 -> 256 -> 10
n_params_two = 784 * 256 + 256 + 256 * 10 + 10
print(f"两层网络参数: {n_params_two}")
print(analyze_model_capacity(n_samples, n_features, n_params_two))
```

### 线性方程组与反向传播

反向传播涉及求解线性方程组（在某种程度上）：

```python
def linear_layer_gradients(X, W, dL_dY):
    """
    计算线性层的梯度
    Y = XW + b
    
    参数:
        X: (batch, in_features)
        W: (in_features, out_features)
        dL_dY: (batch, out_features) - 对输出的梯度
    
    返回:
        dL_dX, dL_dW, dL_db
    """
    # 对 W 的梯度: X^T @ dL_dY
    dL_dW = X.T @ dL_dY
    
    # 对 X 的梯度: dL_dY @ W^T
    dL_dX = dL_dY @ W.T
    
    # 对 b 的梯度: 对 batch 维度求和
    dL_db = np.sum(dL_dY, axis=0)
    
    return dL_dX, dL_dW, dL_db

# 示例
batch, in_feat, out_feat = 32, 784, 128
X = np.random.randn(batch, in_feat)
W = np.random.randn(in_feat, out_feat)
dL_dY = np.random.randn(batch, out_feat)

dL_dX, dL_dW, dL_db = linear_layer_gradients(X, W, dL_dY)
print(f"dL_dX 形状: {dL_dX.shape}")  # (32, 784)
print(f"dL_dW 形状: {dL_dW.shape}")  # (784, 128)
print(f"dL_db 形状: {dL_db.shape}")  # (128,)
```

### 判断权重矩阵的健康状态

```python
def analyze_weight_matrix(W, name="W"):
    """分析权重矩阵的健康状态"""
    rank = np.linalg.matrix_rank(W)
    m, n = W.shape
    
    # 奇异值分析
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    
    # 条件数
    cond = S[0] / S[-1] if S[-1] > 1e-10 else np.inf
    
    print(f"\n{name} 分析:")
    print(f"  形状: {m} x {n}")
    print(f"  秩: {rank} / {min(m, n)}")
    print(f"  条件数: {cond:.2e}")
    print(f"  最大奇异值: {S[0]:.4f}")
    print(f"  最小奇异值: {S[-1]:.4e}")
    
    # 判断
    if rank < min(m, n):
        print(f"  警告: 矩阵不满秩!")
    if cond > 1e6:
        print(f"  警告: 条件数过大，可能数值不稳定!")
    
    return rank, cond, S

# 健康的权重
W_healthy = np.random.randn(256, 128) * np.sqrt(2.0 / 256)
analyze_weight_matrix(W_healthy, "健康权重")

# 不健康的权重（低秩）
W_lowrank = np.random.randn(256, 10) @ np.random.randn(10, 128)
analyze_weight_matrix(W_lowrank, "低秩权重")
```

### 线性层的等价性

```python
def check_linear_layer_equivalence(W1, W2, b1, b2, X, tol=1e-6):
    """检查两个线性层是否等价"""
    Y1 = X @ W1 + b1
    Y2 = X @ W2 + b2
    return np.allclose(Y1, Y2, atol=tol)

# 线性层的等价变换
d = 64
W = np.random.randn(d, d)

# 正交变换后的等价层
Q = np.linalg.qr(np.random.randn(d, d))[0]  # 正交矩阵
W_equiv = Q.T @ W @ Q

# 验证：在某些输入上可能不等价（因为变换不同）
X = np.random.randn(10, d)
Y_original = X @ W
Y_transformed = X @ W_equiv
print(f"正交变换后输出相近? {np.allclose(Y_original, Y_transformed)}")  # 通常 False
```

---

## 小结

本章介绍了线性方程组、行列式和矩阵秩的核心概念：

| 概念 | 定义 | 在深度学习中的应用 |
|------|------|-------------------|
| 线性方程组 | $\mathbf{Ax} = \mathbf{b}$ | 求解、最小二乘、反向传播 |
| 行列式 | 矩阵的"体积缩放因子" | 判断可逆性、初始化 |
| 矩阵秩 | 线性无关行/列的最大数目 | 分析过参数化、低秩近似 |
| 线性相关 | 存在非平凡线性组合为零 | 理解冗余、正则化 |
| 零空间 | $\mathbf{Ax} = \mathbf{0}$ 的解空间 | 理解约束、梯度 |

### 关键公式总结

| 公式 | 含义 |
|------|------|
| $\text{rank}(\mathbf{A}) + \dim(\text{Null}(\mathbf{A})) = n$ | 秩-零化度定理 |
| $\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})$ | 行列式的乘法性 |
| $\text{rank}(\mathbf{A}) = n \iff \det(\mathbf{A}) \neq 0$ | 满秩与行列式的关系 |
| $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$ | 唯一解（当 $\det(\mathbf{A}) \neq 0$） |
| $\mathbf{x}^* = \arg\min \|\mathbf{Ax} - \mathbf{b}\|_2^2$ | 最小二乘解 |

---

**上一节**：[第一章（a）：向量与矩阵基础](01a-向量与矩阵基础.md)

**下一节**：[第一章（c）：特征值与矩阵分解](01c-特征值与矩阵分解.md) - 学习特征值、特征向量和各种矩阵分解方法。

**返回**：[数学基础教程目录](../math-fundamentals.md)
