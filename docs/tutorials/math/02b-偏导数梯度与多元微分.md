# 第二章（b）：偏导数、梯度与多元微分

在深度学习中，我们经常处理多元函数——损失函数依赖于成千上万个参数。偏导数和梯度是理解和优化这些函数的关键工具。本章将系统介绍多元微分及其在梯度下降中的应用。

---

## 目录

1. [偏导数](#偏导数)
2. [方向导数](#方向导数)
3. [梯度](#梯度)
4. [梯度下降算法](#梯度下降算法)
5. [多变量链式法则](#多变量链式法则)
6. [全微分](#全微分)
7. [在深度学习中的应用](#在深度学习中的应用)
8. [小结](#小结)

---

## 偏导数

### 🎯 生活类比：调节音响

想象你在调节音响，有低音(Bass)和高音(Treble)两个旋钮：

- **偏导数（对低音）**：固定高音不动，只调低音，声音变化多少？
- **偏导数（对高音）**：固定低音不动，只调高音，声音变化多少？

**偏导数 = 只改变一个因素，看结果怎么变**

| 场景 | 偏导数含义 |
|------|-----------|
| 房价（面积、位置、楼层） | 面积每增加1平米，价格涨多少？ |
| 成绩（学习时间、睡眠、饮食） | 多学1小时，成绩提高多少？ |
| 温度（加热功率、风扇转速） | 功率每增加1W，温度升多少？ |

### 📝 手把手计算

设 $f(x, y) = x^2 + 2xy + y^2$，在点 $(1, 2)$ 处求偏导数。

**Step 1：求对 $x$ 的偏导**（把 $y$ 当常数）

$$\frac{\partial f}{\partial x} = 2x + 2y + 0 = 2x + 2y$$

**Step 2：求对 $y$ 的偏导**（把 $x$ 当常数）

$$\frac{\partial f}{\partial y} = 0 + 2x + 2y = 2x + 2y$$

**Step 3：在点 $(1, 2)$ 处求值**

$$\frac{\partial f}{\partial x}\bigg|_{(1,2)} = 2(1) + 2(2) = 6$$
$$\frac{\partial f}{\partial y}\bigg|_{(1,2)} = 2(1) + 2(2) = 6$$

**解释**：
- 在 $(1,2)$ 处，$x$ 增加 1，函数值增加约 6
- 在 $(1,2)$ 处，$y$ 增加 1，函数值增加约 6

### 定义

多元函数 $f(x_1, x_2, \ldots, x_n)$ 对 $x_i$ 的**偏导数**是固定其他变量，只对 $x_i$ 求导：

$$
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}
$$

**通俗翻译**：偏导数 = 只看一个变量的变化，假装其他变量都不变。

### 表示法

| 表示法 | 含义 |
|--------|------|
| $\frac{\partial f}{\partial x_i}$ | 莱布尼茨记号 |
| $f_{x_i}$ | 下标记号 |
| $\partial_{x_i} f$ | 算子记号 |
| $D_i f$ | 另一种记号 |

在点 $\mathbf{a}$ 处的偏导：$\frac{\partial f}{\partial x_i}\bigg|_{\mathbf{x}=\mathbf{a}}$

### 计算方法

计算 $\frac{\partial f}{\partial x_i}$ 时，将其他变量 $x_j$ ($j \neq i$) 视为常数。

### 示例

**例1**：设 $f(x, y) = x^2 y + xy^3$

求 $\frac{\partial f}{\partial x}$（将 $y$ 视为常数）：

$$
\frac{\partial f}{\partial x} = 2xy + y^3
$$

求 $\frac{\partial f}{\partial y}$（将 $x$ 视为常数）：

$$
\frac{\partial f}{\partial y} = x^2 + 3xy^2
$$

**例2**：设 $f(x, y, z) = x^2 + 2y^2 + 3z^2 + xy - yz$

$$
\frac{\partial f}{\partial x} = 2x + y
$$

$$
\frac{\partial f}{\partial y} = 4y + x - z
$$

$$
\frac{\partial f}{\partial z} = 6z - y
$$

### 高阶偏导数

二阶偏导数：

$$
\frac{\partial^2 f}{\partial x_i^2} = \frac{\partial}{\partial x_i}\left(\frac{\partial f}{\partial x_i}\right)
$$

混合偏导数：

$$
\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial}{\partial x_i}\left(\frac{\partial f}{\partial x_j}\right)
$$

**Schwarz 定理**（混合偏导数交换律）：

如果 $f$ 的二阶偏导数在区域内连续，则：

$$
\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}
$$

```python
import numpy as np

def partial_derivative(f, var_idx, x, h=1e-5):
    """计算 f 对第 var_idx 个变量的偏导数"""
    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[var_idx] += h
    x_minus[var_idx] -= h
    return (f(x_plus) - f(x_minus)) / (2 * h)

# 示例1: f(x, y) = x^2*y + x*y^3
def f1(x):
    return x[0]**2 * x[1] + x[0] * x[1]**3

# 解析偏导
def df1_dx(x):
    return 2*x[0]*x[1] + x[1]**3

def df1_dy(x):
    return x[0]**2 + 3*x[0]*x[1]**2

x = np.array([2.0, 3.0])

# 数值计算
partial_x = partial_derivative(f1, 0, x)
partial_y = partial_derivative(f1, 1, x)

print(f"f(x,y) = x²y + xy³ 在 ({x[0]}, {x[1]}) 处:")
print(f"  ∂f/∂x: 数值={partial_x:.4f}, 解析={df1_dx(x):.4f}")
print(f"  ∂f/∂y: 数值={partial_y:.4f}, 解析={df1_dy(x):.4f}")

# 示例2: 二阶偏导验证
def f2(x):
    return x[0]**2 * x[1]**2

# ∂²f/∂x∂y = ∂²f/∂y∂x = 4xy
def d2f_dxdy(x):
    return 4 * x[0] * x[1]

# 数值计算 ∂²f/∂x∂y
def mixed_second_partial(f, x, h=1e-5):
    """计算混合二阶偏导"""
    def first_partial_x(x):
        return partial_derivative(f, 0, x, h)
    return partial_derivative(first_partial_x, 1, x, h)

x = np.array([2.0, 3.0])
mixed = mixed_second_partial(f2, x)
print(f"\n∂²f/∂x∂y = ∂²f/∂y∂x 验证:")
print(f"  数值: {mixed:.4f}")
print(f"  解析: {d2f_dxdy(x):.4f}")
```

---

## 方向导数

### 定义

函数 $f(\mathbf{x})$ 沿单位向量 $\mathbf{u}$ 方向的**方向导数**为：

$$
D_{\mathbf{u}} f(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{u}) - f(\mathbf{x})}{h}
$$

### 方向导数与梯度的关系

$$
D_{\mathbf{u}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u}
$$

即方向导数是梯度在方向 $\mathbf{u}$ 上的投影。

**方向导数与梯度关系的推导**：

**第一步**：定义方向导数。

$$D_{\mathbf{u}} f(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{u}) - f(\mathbf{x})}{h}$$

**第二步**：使用多元泰勒展开（一阶）。

$$f(\mathbf{x} + h\mathbf{u}) = f(\mathbf{x}) + \nabla f(\mathbf{x}) \cdot (h\mathbf{u}) + o(h)$$

$$= f(\mathbf{x}) + h \nabla f(\mathbf{x}) \cdot \mathbf{u} + o(h)$$

**第三步**：代入方向导数定义。

$$D_{\mathbf{u}} f(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x}) + h \nabla f(\mathbf{x}) \cdot \mathbf{u} + o(h) - f(\mathbf{x})}{h}$$

$$= \lim_{h \to 0} \left( \nabla f(\mathbf{x}) \cdot \mathbf{u} + \frac{o(h)}{h} \right)$$

**第四步**：取极限。

$$\boxed{D_{\mathbf{u}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u}}$$

**推论**：最大方向导数。

由于 $\mathbf{u}$ 是单位向量，由 Cauchy-Schwarz 不等式：

$$|D_{\mathbf{u}} f| = |\nabla f \cdot \mathbf{u}| \leq \|\nabla f\| \|\mathbf{u}\| = \|\nabla f\|$$

等号当且仅当 $\mathbf{u}$ 与 $\nabla f$ 同向时取得。因此：

$$\max_{\|\mathbf{u}\|=1} D_{\mathbf{u}} f = \|\nabla f\|$$

### 性质

| 方向 | 方向导数值 | 含义 |
|------|-----------|------|
| 梯度方向 $\mathbf{u} = \frac{\nabla f}{\|\nabla f\|}$ | $\|\nabla f\|$ | 最大变化率 |
| 负梯度方向 $\mathbf{u} = -\frac{\nabla f}{\|\nabla f\|}$ | $-\|\nabla f\|$ | 最小变化率（下降最快） |
| 等高线方向 $\mathbf{u} \perp \nabla f$ | $0$ | 不变化 |

### 最大变化率方向

方向导数取最大值当且仅当 $\mathbf{u}$ 与 $\nabla f$ 同向：

$$
\max_{\|\mathbf{u}\|=1} D_{\mathbf{u}} f = \|\nabla f\|
$$

```python
import numpy as np

def directional_derivative(f, x, u, h=1e-5):
    """计算方向导数"""
    u = u / np.linalg.norm(u)  # 归一化
    return (f(x + h*u) - f(x - h*u)) / (2*h)

def gradient(f, x, h=1e-5):
    """计算梯度"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2*h)
    return grad

# 示例：f(x,y) = x^2 + y^2
def f(x):
    return x[0]**2 + x[1]**2

x = np.array([3.0, 4.0])
grad = gradient(f, x)

# 不同方向的方向导数
u1 = np.array([1.0, 0.0])  # x 方向
u2 = np.array([0.0, 1.0])  # y 方向
u3 = grad / np.linalg.norm(grad)  # 梯度方向（归一化）
u4 = -grad / np.linalg.norm(grad)  # 负梯度方向

print(f"f(x,y) = x² + y² 在 ({x[0]}, {x[1]}) 处:")
print(f"  梯度: {grad}")
print(f"  梯度方向: {u3}")
print(f"\n方向导数:")
print(f"  x方向: {directional_derivative(f, x, u1):.4f}")
print(f"  y方向: {directional_derivative(f, x, u2):.4f}")
print(f"  梯度方向: {directional_derivative(f, x, u3):.4f} (最大)")
print(f"  负梯度方向: {directional_derivative(f, x, u4):.4f} (最小)")

# 验证：方向导数 = 梯度 · 方向
print(f"\n验证 D_u f = ∇f · u:")
print(f"  x方向: {directional_derivative(f, x, u1):.4f} ≈ {np.dot(grad, u1/np.linalg.norm(u1)):.4f}")
print(f"  梯度方向: {directional_derivative(f, x, u3):.4f} = ||∇f|| = {np.linalg.norm(grad):.4f}")
```

---

## 梯度

### 🎯 生活类比：山坡上的指南针

想象你站在山坡上，想知道哪个方向上坡最陡：

- **梯度** = 指向最陡上坡方向的箭头
- **负梯度** = 指向最陡下坡方向的箭头（这就是梯度下降！）

```
           山顶（高）
             /|\
            / | \  梯度指向上
           /  |  \
          /   |   \
         /    |    \
        /     |     \
       /      |      \
      /       |       \
     /        |        \
    /    等高线（平地）  \
   /_____________________\
        你在这里 → ●
```

**梯度 = 把所有偏导数"打包"成一个向量，告诉你"往哪走上升最快"**

### 📖 梯度 vs 偏导数

| 概念 | 含义 | 类比 |
|------|------|------|
| 偏导数 $\frac{\partial f}{\partial x}$ | 只看x方向的变化率 | 只关心东西方向 |
| 偏导数 $\frac{\partial f}{\partial y}$ | 只看y方向的变化率 | 只关心南北方向 |
| **梯度** $\nabla f$ | 所有偏导数的组合 | 知道所有方向的情况 |

### 📝 手把手计算

设 $f(x, y) = x^2 + y^2$，在点 $(3, 4)$ 处求梯度。

**Step 1**：求偏导数
- $\frac{\partial f}{\partial x} = 2x$
- $\frac{\partial f}{\partial y} = 2y$

**Step 2**：组成梯度向量
$$\nabla f = \begin{bmatrix} 2x \\ 2y \end{bmatrix}$$

**Step 3**：在点 $(3, 4)$ 处
$$\nabla f(3, 4) = \begin{bmatrix} 2 \times 3 \\ 2 \times 4 \end{bmatrix} = \begin{bmatrix} 6 \\ 8 \end{bmatrix}$$

**解释**：从 $(3,4)$ 出发，沿着 $(6,8)$ 方向走，函数值增长最快。

### 定义

多元函数 $f: \mathbb{R}^n \to \mathbb{R}$ 的**梯度**（gradient）是所有偏导数组成的向量：

$$
\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}
$$

**通俗翻译**：梯度 = 把每个变量的偏导数收集起来，形成一个"方向指南针"。

或写成：$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)^\top$

### 梯度算子

$\nabla$（nabla）是一个向量微分算子：

$$
\nabla = \begin{bmatrix} \frac{\partial}{\partial x_1} \\ \frac{\partial}{\partial x_2} \\ \vdots \\ \frac{\partial}{\partial x_n} \end{bmatrix}
$$

### 梯度的几何意义

1. **方向**：梯度指向函数值**增长最快**的方向
2. **大小**：$\|\nabla f\|$ 是该方向的方向导数值
3. **负梯度**：$-\nabla f$ 指向函数值**下降最快**的方向
4. **等高线**：梯度与等高线（等值面）垂直

### 梯度运算法则

设 $f, g$ 是可微函数，$c$ 是常数：

$$
\nabla(cf) = c\nabla f
$$

$$
\nabla(f + g) = \nabla f + \nabla g
$$

$$
\nabla(fg) = g\nabla f + f\nabla g
$$

$$
\nabla(f/g) = \frac{g\nabla f - f\nabla g}{g^2}
$$

### 常见函数的梯度

**线性函数**：$f(\mathbf{x}) = \mathbf{a}^\top \mathbf{x}$

$$
\nabla f = \mathbf{a}
$$

**二次型**：$f(\mathbf{x}) = \mathbf{x}^\top \mathbf{A} \mathbf{x}$（$\mathbf{A}$ 对称）

$$
\nabla f = 2\mathbf{A}\mathbf{x}
$$

**二次型梯度的推导**：

**第一步**：展开二次型。

设 $\mathbf{x} = (x_1, \ldots, x_n)^\top$，$\mathbf{A} = (a_{ij})$，则：

$$f(\mathbf{x}) = \mathbf{x}^\top \mathbf{A} \mathbf{x} = \sum_{i=1}^n \sum_{j=1}^n a_{ij} x_i x_j$$

**第二步**：计算 $\frac{\partial f}{\partial x_k}$。

$$\frac{\partial f}{\partial x_k} = \frac{\partial}{\partial x_k} \sum_{i,j} a_{ij} x_i x_j$$

对于包含 $x_k$ 的项：
- 当 $i = k, j \neq k$：$a_{kj} x_k x_j$，偏导为 $a_{kj} x_j$
- 当 $j = k, i \neq k$：$a_{ik} x_i x_k$，偏导为 $a_{ik} x_i$
- 当 $i = j = k$：$a_{kk} x_k^2$，偏导为 $2a_{kk} x_k$

因此：

$$\frac{\partial f}{\partial x_k} = \sum_{j \neq k} a_{kj} x_j + \sum_{i \neq k} a_{ik} x_i + 2a_{kk} x_k$$

**第三步**：利用 $\mathbf{A}$ 的对称性（$a_{ik} = a_{ki}$）。

$$\frac{\partial f}{\partial x_k} = 2 \sum_{j=1}^n a_{kj} x_j$$

**第四步**：写成向量形式。

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix} = \begin{bmatrix} 2\sum_j a_{1j} x_j \\ \vdots \\ 2\sum_j a_{nj} x_j \end{bmatrix} = 2\mathbf{A}\mathbf{x}$$

$$\boxed{\nabla (\mathbf{x}^\top \mathbf{A} \mathbf{x}) = 2\mathbf{A}\mathbf{x} \quad (\mathbf{A} \text{ 对称})}$$

**一般情况**（$\mathbf{A}$ 不对称）：

$$\nabla (\mathbf{x}^\top \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$$

**二次函数**：$f(\mathbf{x}) = \frac{1}{2}\|\mathbf{x}\|^2$

$$
\nabla f = \mathbf{x}
$$

**范数平方**：$f(\mathbf{x}) = \|\mathbf{x} - \mathbf{a}\|^2$

$$
\nabla f = 2(\mathbf{x} - \mathbf{a})
$$

```python
import numpy as np

def compute_gradient(f, x, h=1e-5):
    """数值计算梯度"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2*h)
    return grad

# 示例1: f(x) = ||x||^2 / 2
def f1(x):
    return np.sum(x**2) / 2

# 解析梯度: ∇f = x
x = np.array([1.0, 2.0, 3.0])
grad_numeric = compute_gradient(f1, x)
grad_analytic = x
print(f"f(x) = ||x||²/2 的梯度:")
print(f"  数值: {grad_numeric}")
print(f"  解析: {grad_analytic}")

# 示例2: f(x) = x^T A x (A对称)
A = np.array([[2, 1], [1, 3]], dtype=float)
def f2(x):
    return x @ A @ x

x = np.array([1.0, 2.0])
grad_numeric = compute_gradient(f2, x)
grad_analytic = 2 * A @ x
print(f"\nf(x) = x^T A x 的梯度:")
print(f"  数值: {grad_numeric}")
print(f"  解析: {grad_analytic}")
```

---

## 梯度下降算法

### 基本原理

利用负梯度方向进行优化，因为负梯度方向是函数值**下降最快**的方向。

**更新规则**：

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)
$$

其中：
- $\mathbf{x}_t$：当前参数
- $\eta > 0$：学习率（步长）
- $\nabla f(\mathbf{x}_t)$：当前梯度

### 收敛条件

梯度下降保证收敛的条件：
1. $f$ 是凸函数
2. 梯度 Lipschitz 连续
3. 学习率足够小：$\eta < \frac{1}{L}$（$L$ 是 Lipschitz 常数）

### 学习率的影响

| 学习率 | 行为 |
|--------|------|
| 过小 | 收敛太慢 |
| 合适 | 稳定收敛 |
| 过大 | 震荡或发散 |

### Python 实现

```python
import numpy as np

def gradient_descent(grad_fn, x_init, lr=0.01, max_iter=1000, tol=1e-6, verbose=False):
    """
    梯度下降优化
    
    参数:
        grad_fn: 梯度函数
        x_init: 初始点
        lr: 学习率
        max_iter: 最大迭代次数
        tol: 收敛阈值
        verbose: 是否打印进度
    
    返回:
        x: 最优解
        history: 迭代历史
    """
    x = x_init.copy()
    history = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_fn(x)
        x_new = x - lr * grad
        
        # 检查收敛
        if np.linalg.norm(x_new - x) < tol:
            if verbose:
                print(f"在第 {i+1} 次迭代收敛")
            break
        
        x = x_new
        history.append(x.copy())
        
        if verbose and (i+1) % 100 == 0:
            print(f"迭代 {i+1}: x = {x}, ||∇f|| = {np.linalg.norm(grad):.6f}")
    
    return x, history

# 示例1: 最小化 f(x,y) = x^2 + y^2
def grad_f1(x):
    return np.array([2*x[0], 2*x[1]])

x_init = np.array([5.0, 3.0])
x_min, history = gradient_descent(grad_f1, x_init, lr=0.1, verbose=True)
print(f"\n最小值点: {x_min}")
print(f"最小值: {x_min[0]**2 + x_min[1]**2:.6f}")

# 示例2: 最小化 Rosenbrock 函数
def rosenbrock(x):
    """f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    df_dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    df_dy = 200*(x[1] - x[0]**2)
    return np.array([df_dx, df_dy])

x_init = np.array([-1.0, 1.0])
x_min, history = gradient_descent(grad_rosenbrock, x_init, lr=0.001, max_iter=10000)
print(f"\nRosenbrock 最小值点: {x_min}")
print(f"最小值: {rosenbrock(x_min):.6f}")
```

### 动量法

加速收敛并减少震荡：

$$
\mathbf{v}_{t+1} = \gamma \mathbf{v}_t + \eta \nabla f(\mathbf{x}_t)
$$

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \mathbf{v}_{t+1}
$$

```python
def gradient_descent_with_momentum(grad_fn, x_init, lr=0.01, momentum=0.9, 
                                    max_iter=1000, tol=1e-6):
    """带动量的梯度下降"""
    x = x_init.copy()
    v = np.zeros_like(x)
    history = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_fn(x)
        v = momentum * v + lr * grad
        x_new = x - v
        
        if np.linalg.norm(x_new - x) < tol:
            break
        
        x = x_new
        history.append(x.copy())
    
    return x, history
```

---

## 多变量链式法则

### 标量情况

对于复合函数 $z = f(x, y)$，其中 $x = g(t)$，$y = h(t)$：

$$
\frac{dz}{dt} = \frac{\partial z}{\partial x}\frac{dx}{dt} + \frac{\partial z}{\partial y}\frac{dy}{dt}
$$

### 向量情况

对于 $z = f(\mathbf{x})$，其中 $\mathbf{x} = \mathbf{g}(t)$：

$$
\frac{dz}{dt} = \nabla f(\mathbf{x}) \cdot \frac{d\mathbf{x}}{dt} = \sum_i \frac{\partial f}{\partial x_i} \frac{dx_i}{dt}
$$

### 一般形式

对于 $\mathbf{y} = \mathbf{f}(\mathbf{x})$，$\mathbf{x} = \mathbf{g}(\mathbf{t})$：

$$
\frac{\partial \mathbf{y}}{\partial \mathbf{t}} = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \cdot \frac{\partial \mathbf{x}}{\partial \mathbf{t}} = \mathbf{J}_f \cdot \mathbf{J}_g
$$

其中 $\mathbf{J}$ 是雅可比矩阵。

### 示例

设 $z = x^2 y$，$x = \sin t$，$y = e^t$，求 $\frac{dz}{dt}$。

$$
\frac{\partial z}{\partial x} = 2xy, \quad \frac{\partial z}{\partial y} = x^2
$$

$$
\frac{dx}{dt} = \cos t, \quad \frac{dy}{dt} = e^t
$$

$$
\frac{dz}{dt} = 2xy \cdot \cos t + x^2 \cdot e^t = 2\sin t \cdot e^t \cdot \cos t + \sin^2 t \cdot e^t
$$

```python
import numpy as np

# 多变量链式法则示例
def chain_rule_example(t):
    """z = x^2 * y, x = sin(t), y = exp(t)"""
    x = np.sin(t)
    y = np.exp(t)
    z = x**2 * y
    
    # 直接计算 dz/dt
    # z = sin^2(t) * exp(t)
    # dz/dt = 2*sin(t)*cos(t)*exp(t) + sin^2(t)*exp(t)
    dz_dt_direct = 2*np.sin(t)*np.cos(t)*np.exp(t) + np.sin(t)**2*np.exp(t)
    
    # 使用链式法则
    dz_dx = 2*x*y
    dz_dy = x**2
    dx_dt = np.cos(t)
    dy_dt = np.exp(t)
    dz_dt_chain = dz_dx * dx_dt + dz_dy * dy_dt
    
    return z, dz_dt_direct, dz_dt_chain

t = 1.0
z, direct, chain = chain_rule_example(t)
print(f"t = {t}")
print(f"z = {z:.6f}")
print(f"dz/dt 直接: {direct:.6f}")
print(f"dz/dt 链式法则: {chain:.6f}")
```

---

## 全微分

### 定义

多元函数 $f(x_1, \ldots, x_n)$ 的**全微分**：

$$
df = \frac{\partial f}{\partial x_1}dx_1 + \frac{\partial f}{\partial x_2}dx_2 + \cdots + \frac{\partial f}{\partial x_n}dx_n
$$

或写成向量形式：

$$
df = \nabla f \cdot d\mathbf{x}
$$

### 几何意义

全微分描述了函数在各个方向上的**线性近似变化**。

### 近似计算

$$
f(\mathbf{x} + \Delta\mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x}) \cdot \Delta\mathbf{x}
$$

### 示例

设 $f(x, y) = x^2 y$，在点 $(2, 3)$ 处的全微分：

$$
df = 2xy \cdot dx + x^2 \cdot dy = 12 \cdot dx + 4 \cdot dy
$$

```python
import numpy as np

def total_differential_example():
    """全微分示例: f(x,y) = x^2 * y"""
    
    def f(x, y):
        return x**2 * y
    
    # 在 (2, 3) 处
    x0, y0 = 2.0, 3.0
    
    # 偏导数
    df_dx = 2 * x0 * y0  # 12
    df_dy = x0**2        # 4
    
    # 使用全微分近似
    dx, dy = 0.1, -0.05
    df_approx = df_dx * dx + df_dy * dy
    
    # 实际变化
    f_old = f(x0, y0)
    f_new = f(x0 + dx, y0 + dy)
    df_actual = f_new - f_old
    
    print(f"f({x0}, {y0}) = {f_old}")
    print(f"f({x0+dx}, {y0+dy}) = {f_new}")
    print(f"实际变化: {df_actual:.6f}")
    print(f"全微分近似: {df_approx:.6f}")
    print(f"相对误差: {abs(df_actual - df_approx) / abs(df_actual):.4%}")

total_differential_example()
```

---

## 在深度学习中的应用

### 多层网络的梯度传播

设网络结构为：$\mathbf{x} \to \mathbf{h}_1 \to \mathbf{h}_2 \to \mathbf{y} \to L$

其中：
- $\mathbf{h}_1 = f_1(\mathbf{x}; \mathbf{W}_1)$
- $\mathbf{h}_2 = f_2(\mathbf{h}_1; \mathbf{W}_2)$
- $\mathbf{y} = f_3(\mathbf{h}_2; \mathbf{W}_3)$
- $L = \ell(\mathbf{y}$, $\mathbf{y}_{true})$

**反向传播**（链式法则）：

$$
\frac{\partial L}{\partial \mathbf{y}} = \frac{\partial \ell}{\partial \mathbf{y}}
$$

$$
\frac{\partial L}{\partial \mathbf{h}_2} = \frac{\partial L}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{h}_2}
$$

$$
\frac{\partial L}{\partial \mathbf{h}_1} = \frac{\partial L}{\partial \mathbf{h}_2} \cdot \frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1}
$$

$$
\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{h}_1} \cdot \frac{\partial \mathbf{h}_1}{\partial \mathbf{x}}
$$

### 神经网络的梯度计算示例

```python
import numpy as np

class SimpleNN:
    """简单两层神经网络"""
    
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier 初始化
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_grad(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x):
        """前向传播"""
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.h1 = self.relu(self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        return self.z2
    
    def backward(self, grad_output):
        """反向传播"""
        # 第二层
        self.grad_W2 = self.h1.T @ grad_output
        self.grad_b2 = np.sum(grad_output, axis=0)
        grad_h1 = grad_output @ self.W2.T
        
        # ReLU 反向
        grad_z1 = grad_h1 * self.relu_grad(self.z1)
        
        # 第一层
        self.grad_W1 = self.x.T @ grad_z1
        self.grad_b1 = np.sum(grad_z1, axis=0)
        
        return grad_z1 @ self.W1.T
    
    def get_gradients(self):
        """返回所有梯度"""
        return {
            'W1': self.grad_W1, 'b1': self.grad_b1,
            'W2': self.grad_W2, 'b2': self.grad_b2
        }

# 示例
np.random.seed(42)
nn = SimpleNN(10, 20, 5)
x = np.random.randn(32, 10)  # batch of 32

# 前向传播
output = nn.forward(x)

# 假设损失函数的梯度
grad_output = np.random.randn(32, 5)

# 反向传播
grad_input = nn.backward(grad_output)

gradients = nn.get_gradients()
print("梯度形状:")
for name, grad in gradients.items():
    print(f"  {name}: {grad.shape}")
```

### 梯度检查

```python
def gradient_check_neural_network(nn, x, epsilon=1e-5):
    """神经网络梯度检查"""
    # 计算解析梯度
    output = nn.forward(x)
    grad_output = np.ones_like(output)
    nn.backward(grad_output)
    analytic_grads = nn.get_gradients()
    
    print("梯度检查结果:")
    for param_name in ['W1', 'b1', 'W2', 'b2']:
        param = getattr(nn, param_name)
        analytic = analytic_grads[param_name]
        numeric = np.zeros_like(param)
        
        # 数值梯度
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_value = param[idx]
            
            param[idx] = old_value + epsilon
            output_plus = nn.forward(x)
            loss_plus = np.sum(output_plus)
            
            param[idx] = old_value - epsilon
            output_minus = nn.forward(x)
            loss_minus = np.sum(output_minus)
            
            numeric[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            param[idx] = old_value
            it.iternext()
        
        # 计算相对误差
        diff = np.abs(numeric - analytic)
        rel_error = diff / (np.abs(numeric) + np.abs(analytic) + 1e-8)
        max_error = np.max(rel_error)
        
        status = "✓ 通过" if max_error < 1e-5 else "✗ 失败"
        print(f"  {param_name}: 最大相对误差 = {max_error:.2e} {status}")

# 运行梯度检查
np.random.seed(42)
nn = SimpleNN(5, 10, 3)
x = np.random.randn(4, 5)
gradient_check_neural_network(nn, x)
```

---

## 小结

本章介绍了偏导数、梯度和多元微分：

| 概念 | 定义/公式 | 应用 |
|------|----------|------|
| 偏导数 | $\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(\ldots, x_i+h, \ldots) - f(\ldots)}{h}$ | 多元函数对单变量求导 |
| 梯度 | $\nabla f = [\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}]^\top$ | 优化方向 |
| 方向导数 | $D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u}$ | 沿任意方向的变化率 |
| 梯度下降 | $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f$ | 参数优化 |
| 链式法则 | $\frac{dz}{dt} = \sum_i \frac{\partial z}{\partial x_i} \frac{dx_i}{dt}$ | 反向传播 |

### 关键概念

1. **梯度方向**：函数值增长最快的方向
2. **负梯度方向**：函数值下降最快的方向，梯度下降的核心
3. **方向导数**：沿任意方向的变化率，是梯度的投影
4. **链式法则**：连接各层梯度传播的基础

---

**上一节**：[第二章（a）：导数与微分基础](02a-导数与微分基础.md)

**下一节**：[第二章（c）：高阶导数与泰勒展开](02c-高阶导数与泰勒展开.md) - 学习二阶导数、海森矩阵和泰勒展开。

**返回**：[数学基础教程目录](../math-fundamentals.md)
