# 第二章（a）：导数与微分基础

微积分是理解**梯度下降**和**反向传播**的数学基础。导数描述了函数在某点的变化率，是优化算法的核心概念。本章将系统介绍导数的基本概念、计算方法和微分法则。

---

## 🎯 生活类比：导数就是"瞬时速度"

想象你在开车，**导数**就像**速度表**显示的数字。

```
你开了1小时，走了60公里。
问：你的速度是多少？
答：平均速度 = 60公里/小时

但是！你中间可能停下来买水、等红灯...
真正的"某一时刻的速度"（比如第30分钟那一瞬间）= 导数！

时间 → 位置函数 f(t)
导数 = f'(t) = 瞬时速度 = "此刻"的变化快慢
```

### 导数的三种理解方式

| 角度 | 解释 | 生活例子 |
|------|------|---------|
| **几何** | 切线的斜率 | 山坡在某处的陡峭程度 |
| **物理** | 瞬时变化率 | 速度表的读数 |
| **经济** | 边际效益 | 多卖一件商品增加的利润 |

### 📖 通俗翻译

| 数学语言 | 通俗翻译 |
|---------|---------|
| $\lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$ | 当变化量趋近于0时的"平均变化率" |
| $f'(x) > 0$ | 函数在增加（上坡） |
| $f'(x) < 0$ | 函数在减少（下坡） |
| $f'(x) = 0$ | 函数不增不减（山顶或山脚） |
| $f''(x) > 0$ | 开口向上（碗形） |
| $f''(x) < 0$ | 开口向下（倒碗形） |

---

## 目录

1. [函数与极限](#函数与极限)
2. [导数的定义](#导数的定义)
3. [导数的几何与物理意义](#导数的几何与物理意义)
4. [基本导数公式](#基本导数公式)
5. [微分法则](#微分法则)
6. [链式法则](#链式法则)
7. [导数与函数性质](#导数与函数性质)
8. [在深度学习中的应用](#在深度学习中的应用)
9. [小结](#小结)

---

## 函数与极限

### 函数的定义

**函数** $f: \mathbb{R} \to \mathbb{R}$ 是一种映射规则，将输入 $x$ 映射到唯一的输出 $y = f(x)$。

**多元函数**：$f: \mathbb{R}^n \to \mathbb{R}$，将 $n$ 维向量映射到标量。

$$
f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n)
$$

**向量值函数**：$f: \mathbb{R}^n \to \mathbb{R}^m$，将 $n$ 维向量映射到 $m$ 维向量。

$$
\mathbf{f}(\mathbf{x}) = \begin{bmatrix} f_1(\mathbf{x}) \\ f_2(\mathbf{x}) \\ \vdots \\ f_m(\mathbf{x}) \end{bmatrix}
$$

### 极限

**定义**：函数 $f(x)$ 当 $x$ 趋近于 $a$ 时的极限为 $L$，记作：

$$
\lim_{x \to a} f(x) = L
$$

**ε-δ 定义**：对于任意 $\epsilon > 0$，存在 $\delta > 0$，使得当 $0 < |x - a| < \delta$ 时，有 $|f(x) - L| < \epsilon$。

### 极限的性质

**四则运算**：设 $\lim_{x \to a} f(x) = A$，$\lim_{x \to a} g(x) = B$

$$
\lim_{x \to a} [f(x) \pm g(x)] = A \pm B
$$

$$
\lim_{x \to a} [f(x) \cdot g(x)] = A \cdot B
$$

$$
\lim_{x \to a} \frac{f(x)}{g(x)} = \frac{A}{B} \quad (B \neq 0)
$$

### 重要极限

$$
\lim_{x \to 0} \frac{\sin x}{x} = 1
$$

$$
\lim_{x \to 0} \frac{e^x - 1}{x} = 1
$$

$$
\lim_{x \to 0} \frac{\ln(1+x)}{x} = 1
$$

$$
\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e
$$

$$
\lim_{x \to 0} (1+x)^{\frac{1}{x}} = e
$$

### 连续性

函数 $f$ 在点 $a$ **连续**，当且仅当：

$$
\lim_{x \to a} f(x) = f(a)
$$

**连续的三个条件**：
1. $f(a)$ 存在
2. $\lim_{x \to a} f(x)$ 存在
3. $\lim_{x \to a} f(x) = f(a)$

**连续函数的性质**：
- 连续函数的和、差、积、商（分母不为零）仍连续
- 连续函数的复合函数仍连续
- 闭区间上的连续函数有最大值和最小值

```python
import numpy as np
import matplotlib.pyplot as plt

# 连续函数示例
x = np.linspace(-2, 2, 1000)
f_continuous = x**2
f_discontinuous = np.where(x < 0, -1, 1)  # 阶跃函数在 0 处不连续

print("x^2 在 x=0 处连续")
print("阶跃函数在 x=0 处不连续")

# 验证极限
def limit_example(x):
    """验证 lim_{x->0} sin(x)/x = 1"""
    if x == 0:
        return None
    return np.sin(x) / x

for h in [0.1, 0.01, 0.001, 0.0001]:
    print(f"sin({h})/{h} = {limit_example(h):.6f}")
# 输出趋近于 1
```

---

## 导数的定义

### 🎯 生活类比：汽车速度表

想象你在开车：
- **路程** = 你开了多远（函数值）
- **速度** = 你开得有多快（导数）

**速度表显示的就是"路程对时间的导数"**！

| 时刻 | 已开路程 | 速度（导数） |
|------|----------|-------------|
| 0分钟 | 0公里 | 0公里/小时（刚启动） |
| 10分钟 | 5公里 | 60公里/小时 |
| 30分钟 | 30公里 | 100公里/小时（高速公路） |

**导数的本质**：描述"这一刻"的变化快慢，而不是"平均"变化。

### 📖 几何意义：切线的斜率

在函数图像上，导数 = 过该点**切线**的斜率。

```
        /          切线：刚好碰到曲线的直线
       /|          斜率 = 导数
      / |
     /  |斜率=2
    /   |
   /____|________
  /  ●  |
 / (x,f(x))
```

**直观理解**：
- 导数 > 0：曲线在上升（上坡）
- 导数 < 0：曲线在下降（下坡）
- 导数 = 0：曲线是平的（山顶或谷底）

### 📝 手把手计算示例

求 $f(x) = x^2$ 在 $x = 3$ 处的导数。

**方法**：用导数定义

$$f'(3) = \lim_{h \to 0} \frac{f(3+h) - f(3)}{h}$$

**Step 1**：计算 $f(3) = 3^2 = 9$

**Step 2**：取很小的 $h$，比如 $h = 0.001$
- $f(3 + 0.001) = f(3.001) = 3.001^2 = 9.006001$

**Step 3**：计算差商
$$\frac{9.006001 - 9}{0.001} = \frac{0.006001}{0.001} = 6.001$$

**Step 4**：让 $h$ 更小（趋近于0），差商趋近于 **6**

**验证**：用公式 $f'(x) = 2x$，$f'(3) = 2 \times 3 = 6$ ✓

### 定义

函数 $f(x)$ 在点 $x$ 处的**导数**（derivative）定义为：

$$
f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

**通俗翻译**：导数 = 当变化量趋近于0时，"函数变化量 / 输入变化量"的极限。

**其他表示法**：
- 莱布尼茨记号：$\frac{df}{dx}$，$\frac{d}{dx}f(x)$
- 拉格朗日记号：$f'(x)$
- 牛顿记号：$\dot{f}$（常用于时间导数）

### 差商

导数是**差商的极限**：

$$
\text{差商} = \frac{f(x + h) - f(x)}{h}
$$

当 $h \to 0$ 时，差商趋近于导数。

### 左导数与右导数

**左导数**：

$$
f'_-(x) = \lim_{h \to 0^-} \frac{f(x+h) - f(x)}{h}
$$

**右导数**：

$$
f'_+(x) = \lim_{h \to 0^+} \frac{f(x+h) - f(x)}{h}
$$

### 可导性

函数 $f$ 在点 $x$ **可导**的条件：
1. $f$ 在 $x$ 处连续
2. 左导数等于右导数：$f'_-(x) = f'_+(x)$

**不可导的情况**：
- 尖点：$f(x) = |x|$ 在 $x = 0$
- 垂直切线：$f(x) = \sqrt[3]{x}$ 在 $x = 0$
- 不连续点
- ReLU 在 $x = 0$

```python
import numpy as np

def derivative_definition(f, x, h=1e-8):
    """使用定义计算导数"""
    return (f(x + h) - f(x)) / h

def symmetric_derivative(f, x, h=1e-8):
    """中心差分（更精确）"""
    return (f(x + h) - f(x - h)) / (2 * h)

# 示例：f(x) = x^2
def f(x):
    return x ** 2

# 解析导数：f'(x) = 2x
x = 3.0
print(f"f'({x}) 解析: {2*x}")
print(f"f'({x}) 定义: {derivative_definition(f, x):.6f}")
print(f"f'({x}) 中心差分: {symmetric_derivative(f, x):.6f}")

# ReLU 不可导性
def relu(x):
    return np.maximum(0, x)

x = 0.0
h = 1e-6
left_deriv = (relu(x) - relu(x - h)) / h
right_deriv = (relu(x + h) - relu(x)) / h
print(f"\nReLU 在 x=0:")
print(f"  左导数: {left_deriv}")
print(f"  右导数: {right_deriv}")
print(f"  可导? {np.isclose(left_deriv, right_deriv)}")
```

---

## 导数的几何与物理意义

### 几何意义

导数是函数图像在点 $(x, f(x))$ 处**切线的斜率**。

**切线方程**：

$$
y - f(a) = f'(a)(x - a)
$$

**法线方程**（垂直于切线）：

$$
y - f(a) = -\frac{1}{f'(a)}(x - a) \quad (f'(a) \neq 0)
$$

### 物理意义

| 如果 $f(t)$ 表示 | 则 $f'(t)$ 表示 |
|-----------------|-----------------|
| 位置 | 速度 |
| 速度 | 加速度 |
| 电荷量 | 电流 |
| 能量 | 功率 |

### 瞬时变化率

导数表示函数在某点的**瞬时变化率**：

$$
f'(x) = \text{在 } x \text{ 处的瞬时变化率}
$$

```python
import numpy as np

def tangent_line(f, f_prime, a, x):
    """计算切线在 x 处的值"""
    return f(a) + f_prime(a) * (x - a)

# 示例：f(x) = x^2
def f(x):
    return x ** 2

def f_prime(x):
    return 2 * x

a = 2.0  # 切点
x_vals = np.linspace(0, 4, 100)

# 原函数
y_func = f(x_vals)
# 切线
y_tangent = tangent_line(f, f_prime, a, x_vals)

print(f"在 x = {a} 处:")
print(f"  函数值: f({a}) = {f(a)}")
print(f"  导数值: f'({a}) = {f_prime(a)}")
print(f"  切线方程: y = {f_prime(a)}(x - {a}) + {f(a)} = {f_prime(a)}x - {f_prime(a)*a - f(a)}")
```

---

## 基本导数公式

### 幂函数与多项式

| 函数 $f(x)$ | 导数 $f'(x)$ |
|-------------|--------------|
| $c$（常数） | $0$ |
| $x$ | $1$ |
| $x^n$ | $nx^{n-1}$ |
| $\sqrt{x} = x^{1/2}$ | $\frac{1}{2\sqrt{x}}$ |
| $\frac{1}{x} = x^{-1}$ | $-\frac{1}{x^2}$ |

### 指数与对数

| 函数 $f(x)$ | 导数 $f'(x)$ |
|-------------|--------------|
| $e^x$ | $e^x$ |
| $a^x$ | $a^x \ln a$ |
| $\ln x$ | $\frac{1}{x}$ |
| $\log_a x$ | $\frac{1}{x \ln a}$ |

### 三角函数

| 函数 $f(x)$ | 导数 $f'(x)$ |
|-------------|--------------|
| $\sin x$ | $\cos x$ |
| $\cos x$ | $-\sin x$ |
| $\tan x$ | $\sec^2 x = \frac{1}{\cos^2 x}$ |
| $\cot x$ | $-\csc^2 x = -\frac{1}{\sin^2 x}$ |
| $\sec x$ | $\sec x \tan x$ |
| $\csc x$ | $-\csc x \cot x$ |

### 反三角函数

| 函数 $f(x)$ | 导数 $f'(x)$ |
|-------------|--------------|
| $\arcsin x$ | $\frac{1}{\sqrt{1-x^2}}$ |
| $\arccos x$ | $-\frac{1}{\sqrt{1-x^2}}$ |
| $\arctan x$ | $\frac{1}{1+x^2}$ |
| $\text{arccot } x$ | $-\frac{1}{1+x^2}$ |

### 双曲函数

| 函数 $f(x)$ | 导数 $f'(x)$ |
|-------------|--------------|
| $\sinh x = \frac{e^x - e^{-x}}{2}$ | $\cosh x$ |
| $\cosh x = \frac{e^x + e^{-x}}{2}$ | $\sinh x$ |
| $\tanh x = \frac{\sinh x}{\cosh x}$ | $\text{sech}^2 x = 1 - \tanh^2 x$ |

```python
import numpy as np

# 验证导数公式
def verify_derivative(f_name, f, f_prime, x):
    """验证导数公式"""
    numeric = (f(x + 1e-8) - f(x - 1e-8)) / (2e-8)
    analytic = f_prime(x)
    error = abs(numeric - analytic)
    print(f"{f_name}: 解析={analytic:.6f}, 数值={numeric:.6f}, 误差={error:.2e}")

# 测试点
x = 0.5

verify_derivative("sin(x)", np.sin, np.cos, x)
verify_derivative("exp(x)", np.exp, np.exp, x)
verify_derivative("ln(x)", np.log, lambda x: 1/x, x)
verify_derivative("x^3", lambda x: x**3, lambda x: 3*x**2, x)
verify_derivative("tanh(x)", np.tanh, lambda x: 1 - np.tanh(x)**2, x)
```

---

## 微分法则

### 线性法则

**常数倍法则**：

$$
(cf)' = cf'
$$

**和差法则**：

$$
(f \pm g)' = f' \pm g'
$$

### 乘积法则

$$
(fg)' = f'g + fg'
$$

**记忆口诀**："前导后不导 + 前不导后导"

**乘积法则的推导**：

**第一步**：使用导数的定义。

$$(fg)'(x) = \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x)g(x)}{h}$$

**第二步**：添加并减去 $f(x+h)g(x)$。

$$= \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x+h)g(x) + f(x+h)g(x) - f(x)g(x)}{h}$$

**第三步**：分组提取公因式。

$$= \lim_{h \to 0} \left[ f(x+h) \cdot \frac{g(x+h) - g(x)}{h} + \frac{f(x+h) - f(x)}{h} \cdot g(x) \right]$$

**第四步**：取极限，利用 $f$ 的连续性（$f(x+h) \to f(x)$）。

$$= f(x) \cdot g'(x) + f'(x) \cdot g(x)$$

$$\boxed{(fg)' = f'g + fg'}$$

**扩展**：三个函数的乘积

$$(fgh)' = f'gh + fg'h + fgh'$$

### 商法则

$$
\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}
$$

**记忆口诀**："上导下不导 - 上不导下导，除以下面的平方"

**商法则的推导**：

**第一步**：使用导数的定义。

$$\left(\frac{f}{g}\right)'(x) = \lim_{h \to 0} \frac{\frac{f(x+h)}{g(x+h)} - \frac{f(x)}{g(x)}}{h}$$

**第二步**：通分。

$$= \lim_{h \to 0} \frac{f(x+h)g(x) - f(x)g(x+h)}{h \cdot g(x+h)g(x)}$$

**第三步**：添加并减去 $f(x)g(x)$。

$$= \lim_{h \to 0} \frac{f(x+h)g(x) - f(x)g(x) + f(x)g(x) - f(x)g(x+h)}{h \cdot g(x+h)g(x)}$$

**第四步**：分组。

$$= \lim_{h \to 0} \frac{g(x) \cdot \frac{f(x+h) - f(x)}{h} - f(x) \cdot \frac{g(x+h) - g(x)}{h}}{g(x+h)g(x)}$$

**第五步**：取极限，利用 $g$ 的连续性（$g(x+h) \to g(x)$）。

$$= \frac{g(x)f'(x) - f(x)g'(x)}{g(x)^2}$$

$$\boxed{\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}}$$

```python
import numpy as np

# 验证乘积法则: (f*g)' = f'*g + f*g'
def product_rule_check(f, g, f_prime, g_prime, x):
    """验证乘积法则"""
    fg = lambda x: f(x) * g(x)
    
    # 左边：(f*g)' 的数值导数
    lhs_numeric = (fg(x + 1e-8) - fg(x - 1e-8)) / (2e-8)
    
    # 右边：f'*g + f*g'
    rhs = f_prime(x) * g(x) + f(x) * g_prime(x)
    
    return lhs_numeric, rhs

# 示例：f(x) = x^2, g(x) = sin(x)
f = lambda x: x**2
g = lambda x: np.sin(x)
f_prime = lambda x: 2*x
g_prime = lambda x: np.cos(x)

x = 1.0
lhs, rhs = product_rule_check(f, g, f_prime, g_prime, x)
print(f"乘积法则验证:")
print(f"  (f*g)' 数值: {lhs:.6f}")
print(f"  f'*g + f*g': {rhs:.6f}")
print(f"  匹配? {np.isclose(lhs, rhs)}")

# 验证商法则: (f/g)' = (f'g - fg') / g^2
def quotient_rule_check(f, g, f_prime, g_prime, x):
    """验证商法则"""
    fg = lambda x: f(x) / g(x)
    
    lhs_numeric = (fg(x + 1e-8) - fg(x - 1e-8)) / (2e-8)
    rhs = (f_prime(x) * g(x) - f(x) * g_prime(x)) / (g(x) ** 2)
    
    return lhs_numeric, rhs

lhs, rhs = quotient_rule_check(f, g, f_prime, g_prime, x)
print(f"\n商法则验证:")
print(f"  (f/g)' 数值: {lhs:.6f}")
print(f"  (f'g - fg')/g^2: {rhs:.6f}")
print(f"  匹配? {np.isclose(lhs, rhs)}")
```

---

## 链式法则

### 单变量链式法则

对于复合函数 $y = f(g(x))$：

$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)
$$

**记忆方式**："外导乘内导"

### 链式法则的直观理解

如果 $y$ 依赖 $u$，$u$ 依赖 $x$，则：

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

### 常用复合函数导数

| 函数 $f(x)$ | 导数 $f'(x)$ |
|-------------|--------------|
| $\ln(g(x))$ | $\frac{g'(x)}{g(x)}$ |
| $e^{g(x)}$ | $g'(x) e^{g(x)}$ |
| $(g(x))^n$ | $n(g(x))^{n-1} g'(x)$ |
| $\sin(g(x))$ | $g'(x) \cos(g(x))$ |
| $\cos(g(x))$ | $-g'(x) \sin(g(x))$ |
| $\sqrt{g(x)}$ | $\frac{g'(x)}{2\sqrt{g(x)}}$ |

### 链式法则示例

**例1**：求 $(e^{x^2})'$

设 $u = x^2$，则 $y = e^u$：

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = e^u \cdot 2x = 2xe^{x^2}
$$

**例2**：求 $(\sin(3x))'$

$$
\frac{d}{dx}\sin(3x) = \cos(3x) \cdot 3 = 3\cos(3x)
$$

**例3**：求 $(\ln(x^2 + 1))'$

$$
\frac{d}{dx}\ln(x^2 + 1) = \frac{1}{x^2 + 1} \cdot 2x = \frac{2x}{x^2 + 1}
$$

```python
import numpy as np

def chain_rule_example():
    """链式法则示例"""
    
    # 例1: y = e^{x^2}
    def y1(x):
        return np.exp(x**2)
    
    def dy1_dx(x):
        # dy/du * du/dx = e^{x^2} * 2x
        return np.exp(x**2) * 2 * x
    
    # 例2: y = sin(3x)
    def y2(x):
        return np.sin(3*x)
    
    def dy2_dx(x):
        return np.cos(3*x) * 3
    
    # 例3: y = ln(x^2 + 1)
    def y3(x):
        return np.log(x**2 + 1)
    
    def dy3_dx(x):
        return 2*x / (x**2 + 1)
    
    # 验证
    x = 1.5
    
    for name, y, dy_dx in [("e^{x^2}", y1, dy1_dx), 
                            ("sin(3x)", y2, dy2_dx),
                            ("ln(x^2+1)", y3, dy3_dx)]:
        numeric = (y(x + 1e-8) - y(x - 1e-8)) / (2e-8)
        analytic = dy_dx(x)
        print(f"{name}: 解析={analytic:.6f}, 数值={numeric:.6f}")

chain_rule_example()
```

---

## 导数与函数性质

### 单调性

| 条件 | 结论 |
|------|------|
| $f'(x) > 0$ | $f$ 单调递增 |
| $f'(x) < 0$ | $f$ 单调递减 |
| $f'(x) = 0$ | 可能是极值点 |

### 凹凸性

| 条件 | 结论 |
|------|------|
| $f''(x) > 0$ | $f$ 凹（convex，开口向上） |
| $f''(x) < 0$ | $f$ 凸（concave，开口向下） |
| $f''(x) = 0$ | 可能是拐点 |

### 极值判定

**一阶导数判别法**：
- 若 $f'$ 在 $x_0$ 左正右负，则 $x_0$ 是极大值点
- 若 $f'$ 在 $x_0$ 左负右正，则 $x_0$ 是极小值点

**二阶导数判别法**：
- 若 $f'(x_0) = 0$ 且 $f''(x_0) > 0$，则 $x_0$ 是极小值点
- 若 $f'(x_0) = 0$ 且 $f''(x_0) < 0$，则 $x_0$ 是极大值点
- 若 $f'(x_0) = 0$ 且 $f''(x_0) = 0$，需进一步判断

### 拐点

函数凹凸性改变的点。$x_0$ 是拐点的**必要条件**：

$$
f''(x_0) = 0 \text{ 或 } f''(x_0) \text{ 不存在}
$$

```python
import numpy as np

def analyze_function(f, f_prime, f_double_prime, x_range):
    """分析函数的单调性和凹凸性"""
    x = np.linspace(x_range[0], x_range[1], 1000)
    
    # 找极值点 (f' = 0)
    deriv = f_prime(x)
    sign_changes = np.where(np.diff(np.sign(deriv)))[0]
    
    print("极值点:")
    for idx in sign_changes:
        x_extreme = x[idx]
        if f_double_prime(x_extreme) > 0:
            print(f"  x = {x_extreme:.4f}: 极小值 = {f(x_extreme):.4f}")
        else:
            print(f"  x = {x_extreme:.4f}: 极大值 = {f(x_extreme):.4f}")
    
    # 找拐点 (f'' = 0)
    second_deriv = f_double_prime(x)
    inflection_indices = np.where(np.diff(np.sign(second_deriv)))[0]
    
    print("\n拐点:")
    for idx in inflection_indices:
        x_inf = x[idx]
        print(f"  x = {x_inf:.4f}: f(x) = {f(x_inf):.4f}")

# 示例：f(x) = x^3 - 3x
f = lambda x: x**3 - 3*x
f_prime = lambda x: 3*x**2 - 3
f_double_prime = lambda x: 6*x

analyze_function(f, f_prime, f_double_prime, [-3, 3])
```

---

## 在深度学习中的应用

### 激活函数的导数

**Sigmoid**：$\sigma(x) = \frac{1}{1+e^{-x}}$

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

**Sigmoid 导数的推导**：

**第一步**：设 $u = 1 + e^{-x}$，则 $\sigma(x) = \frac{1}{u} = u^{-1}$。

**第二步**：使用商法则（或链式法则）。

$$\sigma'(x) = \frac{d}{dx}\left(\frac{1}{1+e^{-x}}\right)$$

使用商法则 $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$，其中 $f = 1$，$g = 1+e^{-x}$：

$$\sigma'(x) = \frac{0 \cdot (1+e^{-x}) - 1 \cdot \frac{d}{dx}(1+e^{-x})}{(1+e^{-x})^2}$$

$$= \frac{-\frac{d}{dx}(e^{-x})}{(1+e^{-x})^2}$$

**第三步**：计算 $\frac{d}{dx}(e^{-x})$。

使用链式法则：$\frac{d}{dx}(e^{-x}) = e^{-x} \cdot \frac{d}{dx}(-x) = e^{-x} \cdot (-1) = -e^{-x}$

**第四步**：代入。

$$\sigma'(x) = \frac{-(-e^{-x})}{(1+e^{-x})^2} = \frac{e^{-x}}{(1+e^{-x})^2}$$

**第五步**：整理为 $\sigma(x)$ 的形式。

注意到 $\sigma(x) = \frac{1}{1+e^{-x}}$，所以 $1 - \sigma(x) = 1 - \frac{1}{1+e^{-x}} = \frac{e^{-x}}{1+e^{-x}}$。

因此：

$$\sigma'(x) = \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} = \sigma(x) \cdot (1 - \sigma(x))$$

$$\boxed{\sigma'(x) = \sigma(x)(1 - \sigma(x))}$$

**直观理解**：
- 当 $\sigma(x) \approx 0.5$ 时（$x \approx 0$），导数最大，约为 $0.5 \times 0.5 = 0.25$
- 当 $\sigma(x) \approx 0$ 或 $\sigma(x) \approx 1$ 时（$|x|$ 很大），导数接近 0

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# 验证
x = 1.0
numeric = (sigmoid(x + 1e-8) - sigmoid(x - 1e-8)) / (2e-8)
analytic = sigmoid_derivative(x)
print(f"Sigmoid'({x}): 解析={analytic:.6f}, 数值={numeric:.6f}")
```

**ReLU**：$\text{ReLU}(x) = \max(0, x)$

$$
\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \\ \text{未定义} & x = 0 \end{cases}
$$

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

# ReLU 在 0 处不可导，但实践中通常取 0 或 1
```

**Tanh**：$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

$$
\tanh'(x) = 1 - \tanh^2(x) = \text{sech}^2(x)
$$

**Tanh 导数的推导**：

**方法一：使用定义**

**第一步**：设 $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{u}{v}$，其中 $u = e^x - e^{-x}$，$v = e^x + e^{-x}$。

**第二步**：使用商法则 $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$。

计算 $u'$ 和 $v'$：
- $u' = e^x - (-e^{-x}) = e^x + e^{-x} = v$
- $v' = e^x - e^{-x} = u$

**第三步**：代入商法则。

$$\tanh'(x) = \frac{v \cdot v - u \cdot u}{v^2} = \frac{v^2 - u^2}{v^2} = 1 - \frac{u^2}{v^2} = 1 - \tanh^2(x)$$

$$\boxed{\tanh'(x) = 1 - \tanh^2(x)}$$

**方法二：利用 Sigmoid**

由于 $\tanh(x) = 2\sigma(2x) - 1$：

$$\tanh'(x) = 2 \cdot \sigma'(2x) \cdot 2 = 4\sigma(2x)(1-\sigma(2x))$$

设 $s = \sigma(2x)$，则 $\tanh(x) = 2s - 1$，$s = \frac{\tanh(x) + 1}{2}$。

$$\tanh'(x) = 4s(1-s) = 4 \cdot \frac{\tanh+1}{2} \cdot \frac{1-\tanh}{2} = (1+\tanh)(1-\tanh) = 1 - \tanh^2$$

**几何意义**：
- 当 $x = 0$ 时，$\tanh(0) = 0$，$\tanh'(0) = 1$（最大导数）
- 当 $|x| \to \infty$ 时，$|\tanh| \to 1$，$\tanh' \to 0$（饱和区）

### 数值导数计算

```python
def numerical_derivative(f, x, h=1e-5):
    """中心差分数值导数"""
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_gradient(f, x, h=1e-5):
    """计算向量 x 处的数值梯度"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# 示例
def f(x):
    return x[0]**2 + x[1]**2

x = np.array([1.0, 2.0])
grad = numerical_gradient(f, x)
print(f"∇f({x}) = {grad}")  # 应接近 [2, 4]
```

### 梯度检查

```python
def gradient_check(f, analytic_grad, x, h=1e-5, threshold=1e-7):
    """检查解析梯度是否正确"""
    numeric_grad = numerical_gradient(f, x, h)
    
    diff = np.abs(numeric_grad - analytic_grad(x))
    rel_error = diff / (np.abs(numeric_grad) + np.abs(analytic_grad(x)) + 1e-8)
    
    max_error = np.max(rel_error)
    passed = max_error < threshold
    
    return passed, max_error, numeric_grad

# 示例
def f(x):
    return np.sum(x ** 2)

def df(x):
    return 2 * x

x = np.random.randn(5)
passed, error, numeric = gradient_check(f, df, x)
print(f"梯度检查: {'通过' if passed else '失败'}, 最大相对误差: {error:.2e}")
```

---

## 小结

本章介绍了导数与微分的基础知识：

| 概念 | 定义/公式 | 应用 |
|------|----------|------|
| 导数 | $f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}$ | 计算变化率 |
| 几何意义 | 切线斜率 | 理解函数行为 |
| 乘积法则 | $(fg)' = f'g + fg'$ | 复合函数求导 |
| 商法则 | $(f/g)' = \frac{f'g - fg'}{g^2}$ | 分式函数求导 |
| 链式法则 | $\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$ | 反向传播基础 |

### 关键公式

**基本导数**：
- $(x^n)' = nx^{n-1}$
- $(e^x)' = e^x$
- $(\ln x)' = \frac{1}{x}$
- $(\sin x)' = \cos x$

**微分法则**：
- $(f \pm g)' = f' \pm g'$
- $(fg)' = f'g + fg'$
- $(f/g)' = \frac{f'g - fg'}{g^2}$
- $[f(g(x))]' = f'(g(x)) \cdot g'(x)$

---

**下一节**：[第二章（b）：偏导数、梯度与多元微分](02b-偏导数梯度与多元微分.md) - 学习多元函数的微分、梯度下降和方向导数。

**返回**：[数学基础教程目录](../math-fundamentals.md)
