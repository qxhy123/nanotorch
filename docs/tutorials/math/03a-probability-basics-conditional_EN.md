# Chapter 3 (a): Probability Basics and Conditional Probability

Probability theory is the foundation for understanding **uncertainty in machine learning** and **statistical learning theory**. From Softmax outputs in classification problems to generative models, probability theory is everywhere. This chapter will systematically introduce the basic concepts of probability theory, conditional probability, and Bayes' theorem.

---

## 🎯 Life Analogy: Probability is "How Likely Something Happens"

### Probability = Betting Odds

Imagine you're betting with a friend about whether it will rain tomorrow:
- Weather forecast says "70% chance of precipitation"
- Meaning: If the same weather conditions repeated 100 days, about 70 days would have rain

```
Probability 0    ────────────────────────→    Probability 1
   │                                              │
Never happens                               Always happens

0.1 = 10%: Very rare (winning lottery)
0.5 = 50%: Coin flip (equally likely)
0.9 = 90%: Very likely (sun rises tomorrow)
```

### Conditional Probability = "Probability After Getting Information"

**Example**: You see someone wearing glasses, are they more likely to be a programmer?

$$P(\text{Programmer} | \text{Wears Glasses})$$

Read as: "The probability of being a programmer, given they wear glasses"

### Bayes' Theorem = "Updating Beliefs with New Evidence"

```
┌─────────────────────────────────────────────────────────┐
│  Doctor's Diagnosis Example                              │
├─────────────────────────────────────────────────────────┤
│  Prior probability P(Disease): 1% of population has it  │
│  Likelihood P(Positive|Disease): 95% of sick test +    │
│  Likelihood P(Positive|Healthy): 5% false positive     │
│                                                         │
│  Question: After testing positive, what's the chance    │
│           of actually having the disease?               │
│  Answer: P(Disease|Positive) ≈ 16% (far below intuition!)│
│                                                         │
│  Lesson: Positive results for rare diseases are mostly  │
│          false positives                                │
└─────────────────────────────────────────────────────────┘
```

### 📖 Plain English Translation

| Probability Term | Plain English |
|-----------------|---------------|
| Sample space $\Omega$ | All possible outcomes |
| Event A | Some outcomes we care about |
| $P(A)$ | How likely A happens (between 0 and 1) |
| $P(A \cap B)$ | A and B both happen |
| $P(A \cup B)$ | A or B (at least one) happens |
| $P(A \| B)$ | Probability of A given B has occurred |
| Independent | One event doesn't affect another |

---

## Table of Contents

1. [Random Experiments and Sample Space](#random-experiments-and-sample-space)
2. [Events and Event Operations](#events-and-event-operations)
3. [Definition of Probability](#definition-of-probability)
4. [Basic Properties of Probability](#basic-properties-of-probability)
5. [Conditional Probability](#conditional-probability)
6. [Multiplication Rule](#multiplication-rule)
7. [Independence](#independence)
8. [Law of Total Probability](#law-of-total-probability)
9. [Bayes' Theorem](#bayes-theorem)
10. [Applications in Deep Learning](#applications-in-deep-learning)
11. [Summary](#summary)

---

## Random Experiments and Sample Space

### Random Experiments

A **random experiment** is an experiment that satisfies the following three conditions:
1. Can be repeated under the same conditions
2. All possible outcomes are known in advance
3. The result of each trial is uncertain in advance

**Examples**:
- Rolling a die
- Flipping a coin
- Measuring a person's height
- Observing the output of a neural network

### Sample Space

**Definition**: The set of all possible outcomes of a random experiment, denoted as $\Omega$ or $S$.

**Examples**:
- Rolling a die: $\Omega = \{1, 2, 3, 4, 5, 6\}$
- Flipping a coin: $\Omega = \{Heads, Tails\}$
- Measuring height: $\Omega = (0$, $+\infty)$
- Binary classification output: $\Omega = \{0, 1\}$

### Sample Points

Each element in the sample space is called a **sample point** or **elementary event**, denoted as $\omega$.

```python
import numpy as np

# Sample space examples
# Rolling a die
omega_dice = {1, 2, 3, 4, 5, 6}
print(f"Die roll sample space: {omega_dice}")

# Flipping two coins
omega_two_coins = {(h1, h2) for h1 in ['H', 'T'] for h2 in ['H', 'T']}
print(f"Two coin flip sample space: {omega_two_coins}")

# Continuous sample space (height measurement)
# Represented as intervals
height_range = (0, 300)  # Unit: centimeters
print(f"Height measurement sample space: (0, 300) cm"
```

---

## Events and Event Operations

### Definition of Events

An **event** is a subset of the sample space, i.e., a collection of some possible outcomes. An event occurs if and only if the trial result belongs to that subset.

### Classification of Events

| Type | Definition | Example |
|------|------|------|
| Certain event | $\Omega$ (sample space itself) | Die roll $\leq 6$ |
| Impossible event | $\emptyset$ (empty set) | Die roll $> 6$ |
| Elementary event | Single sample point | Die roll equals 3 |

### Relations and Operations of Events

**Inclusion**: $A \subset B$, event A occurring implies B occurs

**Union (Sum)**: $A \cup B$ or $A + B$, at least one of A or B occurs

**Intersection (Product)**: $A \cap B$ or $AB$, both A and B occur

**Complement (Opposite)**: $A^c$ or $\bar{A}$, A does not occur

**Difference**: $A - B = A \cap B^c$, A occurs but B does not

**Mutually Exclusive (Disjoint)**: $A \cap B = \emptyset$, A and B cannot occur simultaneously

### Properties of Event Operations

**Commutative laws**:

$$
A \cup B = B \cup A, \quad A \cap B = B \cap A
$$

**Associative laws**:

$$
(A \cup B) \cup C = A \cup (B \cup C)
$$

$$
(A \cap B) \cap C = A \cap (B \cap C)
$$

**Distributive laws**:

$$
A \cap (B \cup C) = (A \cap B) \cup (A \cap C)
$$

$$
A \cup (B \cap C) = (A \cup B) \cap (A \cup C)
$$

**De Morgan's Laws**:

$$
\overline{A \cup B} = \bar{A} \cap \bar{B}
$$

$$
\overline{A \cap B} = \bar{A} \cup \bar{B}
$$

```python
# Event operations examples
omega = {1, 2, 3, 4, 5, 6}
A = {1, 2, 3, 4}  # Points ≤ 4
B = {3, 4, 5, 6}  # Points ≥ 3

# Union
union = A | B
print(f"A ∪ B = {union}")  # {1, 2, 3, 4, 5, 6}

# Intersection
intersection = A & B
print(f"A ∩ B = {intersection}")  # {3, 4}

# Complement
complement_A = omega - A
print(f"A^c = {complement_A}")  # {5, 6}

# Difference
difference = A - B
print(f"A - B = {difference}")  # {1, 2}

# De Morgan's law verification
de_morgan_1 = (A | B) == ((omega - A) & (omega - B))  # Note: this is a wrong demonstration
de_morgan_1 = omega - (A | B)  # {}
de_morgan_2 = (omega - A) & (omega - B)  # {}
print(f"De Morgan's law: Ω - (A∪B) = (Ω-A) ∩ (Ω-B): {de_morgan_1 == de_morgan_2}")
```

---

## Definition of Probability

### Classical Probability Model

When the sample space is finite and each elementary event is equally likely:

$$
P(A) = \frac{\text{Number of elementary events in } A}{\text{Total number of elementary events in } \Omega} = \frac{|A|}{|\Omega|}
$$

**Example**: Roll a die, find the probability that the number is even.

$$
A = \{2, 4, 6\}, \quad |\Omega| = 6
$$

$$
P(A) = \frac{3}{6} = \frac{1}{2}
$$

### Geometric Probability Model

When the sample space is a continuous region:

$$
P(A) = \frac{\text{Measure of } A}{\text{Measure of } \Omega}
$$

(Measure can be length, area, volume, etc.)

### Axiomatic Definition (Kolmogorov Axioms)

Probability $P$ is a function defined on the set of events that satisfies the following three axioms:

**Axiom 1 (Non-negativity)**:

$$
P(A) \geq 0, \quad \forall A
$$

**Axiom 2 (Normalization)**:

$$
P(\Omega) = 1
$$

**Axiom 3 (Countable Additivity)**:

If $A_1, A_2, \ldots$ are pairwise mutually exclusive, then:

$$
P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)
$$

### Frequentist vs Bayesian Schools

| School | Probability Interpretation | Characteristics |
|------|----------|------|
| Frequentist | Long-run frequency | Objective, based on repeated trials |
| Bayesian | Degree of belief | Subjective, can be updated |

```python
import numpy as np

# Classical probability example: lottery
# 100 tickets, 10 winners
n_total = 100
n_win = 10

p_win = n_win / n_total
print(f"Winning probability: {p_win:.2%}")

# Simulation verification
n_trials = 100000
results = np.random.choice([0, 1], size=n_trials, p=[1-p_win, p_win])
frequency = results.mean()
print(f"Simulated frequency: {frequency:.4f}")
print(f"Theoretical probability: {p_win:.4f}")
```

---

## Basic Properties of Probability

The following properties can be derived from the axioms:

### Basic Properties

**Probability of empty set is zero**:

$$
P(\emptyset) = 0
$$

**Complementary event**:

$$
P(A^c) = 1 - P(A)
$$

**Addition rule**:

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

**Monotonicity**:

$$
A \subset B \Rightarrow P(A) \leq P(B)
$$

**Boundedness**:

$$
0 \leq P(A) \leq 1
$$

### Generalized Addition Rule

**Three events**:

$$
P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)
$$

**Inclusion-Exclusion Principle** (n events):

$$
P\left(\bigcup_{i=1}^n A_i\right) = \sum_{i} P(A_i) - \sum_{i<j} P(A_i \cap A_j) + \sum_{i<j<k} P(A_i \cap A_j \cap A_k) - \cdots
$$

```python
# Addition rule verification
# P(A∪B) = P(A) + P(B) - P(A∩B)
omega = {1, 2, 3, 4, 5, 6}
A = {1, 2, 3, 4}  # P(A) = 4/6
B = {3, 4, 5, 6}  # P(B) = 4/6

p_A = len(A) / len(omega)
p_B = len(B) / len(omega)
p_AB = len(A & B) / len(omega)
p_union = len(A | B) / len(omega)

# Verify addition rule
p_union_formula = p_A + p_B - p_AB
print(f"P(A∪B) direct calculation: {p_union:.4f}")
print(f"P(A∪B) formula calculation: {p_union_formula:.4f}")
print(f"Verification: {abs(p_union - p_union_formula) < 1e-10}")

# Complementary event
p_complement_A = 1 - p_A
p_complement_A_direct = len(omega - A) / len(omega)
print(f"\nP(A^c) = 1 - P(A): {p_complement_A:.4f}")
print(f"P(A^c) direct calculation: {p_complement_A_direct:.4f}")
```

---

## Conditional Probability

### Definition

The **conditional probability** of event A occurring given that event B has occurred:

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0
$$

### Geometric Understanding

Conditional probability can be understood as the probability of A occurring in the **reduced sample space** (B).

### Properties

1. **Generally not equal to reverse conditional probability**: $P(A|B) \neq P(B|A)$ (usually)

2. **Multiplication rule**:

$$
P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)
$$

3. **Conditional probability is still a probability**: satisfies all axioms of probability

$$
P(A|B) \geq 0
$$

$$
P(\Omega|B) = 1
$$

$$
P(A_1 \cup A_2|B) = P(A_1|B) + P(A_2|B) \quad \text{(if } A_1, A_2 \text{ are mutually exclusive)}
$$

### Example

A bag contains 5 balls: 3 red and 2 white. Draw 2 balls without replacement. Find the probability that the second ball is red given that the first ball is red.

Let $A$ = "first ball is red", $B$ = "second ball is red"

$$
P(B|A) = \frac{2}{4} = \frac{1}{2}
$$

Because the first ball drawn is red, there are 2 red balls remaining among the 4 remaining balls.

```python
import numpy as np

# Conditional probability example
# Bag has 5 balls: 3 red, 2 white
# Draw 2 balls without replacement

# Method 1: Direct calculation
# P(second red | first red) = 2/4 = 0.5
p_second_red_given_first_red = 2 / 4
print(f"Conditional probability (direct): {p_second_red_given_first_red:.4f}")

# Method 2: Using definition
# P(first red) = 3/5
# P(both red) = 3/5 * 2/4 = 6/20 = 3/10
p_first_red = 3 / 5
p_both_red = 3 / 5 * 2 / 4
p_second_given_first = p_both_red / p_first_red
print(f"Conditional probability (definition): {p_second_given_first:.4f}")

# Simulation verification
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
print(f"Simulated probability: {simulated_prob:.4f}")
```

---

## Multiplication Rule

### Two Events

$$
P(A \cap B) = P(A|B) \cdot P(B) = P(B|A) \cdot P(A)
$$

### Multiple Events (Chain Rule)

$$
P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \cdot P(A_2|A_1) \cdot P(A_3|A_1 \cap A_2) \cdots P(A_n|A_1 \cap \cdots \cap A_{n-1})
$$

### Application Example

A bag contains 5 balls: 3 red and 2 white. Draw 3 balls without replacement. Find the probability of getting red, red, white in order.

$$
P(\text{Red}_1 \cap \text{Red}_2 \cap \text{White}_3) = \frac{3}{5} \times \frac{2}{4} \times \frac{2}{3} = \frac{12}{60} = \frac{1}{5}
$$

```python
# Multiplication rule example
# Bag has 5 balls: 3 red, 2 white, draw 3 without replacement

# P(red1, red2, white3)
p_r1 = 3 / 5
p_r2_given_r1 = 2 / 4  # After drawing red, 2 red and 2 white remain
p_w3_given_r1r2 = 2 / 3  # After drawing red-red, 1 red and 2 white remain

p_rrw = p_r1 * p_r2_given_r1 * p_w3_given_r1r2
print(f"P(red, red, white) = {p_rrw:.4f}")

# Verification: All possible permutations
from itertools import permutations

balls = ['R', 'R', 'R', 'W', 'W']
all_sequences = list(permutations(balls, 3))
rrw_sequences = [seq for seq in all_sequences if seq == ('R', 'R', 'W')]

# Note: permutations double counts identical balls, need combination method
# Simpler method: Direct combination calculation
from math import comb
total = comb(5, 3)  # All combinations of choosing 3 from 5
favorable = comb(3, 2) * comb(2, 1)  # Choose 2 red from 3, 1 white from 2
print(f"Combination method verification: {favorable / total:.4f}")
```

---

## Independence

### Definition

Events A and B are **mutually independent** if and only if:

$$
P(A \cap B) = P(A) \cdot P(B)
$$

Equivalent conditions:

$$
P(A|B) = P(A) \quad \text{(B occurring does not affect A's probability)}
$$

### Independent vs Mutually Exclusive

| Concept | Condition | Meaning |
|------|------|------|
| Independent | $P(A \cap B) = P(A)P(B)$ | One event occurring doesn't affect the other |
| Mutually Exclusive | $A \cap B = \emptyset$ | Two events cannot occur simultaneously |

**Important distinction**:
- If $P(A) > 0$ and $P(B) > 0$, then independent and mutually exclusive **cannot both be true**
- Mutually exclusive means: if one occurs, the other must not occur (strong correlation)
- Independent means: one occurring has no effect on the other (no correlation)

### Independence of Multiple Events

$A_1, A_2, \ldots, A_n$ are **mutually independent** if and only if for any subset $I \subseteq \{1, 2, \ldots, n\}$:

$$
P\left(\bigcap_{i \in I} A_i\right) \prod_{i \in I} P(A_i)
$$

**Note**: Pairwise independence $\neq$ Mutual independence

### Conditional Independence

A and B are conditionally independent given C occurs:

$$
P(A \cap B | C) = P(A|C) \cdot P(B|C)
$$

```python
import numpy as np

# Independence verification example
# Roll two dice, A=first is 6, B=second is 6

# P(A) = P(B) = 1/6
p_A = 1 / 6
p_B = 1 / 6

# P(A∩B) = 1/36
p_AB = 1 / 36

# Verify independence
is_independent = abs(p_AB - p_A * p_B) < 1e-10
print(f"P(A) × P(B) = {p_A * p_B:.6f}")
print(f"P(A∩B) = {p_AB:.6f}")
print(f"Independent? {is_independent}")

# Independent vs mutually exclusive
# If A and B are mutually exclusive and P(A) > 0, P(B) > 0
# Then P(A∩B) = 0 ≠ P(A)P(B) > 0
# So mutually exclusive ⇒ not independent

# Example: Roll one die
# A = point is 1, B = point is 2
# Mutually exclusive (cannot occur simultaneously)
# But not independent: P(A)P(B) = 1/36 ≠ 0 = P(A∩B)
p_A = 1 / 6
p_B = 1 / 6
p_AB_mutual = 0
print(f"\nMutually exclusive events:")
print(f"P(A) × P(B) = {p_A * p_B:.6f}")
print(f"P(A∩B) = {p_AB_mutual:.6f}")
print(f"Independent? {abs(p_AB_mutual - p_A * p_B) < 1e-10}")

# Simulation verification of independence
def test_independence(n_trials=100000):
    # Roll two dice
    dice1 = np.random.randint(1, 7, n_trials)
    dice2 = np.random.randint(1, 7, n_trials)

    # A: first is 6, B: second is 6
    count_A = np.sum(dice1 == 6)
    count_B = np.sum(dice2 == 6)
    count_AB = np.sum((dice1 == 6) & (dice2 == 6))

    p_A_sim = count_A / n_trials
    p_B_sim = count_B / n_trials
    p_AB_sim = count_AB / n_trials

    print(f"\nSimulation verification of independence:")
    print(f"P(A) ≈ {p_A_sim:.4f}")
    print(f"P(B) ≈ {p_B_sim:.4f}")
    print(f"P(A∩B) ≈ {p_AB_sim:.4f}")
    print(f"P(A)P(B) ≈ {p_A_sim * p_B_sim:.4f}")

test_independence()
```

---

## Law of Total Probability

### Partition

Let $B_1, B_2, \ldots, B_n$ be a **partition** of the sample space, satisfying:
1. $B_i \cap B_j = \emptyset$ (for $i \neq j$) — pairwise mutually exclusive
2. $\bigcup_{i=1}^n B_i = \Omega$ — covers the entire sample space

### Law of Total Probability

For any event A and partition $B_1, \ldots, B_n$:

$$
P(A) = \sum_{i=1}^n P(A|B_i) \cdot P(B_i)
$$

### Intuitive Understanding

Decompose event A into mutually exclusive cases, calculate probabilities separately, and sum them.

### Application Example

A factory has three workshops producing the same product:
- Workshop 1 produces 50%, defect rate 2%
- Workshop 2 produces 30%, defect rate 3%
- Workshop 3 produces 20%, defect rate 4%

Find: The probability that a randomly selected product is defective.

**Solution**:

Let $A$ = "product is defective", $B_i$ = "product is produced by workshop i"

$$
P(A) = P(A|B_1)P(B_1) + P(A|B_2)P(B_2) + P(A|B_3)P(B_3)
$$

$$
= 0.02 \times 0.5 + 0.03 \times 0.3 + 0.04 \times 0.2 = 0.027
$$

```python
# Law of total probability example
# Three workshops producing products

p_B1, p_B2, p_B3 = 0.5, 0.3, 0.2  # Production proportion of each workshop
p_A_given_B1 = 0.02  # Workshop 1 defect rate
p_A_given_B2 = 0.03  # Workshop 2 defect rate
p_A_given_B3 = 0.04  # Workshop 3 defect rate

# Law of total probability
p_A = (p_A_given_B1 * p_B1 +
       p_A_given_B2 * p_B2 +
       p_A_given_B3 * p_B3)

print(f"Defective probability P(A) = {p_A:.4f} = {p_A:.2%}")

# Simulation verification
def simulate_production(n_trials=100000):
    # Randomly select workshop
    workshops = np.random.choice([1, 2, 3], size=n_trials, p=[p_B1, p_B2, p_B3])

    # Defect probability for each workshop
    defect_probs = {1: p_A_given_B1, 2: p_A_given_B2, 3: p_A_given_B3}

    # Simulate defective products
    is_defective = np.array([
        np.random.random() < defect_probs[w] for w in workshops
    ])

    return is_defective.mean()

simulated_prob = simulate_production()
print(f"Simulated defect rate: {simulated_prob:.4f}")
```

---

## Bayes' Theorem

### Bayes' Formula

$$
P(B_i|A) = \frac{P(A|B_i) \cdot P(B_i)}{P(A)} = \frac{P(A|B_i) \cdot P(B_i)}{\sum_{j=1}^n P(A|B_j) \cdot P(B_j)}
$$

**Derivation of Bayes' Formula**:

**Step 1**: Start from the definition of conditional probability.

Conditional probability is defined as:

$$P(B|A) = \frac{P(A \cap B)}{P(A)}$$

**Step 2**: Use the multiplication rule to expand the numerator.

$$P(A \cap B_i) = P(A|B_i) \cdot P(B_i)$$

Therefore:

$$P(B_i|A) = \frac{P(A|B_i) \cdot P(B_i)}{P(A)}$$

**Step 3**: Use the law of total probability to expand the denominator $P(A)$.

$$P(A) = \sum_{j=1}^n P(A|B_j) \cdot P(B_j)$$

**Step 4**: Combine to get Bayes' formula.

$$\boxed{P(B_i|A) = \frac{P(A|B_i) \cdot P(B_i)}{\sum_{j=1}^n P(A|B_j) \cdot P(B_j)}}$$

**Intuitive understanding**:
- $P(B_i)$ is the **prior probability**: the probability of $B_i$ occurring before observing $A$
- $P(B_i|A)$ is the **posterior probability**: the updated probability of $B_i$ occurring after observing $A$
- Bayes' formula describes **how to update beliefs based on new evidence**

### Bayesian Interpretation

$$
\text{Posterior Probability} = \frac{\text{Likelihood} \times \text{Prior Probability}}{\text{Evidence (normalization constant)}}
$$

| Term | Symbol | Meaning |
|------|------|------|
| Prior probability | $P(B_i)$ | Belief before observing data |
| Likelihood | $P(A|B_i)$ | Probability of observing data given hypothesis |
| Evidence | $P(A)$ | Total probability of observing data |
| Posterior probability | $P(B_i|A)$ | Updated belief after observing data |

### Essence of Bayesian Inference

Bayes' theorem describes **how to update beliefs based on new evidence**:

$$
\text{Posterior} \propto \text{Likelihood} \times \text{Prior}
$$

### Application Example: Spam Classification

Let:
- $S$: Email is spam
- $W$: Email contains the word "free"

Given:
- $P(S) = 0.3$ (prior: 30% of emails are spam)
- $P(W|S) = 0.8$ (likelihood: 80% of spam emails contain "free")
- $P(W|\bar{S}) = 0.1$ (10% of normal emails contain "free")

Find: $P(S|W)$ (probability that an email containing "free" is spam)

**Solution**:

**Step 1**: Calculate evidence $P(W)$

$$
P(W) = P(W|S)P(S) + P(W|\bar{S})P(\bar{S}) = 0.8 \times 0.3 + 0.1 \times 0.7 = 0.31
$$

**Step 2**: Apply Bayes' formula

$$
P(S|W) = \frac{P(W|S)P(S)}{P(W)} = \frac{0.8 \times 0.3}{0.31} \approx 0.774
$$

**Interpretation**: After seeing "free", the probability of spam updates from 30% to 77.4%.

```python
import numpy as np

# Bayes' theorem example: spam classification

# Prior probabilities
p_spam = 0.3
p_not_spam = 1 - p_spam

# Likelihood
p_free_given_spam = 0.8
p_free_given_not_spam = 0.1

# Calculate evidence (law of total probability)
p_free = (p_free_given_spam * p_spam +
          p_free_given_not_spam * p_not_spam)

print(f"Evidence P(free): {p_free:.4f}")

# Bayes' formula
p_spam_given_free = (p_free_given_spam * p_spam) / p_free

print(f"\nPrior P(spam): {p_spam:.2%}")
print(f"Posterior P(spam|free): {p_spam_given_free:.2%}")
print(f"Belief update: {p_spam:.2%} → {p_spam_given_free:.2%}")

# Simulation verification
def simulate_email(n_trials=100000):
    # Generate email types
    is_spam = np.random.random(n_trials) < p_spam

    # Determine if contains "free" based on type
    contains_free = np.array([
        np.random.random() < (p_free_given_spam if spam else p_free_given_not_spam)
        for spam in is_spam
    ])

    # Calculate P(spam | contains free)
    free_emails = contains_free == True
    spam_given_free = is_spam[free_emails].mean()

    return spam_given_free

simulated = simulate_email()
print(f"\nSimulated P(spam|free): {simulated:.2%}")
```

---

## Applications in Deep Learning

### Softmax and Class Probabilities

Softmax converts network outputs to a probability distribution:

$$
P(y = i|x) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

```python
import numpy as np

def softmax(logits):
    """Convert logits to probability distribution"""
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    return exp_logits / np.sum(exp_logits)

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(f"Logits: {logits}")
print(f"Probability distribution: {probs}")
print(f"Sum of probabilities: {probs.sum():.6f}")
```

### Probabilistic Interpretation of Dropout

During training, each neuron is retained with probability $p$:

$$
\tilde{h}_i = \frac{h_i \cdot m_i}{p}, \quad m_i \sim \text{Bernoulli}(p)
$$

Expectation remains unchanged: $\mathbb{E}[\tilde{h}_i] = h_i$

```python
def dropout(x, p=0.5, training=True):
    """Dropout implementation"""
    if not training:
        return x

    mask = (np.random.random(x.shape) > p).astype(float)
    return x * mask / p

# Example
x = np.random.randn(1000)
x_dropped = dropout(x, p=0.5)

print(f"Original mean: {x.mean():.4f}")
print(f"Mean after Dropout: {x_dropped.mean():.4f}")  # Should be close
```

### Bayesian Neural Networks

Posterior distribution of parameters:

$$
P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}
$$

Prediction distribution (integral over parameters):

$$
P(y|x,D) = \int P(y|x,\theta) P(\theta|D) d\theta
$$

---

## Summary

This chapter introduced the basic concepts of probability theory:

| Concept | Definition/Formula | Applications |
|------|----------|------|
| Conditional probability | $P(A|B) = P(A \cap B)/P(B)$ | Bayesian inference |
| Independence | $P(A \cap B) = P(A)P(B)$ | Dropout, data assumptions |
| Multiplication rule | $P(AB) = P(A|B)P(B)$ | Chain calculation |
| Law of total probability | $P(A) = \sum P(A|B_i)P(B_i)$ | Decompose complex events |
| Bayes' theorem | $P(B|A) = P(A|B)P(B)/P(A)$ | Belief updating |

### Key Concepts

1. **Conditional probability**: Update probabilities given known information
2. **Independence**: No mutual influence between events
3. **Bayesian inference**: Prior + evidence → posterior
4. **Law of total probability**: Decompose complex problems into simple subproblems

---

**Next section**: [Chapter 3 (b): Random Variables and Common Distributions](03b-random-variables-distributions_EN.md) - Learn about discrete and continuous random variables and their common distributions.

**Return**: [Mathematics Fundamentals Tutorial Table of Contents](../math-fundamentals.md)
