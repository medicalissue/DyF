# DyF: Dynamic Floor — formal specification

This is the as-proposed mathematical derivation. The codebase implements
exactly the function defined in §2.3 plus the orchestration around it.

---

## 1. Problem setup

Pre-norm Transformer sublayer:

$$x_{l+1} = x_l + \mathrm{Sublayer}_l(\mathcal{N}(x_l))$$

with $x_l \in \mathbb{R}^{B\times T\times C}$ and $\mathcal{N}: \mathbb{R}^C \to \mathbb{R}^C$
applied per token.

### Goals

| | Property |
|----|----------|
| **(P1)** | Element-wise: no $\mu$, $\sigma$ over tokens (parallelism, quantization-friendly) |
| **(P2)** | Positive information preserved: $\lim_{x_k \to +\infty} \mathcal{N}(x)_k = +\infty$ |
| **(P3)** | Negative stabilized: $\mathcal{N}(x)_k$ converges to a finite value as $x_k \to -\infty$ |
| **(P4)** | Affine degrees of freedom ≥ inference-time LN |
| **(P5)** | Contains ReLU and identity as special cases |

---

## 2. Function form

### 2.1 Asymmetric floor

Parameterize element-wise transition at $a$:

$$\phi_a(u) = L(a) + \psi(u - a), \qquad \lim_{v \to -\infty} \psi(v) = 0,\ \lim_{v \to +\infty} \psi(v) = +\infty$$

We instantiate three monotone $\psi$ candidates:

- **SiLU (soft):** $\psi(v) = v \cdot \sigma(v)$
- **GELU (soft):** $\psi(v) = v \cdot \Phi(v)$ where $\Phi$ is the standard-normal CDF
- **Hard (ReLU):** $\psi(v) = \max(0, v)$

Setting $L(a) = a$ (so the floor sits at $a$, satisfying P5):

$$\phi_a(u) = \psi(u - a) + a$$

### 2.3 DyF definition

Per-module **scalar** $a \in \mathbb{R}$, per-channel $\gamma_k, \beta_k \in \mathbb{R}$:

$$\boxed{\ \mathrm{DyF}(x)_k \;=\; \gamma_k \cdot \big[\psi(x_k - a) + a\big] + \beta_k\ }$$

Three variants by choice of $\psi$:

- **SiLU:**  $\mathrm{DyF}(x)_k = \gamma_k \cdot \big[(x_k - a)\,\sigma(x_k - a) + a\big] + \beta_k$
- **GELU:**  $\mathrm{DyF}(x)_k = \gamma_k \cdot \big[\mathrm{GELU}(x_k - a) + a\big] + \beta_k$
- **Hard:**  $\mathrm{DyF}(x)_k = \gamma_k \cdot \max(a, x_k) + \beta_k$

Parameter count per module: **$2C + 1$** ($\gamma, \beta$ per-channel; one scalar $a$).

---

## 3. Properties

### 3.1 Asymptotics (soft)

- $u \to +\infty$: $\sigma(u-a) \to 1 \Rightarrow f_a(u) \to u$ — (P2) ✓
- $u \to -\infty$: Swish term $\to 0 \Rightarrow f_a(u) \to a$ — (P3) ✓

### 3.2 Special cases

| Setting | $\mathrm{DyF}(x)_k$ | Name |
|---|---|---|
| $a \to -\infty$, SiLU/GELU | $\gamma_k x_k + \beta_k$ | pure affine (norm-free) |
| $a = 0$, $\gamma_k=1$, $\beta_k=0$, SiLU | $x_k \sigma(x_k)$ | Swish/SiLU |
| $a = 0$, $\gamma_k=1$, $\beta_k=0$, GELU | $\mathrm{GELU}(x_k)$ | GELU |
| $a = 0$, hard | $\max(0, x_k)$ | ReLU |

(P5) ✓

### 3.3 Affine DOF

Inference LN with frozen statistics is per-channel affine:
$\mathrm{LN}_\text{infer}(x)_k = \gamma'_k x_k + \beta'_k$, DOF = $2C$.

DyF DOF = $2C + 1$ (per-channel $\gamma, \beta$ plus a single scalar floor $a$).
Strict superset of inference LN's DOF.

(P4) satisfied. ✓

> **Design note (scalar vs per-channel `a`).** An earlier iteration
> made `a` per-channel (DOF = $3C$), giving every channel its own
> learned floor. We later collapsed `a` to a single scalar shared
> across channels: it removes a degree of freedom that was empirically
> tied to per-channel input variance (which $\gamma_k$ already absorbs)
> and makes the "transition at $a$" interpretable at the module level.
> Per-channel `a` remains a future ablation; the spec and code support
> recovering it by changing one shape constant.

### 3.4 Gradients

For $f_a(u) = \psi(u-a) + a$:

| kernel | $\dfrac{df}{du}$ at $u=a$ | $u \to +\infty$ | $u \to -\infty$ |
|---|---|---|---|
| SiLU  | $1/2$ | $\to 1$ | $\to 0$ (exponential) |
| GELU  | $1/2$ | $\to 1$ | $\to 0$ (faster than SiLU) |
| Hard  | undefined ($\{0,1\}$) | $1$ | $0$ |

Both soft kernels give nonzero negative-side gradient (mitigates the
dead-neuron pathology of pure ReLU); the hard kernel reproduces ReLU's
exact zero gradient on the negative side.

Identity (any kernel): $\dfrac{\partial f}{\partial u} + \dfrac{\partial f}{\partial a} = 1$ — **shift covariance**:
$f(u; a) = f(u + c; a + c)$.

### 3.5 Reparameterization (hard variant)

$\mathrm{DyF}^\text{hard}(x)_k = \gamma_k \mathrm{ReLU}(x_k - a_k) + \beta'_k$ where $\beta'_k = \gamma_k a_k + \beta_k$.

Expressively equivalent to $(\gamma_k, a_k, \beta'_k)$, but the original
parameterization couples $a$ to two paths in the gradient → tends to
keep $a$ interpretable as "transition + baseline level".

---

## 4. Stability notes

### 4.1 Pre-norm magnitude

In pre-norm transformers $\mathrm{Var}(x_{l+1}) = \mathrm{Var}(x_l) +
\mathrm{Var}(\mathrm{Sublayer}_l(\mathcal{N}(x_l)))$. Sublayer output
variance depends on weights, not on whether $\mathcal{N}$ caps the
positive tail. Therefore residual-stream stability is *not* compromised
by P2.

### 4.2 Post-norm caveat

Post-norm: $x_{l+1} = \mathcal{N}(x_l + \mathrm{Sublayer}_l(x_l))$. With
$\mathcal{N}$ applied to the residual sum directly, boundedness is
load-bearing. DyF's positive-unbounded design is **not appropriate** for
post-norm — predicted as (C2).

---

## 5. Predictions (testable)

| ID  | Statement |
|-----|-----------|
| C1  | DyF ≥ LN, DyT on pre-norm Transformer / ConvNeXt |
| C2  | DyF underperforms or diverges in post-norm |
| C3  | Learned $a_k$ correlates positively with the channel's input lower tail |
| C4  | Per-module/per-block $a_\text{init}$ matters at LLM scale |
| C5  | ViT-B / ImageNet-1K accuracy within ±0.5%p of DyT, ±0.3%p of LN |

See README §"Predictions to validate" for which job rows test which
prediction.
