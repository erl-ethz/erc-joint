# ERC Model

### Torque Formulation

For a forced rotation of the rolling cam, the mechanical work is written as follows.

```math
W = \int_{\theta_1}^{\theta_2}\tau(\theta)d\theta =
\int_{\theta_1}^{\theta_2}F(x)dx =
\int_{\theta_1}^{\theta_2}k(2r(\theta) - d_{0})2dr
```

Extracting the integrands gives the torque profile.

```math
\begin{align}
\tau(\theta)d \theta &=2k(2r(\theta) - d_{0})dr\\
\tau(\theta) &= 2k\left(2r(\theta)-d_{0}\right)r'(\theta)
\end{align}
```

### Energy and Support Function

Integrating the torque gives the energy stored in or released by the spring w.r.t. the reference angle $\theta_{0}$

```math
\begin{align}
E(\theta)  &:= \int_{\theta_0}^{\theta}\tau(u)du=
2k\left[
\int_{\theta_0}^{\theta}2r(u)r'(u)du -
d_{0}\int_{\theta_0}^{\theta}r'(u)du
\right] \\
&= 2k\left[r(u)^2 - d_0r(u)
\right]_{\theta_0}^{\theta} \\
&= 2k\left[
r(\theta)^2 - {r_0}^2 - d_0r(\theta) + d_0r_0
\right]
\end{align}
```

with $r_{0} = r(\theta_{0})$, the support function constant. Finally the support function can be computed from the resulting second order polynomial

```math
r(\theta)^2 - r(\theta)d_{0} + \left(d_0r_0 - {r_0}^2 - \frac{E(\theta))}{2k}\right) = 0
```

```math
r(\theta) = \frac{1}{2}\left[d_{0} + \sqrt{(2r_0 - d_{0})^2 + \frac{2E(\theta)}{k}}\right]
```

with the positive root corresponding to the case where $2r_{0} \geq d_{0}$ which is always the case since the spring only admits tension and not compression.

### Energy Scaling

For a spring with constant $k$, length at rest $d_{0}$, maximum length $d_{\text{max}}$, and a safety factor $\text{SF}$, the energy landscape is bounded by how much energy was initially stored in the spring (depending on r0), and by the energy at maximum elongation.

```math
-\frac{k}{2}\left(2r_{0} - d_{0}\right)^2 \leq E(\theta) \leq \frac{k}{2}\left(\frac{d_{\text{max}}}{\text{SF}} - d_{0}\right)^2
```

If one of the two bounds is not satisfied by the target energy profile, a scalling factor $\alpha \in [0 ,1]$ should be introduced to enforce the bounds.

```math
-\frac{k}{2}\left(2r_{0} - d_{0}\right)^2 \leq \alpha E(\theta) \leq \frac{k}{2}\left(\frac{d_{\text{max}}}{\text{SF}} - d_{0}\right)^2
```

By defining $E_{\text{min}} := \min_{\theta}{E(\theta)}$, $E_{\text{max}} := \max_{\theta}{E(\theta)}$, and setting

```math
r_{\text{max}} = \frac{d_{\text{max}}}{2\text{SF}} = \frac{1}{2}\left[d_{0} + \sqrt{(2r_0 - d_{0})^2 + \frac{2E_{\text{max}}}{k}}\right]
```

then

```math
r_{0} = \frac{1}{2}\left[\sqrt{\left(\frac{d_{\text{max}}}{\text{SF}} - d_{0}\right)^2 - \frac{2E_{\text{max}}}{k}} + d_{0}\right]
```

which can be substituted in

```math
\begin{cases}
\alpha E_{\text{min}} &\geq -\frac{k}{2}\left(2r_{0}-d_{0}\right)^2 \\
\alpha E_{\text{max}} &\leq \frac{k}{2}\left(\frac{d_{\text{max}}}{\text{SF}} - d_{0}\right)^2 \\
\end{cases}
```

```math
\begin{cases}
\alpha E_{\text{min}} &\geq -\frac{k}{2}\left(2\frac{1}{2}\left[\sqrt{\left(\frac{d_{\text{max}}}{\text{SF}} - d_{0}\right)^2 - \frac{2\alpha E_{\text{max}}}{k}} + d_{0}\right]-d_{0}\right)^2\\
\alpha E_{\text{max}} &\leq \frac{k}{2}\left(\frac{d_{\text{max}}}{\text{SF}} - d_{0}\right)^2 \\
\end{cases}
```

```math
\begin{cases}
\alpha &\leq \frac{k\left(\frac{d_{\text{max}}}{\text{SF}} - d_{0}\right)^2}{2\left( E_{\text{max}} -  E_{\text{min}}\right)}\\
\alpha &\leq \frac{k\left(\frac{d_{\text{max}}}{\text{SF}} - d_{0}\right)^2}{2E_{\text{max}}} \\
\end{cases}
```

Finally, taking $\alpha$ as close to $1$ as possible while satisfying both constraints gives

```math
\alpha = \frac{k\left(\frac{d_{\text{max}}}{\text{SF}} - d_{0}\right)^2}{2\max\left( E_{\text{max}},E_{\text{max}} -  E_{\text{min}}\right)}
```

Finally the support function constant is computed using the scaling factor and the scaled support function is given by

```math
\begin{align}
r_{s}(\theta) &= \frac{1}{2}\left[d_{0} + \sqrt{(2r_0 - d_{0})^2 + \frac{2\alpha E(\theta)}{k}}\right], \quad
r_{0} = \frac{1}{2}\left[d_{0}+\sqrt{\left(\frac{d_{\text{max}}}{\text{SF}} - d_{0}\right)^2 - \frac{2\alpha E_{\text{max}}}{k}}\right] \\
r_{s}(\theta) &= \frac{1}{2}\left[d_{0} + \sqrt{\left(\frac{d_{\text{max}}}{\text{SF}} - d_{0}\right)^2 - \frac{2\alpha}{k}\left(E_{\text{max}} - E(\theta)\right)}\right] \\
r_{s}(\theta) &= \frac{1}{2}\left[d_{0} + \left(\frac{d_{\text{max}}}{\text{SF}} - d_{0}\right) \sqrt{1 - \frac{E_{\text{max}} - E(\theta)}{\max\left( E_{\text{max}},E_{\text{max}} -  E_{\text{min}}\right)}}\right]
\end{align}
```

### Enforcing Convexity of the Cam Profile

The last condition on the support function is that the resulting cam profile must be convex. For a cuve defined by its support function, the radius of curvature is given by the sum of the support function and its second derivative. The convexity condition is then

```math
R(\theta) = r_{s}(\theta) + r_{s}''(\theta) \geq 0
```

To "repair" a function $r(\theta)$ that does not satisfy convexity on must solve an ODE to find a new support function $r^*(\theta)$ that satisfies it while being as close as possible to the original function, ideally only adressing problematic regions with negative curvature. A fast optimization-free method to solve the boundary value problem is to discretize the differential operator into a matrix (finite differences) and solve the resulting linear system. First let's define the target function

```math
g(\theta) = \max(r_{s}(\theta) + r_{s}''(\theta),\epsilon)
```

with $\epsilon > 0$ a small tolerance value, essentially clipping the orginial radius of curvature to positive values. One now needs to solve

```math
r^{*}(\theta) + {r^{*}}''(\theta) = g(\theta)
```

With a discretized differential operator, the ODE can be written as

```math
 A\mathbf{r^*} = \mathbf{g},\quad A = \left(D_{2} + I\right)
```

with

```math
D_2 = \frac{1}{\Delta \theta^2}
\begin{bmatrix}
-2 & 1  &        &        &        \\
1  & -2 & 1      &        &        \\
   & 1  & -2     & \ddots &        \\
   &    & \ddots & \ddots & 1      \\
   &    &        & 1      & -2
\end{bmatrix}
```

Boundary conditions can then be enforced by replacing the first and last two rows of the system.

```math
A \mathbf{r^*} = \mathbf{g},\quad
A =\begin{bmatrix}
1 & 0 & 0 & \cdots & 0 \\
-\frac{3}{2\Delta \theta} & \frac{4}{2\Delta \theta} & -\frac{1}{2\Delta \theta} & \cdots & 0 \\
\frac{1}{\Delta \theta^2} & -\frac{2}{\Delta \theta^2} + 1 & \frac{1}{\Delta \theta^2} & \ddots & \vdots \\
\vdots & \ddots & \ddots & \ddots & \frac{1}{\Delta \theta^2} \\
0 & \cdots & \frac{1}{2\Delta \theta} & -\frac{4}{2\Delta \theta} & \frac{3}{2\Delta \theta} \\
0 & \cdots & 0 & 0 & 1
\end{bmatrix},\quad
\mathbf{g} =
\begin{bmatrix}
r(\theta_1) \\
r'(\theta_1) \\
g_3 \\
\vdots \\
g_{n-2} \\
r'(\theta_n) \\
r(\theta_n)
\end{bmatrix}
```

Solving the linear system gives $r^*(\theta)$, the repaired support function. While this method does not formally minimize the difference between the original and repaired functions, in practice it is a fast and robust way to enforce the convexity constraint and results in close-to-optimal repair.

### Cam Profile from the Support Function

Finally, the repaired support function $r^*(\theta)$, generates the cam profile $\gamma(\theta) = \left(x_{\theta},y_{\theta}\right)$. From the definition of a support function we have

```math
r^*(\theta)  = \max_{(x,y) \in C}\left(x\cos(\theta) + y \sin(\theta)\right) = x_{\theta}\cos(\theta) + y_{\theta} \sin(\theta)
```

and differentiating once

```math
{r^*}'(\theta)  = -x_{\theta}\sin(\theta) + y_{\theta} \cos(\theta)
```

solving the linear system yields the cam profile.

```math
\begin{bmatrix}
\cos\theta & \sin\theta\\
-\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
x_{\theta} \\
y_{\theta}
\end{bmatrix}=
\begin{bmatrix}
r(\theta) \\
r'(\theta)
\end{bmatrix}
```

```math
\begin{bmatrix}
x_{\theta} \\
y_{\theta}
\end{bmatrix}=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
r(\theta) \\
r'(\theta)
\end{bmatrix}
```

```math
\begin{aligned}
x_\theta &= r^*(\theta)\cos\theta - {r^*}'(\theta)\sin\theta \\
y_\theta &= r^*(\theta)\sin\theta + {r^*}'(\theta)\cos\theta
\end{aligned}
```
