# PDE Benchmark Summary

All PDEs implemented in this project with their mathematical formulation, domain configuration, and state-of-the-art PINN results from the literature.

---

## Quick Reference

| # | PDE | Equation | Domain | Key Parameters | Output Dim |
|---|-----|----------|--------|----------------|------------|
| 1 | Burgers 1D | $h_t + h h_x - \frac{\nu}{\pi} h_{xx} = 0$ | $x \in [-1,1],\; t \in [0,1]$ | $\nu = 0.01$ | 1 |
| 2 | Schrödinger (NLS) | $i h_t + \tfrac{1}{2} h_{xx} + \|h\|^2 h = 0$ | $x \in [-5,5],\; t \in [0,\pi/2]$ | — | 2 |
| 3 | Wave 1D | $h_{tt} - h_{xx} = 0$ | $x \in [-5,5],\; t \in [0,2\pi]$ | — | 1 |
| 4 | Burgers 2D | $h_t + h(h_{x_0} + h_{x_1}) - \nu(h_{x_0 x_0} + h_{x_1 x_1}) = 0$ | $(x_0,x_1) \in [0,1]^2,\; t \in [0,2]$ | $\nu = 0.1$ | 1 |
| 5 | Allen-Cahn | $h_t - D h_{xx} - 5(h - h^3) = 0$ | $x \in [-1,1],\; t \in [0,1]$ | $D = 0.0001$ | 1 |
| 6 | KdV | $h_t + h h_x + \mu^2 h_{xxx} = 0$ | $x \in [-1,1],\; t \in [0,1]$ | $\mu = 0.022$ ($\mu^2 = 4.84 \times 10^{-4}$) | 1 |
| 7 | Fisher-KPP | $h_t - D h_{xx} - \kappa h(1-h) = 0$ | $x \in [0,1],\; t \in [0,1]$ | $D=1,\; \kappa=25$ | 1 |
| 8 | Convection-Diffusion | $h_t + \beta h_x - \varepsilon h_{xx} = 0$ | $x \in [-1,1],\; t \in [0,1]$ | $\beta=1,\; \varepsilon=0.01$ | 1 |
| 9 | Kuramoto-Sivashinsky | $h_t + \alpha h h_x + \beta h_{xx} + \gamma h_{xxxx} = 0$ | $x \in [0,2\pi],\; t \in [0,1]$ | $\alpha=100/16,\; \beta=100/16^2,\; \gamma=100/16^4$ | 1 |

---

## Detailed PDE Descriptions

### 1. Burgers 1D (Hard Variant)

$$h_t + h\, h_x - \frac{\nu}{\pi}\, h_{xx} = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [-1, 1]$ |
| **Temporal domain** | $t \in [0, 1]$ |
| **Parameters** | $\nu = 0.01$ (effective viscosity $\nu/\pi \approx 0.00318$) |
| **Initial condition** | $h(x, 0) = -\sin(\pi x)$ |
| **Boundary conditions** | Dirichlet: $h(-1, t) = h(1, t) = 0$ |
| **Character** | Sharp shock formation with very thin viscous layer; standard "hard" PINN benchmark |

**PINN Benchmark Results (Rel. L₂ Error):**

| Method | Year | Optimizer | Rel. L₂ | Architecture | Params | Time Windows | Reference |
|--------|------|-----------|---------|--------------|--------|--------------|-----------|
| SSBroyden PINN | 2025 | SSBroyden | **1.62 × 10⁻⁸** | MLP `[2,20×8,1]`, tanh | **3,021** | 1 | [Kiyani et al., 2025](https://arxiv.org/abs/2501.16371) |
| vRBA ($\Phi = r^2$) + FF | 2025 | SSBroyden | 2.68 × 10⁻⁷ | MLP `[2,30×3,1]` + periodic enc | **2,011** | 1 | [Hag et al., 2025](https://www.nature.com/articles/s44387-026-00084-4) |
| SOAP + PirateNet | 2025 | SOAP | 4.03 × 10⁻⁵ | PirateNet 3×256, RWF | *~500K* | 1 | [Wang et al., 2025](https://arxiv.org/abs/2502.00604) |
| RAD | 2023 | Adam+L-BFGS | varies | MLP `[2,64×4,1]`, tanh | **12,737** | 1 | [Wu et al., 2023](https://jmlr.org/papers/v24/22-1258.html) |
| Vanilla PINN | 2019 | L-BFGS | 6.7 × 10⁻⁴ | MLP `[2,20×8,1]`, tanh | **3,021** | 1 | [Raissi et al., 2019](https://doi.org/10.1016/j.jcp.2018.10.045) |

> **Comparability:** Results above are for the standard $\nu = 0.01$ variant (effective viscosity $\nu/\pi \approx 0.00318$) matching our config and all listed papers. For the easier $\nu/\pi = 1/100$ variant, SOTA is vRBA: 8.25 × 10⁻⁹ and PirateNet (Adam): 8.20 × 10⁻⁵ (not directly comparable).

---

### 2. Nonlinear Schrödinger Equation (NLS)

$$i\, h_t + \tfrac{1}{2}\, h_{xx} + |h|^2 h = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [-5, 5]$ |
| **Temporal domain** | $t \in [0, \pi/2]$ |
| **Parameters** | None (coefficients fixed at 1/2 and 1) |
| **Initial condition** | $h(x, 0) = 2\,\mathrm{sech}(x)$ |
| **Boundary conditions** | Periodic: $h(-5,t) = h(5,t)$, $h_x(-5,t) = h_x(5,t)$ |
| **Output** | Complex-valued: $h = [u, v]$ (real, imaginary parts); output dim = 2 |
| **Character** | Peregrine soliton dynamics; amplitude $|h|$ develops sharp localized peaks |

**PINN Benchmark Results (Rel. L₂ Error):**

| Method | Year | Rel. L₂ | Architecture | Params | Time Windows | Reference |
|--------|------|---------|--------------|--------|--------------|-----------|
| Vanilla PINN | 2019 | ~1.97 × 10⁻³ | MLP `[2,100×4,2]`, tanh | **30,802** | None | [Raissi et al., 2019](https://doi.org/10.1016/j.jcp.2018.10.045) |

> **Comparability:** Vanilla PINN uses the canonical NLS benchmark configuration matching our setup. **Note:** PirateNet paper does not include a Schrödinger/NLS benchmark — only Ginzburg-Landau (different PDE).

---

### 3. Wave 1D

$$h_{tt} - h_{xx} = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [-5, 5]$ |
| **Temporal domain** | $t \in [0, 2\pi]$ |
| **Parameters** | None |
| **Initial conditions** | $h(x,0) = \sin(x)$, $h_t(x,0) = 0$ |
| **Boundary conditions** | Dirichlet from analytical solution: $h(\pm 5, t) = \sin(\pm 5)\cos(t)$ |
| **Analytical solution** | $h(x,t) = \sin(x)\cos(t)$ |
| **Character** | Linear; smooth standing wave solution. Useful baseline for method validation |

**PINN Benchmark Results (Rel. L₂ Error):**

| Method | Year | Rel. L₂ | Model Capacity | Reference |
|--------|------|---------|----------------|-----------|
| PINNacle benchmark (multi-method) | 2024 | varies by method | Varies by method | [Hao et al., NeurIPS 2024](https://arxiv.org/abs/2306.08827) |
| Vanilla PINN | 2019 | ~10⁻³ range | *~3K params (est.)* — same MLP style as Burgers (`[2,20×8,1]`). Raissi uses this architecture across 1D problems. | [Raissi et al., J. Comput. Phys. 378](https://doi.org/10.1016/j.jcp.2018.10.045) |

> Note: The 1D wave equation with smooth IC is relatively easy for PINNs. It is included primarily as a validation/baseline problem rather than a challenging benchmark.

---

### 4. Burgers 2D

$$h_t + h(h_{x_0} + h_{x_1}) - \nu(h_{x_0 x_0} + h_{x_1 x_1}) = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $(x_0, x_1) \in [0, 1] \times [0, 1]$ |
| **Temporal domain** | $t \in [0, 2]$ |
| **Parameters** | $\nu = 0.1$ |
| **Initial condition** | $h(x_0, x_1, 0) = \frac{1}{1 + \exp\!\left(\frac{x_0 + x_1}{2\nu}\right)}$ |
| **Boundary conditions** | Dirichlet from analytical solution: $h = \frac{1}{1 + \exp\!\left(\frac{x_0 + x_1 - t}{2\nu}\right)}$ on all edges |
| **Analytical solution** | $h(x_0, x_1, t) = \frac{1}{1 + \exp\!\left(\frac{x_0 + x_1 - t}{2\nu}\right)}$ |
| **Character** | Traveling sigmoid front in 2D; tests multi-dimensional spatial derivative handling |

**PINN Benchmark Results:**

| Method | Year | Notes | Model Capacity | Reference |
|--------|------|-------|----------------|-----------|
| WF-PINNs | 2025 | Weak-form PINNs for Burgers-type models including 2D | *Not reported.* Paper states "simple standard neural network architecture"; dual-network for inverse problems. | [Alghamdi et al., Sci. Reports, 2025](https://www.nature.com/articles/s41598-025-24427-4) |
| PINNacle benchmark | 2024 | Includes 2D Burgers among 20+ PDEs | Varies by method | [Hao et al., NeurIPS 2024](https://arxiv.org/abs/2306.08827) |

> Note: 2D Burgers with $\nu = 0.1$ is moderately diffusive. Few papers report comparable benchmark numbers for this exact configuration.

---

### 5. Allen-Cahn

$$h_t - D\, h_{xx} - 5(h - h^3) = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [-1, 1]$ |
| **Temporal domain** | $t \in [0, 1]$ |
| **Parameters** | $D = 0.0001$ (standard SOTA benchmark) |
| **Initial condition** | $h(x, 0) = x^2 \cos(\pi x)$ |
| **Boundary conditions** | Periodic: $h(-1, t) = h(1, t)$, $h_x(-1, t) = h_x(1, t)$ |
| **Character** | Stiff nonlinear reaction-diffusion; sharp moving interface between $h \approx +1$ and $h \approx -1$ regions. One of the hardest standard PINN benchmarks |

**PINN Benchmark Results (Rel. L₂ Error):**

| Method | Year | Optimizer | Rel. L₂ | Architecture | Params | Time Windows | Reference |
|--------|------|-----------|---------|--------------|--------|--------------|-----------|
| SSBroyden PINN | 2025 | SSBroyden | **9.43 × 10⁻⁷** | MLP `[2,30×3,1]`, tanh | **2,019** | 1 | [Kiyani et al., 2025](https://arxiv.org/abs/2501.16371) |
| vRBA ($\Phi = r^2$) + FF | 2025 | SSBroyden | 1.88 × 10⁻⁶ | MLP `[2,30×3,1]` + periodic enc | **2,011** | 1 | [Hag et al., 2025](https://www.nature.com/articles/s44387-026-00084-4) |
| SOAP + PirateNet | 2025 | SOAP | 3.48 × 10⁻⁶ | PirateNet 3×256, RWF | *~500K* | 1 | [Wang et al., 2025](https://arxiv.org/abs/2502.00604) |
| Causal PINN | 2022 | Adam | 2.46 × 10⁻⁴ | Modified MLP `[2,128×6,1]`, tanh | **~84K** | 1 (causal weighting) | [Wang et al., 2022](https://arxiv.org/abs/2203.07404) |
| vRBA ($\Phi = r^2$) + FF | 2025 | Adam | varies | MLP `[2,64×6,1]` + FF, tanh | **21,318** | 1 | [Hag et al., 2025](https://www.nature.com/articles/s44387-026-00084-4) |
| RBA + mMLP + FF | 2023 | Adam | ~4.55 × 10⁻⁵ | MLP `[2,128×6,1]`, tanh, Xavier | **83,073** | 1 | [Anagnostopoulos et al., 2023](https://arxiv.org/abs/2307.00379) |
| PirateNet + FF + WF + CS | 2024 | Adam | 2.24 × 10⁻⁵ | 9 layers × 256 ch, tanh, FF 2.0, RWF | *~500K+* | 1 (causal 32 chunks) | [Wang et al., 2024](https://jmlr.org/papers/v25/24-0313.html) |
| Vanilla PINN | 2019 | Adam | ~4.98 × 10⁻¹ | MLP `[2,20×8,1]`, tanh | **3,021** | 1 | [Raissi et al., 2019](https://doi.org/10.1016/j.jcp.2018.10.045) |

> **Comparability:** All results above use the standard Allen-Cahn benchmark with $D = 0.0001$ matching our configuration and are directly comparable. Note: RAD (Wu et al. 2023) uses D=0.001, which is a different (easier) problem. "Causal chunks" are training/weighting chunks (not separate NNs).

---

### 6. Korteweg-de Vries (KdV)

$$h_t + \eta\, h\, h_x + \mu^2\, h_{xxx} = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [-1, 1]$ |
| **Temporal domain** | $t \in [0, 1]$ |
| **Parameters** | $\eta = 1$, $\mu = 0.022$ (so $\mu^2 = 4.84 \times 10^{-4}$); classical Zabusky & Kruskal (1965) values. **Note:** Our code uses `mu` directly as the coefficient (i.e., code's `mu` = literature's $\mu^2$) |
| **Initial condition** | $h(x, 0) = \cos(\pi x)$ |
| **Boundary conditions** | Periodic: $h(-1,t) = h(1,t)$ |
| **Character** | Nonlinear dispersive; initial cosine breaks into multi-soliton train with sharp localized peaks on flat background |

**PINN Benchmark Results (Rel. L₂ Error):**

| Method | Year | Optimizer | Rel. L₂ | Architecture | Params | Time Windows | Reference |
|--------|------|-----------|---------|--------------|--------|--------------|-----------|
| vRBA ($\Phi = e^r$) + FF | 2025 | SSBroyden | **2.17 × 10⁻⁶** | MLP `[2,30×3,1]` + periodic enc | **2,011** | 1 | [Hag et al., 2025](https://www.nature.com/articles/s44387-026-00084-4) |
| SOAP + PirateNet | 2025 | SOAP | 3.40 × 10⁻⁴ | PirateNet 3×256, RWF | *~500K* | 1 | [Wang et al., 2025](https://arxiv.org/abs/2502.00604) |
| PirateNet + FF + WF + CS | 2024 | Adam | 4.27 × 10⁻⁴ | 9 layers × 256 ch, tanh, FF 1.0, RWF | *~500K+* | 1 (causal 16 chunks) | [Wang et al., 2024](https://jmlr.org/papers/v25/24-0313.html) |

> **Comparability:** vRBA uses a different KdV formulation (two-soliton, x∈[0,20], t∈[0,5]) than our config. PirateNet/SOAP KdV benchmark matches our config ($\mu^2 = 4.84 \times 10^{-4}$, periodic BC). Note: RAD (Wu et al. 2023) KdV benchmark is an **inverse problem** (parameter discovery), not directly comparable to forward problem.

---

### 7. Fisher-KPP

$$h_t - D\, h_{xx} - \kappa\, h(1 - h) = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [0, 1]$ |
| **Temporal domain** | $t \in [0, 1]$ |
| **Parameters** | $D = 1.0$, $\kappa = 25.0$ |
| **Initial condition** | $h(x, 0) = \frac{1}{1 + \exp\!\left(\sqrt{\kappa/6}\,(x - 0.25)\right)}$ |
| **Boundary conditions** | Dirichlet: $h(0, t) = 1$, $h(1, t) = 0$ |
| **Character** | Nonlinear reaction-diffusion; sharp traveling wavefront propagating rightward. High $\kappa$ makes the front steeper |

**PINN Benchmark Results:**

| Method | Year | Notes | Model Capacity | Reference |
|--------|------|-------|----------------|-----------|
| PINN (various architectures) | 2025 | ~10⁻⁶ errors reported | *Not reported.* Paper tests multiple architectures without specifying param counts. | [Oruç, Accscience, 2025](https://www.accscience.com/journal/NSCE/articles/online_first/6222) |
| Residual-weighted PINN | 2024 | Specialized for steep traveling waves | *Not reported.* | [Hale & Sheraton, 2024](https://arxiv.org/abs/2402.08313) |
| PIKAN (KAN-based PINN) | 2026 | Includes Fisher-type reaction-diffusion | *KAN-based* — uses Kolmogorov-Arnold Networks instead of MLP; param count depends on B-spline grid size. Rigas et al. report KANs can match MLPs with 8.5× fewer params. | [Rigas et al., 2026](https://arxiv.org/abs/2602.15068) |

> Note: Fisher-KPP is less standardized as a PINN benchmark. The large $\kappa = 25$ makes the wavefront very steep, providing a good test case for adaptive methods.

---

### 8. Convection-Diffusion

$$h_t + \beta\, h_x - \varepsilon\, h_{xx} = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [-1, 1]$ |
| **Temporal domain** | $t \in [0, 1]$ |
| **Parameters** | $\beta = 1.0$ (convection), $\varepsilon = 0.01$ (diffusion) |
| **Initial condition** | $h(x, 0) = -\sin(\pi x)$ |
| **Boundary conditions** | Dirichlet: $h(-1, t) = h(1, t) = 0$ |
| **Péclet number** | $Pe = \beta L / \varepsilon = 100$ (convection-dominated) |
| **Character** | Linear but numerically challenging; develops sharp boundary layers where convection pushes solution against boundaries. High Péclet number tests adaptive resolution |

**PINN Benchmark Results:**

| Method | Year | Notes | Model Capacity | Reference |
|--------|------|-------|----------------|-----------|
| PINNacle benchmark (multi-method) | 2024 | Includes convection-diffusion problems | Varies by method; standard baseline: 4 hidden layers × 128-256 neurons | [Hao et al., NeurIPS 2024](https://arxiv.org/abs/2306.08827) |
| Specialized loss functionals | 2024 | 17% L² error reduction vs vanilla for convection-dominated problems | *Not reported.* Paper focuses on loss functional design, not architecture. | [Brüning et al., J. Numer. Math., 2024](https://link.springer.com/article/10.1007/s42967-024-00433-7) |

> Note: Convection-diffusion with $Pe = 100$ is convection-dominated, producing sharp layers. This makes it a natural test case for adaptive expert methods even though it is linear.

---

### 9. Kuramoto-Sivashinsky (KS)

$$h_t + \alpha\, h\, h_x + \beta\, h_{xx} + \gamma\, h_{xxxx} = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [0, 2\pi]$ |
| **Temporal domain** | $t \in [0, 1]$ |
| **Parameters** | $\alpha = 100/16 = 6.25$, $\beta = 100/16^2 = 0.390625$, $\gamma = 100/16^4 \approx 1.526 \times 10^{-3}$ |
| **Initial condition** | $h(x, 0) = \cos(x)(1 + \sin(x))$ |
| **Boundary conditions** | Periodic: $h(0,t) = h(2\pi,t)$, $h_x(0,t) = h_x(2\pi,t)$ |
| **Character** | Nonlinear, chaotic; anti-diffusion ($\beta h_{xx}$) drives instability, hyper-diffusion ($\gamma h_{xxxx}$) stabilizes short wavelengths, nonlinear convection ($\alpha h h_x$) transfers energy. Exhibits complex spatiotemporal patterns with sharp features |

**PINN Benchmark Results (Rel. L₂ Error):**

| Method | Year | Optimizer | Rel. L₂ | Architecture | Params/window | Total Params | Time Windows | Reference |
|--------|------|-----------|---------|--------------|---------------|--------------|--------------|-----------|
| SSBroyden PINN | 2025 | SSBroyden | **2.65 × 10⁻⁵** | MLP `[2,30×5,1]`, tanh | **4,411** | **22,055** | 5 (t∈[0,0.5]) | [Kiyani et al., 2025](https://arxiv.org/abs/2501.16371) |
| SSBroyden PINN | 2025 | SSBroyden | 6.51 × 10⁻⁴ | MLP `[2,30×5,1]`, tanh | **4,411** | **88,220** | 20 (t∈[0,1]) | [Kiyani et al., 2025](https://arxiv.org/abs/2501.16371) |
| Causal PINN (regular) | 2022 | Adam | 3.49 × 10⁻⁴ | Modified MLP `[2,256×5,1]`, tanh | **~267K** | **~2.67M** | 10 (t∈[0,1]) | [Wang et al., 2022](https://arxiv.org/abs/2203.07404) |
| Causal PINN (chaotic) | 2022 | Adam | 2.46 × 10⁻² | Modified MLP `[2,128×10,1]`, tanh | **~150K** | **~750K** | 5 (t∈[0,0.5]) | [Wang et al., 2022](https://arxiv.org/abs/2203.07404) |
| SOAP + PirateNet | 2025 | SOAP | 3.86 × 10⁻² | PirateNet 3×256, RWF | *~500K* | *~5M* | 10 | [Wang et al., 2025](https://arxiv.org/abs/2502.00604) |
| PINNacle benchmark | 2024 | varies | varies | MLP 5 layers × 100 neurons | *~40K* | *~40K* | 1 | [Hao et al., 2024](https://arxiv.org/abs/2306.08827) |

> **Comparability:** KS is one of the most challenging 1D PINN benchmarks due to its chaotic dynamics and 4th-order spatial derivative. **Critical note on time windows:** All high-accuracy KS results use time-marching (separate PINNs per window). Total params = params/window × num_windows. The "Causal" approaches of Wang et al. use time-marching with causal weighting within each window.

---

## Model Capacity Summary

A cross-method summary of network sizes. **Bold** = exact numbers from paper/code (verified). *Italic* = estimate.

| Method | PDE | Rel-L2 Reported | Architecture | Params/NN | Time Windows | Total Params | Source |
|--------|-----|-----------------|-------------|-----------|--------------|--------------|--------|
| **Raissi (Burgers 1D)** | Burgers 1D | 6.7 × 10⁻⁴ | MLP `[2,20×8,1]`, tanh | **3,021** | 1 | **3,021** | [Paper](https://doi.org/10.1016/j.jcp.2018.10.045) + [code](https://github.com/maziarraissi/PINNs) |
| **Raissi (Schrödinger)** | Schrödinger | ~1.97 × 10⁻³ | MLP `[2,100×4,2]`, tanh | **30,802** | 1 | **30,802** | [Paper](https://doi.org/10.1016/j.jcp.2018.10.045) + [code](https://github.com/maziarraissi/PINNs) |
| **SSBroyden (Burgers)** | Burgers 1D | **1.62 × 10⁻⁸** | MLP `[2,20×8,1]`, tanh | **3,021** | 1 | **3,021** | [Kiyani et al., 2025](https://arxiv.org/abs/2501.16371) |
| **SSBroyden (Allen-Cahn)** | Allen-Cahn | **9.43 × 10⁻⁷** | MLP `[2,30×3,1]`, tanh | **2,019** | 1 | **2,019** | [Kiyani et al., 2025](https://arxiv.org/abs/2501.16371) |
| **SSBroyden (KS 5-win)** | KS | **2.65 × 10⁻⁵** | MLP `[2,30×5,1]`, tanh | **4,411** | 5 | **22,055** | [Kiyani et al., 2025](https://arxiv.org/abs/2501.16371) |
| **SSBroyden (KS 20-win)** | KS | 6.51 × 10⁻⁴ | MLP `[2,30×5,1]`, tanh | **4,411** | 20 | **88,220** | [Kiyani et al., 2025](https://arxiv.org/abs/2501.16371) |
| **Causal PINN (Allen-Cahn)** | Allen-Cahn | 2.46 × 10⁻⁴ | Modified MLP `[2,128×6,1]`, tanh | **~84K** | 1 | **~84K** | [Wang et al., 2022](https://arxiv.org/abs/2203.07404) |
| **Causal PINN (KS regular)** | KS | 3.49 × 10⁻⁴ | Modified MLP `[2,256×5,1]`, tanh | **~267K** | 10 | **~2.67M** | [Wang et al., 2022](https://arxiv.org/abs/2203.07404) |
| **Causal PINN (KS chaotic)** | KS | 2.46 × 10⁻² | Modified MLP `[2,128×10,1]`, tanh | **~150K** | 5 | **~750K** | [Wang et al., 2022](https://arxiv.org/abs/2203.07404) |
| **SOAP + PirateNet (Burgers)** | Burgers 1D | 4.03 × 10⁻⁵ | PirateNet 3×256, RWF | *~500K* | 1 | *~500K* | [Wang et al., 2025](https://arxiv.org/abs/2502.00604) |
| **SOAP + PirateNet (Allen-Cahn)** | Allen-Cahn | 3.48 × 10⁻⁶ | PirateNet 3×256, RWF | *~500K* | 1 | *~500K* | [Wang et al., 2025](https://arxiv.org/abs/2502.00604) |
| **SOAP + PirateNet (KdV)** | KdV | 3.40 × 10⁻⁴ | PirateNet 3×256, RWF | *~500K* | 1 | *~500K* | [Wang et al., 2025](https://arxiv.org/abs/2502.00604) |
| **SOAP + PirateNet (KS)** | KS | 3.86 × 10⁻² | PirateNet 3×256, RWF | *~500K* | 10 | *~5M* | [Wang et al., 2025](https://arxiv.org/abs/2502.00604) |
| **vRBA (Adam, Allen-Cahn)** | Allen-Cahn | varies | MLP `[2,64×6,1]` + FF, tanh | **21,318** | 1 | **21,318** | [Hag et al., 2025](https://www.nature.com/articles/s44387-026-00084-4) |
| **vRBA (SSBroyden, Burgers)** | Burgers 1D | 2.68 × 10⁻⁷ | MLP `[2,30×3,1]` + periodic enc | **2,011** | 1 | **2,011** | [Hag et al., 2025](https://www.nature.com/articles/s44387-026-00084-4) |
| **vRBA (SSBroyden, Allen-Cahn)** | Allen-Cahn | 1.88 × 10⁻⁶ | MLP `[2,30×3,1]` + periodic enc | **2,011** | 1 | **2,011** | [Hag et al., 2025](https://www.nature.com/articles/s44387-026-00084-4) |
| **vRBA (SSBroyden, KdV)** | KdV | **2.17 × 10⁻⁶** | MLP `[2,30×3,1]` + periodic enc | **2,011** | 1 | **2,011** | [Hag et al., 2025](https://www.nature.com/articles/s44387-026-00084-4) |
| **RAD (Wu et al.)** | Burgers 1D | varies | MLP `[2,64×4,1]`, tanh | **12,737** | 1 | **12,737** | [Wu et al., 2023](https://jmlr.org/papers/v24/22-1258.html) |
| **RBA (Anagnostopoulos)** | Allen-Cahn | ~4.55 × 10⁻⁵ | MLP `[2,128×6,1]`, tanh, Xavier | **83,073** | 1 | **83,073** | [Anagnostopoulos et al., 2023](https://arxiv.org/abs/2307.00379) |
| **PirateNet (Allen-Cahn)** | Allen-Cahn | 2.24 × 10⁻⁵ | 9×256 ch, tanh, FF 2.0, RWF | *~500K+* | 1 (32 causal chunks) | *~500K+* | [Wang et al., 2024](https://jmlr.org/papers/v25/24-0313.html) |
| **PirateNet (KdV)** | KdV | 4.27 × 10⁻⁴ | 9×256 ch, tanh, FF 1.0, RWF | *~500K+* | 1 (16 causal chunks) | *~500K+* | [Wang et al., 2024](https://jmlr.org/papers/v25/24-0313.html) |

> **How to read**: "MLP `[in,H×L,out]`" = multi-layer perceptron with input dim, H neurons × L hidden layers, output dim. "FF" = Fourier feature embedding. "RWF" = random weight factorization. SSBroyden is a quasi-Newton optimizer that converges with far fewer parameters than Adam. **"Causal chunks"** = training/weighting divisions within a single NN (not separate models). **"Time Windows"** = separate NNs trained sequentially; total params = params/NN × windows.

---

## Key References

| Abbrev. | Full Reference |
|---------|---------------|
| **Raissi et al., 2019** | M. Raissi, P. Perdikaris, G.E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *J. Comput. Phys.* 378, 2019. |
| **Causal PINNs (Wang et al., 2022)** | S. Wang, S. Sankaran, P. Perdikaris. "Respecting causality is all you need for training physics-informed neural networks." arXiv:2203.07404, 2022. |
| **SSBroyden (Kiyani et al., 2025)** | E. Kiyani, K. Shukla, J.F. Urbán, J. Darbon, G.E. Karniadakis. "Optimizing the Optimizer for Physics-Informed Neural Networks and Kolmogorov-Arnold Networks." arXiv:2501.16371, 2025. |
| **SOAP (Wang et al., 2025)** | S. Wang, A.K. Bhartari, B. Li, P. Perdikaris. "Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective." arXiv:2502.00604, 2025. |
| **PirateNet (Wang et al., 2024)** | S. Wang, B. Li, Y. Chen, P. Perdikaris. "PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks." *JMLR* 25, 2024. |
| **RAD (Wu et al., 2023)** | C. Wu, M. Zhu, Q. Tan, Y. Kartha, L. Lu. "A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks." *CMAME* 403, 2023. |
| **vRBA (Hag et al., 2025)** | J. Hag et al. "A variational framework for residual-based adaptivity in neural PDE solvers and operator learning." *npj Artificial Intelligence*, 2025. |
| **PINNacle (Hao et al., 2024)** | Z. Hao et al. "PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs." *NeurIPS Datasets & Benchmarks*, 2024. |
| **DASA-PINN / RBA (Anagnostopoulos et al., 2024)** | S. Anagnostopoulos, J.D. Toscano, N. Stergiopulos, G.E. Karniadakis. "Residual-based attention in physics-informed neural networks." *CMAME* 421, 2024. |

---

*Last updated: 2026-05-30 (Added SSBroyden, Causal PINN, and SOAP papers with verified param counts)*
