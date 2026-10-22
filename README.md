# Scientific Machine Learning (SciML): Coursework

This repository collects my work for a graduate course in Scientific Machine Learning, based on the open course materials by Krishna Kumar ([course site](https://kks32-courses.github.io/sciml/), [materials repo](https://github.com/kks32-courses/sciml)). The course combines numerical analysis, scientific computing, and deep learning to solve problems governed by ordinary and partial differential equations, with particular attention to high-dimensional problems, inverse problems, and systems where the physics is only partly known. Work in this repository is implemented in Python, primarily with PyTorch.

## What the course covers

### 1. Representation and function spaces
The course opens with a basic question: how can a continuous function be represented with finitely many numbers? It covers basis expansions, kernels and parametric families, function spaces (Hilbert, Banach, and Sobolev), norms, and projection. This framework recurs throughout, since every later method is a choice of representation plus a way of fitting it.

### 2. Multilayer perceptrons and their foundations
Neural networks are introduced as function approximators. Topics include the Universal Approximation Theorem, including its proof and its limitations; activation functions such as ReLU, tanh, and sigmoid, and how each shapes what a network can represent; regularization; automatic differentiation; and gradient descent and its variants. Classification is covered through decision trees, MLP classifiers, and binary cross-entropy.

### 3. Physics-informed neural networks (PINNs)
PINNs embed a differential equation directly in the loss function. The course works through forward problems (oscillators, Poisson, Burgers, linear elasticity in a variational form) and inverse problems (heat equation, oscillator parameter identification with a differentiable solver). It also examines why PINNs fail and how to fix them: soft versus hard boundary-condition enforcement, collocation-point sampling, adaptive loss weighting, neural tangent kernel (NTK) analysis of spectral bias, failure modes on convection-dominated problems, and optimizer benchmarks.

### 4. Operator learning
Instead of learning a single solution, operator learning learns the mapping from inputs (forcing terms, initial conditions, coefficients) to solutions. The course covers DeepONet and the Universal Approximation Theorem for operators, physics-informed DeepONet, sensor placement and frequency effects, function encoders and basis-to-basis operator learning, and Fourier Neural Operators (FNOs), which use spectral methods.

### 5. Graph neural networks for simulation
Physical systems such as particles and fluids are represented as graphs and modeled with message-passing networks, including Graph Network-based Simulators (GNS). Symmetry and equivariance are treated as design principles.

### 6. Differentiable programming
Automatic differentiation is studied in depth as the engine behind every SciML method: forward and reverse mode, differentiating through physics simulators and time-stepping schemes, and using these gradients for inverse problems and parameter estimation.

### 7. Equation discovery
Sparse Identification of Nonlinear Dynamics (SINDy) and symbolic regression are used to recover governing equations from data, including a pendulum case study and experimental data.

### 8. Neural ODEs
Neural ODEs treat a network as a continuous-depth dynamical system. Topics include the adjoint method for memory-efficient gradients, constrained optimization, latent ODEs for time-series modeling, and a Van der Pol oscillator modeled with function encoders.

### 9. Uncertainty quantification and generative models
Bayesian neural networks, Gaussian processes, normalizing flows, and physics-constrained variational autoencoders are used to quantify how much a model's predictions can be trusted.

### 10. Transfer learning and transformers
The syllabus also covers domain adaptation across physical regimes, few-shot and meta-learning for new physical systems, multi-task and active learning, and transformer architectures for spatiotemporal scientific data.

## Skills developed

**Mathematical and theoretical skills**

The course builds the ability to analyze approximation methods rigorously: what a theorem like the UAT guarantees and what it does not, how smoothness of a target function interacts with the smoothness of the approximating basis, and how convergence rates differ between representations. It also develops intuition for training dynamics, including spectral bias, vanishing gradients, loss landscapes, and the conditioning of least-squares problems, and for the role of regularization in turning an ill-posed fit into a stable one.

**Modeling skills**

A central skill is formulating a scientific problem as a learning problem: choosing what to learn (a solution, an operator, a parameter, or an equation), how to build known physics into the model (loss terms, hard constraints, architecture, or symmetry), how to sample training points, and how to validate against traditional numerical methods. The course emphasizes matching model structure to the problem's structure, for example decomposing multi-scale targets or choosing activations suited to non-smooth functions.

**Implementation skills**

On the practical side, the course builds fluency in PyTorch for building and training networks, implementing custom physics-informed losses with automatic differentiation, writing optimization loops by hand, and implementing classical algorithms such as soft thresholding for L1 regularization. It also stresses reproducible research practice: fixed seeds, multi-seed comparisons, matched compute budgets, and verifying that reported numbers come from the code that was submitted.

**Critical evaluation**

Finally, the course trains judgment about when SciML methods are worth using. That means assessing accuracy, convergence, cost, and failure modes against conventional solvers, distinguishing approximation error from optimization error, and recognizing when a result is a property of the method versus an artifact of a particular seed or training budget.

## Final project (in progress)

The capstone project applies SciML methods to power-systems planning: estimating how much large data-center load a transmission-constrained grid node can host when paired with on-site battery storage. It combines neural-operator surrogates, physics-embedded power-flow layers, conformal prediction for uncertainty quantification, and transfer learning across weather years.

## Acknowledgments

Course materials and structure are by Krishna Kumar, released under the MIT License. All assignment solutions in this repository are my own work.
