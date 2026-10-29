# Optimization for Machine Learning

**Numerical optimization · Nesterov accelerated gradient · Neural-network learning-rate scaling**

This academic project explores two numerical optimization problems using **PyTorch**: accelerated gradient methods and choosing the gradient-descent step size as a neural network becomes wider.

> 📊 **The main notebook, [`homework.ipynb`](homework.ipynb), contains the experiments, figures and detailed discussion.**

## Part I - Nesterov acceleration and restart

Two implementations of Nesterov's accelerated gradient are compared on the **two-dimensional Rosenbrock function**:

- **Speed restart:** resets momentum when a speed-based criterion is triggered.
- **Vanishing friction:** acceleration without restart, with parameter `b = 3`.

**Reported finding:** with the hyperparameters explored in the notebook, vanishing friction gives faster initial progress but shows oscillations; speed restart is more regular and stable. This conclusion concerns the experimental conditions tested, not every possible optimization problem.

## Part II - Learning-rate scaling in neural networks

A one-hidden-layer **ReLU neural network** is trained using full-batch gradient descent and **mean squared error** to approximate a noisy sine function.

The experiments first examine **d = 16** hidden units, then compare widths **d ∈ {5, 20, 30, 50, 80}**. The aim is to estimate an empirical relationship between the preferred step size α and network width d:

**α*(d) ≈ ν d^γ**

The notebook reports **ν ≈ 0.147** and **γ ≈ −0.472**, giving:

**α*(d) ≈ 0.147 × d^(-0.472)**.

In these experiments, the selected learning rate decreases with increasing hidden-layer width. This is an **experimental fit**, not a universal law for neural networks.

## Results and visualizations

📈 **Open [`homework.ipynb`](homework.ipynb)** to see the convergence curves, optimizer comparisons, step-size experiments, figures and methodological interpretations. The [original assignment](homework_opt_ml.pdf) gives the precise study objectives.

## Files

| File | Description |
| --- | --- |
| [`homework.ipynb`](homework.ipynb) | Main documented notebook: implementations, experiments, figures and interpretations |
| [`train_nn.ipynb`](train_nn.ipynb) | Provided neural-network training code |
| [`homework_opt_ml.pdf`](homework_opt_ml.pdf) | Assignment instructions |
| [`requirements.txt`](requirements.txt) | Python dependencies |
| [`.gitignore`](.gitignore) | Local-file exclusions |

## Reproduction

Install dependencies in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open `homework.ipynb` and execute cells in order. Figures and metrics recorded in the notebook are from the submitted coursework; they have **not** been independently re-run during preparation of this repository.

## Scope and limitations

Numerical comparisons depend on initialization, optimization parameters, step-size search grids and training budget. The fitted scaling exponent describes the tested architecture and experimental setup.

**Tools:** Python · PyTorch · NumPy · Matplotlib · Jupyter · Automatic differentiation · Numerical optimization
