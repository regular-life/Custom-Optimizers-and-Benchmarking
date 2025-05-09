# Custom Optimizers and Benchmarking

A PyTorch‐based suite of advanced optimization algorithms and a systematic benchmarking framework on both deep‐learning and synthetic‐function tasks.

> **Course:** This work was completed as part of **CSE517: Applied Optimization Methods in Machine Learning** at IIIT-Delhi under Prof. Bapi Chatterjee in Winter 2025 Semester.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Layout](#repository-layout)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Implemented Optimizers](#implemented-optimizers)
6. [Benchmark Suite](#benchmark-suite)
7. [Report & Notebooks](#report--notebooks)
8. [Requirements](#requirements)

---

## Project Overview

This repository implements 13 optimization algorithms—including classical, adaptive, zero-order, proximal, and quasi-Newton variants—in PyTorch, and benchmarks them on:

* **Image classification** (MNIST, CIFAR-10) using a standard 9-layer CNN.
* **Synthetic optimization** over challenging nonconvex functions (e.g., bowl-shaped, valley-shaped, plate-shaped, steep ridges, multiple local minima).

Our goal is to compare convergence behavior, final objective values, and hyperparameter robustness across both controlled synthetic landscapes and real-world vision tasks.

---

## Repository Layout

```
├── README.md
├── report.pdf
├── requirements.txt
├── main.ipynb               # End-to-end benchmarking pipeline (images + synthetic)
├── interpret.ipynb          # Visualization & result analysis
├── results/                 # Saved training logs, metrics, and plots
│   ├── cifar10/
│   ├── mnist/
│   └── functions/
├── self_benchmarks/         # Definitions of synthetic benchmark functions
│   ├── bowl_shaped.py
│   ├── valley_shaped.py
│   ├── plate_shaped.py
│   ├── steep_ridges.py
│   ├── multiple_local_minima.py
│   └── others.py
├── self_optimizers/         # Custom PyTorch optimizer implementations
│   ├── adam_qn.py
│   ├── amsgrad_mirror.py
│   ├── proximal_hb.py
│   ├── prox_yogi.py
│   ├── rprop_momentum.py
│   ├── rprop_polyak.py
│   └── sgd_polyak_momentum.py
```

> **Note:** While individual benchmark scripts live under `self_benchmarks/` and optimizer classes under `self_optimizers/`, the *only* runnable pipeline is in `main.ipynb`.

---

## Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/regular-life/Custom-Optimizers-and-Benchmarking.git
   cd Custom-Optimizers-and-Benchmarking
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

All experiments—from model training on MNIST/CIFAR-10 to synthetic-function runs—are organized in the interactive notebook:

* **Open and run** `main.ipynb` in JupyterLab or VS Code.
* The notebook loads your custom optimizers from `self_optimizers/`, runs them on both vision datasets and each synthetic test function in `self_benchmarks/`, and saves metrics/plots under `results/`.
* Use parameters at the top of the notebook to switch optimizers, adjust learning rates, epochs, and select which benchmarks to execute.

For detailed visualization of results on the synthetic functions, open **`interpret.ipynb`** to reproduce the various curves.

---

## Implemented Optimizers

All optimizers inherit from `torch.optim.Optimizer` and are importable via:

```python
from self_optimizers import AdamQN, ProxYogi, ProximalHB, AMSGradMirror, RPropMomentum, RPropPolyak, SGDPolyakMomentum
```

* **First-order:** SGD, SGD + Polyak momentum
* **Adaptive:** Adam, AdamW, RMSprop, Yogi
* **Zero-order:** Rprop, Rprop + momentum, Rprop + Polyak averaging
* **Proximal/mirror:** ProxYogi, ProximalHB, AMSGradMirror
* **Quasi-Newton hybrid:** AdamQN

See each `.py` file in `self_optimizers/` for API details and update rules.

---

## Benchmark Suite

Synthetic functions are defined in `self_benchmarks/`:

* **Bowl-shaped** (`bowl_shaped.py`)
* **Valley-shaped** (`valley_shaped.py`)
* **Plate-shaped** (`plate_shaped.py`)
* **Steep ridges** (`steep_ridges.py`)
* **Multiple local minima** (`multiple_local_minima.py`)
* **Other standard tests** (`others.py`)

Each function module exposes a callable `evaluate(x)` returning both loss value and gradient.

---

## Report & Notebooks

* **`report.pdf`** — Full LaTeX write-up with methodology, equations, plots, and discussion.
* **`main.ipynb`** — Runs all benchmarks and saves outputs under `results/`.
* **`interpret.ipynb`** — Loads saved results and generates publication-quality figures.

---

## Requirements

All Python packages are listed in **`requirements.txt`**. Install via:

```bash
pip install -r requirements.txt
```

---

Enjoy exploring and extending these custom optimizers and benchmarks!
