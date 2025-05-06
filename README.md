# 🧠 Machine Learning Models from Scratch

This repository showcases a series of core machine learning models and training utilities implemented entirely from scratch using **NumPy** and **CVXOPT**. It covers both **Support Vector Machines (SVMs)** — in primal and dual forms — and a **modular two-layer neural network** framework. The implementation avoids using high-level machine learning libraries to provide a deeper understanding of how these models work under the hood.

---

## 🚀 Highlights

- ✅ Batch Gradient Descent for Soft-Margin SVM (Primal)
- ✅ Dual-Form SVM solved using Quadratic Programming (CVXOPT)
- ✅ Fully connected Two-Layer Neural Network (fc → ReLU → fc → Softmax)
- ✅ Modular training engine with custom optimization
- ✅ Gradient checking for debugging backpropagation
- ✅ All implemented with **NumPy only** (except CVXOPT for dual SVM)

---

## 📁 Project Structure

```
├── soft_margin_svm.py           # SVM (primal form with gradient descent)
├── cvxopt_svm.py                # Kernelized SVM using CVXOPT QP solver
├── two_layer_net.py             # Neural network module (forward/backward)
├── solver.py                    # Training loop engine for neural networks
├── optim.py                     # Optimizers (e.g., SGD)
├── gradient_check.py            # Numerical gradient checking utilities
├── *.ipynb                      # Jupyter notebooks demonstrating each model
```

---

## 📦 Components

### 🔹 1. Soft-Margin SVM (Primal Form)

**Files**: `soft_margin_svm.py`, `soft_margin_svm.ipynb`  
Implements a soft-margin SVM using **batch gradient descent** to minimize hinge loss with L2 regularization.

```python
W, b = svm_train_bgd(X_train, y_train, num_epochs=100, C=5.0, eta=0.001)
accuracy = svm_test(W, b, X_test, y_test)
```

✅ Vectorized gradient updates  
✅ Slack penalty (C) support  
✅ Returns trained weights & accuracy

---

### 🔹 2. CVXOPT SVM (Dual Form + Kernelized)

**Files**: `cvxopt_svm.py`, `cvxopt_svm.ipynb`  
Formulates the SVM as a **convex quadratic programming** problem and solves it using `cvxopt`.

```python
model = CVXOPTSVC(C=1.0, kernel='rbf', gamma=0.5)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
```

✅ Supports kernels: `'linear'`, `'rbf'`, `'poly'`, `'sigmoid'`  
✅ Solves dual QP with kernel trick  
✅ Supports margin visualization in notebook

---

### 🔹 3. Two-Layer Neural Network

**Files**: `two_layer_net.py`, `solver.py`, `optim.py`, `two_layer_net.ipynb`  
Implements a two-layer neural network with modular forward/backward passes and a training framework.

```python
model = TwoLayerNet(input_dim=784, hidden_dim=100, num_classes=10)
solver = Solver(model, data={
    'X_train': X_train, 'y_train': y_train,
    'X_val': X_val, 'y_val': y_val
}, update_rule='sgd', optim_config={'learning_rate': 1e-3})
solver.train()
```

🧱 Architecture: fc → ReLU → fc → Softmax  
🔁 Built-in training/validation tracking  
🧪 Compatible with gradient checking

---

### 🔹 4. Gradient Checking

**File**: `gradient_check.py`  
Implements numerical gradient computation via finite differences to verify backpropagation correctness.

```python
from gradient_check import eval_numerical_gradient
grad = eval_numerical_gradient(loss_func, param_array)
```

✅ Works on scalar, vector, or blob-style gradients  
✅ Helps debug forward/backward implementation

---

### 🔹 5. Optimization Utilities

**File**: `optim.py`  
Includes basic optimizers such as **stochastic gradient descent (SGD)**.

```python
from optim import sgd
next_w, config = sgd(w, dw, config={'learning_rate': 1e-2})
```

---

## 🧪 Summary Table

| File                | Description                                      |
|---------------------|--------------------------------------------------|
| `soft_margin_svm.py`| SVM using batch gradient descent (primal form)  |
| `cvxopt_svm.py`     | SVM with kernelization via QP (dual form)       |
| `two_layer_net.py`  | Modular two-layer neural network                |
| `solver.py`         | Training loop for neural network                |
| `optim.py`          | Optimizer utilities (SGD)                       |
| `gradient_check.py` | Finite difference gradient checker              |

---

## 📌 Requirements

- Python 3.7+
- NumPy
- [cvxopt](https://cvxopt.org/) (only for dual-form SVM)

```bash
pip install numpy cvxopt
```

---

## 💻 Getting Started

```bash
git clone https://github.com/your-username/ml-models-from-scratch.git
cd ml-models-from-scratch
```

Run Jupyter notebook for SVM or neural network demos:

```bash
jupyter notebook
```

Run SVM from script:

```bash
python soft_margin_svm.py
```

---

## 🎓 Educational Purpose

This project was developed to **explore and teach the core mechanics of machine learning models** — how gradients, optimization, and classification work at the lowest level. No high-level ML frameworks were used to encourage full control and deeper learning.

- Understand SVM primal vs. dual optimization  
- Visualize classification boundaries and loss  
- Debug gradients with numerical checks  
- Build and train neural networks manually

---

## 👨‍💻 Author

**Hao-Chun Shih (Oscar)**  
Master’s in Data Science, University of Michigan  
📧 oscar10408@gmail.com

---

## 🪪 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and share this code for personal or educational purposes.
