# Gradient Descent Optimization & Logistic Regression | Python

A modular implementation of **Gradient-Based Optimization** algorithms built from scratch. This project features a custom **Gradient Descent (GD)** solver with dynamic learning rates, used to train **L1/L2 Regularized Logistic Regression** models on the South Africa Heart Disease dataset.

## 🚀 Key Features
### 1. Custom Optimization Engine
* **Gradient Descent:** Full implementation of Batch Gradient Descent with support for callback functions to track convergence.
* **Dynamic Learning Rates:** Implemented **Fixed** and **Exponential Decay** learning rate schedulers to optimize convergence speed.
* **Modular Design:** Decoupled the **Solver** (GD), **Model** (Logistic Regression), and **Regularization** (L1/L2) into separate, interchangeable modules (Strategy Pattern).

### 2. Regularized Logistic Regression
* **Elastic Net Capabilities:** Supports both **L1 (Lasso)** and **L2 (Ridge)** penalties to handle sparse data and prevent overfitting.
* **ROC Analysis:** Custom implementation of Receiver Operating Characteristic (ROC) curve evaluation to select optimal decision thresholds ($\alpha$).
* **Cross-Validation:** Automated hyperparameter tuning ($\lambda$) to minimize test error using a custom K-Fold engine.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** NumPy (Linear Algebra), Pandas (Data Processing), Plotly (Interactive Graphs).
* **Concepts:** Convex Optimization, Gradient Descent, Regularization, ROC/AUC.

## 📂 Project Structure
* `gradient_descent.py`: The core optimization engine.
* `logistic_regression.py`: Classifier implementation using the custom solver.
* `learning_rate.py`: Classes for Fixed and Exponential decay schedules.
* `base_module.py`: Abstract base class defining the interface for differentiable modules.
* `modules.py`: Mathematical implementations of L1/L2 regularization terms and their gradients.
* `gradient_descent_investigation.py`: Main script for running experiments and visualizing descent paths.

## 🧠 Algorithmic Implementation
* **Jacobian Computation:** The `LogisticRegression` class manually computes the Jacobian (gradient) of the Cross-Entropy loss w.r.t weights.
* **Descent Path Visualization:** Visualizes the trajectory of weights during training to analyze convergence behavior across different learning rates.
