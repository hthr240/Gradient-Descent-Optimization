
import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from base_module import BaseModule
from base_learning_rate import  BaseLR
from gradient_descent import GradientDescent
from learning_rate import FixedLR
from cross_validate import cross_validate
from loss_functions import misclassification_error

from sklearn.metrics import roc_curve, auc, zero_one_loss

# from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))

def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights_history = []

    def callback(solver, weights, val, grad, it, eta, delta):
        values.append(val)
        weights_history.append(weights.copy())

    return callback, values, weights_history

def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    
    l1_convergence, l2_convergence = {}, {}
    l1_norms_by_eta, l2_norms_by_eta = {}, {}
    all_l1_min, all_l2_min = [], []

    for eta in etas:

        # Initialize L1 and L2 modules
        l1 = L1(weights=init.copy())
        l2 = L2(weights=init.copy())

        # Initialize GD solvers for L1 and L2 modules
        callback_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
        callback_l2, values_l2, weights_l2 = get_gd_state_recorder_callback()

        # Create GradientDescent instances for L1 and L2 modules
        gd1 = GradientDescent(learning_rate=FixedLR(eta), callback=callback_l1)
        gd2 = GradientDescent(learning_rate=FixedLR(eta), callback=callback_l2)

        # Fit the L1 and L2 modules using the GradientDescent instances
        gd1.fit(l1,X=None, y=None)
        gd2.fit(l2,X=None, y=None) 

        # Plot the descent path for L1 and L2 modules
        l1_convergence[eta] = np.array(weights_l1)
        l2_convergence[eta] = np.array(weights_l2)

        # Store losses for final Plotly graph
        l1_norms = [np.linalg.norm(w, ord=1) for w in weights_l1]
        l2_norms = [np.linalg.norm(w, ord=2) for w in weights_l2]
        l1_norms_by_eta[eta] = l1_norms
        l2_norms_by_eta[eta] = l2_norms
        min_l1 = min(values_l1).item() if hasattr(min(values_l1), "item") else float(min(values_l1))
        min_l2 = min(values_l2).item() if hasattr(min(values_l2), "item") else float(min(values_l2))
        all_l1_min.append(min_l1)
        all_l2_min.append(min_l2)

        # Plot descent paths
        fig_l1 = plot_descent_path(L1, l1_convergence[eta], title=f"L1 module with η={eta}")
        fig_l2 = plot_descent_path(L2, l2_convergence[eta], title=f"L2 module with η={eta}")

        fig_l1.show()
        fig_l2.show()

    print(f"\nLowest L1 loss across all etas: {min(all_l1_min):.18f}")
    print(f"Lowest L2 loss across all etas: {min(all_l2_min):.18f}")

    # L1 norm convergence
    fig_l1_norm = go.Figure()
    for eta in etas:
        fig_l1_norm.add_trace(go.Scatter(
            x=list(range(len(l1_norms_by_eta[eta]))),
            y=l1_norms_by_eta[eta],
            mode='lines',
            name=f'η={eta}'
        ))
    fig_l1_norm.update_layout(
        title='L1 GD convergence for different learning rates',
        xaxis_title='Iteration',
        yaxis_title='||w||₁'
    )
    fig_l1_norm.show()

    # L2 norm convergence
    fig_l2_norm = go.Figure()
    for eta in etas:
        fig_l2_norm.add_trace(go.Scatter(
            x=list(range(len(l2_norms_by_eta[eta]))),
            y=l2_norms_by_eta[eta],
            mode='lines',
            name=f'η={eta}'
        ))
    fig_l2_norm.update_layout(
        title='L2 GD convergence for different learning rates',
        xaxis_title='Iteration',
        yaxis_title='||w||₂'
    )
    fig_l2_norm.show()
        
def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)

def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Fit logistic regression model with fixed learning rate and no regularization
    model = LogisticRegression(solver=GradientDescent(FixedLR(1e-4), max_iter=20000),
                               penalty="none", alpha=0.5)
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    probs = model.predict_proba(X_test.to_numpy())
    
    # Q5
    # Plotting convergence rate of logistic regression over SA heart disease data
    fpr, tpr, thresholds = roc_curve(y_test.to_numpy(), probs)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guessing', line=dict(dash='dash')))
    fig.update_layout(title=f'ROC Curve (AUC = {roc_auc:.2f})',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      width=800, height=600)
    fig.show()

    # Q6
    index = tpr - fpr
    best_idx = np.argmax(index)
    best_alpha = thresholds[best_idx]
    print(f"Optimal alpha (α*) = {best_alpha:.10f}")

    y_pred = (probs >= best_alpha).astype(int)
    test_error = np.mean(y_pred != y_test.to_numpy())
    print(f"Test error using α* = {test_error:.10f}")

    # Q7
    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lam_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    best_val_error = float('inf')
    best_lambda = None

    for lam in lam_values:
        model = LogisticRegression(
            solver=GradientDescent(FixedLR(1e-4), max_iter=20000),
            penalty="l1",
            lam=lam,
            alpha=0.5
        )

        train_score, val_score = cross_validate(
            model,
            X_train.to_numpy(),
            y_train.to_numpy(),
            scoring=lambda y_true, y_pred: misclassification_error(y_true, y_pred),
            cv=5
        )

        print(f"λ={lam:.3f} | Train Error={train_score:.10f} | Validation Error={val_score:.10f}")

        if val_score < best_val_error:
            best_val_error = val_score
            best_lambda = lam

    print(f"\nBest λ from cross-validation: {best_lambda}")

    # Fit model on full training set using best λ
    best_model = LogisticRegression(
        solver=GradientDescent(FixedLR(1e-4), max_iter=20000),
        penalty="l1",
        lam=best_lambda,
        alpha=0.5
    )
    best_model.fit(X_train.to_numpy(), y_train.to_numpy())
    y_test_pred = best_model.predict(X_test.to_numpy())
    test_error = misclassification_error(y_test.to_numpy(), y_test_pred)
    print(f"Test error with best λ = {test_error:.10f}")

    
    


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    fit_logistic_regression()
