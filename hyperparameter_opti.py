from bayes_opt import BayesianOptimization
import subprocess
import re

# ──────────────────────────────────────────────────────────────────────────────
#  ACTIVATION FUNCTIONS (indexed for CLI)
# ──────────────────────────────────────────────────────────────────────────────
activation_list = [
    "relu",
    "sigmoid",
    "softplus",
    "softsign",
    "tanh",
    "selu",
    "elu",
    "exponential",
]


# ──────────────────────────────────────────────────────────────────────────────
#  OBJECTIVE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def nn_cl_bo(neurons, act_idx, lr, batch_size, epochs, dropout_rate):
    """
    Runs train.py with given hyperparameters and returns validation accuracy.
    CLI args to train.py (in order):
      neurons, activation_index, learning_rate, batch_size, epochs, dropout_rate
    """
    # Build command to call training script
    cmd = [
        "python",
        "train.py",
        str(int(neurons)),  # neurons → int
        str(int(act_idx)),  # activation index → int
        str(lr),  # learning rate → float
        str(int(batch_size)),  # batch size → int
        str(int(epochs)),  # epochs → int
        str(dropout_rate),  # dropout rate → float
    ]

    # Execute training and capture stdout
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout

    # Parse the final validation accuracy from the last epoch line: "val-acc xx.xxxx"
    matches = re.findall(r"val-acc ([0-9\.]+)", out)
    if matches:
        # Use the last reported validation accuracy
        return float(matches[-1])
    else:
        # Fallback if parsing fails
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  HYPERPARAMETER BOUNDS
# ──────────────────────────────────────────────────────────────────────────────
param_bounds = {
    "neurons": (8, 256),  # number of GroupKAN neurons
    "act_idx": (0, len(activation_list) - 1),
    "lr": (1e-6, 1e-2),  # learning rate
    "batch_size": (16, 128),  # batch size
    "epochs": (10, 50),  # training epochs
    "dropout_rate": (0.0, 0.6),  # dropout probability
}

# ──────────────────────────────────────────────────────────────────────────────
#  BAYESIAN OPTIMIZATION
# ──────────────────────────────────────────────────────────────────────────────
optimizer = BayesianOptimization(
    f=nn_cl_bo,
    pbounds=param_bounds,
    random_state=42,
)

# Initial exploration, then iterative optimization
optimizer.maximize(
    init_points=50,
    n_iter=200,
)

# ──────────────────────────────────────────────────────────────────────────────
#  RESULTS
# ──────────────────────────────────────────────────────────────────────────────
print("\nBest hyperparameters found:")
print(optimizer.max["params"])
print("Best validation accuracy:", optimizer.max["target"])
