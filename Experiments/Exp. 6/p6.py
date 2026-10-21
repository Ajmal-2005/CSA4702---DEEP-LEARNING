# === Module 1 : Importing the Required Packages ===
import numpy as np
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")   # print the tables safely
except Exception:
    pass
import matplotlib.pyplot as plt

plt.rcParams["font.family"]          = "serif"
plt.rcParams["font.serif"]           = ["Cambria", "Caladea", "DejaVu Serif"]
plt.rcParams["axes.spines.top"]      = False
plt.rcParams["axes.spines.right"]    = False
plt.rcParams["legend.frameon"]       = False

# === Module 2 : Building an Ill-Conditioned Training Problem ===
np.random.seed(42)
N = 200                                             # number of training samples

x1 = np.random.normal(0, 1.0, N)                     # well-scaled feature
x2 = np.random.normal(0, 0.07, N)                    # badly scaled feature
X  = np.column_stack([x1, x2])                   # input matrix , shape (200, 2)

W_TRUE = np.array([2.5, -4.0])                       # weights to be recovered
y = X @ W_TRUE                                    # target vector , shape (200,)

print("Input matrix X :", X.shape, "  Target y :", y.shape)
print("Standard deviation of the two features :", np.round(X.std(axis=0), 4))
print("True weights :", W_TRUE)

# === Module 3 : Loss Function, Gradient and the Conditioning of the Surface ===
def loss(w):
    """Mean squared error of the single neuron for the weight vector w."""
    r = X @ w - y
    return float(np.mean(r ** 2))

def grad(w):
    """Gradient of the mean squared error with respect to w."""
    r = X @ w - y
    return (2.0 / N) * (X.T @ r)

H      = (2.0 / N) * (X.T @ X)                     # Hessian of the loss surface
eig    = np.linalg.eigvalsh(H)
kappa  = eig.max() / eig.min()                       # condition number

print("Hessian H :\n", np.round(H, 4))
print("Eigenvalues of H :", np.round(eig, 6))
print("Condition number : %.2f  (large value -> elongated valley)" % kappa)
print("Stable learning-rate limit 2/L : %.4f" % (2.0 / eig.max()))

# === Module 4 : Standard Gradient Descent ===
W0      = np.array([-6.0, 8.0])                   # same starting point for both
LR      = 1.05                            # a rate that makes plain GD oscillate
EPOCHS  = 300

def gradient_descent(lr, epochs):
    w, hist, path = W0.copy(), [], [W0.copy()]
    for _ in range(epochs):
        g  = grad(w)
        w -= lr * g                                  # w := w - alpha * grad
        hist.append(loss(w));  path.append(w.copy())
    return w, np.array(hist), np.array(path)

w_gd, loss_gd, path_gd = gradient_descent(LR, EPOCHS)
print("Standard GD  -> final loss %.6e  final weights %s"
      % (loss_gd[-1], np.round(w_gd, 4)))

# === Module 5 : Gradient Descent with Momentum ===
def momentum_descent(lr, epochs, beta):
    w, v, hist, path = W0.copy(), np.zeros(2), [], [W0.copy()]
    for _ in range(epochs):
        g = grad(w)
        v = beta * v + g                    # velocity accumulates the gradients
        w -= lr * v                                  # w := w - alpha * v
        hist.append(loss(w));  path.append(w.copy())
    return w, np.array(hist), np.array(path)

BETA = 0.70
w_mo, loss_mo, path_mo = momentum_descent(LR, EPOCHS, BETA)
print("Momentum GD  -> final loss %.6e  final weights %s"
      % (loss_mo[-1], np.round(w_mo, 4)))

# === Module 6 : Measuring the Oscillation and the Speed of Convergence ===
def oscillations(path):
    """Times the step along the steep weight w1 reverses direction."""
    s = np.diff(path, axis=0)[:, 0]
    return int(np.sum(np.sign(s[1:]) != np.sign(s[:-1])))

def epochs_to(hist, tol):
    idx = np.where(hist < tol)[0]
    return int(idx[0] + 1) if idx.size else -1

TOL = 1e-3
for name, hist, path, w in [("Standard GD", loss_gd, path_gd, w_gd),
                            ("Momentum GD", loss_mo, path_mo, w_mo)]:
    print("%-12s | osc %4d | epochs to loss < %.0e : %4d | final %.3e"
          % (name, oscillations(path), TOL, epochs_to(hist, TOL), hist[-1]))

# === Module 7 : Effect of the Momentum Coefficient ===
betas, table = [0.0, 0.50, 0.70, 0.90, 0.95, 0.99], []
for b in betas:
    w_b, h_b, p_b = momentum_descent(LR, EPOCHS, b)
    table.append((b, h_b[-1], epochs_to(h_b, TOL), oscillations(p_b),
                  float(np.linalg.norm(w_b - W_TRUE))))
    print("beta %.2f | loss %.4e | epochs %4d | osc %4d | |w-w*| %.4f"
          % table[-1])

# === Module 8 : Visualisation of the Convergence ===
ep = np.arange(1, EPOCHS + 1)
fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.plot(ep, loss_gd, color="#C0702A", lw=1.5, label="Standard gradient descent")
ax.plot(ep, loss_mo, color="#2F6DB5", lw=1.5,
        label="Gradient descent with momentum ($\\beta$ = %.2f)" % BETA)
ax.set_yscale("log")
ax.set_xlabel("Epoch");  ax.set_ylabel("Mean squared error (log scale)")
ax.set_xlim(0, EPOCHS)
ax.grid(alpha=0.55);  ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("plot1_loss_curves.png", dpi=300, bbox_inches="tight")

g1 = np.linspace(-8, 11, 220)
g2 = np.linspace(-9, 10, 220)
G1, G2 = np.meshgrid(g1, g2)
Z = np.array([[loss(np.array([a, b])) for a in g1] for b in g2])

fig, ax = plt.subplots(figsize=(6.0, 4.4))
cs = ax.contour(G1, G2, Z, levels=np.geomspace(Z.min() + 1e-3, Z.max(), 16),
                colors="#B8B8B8", linewidths=0.6)
ax.plot(path_gd[:60, 0], path_gd[:60, 1], "-o", color="#C0702A", lw=1.0, ms=2.4,
        label="Standard gradient descent")
ax.plot(path_mo[:60, 0], path_mo[:60, 1], "-o", color="#2F6DB5", lw=1.0, ms=2.4,
        label="Gradient descent with momentum")
ax.scatter(*W0, s=42, c="#4E8F59", marker="s", zorder=4, label="Starting point")
ax.scatter(*W_TRUE, s=60, c="#1A1A1A", marker="*", zorder=4, label="Optimum")
ax.set_xlabel("Weight $w_1$");  ax.set_ylabel("Weight $w_2$")
ax.set_title("First 60 epochs on the loss surface", fontsize=10)
ax.set_xlim(-8, 11);  ax.set_ylim(-9, 10)
ax.legend(loc="lower left", fontsize=7.5)
fig.tight_layout()
fig.savefig("plot2_trajectory.png", dpi=300, bbox_inches="tight")

# === Module 9 : Printing the Output Tables ===
def show(title, headers, rows):
    """Print one output table with its columns aligned."""
    w = [
        max(len(str(h)), max(len(str(r[i])) for r in rows))
        for i, h in enumerate(headers)
    ]
    print("\n" + title)
    print(" | ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * x for x in w))
    for r in rows:
        print(" | ".join(str(v).ljust(w[i]) for i, v in enumerate(r)))


snap = [1, 5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300]
tables = [
    {
        "title": "Table 1 - Training Data Sample (First 10 Samples)",
        "headers": ["S.No", "Sample", "Feature x₁", "Feature x₂", "Target y"],
        "rows": [
            [
                str(i + 1),
                "S%03d" % (i + 1),
                "%+.4f" % X[i, 0],
                "%+.4f" % X[i, 1],
                "%+.4f" % y[i],
            ]
            for i in range(10)
        ],
    },
    {
        "title": "Table 2 - Conditioning of the Loss Surface (the Cause of the "
                 "Oscillation)",
        "headers": ["Parameter", "Value", "Remark"],
        "rows": [
            ["Standard deviation of x₁", "%.4f" % X[:, 0].std(), "Well scaled"],
            [
                "Standard deviation of x₂",
                "%.4f" % X[:, 1].std(),
                "Badly scaled",
            ],
            [
                "Hessian H = (2/m) XᵀX",
                "[[%.4f, %.4f], [%.4f, %.4f]]"
                % (H[0, 0], H[0, 1], H[1, 0], H[1, 1]),
                "Curvature of the loss",
            ],
            ["Larger eigenvalue λₘₐₓ", "%.6f" % eig.max(), "Steep direction"],
            ["Smaller eigenvalue λₘᵢₙ", "%.6f" % eig.min(), "Flat direction"],
            ["Condition number κ", "%.2f" % kappa, "Elongated narrow valley"],
            [
                "Stability limit 2 / λₘₐₓ",
                "%.4f" % (2.0 / eig.max()),
                "Learning rate used is %.2f" % LR,
            ],
            [
                "Starting point w₀",
                "[%.1f, %.1f]" % tuple(W0),
                "Same for both methods",
            ],
            ["Optimum w*", "[%.1f, %.1f]" % tuple(W_TRUE), "True weights"],
        ],
    },
    {
        "title": "Table 3 - Loss per Epoch : Standard Gradient Descent against "
                 "Momentum",
        "headers": [
            "S.No",
            "Epoch",
            "Standard GD Loss",
            "Momentum GD Loss",
            "Reduction Factor",
        ],
        "rows": [
            [
                str(i + 1),
                "%d" % e,
                "%.6f" % loss_gd[e - 1],
                "%.6f" % loss_mo[e - 1],
                (
                    "%.2f ×" % (loss_gd[e - 1] / loss_mo[e - 1])
                    if loss_mo[e - 1] > 0
                    else "—"
                ),
            ]
            for i, e in enumerate(snap)
        ],
    },
    {
        "title": "Table 4 - Convergence Summary of the Two Methods",
        "headers": [
            "Parameter",
            "Standard Gradient Descent",
            "Gradient Descent with Momentum",
        ],
        "rows": [
            ["Learning rate α", "%.2f" % LR, "%.2f" % LR],
            ["Momentum coefficient β", "— (not used)", "%.2f" % BETA],
            ["Epochs executed", "%d" % EPOCHS, "%d" % EPOCHS],
            ["Loss at epoch 1", "%.6f" % loss_gd[0], "%.6f" % loss_mo[0]],
            ["Loss at epoch 50", "%.6f" % loss_gd[49], "%.6f" % loss_mo[49]],
            ["Loss at epoch 300", "%.3e" % loss_gd[-1], "%.3e" % loss_mo[-1]],
            [
                "Epochs to reach loss < 10⁻³",
                (
                    "%d" % epochs_to(loss_gd, TOL)
                    if epochs_to(loss_gd, TOL) > 0
                    else "Not reached"
                ),
                (
                    "%d" % epochs_to(loss_mo, TOL)
                    if epochs_to(loss_mo, TOL) > 0
                    else "Not reached"
                ),
            ],
            [
                "Direction reversals (oscillations)",
                "%d" % oscillations(path_gd),
                "%d" % oscillations(path_mo),
            ],
            [
                "Epochs in which the loss rose",
                "%d" % int(np.sum(np.diff(loss_gd) > 0)),
                "%d" % int(np.sum(np.diff(loss_mo) > 0)),
            ],
            ["Final weight w₁", "%.4f" % w_gd[0], "%.4f" % w_mo[0]],
            ["Final weight w₂", "%.4f" % w_gd[1], "%.4f" % w_mo[1]],
            [
                "Distance from the optimum ‖w − w*‖",
                "%.4f" % np.linalg.norm(w_gd - W_TRUE),
                "%.4f" % np.linalg.norm(w_mo - W_TRUE),
            ],
            [
                "Observed behaviour",
                "Zig-zags across the narrow valley",
                "Damps the zig-zag and glides along the valley",
            ],
        ],
    },
    {
        "title": "Table 5 - Effect of the Momentum Coefficient β",
        "headers": [
            "S.No",
            "Momentum β",
            "Final Loss",
            "Epochs to Loss < 10⁻³",
            "Oscillations",
            "‖w − w*‖",
            "Remark",
        ],
        "rows": [
            [
                str(i + 1),
                "%.2f" % b,
                "%.4e" % fl,
                "%d" % ep_t if ep_t > 0 else "Not reached",
                "%d" % osc,
                "%.4f" % dist,
                (
                    "Plain gradient descent, does not converge"
                    if b == 0
                    else (
                        "Oscillation damped, tolerance reached"
                        if b <= 0.5
                        else (
                            "Best balance of damping and speed"
                            if b <= 0.7
                            else (
                                "Fastest loss decay but rings in w₁"
                                if b <= 0.9
                                else (
                                    "Heavy ringing, convergence slows"
                                    if b <= 0.95
                                    else "Unstable, diverges from the optimum"
                                )
                            )
                        )
                    )
                ),
            ]
            for i, (b, fl, ep_t, osc, dist) in enumerate(table)
        ],
    },
]

for t in tables:
    show(t["title"], t["headers"], t["rows"])
