# === Module 1 : Importing the Required Packages ===
import numpy as np
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")   # print the tables safely
except Exception:
    pass
from scipy.special import softmax
from scipy.stats import entropy
import matplotlib.pyplot as plt

plt.rcParams["font.family"]          = "serif"
plt.rcParams["font.serif"]           = ["Cambria", "Caladea", "DejaVu Serif"]
plt.rcParams["axes.spines.top"]      = False
plt.rcParams["axes.spines.right"]    = False
plt.rcParams["legend.frameon"]       = False

# === Module 2 : Defining the Object Categories and the True Labels ===
np.random.seed(42)
CLASSES = ["Cat", "Dog", "Car", "Bird", "Aeroplane"]
K       = len(CLASSES)                            # number of object categories
M       = 10                                     # number of images in the batch

y_true  = np.array([0, 2, 1, 4, 3, 2, 0, 3, 1, 4])        # true class index
Y       = np.eye(K)[y_true]                           # one-hot matrix , (10, 5)

print("Categories :", CLASSES)
print("True class index of every image :", y_true)
print("One-hot label matrix Y :", Y.shape)
print(Y.astype(int))

# === Module 3 : Converting the Network Scores into Probabilities ===
logits = np.array([
    [ 4.2,  1.1, -1.0,  0.6, -0.8],   # confident and correct
    [-0.9,  0.2,  3.8, -1.1,  0.4],   # confident and correct
    [ 1.6,  2.0, -0.7,  0.3, -0.5],   # correct but not confident
    [-0.6,  0.1, -0.2,  1.0,  2.9],   # confident and correct
    [ 0.2, -0.4,  0.5,  2.6,  0.9],   # confident and correct
    [ 1.9, -0.3,  1.7,  0.2, -0.6],   # wrong : Cat preferred over Car
    [ 2.8,  1.4, -0.9,  0.1, -0.7],   # confident and correct
    [ 0.4,  0.9,  0.3,  1.1,  0.7],   # uncertain but correct
    [ 2.4, -0.2,  0.5,  0.8, -0.3],   # wrong : Cat preferred over Dog
    [-0.5,  0.3,  0.7,  1.2,  3.4]])  # confident and correct

P = softmax(logits, axis=1)                       # predicted probability matrix
print("Probability matrix P :", P.shape)
print("Row sums of P :", np.round(P.sum(axis=1), 10))
print(np.round(P, 4))

# === Module 4 : Softmax Computation for a Single Image ===
z   = logits[0]
ez  = np.exp(z - z.max())                      # shifted for numerical stability
p0  = ez / ez.sum()
print("Image 1 logits        :", np.round(z, 3))
print("Image 1 exponentials  :", np.round(ez, 6))
print("Image 1 probabilities :", np.round(p0, 6), " sum :", p0.sum())

# === Module 5 : Categorical Cross-Entropy of every Image ===
EPS   = 1e-12
P_cl  = np.clip(P, EPS, 1.0)
losses = -np.sum(Y * np.log(P_cl), axis=1)        # L_i = - sum_k y_ik log p_ik

p_true = P[np.arange(M), y_true]           # probability given to the true class
y_pred = P.argmax(axis=1)                         # predicted class index

for i in range(M):
    print("Image %2d | true %-9s | predicted %-9s | p %.4f | loss %.4f | %s" % (
        i + 1, CLASSES[y_true[i]], CLASSES[y_pred[i]], p_true[i], losses[i],
        "correct" if y_pred[i] == y_true[i] else "wrong"))

# === Module 6 : Batch Loss, Accuracy and Verification ===
batch_loss = losses.mean()              # categorical cross-entropy of the batch
accuracy   = np.mean(y_pred == y_true)

print("Total cross-entropy of the batch : %.6f" % losses.sum())
print("Mean categorical cross-entropy   : %.6f" % batch_loss)
print("Classification accuracy          : %.2f %%" % (100 * accuracy))

simple_check = -np.mean(np.log(p_true))       # same as the mean loss
kl_check     = np.mean([entropy(Y[i], P_cl[i]) for i in range(M)])
print("Loss from -log(p_true) only : %.6f" % simple_check)
print("Mean KL divergence  D(Y||P) : %.6f" % kl_check)
print("Entropy of a one-hot label  : %.6f" % entropy(Y[0]))

# === Module 7 : Effect of the Predicted Confidence on the Loss ===
conf = np.array([0.99, 0.90, 0.70, 0.50, 0.30, 0.10, 0.01])
conf_loss = -np.log(conf)
for c, l in zip(conf, conf_loss):
    print("p(true class) = %.2f -> loss = %.4f" % (c, l))

# === Module 8 : Visualisation of the Loss ===
fig, ax = plt.subplots(figsize=(6.6, 3.3))
ok = y_pred == y_true
ax.bar(np.arange(1, M + 1)[ok],  losses[ok],  color="#2F6DB5",
       label="Correctly classified")
ax.bar(np.arange(1, M + 1)[~ok], losses[~ok], color="#C0702A",
       label="Misclassified")
ax.axhline(batch_loss, color="#4E8F59", lw=1.3, ls="--")
ax.text(0.7, batch_loss + 0.08, "batch mean = %.4f" % batch_loss,
        fontsize=8, color="#4E8F59")
ax.set_xlabel("Image number in the batch")
ax.set_ylabel("Categorical cross-entropy loss")
ax.set_xticks(np.arange(1, M + 1))
ax.grid(axis="y", alpha=0.55);  ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("plot1_per_image_loss.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6.6, 3.3))
grid = np.linspace(0.01, 1.0, 400)
ax.plot(grid, -np.log(grid), color="#2F6DB5", lw=1.8,
        label="Loss curve  L = $-\\log\\,p$")
ax.scatter(p_true[ok],  losses[ok],  s=30, c="#4E8F59", zorder=3,
           label="Correctly classified images")
ax.scatter(p_true[~ok], losses[~ok], s=30, c="#C0702A", zorder=3,
           label="Misclassified images")
ax.set_xlabel("Probability assigned to the true category,  p")
ax.set_ylabel("Cross-entropy loss of the image")
ax.set_xlim(0, 1.02);  ax.set_ylim(0, 4.8)
ax.set_xticks(np.arange(0, 1.01, 0.1))
ax.grid(alpha=0.55);  ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("plot2_loss_curve.png", dpi=300, bbox_inches="tight")

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


tables = [
    {
        "title": "Table 1 - Softmax Conversion for Image 1 (Logits → "
                 "Probabilities)",
        "headers": [
            "S.No",
            "Category",
            "Logit zₖ",
            "exp(zₖ − zₘₐₓ)",
            "Probability pₖ",
            "One-hot yₖ",
        ],
        "rows": [
            [
                str(k + 1),
                CLASSES[k],
                "%+.2f" % z[k],
                "%.6f" % ez[k],
                "%.6f" % p0[k],
                "%d" % Y[0, k],
            ]
            for k in range(K)
        ]
        + [["—", "Total", "—", "%.6f" % ez.sum(), "%.6f" % p0.sum(), "1"]],
    },
    {
        "title": "Table 2 - Predicted Probability Matrix P of the Batch (order "
                 "10 × 5)",
        "headers": ["S.No", "Image"] + CLASSES + ["True Category"],
        "rows": [
            [str(i + 1), "IMG-%02d" % (i + 1)]
            + ["%.4f" % P[i, k] for k in range(K)]
            + [CLASSES[y_true[i]]]
            for i in range(M)
        ],
    },
    {
        "title": "Table 3 - Categorical Cross-Entropy of every Image",
        "headers": [
            "S.No",
            "Image",
            "True Category",
            "Predicted Category",
            "p(true class)",
            "Loss  −log p",
            "Outcome",
        ],
        "rows": [
            [
                str(i + 1),
                "IMG-%02d" % (i + 1),
                CLASSES[y_true[i]],
                CLASSES[y_pred[i]],
                "%.4f" % p_true[i],
                "%.4f" % losses[i],
                "Correct" if y_pred[i] == y_true[i] else "Wrong",
            ]
            for i in range(M)
        ]
        + [["—", "Batch total", "—", "—", "—", "%.4f" % losses.sum(), "—"]],
    },
    {
        "title": "Table 4 - Loss Summary and Verification of the Batch",
        "headers": ["Parameter", "Value"],
        "rows": [
            ["Number of images in the batch (m)", "%d" % M],
            ["Number of object categories (K)", "%d" % K],
            ["Order of the probability matrix P", "%d × %d" % P.shape],
            ["Order of the one-hot label matrix Y", "%d × %d" % Y.shape],
            ["Row sums of P", "1.000000 for every image"],
            ["Total cross-entropy Σ Lᵢ", "%.6f" % losses.sum()],
            ["Mean categorical cross-entropy", "%.6f" % batch_loss],
            ["Same value from −mean(log p_true)", "%.6f" % simple_check],
            ["Mean KL divergence D(Y ‖ P)", "%.6f" % kl_check],
            ["Entropy of a one-hot label H(Y)", "%.6f" % entropy(Y[0])],
            ["Identity verified", "CE = H(Y) + D(Y ‖ P) = D(Y ‖ P)"],
            ["Classification accuracy", "%.2f %%" % (100 * accuracy)],
            [
                "Lowest loss (best prediction)",
                "IMG-%02d , %.4f" % (losses.argmin() + 1, losses.min()),
            ],
            [
                "Highest loss (worst prediction)",
                "IMG-%02d , %.4f" % (losses.argmax() + 1, losses.max()),
            ],
        ],
    },
    {
        "title": "Table 5 - Effect of the Predicted Confidence on the Loss",
        "headers": [
            "S.No",
            "Probability given to the True Category",
            "Loss  −log p",
            "Interpretation",
        ],
        "rows": [
            [
                str(i + 1),
                "%.2f" % c,
                "%.4f" % l,
                (
                    "Almost no penalty"
                    if c >= 0.9
                    else (
                        "Small penalty"
                        if c >= 0.7
                        else "Moderate penalty" if c >= 0.3 else "Heavy penalty"
                    )
                ),
            ]
            for i, (c, l) in enumerate(zip(conf, conf_loss))
        ],
    },
]

for t in tables:
    show(t["title"], t["headers"], t["rows"])
