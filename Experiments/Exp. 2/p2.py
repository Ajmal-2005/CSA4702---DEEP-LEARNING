# === Module 1 : Importing the Required Packages ===
import numpy as np
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")   # print the tables safely
except Exception:
    pass
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"]          = "serif"
plt.rcParams["font.serif"]           = ["Cambria", "Caladea", "DejaVu Serif"]
plt.rcParams["axes.spines.top"]      = False
plt.rcParams["axes.spines.right"]    = False
plt.rcParams["legend.frameon"]       = False

# === Module 2 : Building the Retail Customer Dataset ===
np.random.seed(42)
N = 150                                           # number of customers

income    = np.clip(np.random.normal(6.5, 1.9, N), 2.2, 13.0)      # lakh / year
spending  = np.clip(0.42*income + np.random.normal(0, 0.28, N),
                    0.4, 6.0)               # lakh / year
frequency = np.clip(44 - 3.1*income + np.random.normal(0, 3.2, N),
                    2, 60)                 # visits / year

df = pd.DataFrame({
    "Customer ID"              : ["C%03d" % (i + 1) for i in range(N)],
    "Annual Income (Rs lakh)"  : np.round(income, 2),
    "Annual Spending (Rs lakh)": np.round(spending, 2),
    "Purchase Frequency (/yr)" : np.round(frequency, 1)})

FEATURES = ["Annual Income (Rs lakh)",
            "Annual Spending (Rs lakh)",
            "Purchase Frequency (/yr)"]
SHORT    = ["Income", "Spending", "Frequency"]

X = df[FEATURES].to_numpy(dtype=float)            # data matrix , shape (150, 3)
print("Data matrix X :", X.shape)
print(df.head(10).to_string(index=False))

# === Module 3 : Descriptive Statistics of the Three Features ===
desc = df[FEATURES].describe().round(3)
print(desc)

# === Module 4 : Mean Vector and the Mean-Centred Deviation Matrix ===
mean_vec = X.mean(axis=0)                         # mean vector , shape (3,)
Xc       = X - mean_vec                      # deviation matrix , shape (150, 3)

print("Mean vector      :", np.round(mean_vec, 3))
print("Column means of Xc :", np.round(Xc.mean(axis=0), 12))

# === Module 5 : Covariance Matrix from First Principles and with NumPy ===
n         = X.shape[0]
cov_manual = (Xc.T @ Xc) / (n - 1)                # S = (1/(n-1)) * Xc^T . Xc
cov_numpy  = np.cov(X, rowvar=False)              # same result from NumPy

print("Covariance matrix (manual) :\n", np.round(cov_manual, 4))
print("Matches np.cov :", np.allclose(cov_manual, cov_numpy))
print("Variances on the diagonal :", np.round(np.diag(cov_manual), 4))

# === Module 6 : Correlation Matrix and Strength of the Relationships ===
sd        = np.sqrt(np.diag(cov_manual))
corr      = cov_manual / np.outer(sd, sd)         # r_ij = s_ij / (s_i * s_j)

def strength(r):
    a = abs(r)
    return ("Very strong" if a >= 0.80 else "Strong" if a >= 0.60 else
            "Moderate"    if a >= 0.40 else
            "Weak"        if a >= 0.20 else "Negligible")

pairs = []
for i in range(3):
    for j in range(i + 1, 3):
        r = corr[i, j]
        pairs.append((SHORT[i], SHORT[j], cov_manual[i, j], r,
                      "Positive" if r > 0 else "Negative", strength(r)))
        print("%-10s vs %-10s  cov %8.4f  r %+6.3f  %-8s  %s" % pairs[-1])

print("Matches pandas .corr() :",
      np.allclose(corr, df[FEATURES].corr().to_numpy()))

# === Module 7 : Visualisation of the Pairwise Relationships ===
COLORS = ["#2F6DB5", "#C0702A", "#4E8F59"]
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
for ax, (a, b, cv, r, sign, st), cl in zip(axes, pairs, COLORS):
    xa, xb = X[:, SHORT.index(a)], X[:, SHORT.index(b)]
    ax.scatter(xa, xb, s=12, c=cl, alpha=0.85)
    m, c = np.polyfit(xa, xb, 1)
    xs = np.linspace(xa.min(), xa.max(), 50)
    ax.plot(xs, m*xs + c, color="#404040", lw=1.2, ls="--")
    ax.set_xlabel(a);  ax.set_ylabel(b)
    ax.set_title("%s vs %s   (r = %+.3f)" % (a, b, r), fontsize=9)
    ax.grid(alpha=0.5);  ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("plot1_scatter.png", dpi=300, bbox_inches="tight")

# === Module 8 : Visualisation of the Covariance and Correlation Maps ===
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
for ax, M, name, cmap, vlim in [
        (axes[0], cov_manual, "(a) Covariance matrix", "PuOr", None),
        (axes[1], corr,       "(b) Correlation matrix", "RdBu", 1.0)]:
    lim = vlim if vlim else np.abs(M).max()
    im  = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_xticks(range(3), SHORT, rotation=20, ha="right")
    ax.set_yticks(range(3), SHORT)
    ax.set_title(name, fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, "%.3f" % M[i, j], ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(M[i, j]) > 0.55*lim else "#1A1A1A")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig("plot2_covariance_map.png", dpi=300, bbox_inches="tight")

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


stat_names = [
    "Count",
    "Mean",
    "Std. Deviation",
    "Minimum",
    "25th Percentile",
    "Median (50%)",
    "75th Percentile",
    "Maximum",
]
tables = [
    {
        "title": "Table 1 - Retail Customer Dataset (First 10 Customers)",
        "headers": [
            "S.No",
            "Customer ID",
            "Annual Income (Rs lakh)",
            "Annual Spending (Rs lakh)",
            "Purchase Frequency (/yr)",
        ],
        "rows": [
            [
                str(i + 1),
                df["Customer ID"][i],
                "%.2f" % df[FEATURES[0]][i],
                "%.2f" % df[FEATURES[1]][i],
                "%.1f" % df[FEATURES[2]][i],
            ]
            for i in range(10)
        ],
    },
    {
        "title": "Table 2 - Descriptive Statistics of the Three Features",
        "headers": ["Statistic"] + FEATURES,
        "rows": [
            [stat_names[k]]
            + [
                "%.3f" % desc.iloc[k, c] if k else "%d" % desc.iloc[k, c]
                for c in range(3)
            ]
            for k in range(8)
        ],
    },
    {
        "title": "Table 3 - Covariance Matrix S (order 3 × 3)",
        "headers": ["Feature"] + SHORT,
        "rows": [
            [SHORT[i]] + ["%.4f" % cov_manual[i, j] for j in range(3)]
            for i in range(3)
        ],
    },
    {
        "title": "Table 4 - Correlation Matrix R (order 3 × 3)",
        "headers": ["Feature"] + SHORT,
        "rows": [
            [SHORT[i]] + ["%+.4f" % corr[i, j] for j in range(3)]
            for i in range(3)
        ],
    },
    {
        "title": "Table 5 - Relationship of every Feature Pair",
        "headers": [
            "S.No",
            "Feature Pair",
            "Covariance",
            "Correlation r",
            "Direction",
            "Strength",
        ],
        "rows": [
            [str(i + 1), "%s – %s" % (a, b), "%.4f" % cv, "%+.3f" % r, sign, st]
            for i, (a, b, cv, r, sign, st) in enumerate(pairs)
        ],
    },
    {
        "title": "Table 6 - Verification and Summary of the Analysis",
        "headers": ["Parameter", "Value"],
        "rows": [
            ["Number of customers (n)", "%d" % n],
            ["Order of the data matrix X", "%d × %d" % X.shape],
            ["Order of the covariance matrix S", "3 × 3"],
            ["Mean vector", "[%s]" % ", ".join("%.3f" % v for v in mean_vec)],
            ["Column means of the deviation matrix", "[0.000, 0.000, 0.000]"],
            ["Variance of Income", "%.4f" % cov_manual[0, 0]],
            ["Variance of Spending", "%.4f" % cov_manual[1, 1]],
            ["Variance of Frequency", "%.4f" % cov_manual[2, 2]],
            ["Manual S equals np.cov(X)", "Verified"],
            ["Manual R equals pandas .corr()", "Verified"],
            [
                "Strongest positive relationship",
                "Income – Spending (r = %+.3f)" % corr[0, 1],
            ],
            [
                "Strongest negative relationship",
                "Income – Frequency (r = %+.3f)" % corr[0, 2],
            ],
        ],
    },
]

for t in tables:
    show(t["title"], t["headers"], t["rows"])
