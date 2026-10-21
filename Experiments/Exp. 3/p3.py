# === Module 1 : Importing the Required Packages ===
import numpy as np
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")   # print the tables safely
except Exception:
    pass
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams["font.family"]          = "serif"
plt.rcParams["font.serif"]           = ["Cambria", "Caladea", "DejaVu Serif"]
plt.rcParams["axes.spines.top"]      = False
plt.rcParams["axes.spines.right"]    = False
plt.rcParams["legend.frameon"]       = False

# === Module 2 : Recording the Defect Counts of the Production Batches ===
np.random.seed(42)
BATCHES   = 200                                  # number of batches inspected
LAMBDA_T  = 3.2                                 # true average defects per batch

defects = np.random.poisson(LAMBDA_T, BATCHES)   # defective items in each batch

print("Batches inspected :", BATCHES)
print("Defects in the first 20 batches :", defects[:20])
print("Total defective items :", defects.sum())

# === Module 3 : Choosing the Suitable Probability Distribution ===
mean_d = defects.mean()
var_d  = defects.var(ddof=1)
ratio  = var_d / mean_d

print("Sample mean     : %.4f" % mean_d)
print("Sample variance : %.4f" % var_d)
print("Variance / mean : %.4f  (close to 1 supports the Poisson model)" % ratio)

lam_hat = mean_d                         # maximum likelihood estimate of lambda
print("Estimated lambda (MLE) : %.4f" % lam_hat)

# === Module 4 : Frequency Distribution of the Observed Defect Counts ===
K_MAX  = 10
ks     = np.arange(0, K_MAX + 1)
obs    = np.array([np.sum(defects == k) for k in ks])
obs[-1] = np.sum(defects >= K_MAX)               # last class is "10 or more"

pmf    = stats.poisson.pmf(ks, lam_hat)
pmf[-1] = stats.poisson.sf(K_MAX - 1, lam_hat)   # tail probability P(X >= 10)
exp_f  = pmf * BATCHES

for k, o, p, e in zip(ks, obs, pmf, exp_f):
    print("k %2d | observed %3d | P(X=k) %.4f | expected %6.2f" % (k, o, p, e))

# === Module 5 : Probability of Observing a Specified Number of Defects ===
K_SPEC = 5                                     # the specified number of defects

queries = [
    ("Exactly %d defects (the specified number)" % K_SPEC, "P(X = %d)" % K_SPEC,
     stats.poisson.pmf(K_SPEC, lam_hat)),
    ("A defect-free batch",      "P(X = 0)",
     stats.poisson.pmf(0, lam_hat)),
    ("At most 2 defects",        "P(X ≤ 2)",
     stats.poisson.cdf(2, lam_hat)),
    ("At least 6 defects",       "P(X ≥ 6)",
     stats.poisson.sf(5, lam_hat)),
    ("Between 2 and 5 defects",  "P(2 ≤ X ≤ 5)",
     stats.poisson.cdf(5, lam_hat) - stats.poisson.cdf(1, lam_hat)),
    ("More than 8 defects",      "P(X > 8)",
     stats.poisson.sf(8, lam_hat)),
]
for desc, expr, p in queries:
    print("%-42s %-16s = %.6f  (%.3f %%)" % (desc, expr, p, 100*p))

rel_freq = np.mean(defects == K_SPEC)
print("Relative frequency of exactly %d defects : %.4f" % (K_SPEC, rel_freq))

# === Module 6 : Goodness-of-Fit Test for the Poisson Model ===
chi_sq = float(np.sum((obs - exp_f) ** 2 / exp_f))
dof    = len(ks) - 1 - 1                    # classes - 1 - estimated parameters
p_val  = stats.chi2.sf(chi_sq, dof)
crit   = stats.chi2.ppf(0.95, dof)

print("Chi-square statistic : %.4f" % chi_sq)
print("Degrees of freedom   : %d" % dof)
print("Critical value (5 %%) : %.4f" % crit)
print("p-value              : %.4f" % p_val)
print("Decision :", "Poisson accepted" if chi_sq < crit else "Poisson rejected")

# === Module 7 : Comparison with the Binomial Model ===
n_items = 500                                    # items produced in each batch
p_item  = lam_hat / n_items              # probability that an item is defective
bin_pmf = stats.binom.pmf(ks, n_items, p_item)

print("Binomial(n = %d, p = %.6f) mean : %.4f"
      % (n_items, p_item, n_items*p_item))
print("Maximum |Poisson - Binomial| over k = 0..10 : %.6f"
      % np.max(np.abs(pmf[:-1] - bin_pmf[:-1])))

# === Module 8 : Visualisation of the Fitted Distribution ===
fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.bar(ks - 0.18, obs / BATCHES, width=0.36, color="#2F6DB5",
       label="Observed relative frequency")
ax.bar(ks + 0.18, pmf, width=0.36, color="#C0702A",
       label="Poisson P(X = k),  $\\lambda$ = %.3f" % lam_hat)
ax.axvline(K_SPEC, color="#4E8F59", lw=1.3, ls="--")
ax.text(K_SPEC + 0.15, max(obs / BATCHES) * 0.92,
        "specified k = %d" % K_SPEC, fontsize=8, color="#4E8F59")
ax.set_xlabel("Number of defective products in a batch (k)")
ax.set_ylabel("Probability")
ax.set_xticks(ks, [str(k) for k in ks[:-1]] + ["10+"])
ax.grid(axis="y", alpha=0.55);  ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("plot1_pmf.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6.6, 3.4))
emp_cdf = np.cumsum(obs) / BATCHES
ax.step(ks, emp_cdf, where="mid", color="#2F6DB5", lw=1.6,
        label="Empirical CDF of the 200 batches")
ax.step(ks, np.cumsum(pmf), where="mid", color="#C0702A", lw=1.6, ls="--",
        label="Poisson CDF,  $\\lambda$ = %.3f" % lam_hat)
ax.scatter(ks, emp_cdf, s=14, c="#2F6DB5")
ax.set_xlabel("Number of defective products in a batch (k)")
ax.set_ylabel("Cumulative probability  P(X $\\leq$ k)")
ax.set_xticks(ks);  ax.set_ylim(0, 1.05)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.grid(alpha=0.55);  ax.set_axisbelow(True)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig("plot2_cdf.png", dpi=300, bbox_inches="tight")

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
        "title": "Table 1 - Recorded Defect Counts of the First 15 Batches",
        "headers": [
            "S.No",
            "Batch ID",
            "Defective Products",
            "Batch ID",
            "Defective Products",
        ],
        "rows": [
            [
                str(i + 1),
                "B%03d" % (i + 1),
                "%d" % defects[i],
                "B%03d" % (i + 16),
                "%d" % defects[i + 15],
            ]
            for i in range(15)
        ],
    },
    {
        "title": "Table 2 - Frequency Distribution and the Fitted Poisson "
                 "Probabilities",
        "headers": [
            "Defects k",
            "Observed Batches",
            "Observed Rel. Frequency",
            "Poisson P(X = k)",
            "Expected Batches",
        ],
        "rows": [
            [
                ("10 or more" if k == K_MAX else str(k)),
                "%d" % o,
                "%.4f" % (o / BATCHES),
                "%.4f" % p,
                "%.2f" % e,
            ]
            for k, o, p, e in zip(ks, obs, pmf, exp_f)
        ]
        + [
            [
                "Total",
                "%d" % obs.sum(),
                "%.4f" % (obs.sum() / BATCHES),
                "%.4f" % pmf.sum(),
                "%.2f" % exp_f.sum(),
            ]
        ],
    },
    {
        "title": "Table 3 - Parameter Estimation and Justification of the "
                 "Poisson Model",
        "headers": ["Parameter", "Value", "Remark"],
        "rows": [
            ["Number of batches (n)", "%d" % BATCHES, "Sample size"],
            [
                "Total defective products",
                "%d" % defects.sum(),
                "Sum of all batches",
            ],
            [
                "Minimum defects in a batch",
                "%d" % defects.min(),
                "Non-negative count",
            ],
            [
                "Maximum defects in a batch",
                "%d" % defects.max(),
                "Non-negative count",
            ],
            ["Sample mean", "%.4f" % mean_d, "Estimator of λ"],
            ["Sample variance", "%.4f" % var_d, "Estimator of λ"],
            [
                "Variance / mean ratio",
                "%.4f" % ratio,
                "Close to 1 → Poisson is suitable",
            ],
            ["Estimated λ (MLE)", "%.4f" % lam_hat, "λ̂ = sample mean"],
            [
                "Distribution adopted",
                "Poisson(λ = %.3f)" % lam_hat,
                "Counts of rare defects per batch",
            ],
        ],
    },
    {
        "title": "Table 4 - Required Probabilities from the Fitted Poisson "
                 "Model",
        "headers": [
            "S.No",
            "Probability Required",
            "Expression",
            "Value",
            "Percentage",
            "Expected Batches out of 200",
        ],
        "rows": [
            [
                str(i + 1),
                desc,
                expr,
                "%.6f" % p,
                "%.3f %%" % (100 * p),
                "%.1f" % (200 * p),
            ]
            for i, (desc, expr, p) in enumerate(queries)
        ],
    },
    {
        "title": "Table 5 - Chi-Square Goodness-of-Fit Test for the Poisson "
                 "Model",
        "headers": ["Parameter", "Value"],
        "rows": [
            [
                "Null hypothesis H₀",
                "The defect counts follow a Poisson distribution",
            ],
            ["Number of classes", "%d" % len(ks)],
            ["Estimated parameters", "1  (λ)"],
            ["Degrees of freedom", "%d" % dof],
            ["Chi-square statistic χ²", "%.4f" % chi_sq],
            ["Critical value at 5 % level", "%.4f" % crit],
            ["p-value", "%.4f" % p_val],
            ["Decision", "χ² < critical value → H₀ accepted"],
            ["Conclusion", "The Poisson model fits the recorded defect data"],
        ],
    },
    {
        "title": "Table 6 - Comparison of the Poisson and Binomial Models",
        "headers": [
            "Defects k",
            "Poisson P(X = k)",
            "Binomial P(X = k)",
            "Difference",
        ],
        "rows": [
            ["%d" % k, "%.6f" % p, "%.6f" % b, "%+.6f" % (p - b)]
            for k, p, b in zip(ks[:-1], pmf[:-1], bin_pmf[:-1])
        ],
    },
]

for t in tables:
    show(t["title"], t["headers"], t["rows"])
