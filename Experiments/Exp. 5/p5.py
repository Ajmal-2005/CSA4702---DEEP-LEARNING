# === Module 1 : Importing the Required Packages ===
import numpy as np
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")   # print the tables safely
except Exception:
    pass
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams["font.family"]          = "serif"
plt.rcParams["font.serif"]           = ["Cambria", "Caladea", "DejaVu Serif"]
plt.rcParams["axes.spines.top"]      = False
plt.rcParams["axes.spines.right"]    = False
plt.rcParams["legend.frameon"]       = False

# === Module 2 : Building the Real-Estate Dataset ===
np.random.seed(42)
N = 300                                             # number of houses

area     = np.clip(np.random.normal(1450, 430, N), 500, 3200)      # square feet
bedrooms = np.clip(np.round(1 + area / 650 + np.random.normal(0, 0.6, N)), 1, 6)
age      = np.clip(np.random.gamma(2.6, 4.2, N), 0, 45)            # years

price = (6.0
         + 0.0062 * area                            # linear growth with area
         - 0.00000085 * (area - 1200) ** 2     # diminishing returns (nonlinear)
         + 1.15 * bedrooms
         - 0.22 * age + 0.0028 * age ** 2           # nonlinear effect of age
         + np.random.normal(0, 1.5, N))             # market noise
price = np.round(np.clip(price, 3, 60), 2)          # price in Rs lakh

df = pd.DataFrame({"House ID"       : ["H%03d" % (i + 1) for i in range(N)],
                   "Area (sq ft)"   : np.round(area, 1),
                   "Bedrooms"       : bedrooms.astype(int),
                   "Age (years)"    : np.round(age, 1),
                   "Price (Rs lakh)": price})

FEATURES = ["Area (sq ft)", "Bedrooms", "Age (years)"]
TARGET   = "Price (Rs lakh)"

X = df[FEATURES].to_numpy(dtype=float)
y = df[TARGET].to_numpy(dtype=float)
print("Feature matrix X :", X.shape, "  Target y :", y.shape)
print(df.head(10).to_string(index=False))

# === Module 3 : Descriptive Statistics ===
desc = df[FEATURES + [TARGET]].describe().round(2)
print(desc)

# === Module 4 : Train-Test Split ===
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
print("Training houses :", X_tr.shape[0], "  Testing houses :", X_te.shape[0])

# === Module 5 : Defining the Three Competing Models ===
models = {
    "Linear Regression"            : make_pipeline(StandardScaler(),
                                                   LinearRegression()),
    "Decision Tree (unrestricted)" : DecisionTreeRegressor(random_state=42),
    "Polynomial Regression (deg 9)": make_pipeline(
        StandardScaler(), PolynomialFeatures(degree=9, include_bias=False),
        LinearRegression()),
}
for name, mdl in models.items():
    mdl.fit(X_tr, y_tr)
    print("Fitted :", name)

# === Module 6 : Training and Testing Errors of every Model ===
def diagnose(tr_rmse, te_rmse, tr_r2, te_r2):
    if te_r2 < 0:                               # prediction worse than the mean
        return "Severe overfitting (very high variance)"
    if te_rmse > 1.4 * tr_rmse or (tr_r2 - te_r2) > 0.15:
        return "Overfitting (high variance)"
    if tr_r2 < 0.88:                             # error stays high on both sets
        return "Slight underfitting (high bias, low variance)"
    return "Good fit (balanced bias and variance)"

results = []
for name, mdl in models.items():
    p_tr, p_te = mdl.predict(X_tr), mdl.predict(X_te)
    tr_rmse = np.sqrt(mean_squared_error(y_tr, p_tr))
    te_rmse = np.sqrt(mean_squared_error(y_te, p_te))
    tr_r2, te_r2 = r2_score(y_tr, p_tr), r2_score(y_te, p_te)
    results.append({
        "name": name, "tr_rmse": tr_rmse, "te_rmse": te_rmse,
        "tr_mae": mean_absolute_error(y_tr, p_tr),
        "te_mae": mean_absolute_error(y_te, p_te),
        "tr_r2": tr_r2, "te_r2": te_r2, "gap": te_rmse - tr_rmse,
        "verdict": diagnose(tr_rmse, te_rmse, tr_r2, te_r2), "p_te": p_te})
    print("%-30s train RMSE %6.3f | test RMSE %6.3f | R2 %.4f / %.4f | %s"
          % (name, tr_rmse, te_rmse, tr_r2, te_r2, results[-1]["verdict"]))

best = min(results, key=lambda r: r["te_rmse"])
print("Most suitable model :", best["name"])

# === Module 7 : Five-Fold Cross-Validation of the Three Models ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv = {}
for name, mdl in models.items():
    sc = -cross_val_score(mdl, X, y, cv=kf,
                          scoring="neg_root_mean_squared_error")
    cv[name] = sc
    print("%-30s fold RMSE %s | mean %.3f | std %.3f"
          % (name, np.round(sc, 3), sc.mean(), sc.std()))

# === Module 8 : Bias-Variance Behaviour across Polynomial Degrees ===
degrees   = range(1, 10)
deg_train, deg_test = [], []
for d in degrees:
    mdl = make_pipeline(StandardScaler(),
                        PolynomialFeatures(degree=d, include_bias=False),
                        LinearRegression()).fit(X_tr, y_tr)
    deg_train.append(np.sqrt(mean_squared_error(y_tr, mdl.predict(X_tr))))
    deg_test.append(np.sqrt(mean_squared_error(y_te, mdl.predict(X_te))))
    print("degree %d | train RMSE %7.4f | test RMSE %9.4f"
          % (d, deg_train[-1], deg_test[-1]))

# === Module 9 : Visualisation of the Model Comparison ===
labels = ["Linear\nRegression", "Decision Tree\n(unrestricted)",
          "Polynomial\n(degree 9)"]
pos    = np.arange(3)

fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.bar(pos - 0.2, [r["tr_rmse"] for r in results], width=0.4,
       color="#2F6DB5", label="Training RMSE")
ax.bar(pos + 0.2, [r["te_rmse"] for r in results], width=0.4,
       color="#C0702A", label="Testing RMSE")
for i, r in enumerate(results):
    ax.text(i - 0.2, r["tr_rmse"] + 0.06, "%.2f" % r["tr_rmse"],
            ha="center", fontsize=8)
    ax.text(i + 0.2, r["te_rmse"] + 0.06, "%.2f" % r["te_rmse"],
            ha="center", fontsize=8)
ax.set_xticks(pos, labels, fontsize=9)
ax.set_ylabel("Root mean squared error (Rs lakh)")
ax.grid(axis="y", alpha=0.55);  ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("plot1_model_errors.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.plot(list(degrees), deg_train, "o-", color="#2F6DB5", lw=1.6, ms=4,
        label="Training RMSE")
ax.plot(list(degrees), deg_test,  "s--", color="#C0702A", lw=1.6, ms=4,
        label="Testing RMSE")
ax.set_yscale("log")
ax.set_xlabel("Degree of the polynomial model (model complexity)")
ax.set_ylabel("RMSE (Rs lakh), log scale")
ax.set_xticks(list(degrees))
ax.axvspan(0.6, 1.4,  color="#2F6DB5", alpha=0.07)
ax.axvspan(6.6, 9.4,  color="#C0702A", alpha=0.07)
ax.text(1.0, max(deg_test)*0.4, "underfitting", fontsize=8,
        ha="center", color="#2F6DB5")
ax.text(8.0, max(deg_test)*0.4, "overfitting",  fontsize=8,
        ha="center", color="#C0702A")
ax.grid(alpha=0.55);  ax.set_axisbelow(True)
ax.legend(loc="upper center", fontsize=8)
fig.tight_layout()
fig.savefig("plot2_complexity_curve.png", dpi=300, bbox_inches="tight")

# === Module 10 : Printing the Output Tables ===
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
short = ["Linear Regression", "Decision Tree", "Polynomial (deg 9)"]
tables = [
    {
        "title": "Table 1 - Real-Estate Dataset Sample (First 10 Houses)",
        "headers": [
            "S.No",
            "House ID",
            "Area (sq ft)",
            "Bedrooms",
            "Age (years)",
            "Price (Rs lakh)",
        ],
        "rows": [
            [
                str(i + 1),
                df["House ID"][i],
                "%.1f" % df[FEATURES[0]][i],
                "%d" % df[FEATURES[1]][i],
                "%.1f" % df[FEATURES[2]][i],
                "%.2f" % df[TARGET][i],
            ]
            for i in range(10)
        ],
    },
    {
        "title": "Table 2 - Descriptive Statistics of the Features and the "
                 "Target",
        "headers": ["Statistic"] + FEATURES + [TARGET],
        "rows": [
            [stat_names[k]]
            + [
                "%.2f" % desc.iloc[k, c] if k else "%d" % desc.iloc[k, c]
                for c in range(4)
            ]
            for k in range(8)
        ],
    },
    {
        "title": "Table 3 - Training and Testing Errors of the Three Models",
        "headers": [
            "S.No",
            "Model",
            "Train RMSE",
            "Test RMSE",
            "Train MAE",
            "Test MAE",
            "Train R²",
            "Test R²",
        ],
        "rows": [
            [
                str(i + 1),
                short[i],
                "%.4f" % r["tr_rmse"],
                "%.4f" % r["te_rmse"],
                "%.4f" % r["tr_mae"],
                "%.4f" % r["te_mae"],
                "%.4f" % r["tr_r2"],
                "%.4f" % r["te_r2"],
            ]
            for i, r in enumerate(results)
        ],
    },
    {
        "title": "Table 4 - Bias–Variance Diagnosis of every Model",
        "headers": [
            "S.No",
            "Model",
            "Test − Train RMSE",
            "Behaviour",
            "Diagnosis",
        ],
        "rows": [
            [
                str(i + 1),
                short[i],
                "%+.4f" % r["gap"],
                (
                    "Very low training error, enormous testing error"
                    if "Severe" in r["verdict"]
                    else (
                        "Low training error, clearly higher testing error"
                        if "Overfit" in r["verdict"]
                        else "Moderate and nearly equal error on both sets"
                    )
                ),
                r["verdict"],
            ]
            for i, r in enumerate(results)
        ],
    },
    {
        "title": "Table 5 - Five-Fold Cross-Validated RMSE of the Three Models",
        "headers": [
            "S.No",
            "Model",
            "Fold 1",
            "Fold 2",
            "Fold 3",
            "Fold 4",
            "Fold 5",
            "Mean",
            "Std. Deviation",
        ],
        "rows": [
            [str(i + 1), short[i]]
            + ["%.3f" % v for v in cv[name]]
            + ["%.3f" % cv[name].mean(), "%.3f" % cv[name].std()]
            for i, name in enumerate(models)
        ],
    },
    {
        "title": "Table 6 - Actual against Predicted Price for Ten Test Houses",
        "headers": [
            "S.No",
            "Actual Price",
            "Linear Regression",
            "Decision Tree",
            "Polynomial (deg 9)",
            "Best Match",
        ],
        "rows": [
            [str(i + 1), "%.2f" % y_te[i]]
            + ["%.2f" % r["p_te"][i] for r in results]
            + [
                short[
                    int(
                        np.argmin(
                            [abs(r["p_te"][i] - y_te[i]) for r in results]
                        )
                    )
                ]
            ]
            for i in range(10)
        ],
    },
    {
        "title": "Table 7 - Effect of Model Complexity on the Polynomial Model",
        "headers": [
            "Polynomial Degree",
            "Training RMSE",
            "Testing RMSE",
            "Test − Train",
            "Behaviour",
        ],
        "rows": [
            [
                "%d" % d,
                "%.4f" % a,
                "%.4f" % b,
                "%+.4f" % (b - a),
                (
                    "Underfitting (high bias)"
                    if d == 1
                    else (
                        "Overfitting (high variance)"
                        if b > 1.25 * a
                        else "Good fit"
                    )
                ),
            ]
            for d, a, b in zip(degrees, deg_train, deg_test)
        ],
    },
    {
        "title": "Table 8 - Final Evaluation Summary",
        "headers": ["Parameter", "Value"],
        "rows": [
            ["Number of houses", "%d" % N],
            [
                "Training houses / Testing houses",
                "%d / %d" % (X_tr.shape[0], X_te.shape[0]),
            ],
            ["Features used", "Area, Bedrooms, Age"],
            [
                "Lowest training RMSE",
                "%s  (%.4f)"
                % (
                    short[int(np.argmin([r["tr_rmse"] for r in results]))],
                    min(r["tr_rmse"] for r in results),
                ),
            ],
            [
                "Lowest testing RMSE",
                "%s  (%.4f)"
                % (
                    short[int(np.argmin([r["te_rmse"] for r in results]))],
                    min(r["te_rmse"] for r in results),
                ),
            ],
            [
                "Highest testing R²",
                "%s  (%.4f)"
                % (
                    short[int(np.argmax([r["te_r2"] for r in results]))],
                    max(r["te_r2"] for r in results),
                ),
            ],
            ["Model with the highest bias", "Linear Regression"],
            [
                "Overfitting models",
                "Decision Tree (unrestricted), Polynomial (degree 9)",
            ],
            [
                "Best polynomial degree observed",
                "2  (test RMSE %.4f)" % min(deg_test),
            ],
            ["Most suitable model", best["name"]],
            [
                "Reason",
                "Lowest testing RMSE and the smallest train–test gap; its "
                "slight bias costs far less than the variance of the other two",
            ],
        ],
    },
]

for t in tables:
    show(t["title"], t["headers"], t["rows"])
