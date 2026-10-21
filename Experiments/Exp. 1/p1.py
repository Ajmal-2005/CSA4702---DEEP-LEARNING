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

# === Module 2 : Representing the Grayscale Image as a Matrix ===
np.random.seed(42)
R = C = 8                                  # rows and columns of the image

row, col = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")

I = 40 + 9.0*row + 6.0*col                 # smooth background gradient
I[1:4, 4:7] += 95                          # a bright object in the scene
I[5:8, 0:3] -= 25                          # a dark region in the scene
I = np.clip(np.round(I + np.random.normal(0, 3, (R, C))), 0, 255).astype(int)

print("Image matrix I :", I.shape, " dtype :", I.dtype)
print("Intensity range : %d to %d" % (I.min(), I.max()))
print(I)

# === Module 3 : Brightness Adjustment by Matrix Addition ===
J = np.ones((R, C), dtype=int)             # all-ones matrix of the same order
k = 40                                     # brightness step

B = np.clip(I + k*J, 0, 255)               # brighten : addition of k*J
D = np.clip(I - k*J, 0, 255)               # darken   : subtraction of k*J

print("Brightened B = I + %d*J -> mean %.2f" % (k, B.mean()))
print("Darkened   D = I - %d*J -> mean %.2f" % (k, D.mean()))

# === Module 4 : Contrast Adjustment by Scalar Multiplication ===
c1, c2 = 1.4, 0.6                          # contrast gain and attenuation

S1 = np.clip(np.round(c1 * I), 0, 255).astype(int)      # stretched contrast
S2 = np.clip(np.round(c2 * I), 0, 255).astype(int)      # reduced contrast

print("Scalar %.1f * I -> mean %.2f , std %.2f" % (c1, S1.mean(), S1.std()))
print("Scalar %.1f * I -> mean %.2f , std %.2f" % (c2, S2.mean(), S2.std()))

# === Module 5 : Data Transformation by Transpose ===
T  = I.T                                   # transpose : rows become columns
TT = T.T                                   # transpose of the transpose

print("I shape :", I.shape, "  I^T shape :", T.shape)
print("(I^T)^T equals I :", np.array_equal(TT, I))
print(T)

# === Module 6 : Verifying the Algebraic Properties of the Operations ===
K = np.clip(np.round(np.random.uniform(0, 60, (R, C))), 0, 255).astype(int)

checks = [
    ("(I + K)^T = I^T + K^T",        np.array_equal((I + K).T,  I.T + K.T)),
    ("(c * I)^T = c * I^T",          np.allclose((c1 * I).T,    c1 * I.T)),
    ("(I^T)^T = I",                  np.array_equal(I.T.T,      I)),
    ("I + O = I   (O = zero matrix)",
     np.array_equal(I + np.zeros((R, C), int), I)),
    ("I + K = K + I  (commutative)", np.array_equal(I + K,      K + I)),
    ("1 * I = I   (identity scalar)", np.array_equal(1 * I,     I)),
]
for text, ok in checks:
    print("%-32s : %s" % (text, "Verified" if ok else "Failed"))

# === Module 7 : Statistics of the Original and Transformed Images ===
ops = [("Original  I",          I),
       ("Brightened  I + 40J",  B),
       ("Darkened  I - 40J",    D),
       ("Contrast  1.4 x I",    S1),
       ("Contrast  0.6 x I",    S2),
       ("Transpose  I^T",       T)]

for name, M in ops:
    print("%-22s shape %s  min %3d  max %3d  mean %6.2f  std %5.2f" % (
        name, M.shape, M.min(), M.max(), M.mean(), M.std()))

# === Module 8 : Visualisation of the Transformed Images ===
TITLES = ["Original  $I$", "Brightened  $I + 40J$", "Darkened  $I - 40J$",
          "Contrast  $1.4 \\times I$", "Contrast  $0.6 \\times I$",
          "Transpose  $I^T$"]

fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
for ax, (name, M), ttl in zip(axes.ravel(), ops, TITLES):
    ax.imshow(M, cmap="gray", vmin=0, vmax=255)
    ax.set_title(ttl, fontsize=9)
    ax.set_xticks(range(0, 8, 2));  ax.set_yticks(range(0, 8, 2))
    ax.tick_params(labelsize=7)
fig.suptitle("Grayscale image matrix under addition, scalar"
             " multiplication and transpose", fontsize=10)
fig.tight_layout()
fig.savefig("plot1_operations.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6.4, 3.3))
bins = np.arange(0, 271, 15)
for M, lab, cl in [(I, "Original  I", "#2F6DB5"),
                   (B, "Brightened  I + 40J", "#C0702A"),
                   (D, "Darkened  I - 40J", "#4E8F59")]:
    ax.hist(M.ravel(), bins=bins, histtype="step", lw=1.6, color=cl, label=lab)

ax.set_xlabel("Pixel intensity");  ax.set_ylabel("Number of pixels")
ax.set_xlim(0, 270);  ax.set_xticks(np.arange(0, 271, 30))
ax.grid(axis="y", alpha=0.55);  ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("plot2_histogram.png", dpi=300, bbox_inches="tight")

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


def mat_table(title, M):
    return {
        "title": title,
        "headers": ["Row"] + ["C%d" % j for j in range(C)],
        "rows": [
            ["R%d" % i] + ["%d" % v for v in M[i]] for i in range(M.shape[0])
        ],
    }


tables = [
    mat_table("Table 1 - Original Grayscale Image Matrix I (order 8 x 8)", I),
    mat_table(
        "Table 2 - Brightness Increased by Matrix Addition : B = I + 40J", B
    ),
    mat_table(
        "Table 3 - Brightness Reduced by Matrix Subtraction : D = I - 40J", D
    ),
    mat_table(
        "Table 4 - Contrast Stretched by Scalar Multiplication : 1.4 x I", S1
    ),
    mat_table(
        "Table 5 - Data Transformation by Transpose : Iᵀ (rows become columns)",
        T,
    ),
    {
        "title": "Table 6 - Statistical Comparison of the Original and "
                 "Transformed Images",
        "headers": [
            "S.No",
            "Operation",
            "Order",
            "Minimum",
            "Maximum",
            "Mean",
            "Std. Deviation",
        ],
        "rows": [
            [
                str(i + 1),
                name.replace("^T", "ᵀ").replace("x", "×"),
                "%d × %d" % M.shape,
                "%d" % M.min(),
                "%d" % M.max(),
                "%.2f" % M.mean(),
                "%.2f" % M.std(),
            ]
            for i, (name, M) in enumerate(ops)
        ],
    },
    {
        "title": "Table 7 - Verification of the Matrix Algebra Properties",
        "headers": ["S.No", "Property", "Result"],
        "rows": [
            [
                str(i + 1),
                t.replace("^T", "ᵀ").replace("*", "×"),
                "Verified" if ok else "Failed",
            ]
            for i, (t, ok) in enumerate(checks)
        ],
    },
]

for t in tables:
    show(t["title"], t["headers"], t["rows"])
