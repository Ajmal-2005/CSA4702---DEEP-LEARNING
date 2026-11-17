"""
Samurai vs Ninja Assassins - Conditional GAN (cGAN) data generator
Generates 1000 synthetic records labelled  TRAP (0)  or  ESCAPE (1).

Step 1: build a small "seed" dataset from a rule-based simulation
Step 2: train a conditional GAN on the seed data
Step 3: generate 1000 new records (500 trap + 500 escape) -> CSV
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

SAMURAI = ["strength", "durability", "speed", "skill", "stamina", "stealth"]
NINJA = ["teamwork", "agility", "momentum", "timing", "commander_control", "alertness"]
FEATURES = SAMURAI + NINJA
NOISE_DIM, N_CLASSES = 32, 2          # class 0 = TRAP, class 1 = ESCAPE


# ---------------------------------------------------------------
# 1. Seed data: simple simulation of the narrow path
# ---------------------------------------------------------------
def make_seed_data(n=6000):
    X = np.random.uniform(20, 100, size=(n, len(FEATURES)))
    df = pd.DataFrame(X, columns=FEATURES)

    # Samurai fighting power vs ninja combined threat
    fight = (0.25 * df.strength + 0.20 * df.durability + 0.20 * df.skill
             + 0.20 * df.stamina + 0.15 * df.speed)
    threat = (0.30 * df.teamwork + 0.20 * df.agility + 0.20 * df.momentum
              + 0.15 * df.timing + 0.15 * df.commander_control)

    # Path 1: sneak through (stealth + speed beat alertness)
    avoid_score = 0.6 * df.stealth + 0.4 * df.speed - df.alertness
    # Path 2: defeat everyone
    defeat_score = fight - threat

    best = np.maximum(avoid_score + 10, defeat_score)       # best of the 2 paths
    best = best + np.random.normal(0, 4, n)                  # luck / noise
    df["outcome"] = (best > 0).astype(int)                   # 1 = escape, 0 = trap
    return df


# ---------------------------------------------------------------
# 2. Conditional GAN
# ---------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM + N_CLASSES, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, len(FEATURES)), nn.Tanh())

    def forward(self, z, y):
        return self.net(torch.cat([z, y], dim=1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(FEATURES) + N_CLASSES, 256), nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


def train_gan(df, epochs=3000, batch=128):
    lo, hi = df[FEATURES].min().values, df[FEATURES].max().values
    X = 2 * (df[FEATURES].values - lo) / (hi - lo) - 1            # scale to [-1, 1]
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.eye(N_CLASSES)[torch.tensor(df["outcome"].values)]

    G, D = Generator(), Discriminator()
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    for ep in range(1, epochs + 1):
        idx = torch.randint(0, len(X), (batch,))
        real, y = X[idx], Y[idx]
        z = torch.randn(batch, NOISE_DIM)
        fake = G(z, y)

        # --- train discriminator
        d_loss = bce(D(real, y), torch.ones(batch, 1)) + \
                 bce(D(fake.detach(), y), torch.zeros(batch, 1))
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()

        # --- train generator
        g_loss = bce(D(fake, y), torch.ones(batch, 1))
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()

        if ep % 500 == 0:
            print(f"epoch {ep:5d} | D loss {d_loss.item():.3f} | G loss {g_loss.item():.3f}")
    return G, lo, hi


# ---------------------------------------------------------------
# 3. Generate 1000 records
# ---------------------------------------------------------------
def generate(G, lo, hi, n=1000):
    labels = np.array([0] * (n // 2) + [1] * (n - n // 2))        # 500 trap, 500 escape
    y = torch.eye(N_CLASSES)[torch.tensor(labels)]
    with torch.no_grad():
        fake = G(torch.randn(n, NOISE_DIM), y).numpy()
    data = (fake + 1) / 2 * (hi - lo) + lo                         # back to 20-100 scale
    out = pd.DataFrame(data, columns=FEATURES).round(2)
    out["outcome"] = np.where(labels == 1, "escape", "trap")

    # For escapes, tag which path the samurai most likely took
    avoid = 0.6 * out.stealth + 0.4 * out.speed - out.alertness + 10
    fight = (0.25 * out.strength + 0.20 * out.durability + 0.20 * out.skill
             + 0.20 * out.stamina + 0.15 * out.speed) - \
            (0.30 * out.teamwork + 0.20 * out.agility + 0.20 * out.momentum
             + 0.15 * out.timing + 0.15 * out.commander_control)
    path = np.where(avoid >= fight, "avoid_confrontation", "defeat_all_ninjas")
    out["escape_path"] = np.where(labels == 1, path, "none")
    return out.sample(frac=1, random_state=SEED).reset_index(drop=True)


if __name__ == "__main__":
    seed_df = make_seed_data()
    print("Seed data outcome ratio:\n", seed_df["outcome"].value_counts(normalize=True))
    G, lo, hi = train_gan(seed_df)
    result = generate(G, lo, hi, 1000)
    result.insert(0, "record_id", range(1, len(result) + 1))
    result.to_csv("samurai_ninja_trap_escape_1000.csv", index=False)
    print(result.head())
    print(result["outcome"].value_counts())
    print(result["escape_path"].value_counts())