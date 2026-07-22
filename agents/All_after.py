from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# ======================================================
# LOAD YOUR FEATURES + LABELS
# ======================================================

# Example:
# X = np.load("latent_features.npy")
# y = np.load("labels.npy")

# ------------------------------------------------------
# REMOVE THIS DEMO PART WHEN USING REAL DATA
# ------------------------------------------------------
np.random.seed(42)

n_features = 128

X0 = np.random.normal(0, 0.5, (5006, n_features))   # Benign
X1 = np.random.normal(5, 0.5, (1000, n_features))   # Cerber
X2 = np.random.normal(-5, 0.5, (1000, n_features))  # GandCrab
X3 = np.random.normal(8, 0.5, (1000, n_features))   # Maze
X4 = np.random.normal(-8, 0.5, (90, n_features))    # Shade
X5 = np.random.normal(12, 0.5, (1000, n_features))  # WannaCry

X = np.vstack([X0, X1, X2, X3, X4, X5])

y = np.array(
    [0]*5006 +
    [1]*1000 +
    [2]*1000 +
    [3]*1000 +
    [4]*90 +
    [5]*1000
)

# ======================================================
# STANDARDIZE
# ======================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================================
# t-SNE SETTINGS FOR SEPARATED CLUSTERS
# ======================================================

tsne = TSNE(
    n_components=2,
    perplexity=45,
    learning_rate=500,
    max_iter=5000,
    early_exaggeration=50,
    metric='cosine',
    init='pca',
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

# ======================================================
# PLOT
# ======================================================

plt.figure(figsize=(6,6))

scatter = plt.scatter(
    X_tsne[:,0],
    X_tsne[:,1],
    c=y,
    cmap='viridis',
    s=10,
    alpha=0.9
)

plt.title("t-SNE After Latent Learning", fontsize=12)

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

cbar = plt.colorbar(scatter)
cbar.set_label("Ransomware Family")

plt.tight_layout()

plt.savefig("clustered_tsne.png", dpi=300)

plt.show()