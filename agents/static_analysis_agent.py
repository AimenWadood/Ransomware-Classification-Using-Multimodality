import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# CONFIG
# -------------------------
CSV_PATH = "data_new/static_dataset.csv"
LABEL_COL = "Family"
LATENT_DIM = 64
EPOCHS = 100
BATCH_SIZE = 128
LAMBDA     = 0.5    # weight for CE loss       (from formula: MSE + λ·CE)
DELTA      = 0.3    # weight for contrastive loss
TEMPERATURE = 0.07  # contrastive temperature (lower = sharper separation)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -------------------------
# MODEL — Contrastive Autoencoder
# -------------------------
class ContrastiveAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, n_classes):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

        # Projection head for contrastive loss
        # Projects latent Z into a smaller space where contrastive
        # loss is applied — standard practice from SimCLR/SupCon
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        z        = self.encoder(x)
        x_recon  = self.decoder(z)
        logits   = self.classifier(z)
        z_proj   = self.projector(z)   # projected embedding for contrastive loss
        return z, x_recon, logits, z_proj


# -------------------------
# SUPERVISED CONTRASTIVE LOSS
# Directly pushes same-class points together and
# different-class points apart in the projected space
# -------------------------
def supervised_contrastive_loss(z_proj, labels, temperature=0.07):
    # L2 normalize so cosine similarity = dot product
    z = F.normalize(z_proj, dim=1)

    # Similarity matrix: (B, B)
    sim = torch.matmul(z, z.T) / temperature

    # Mask: 1 where same class, 0 elsewhere
    labels = labels.unsqueeze(1)
    mask = (labels == labels.T).float()
    mask.fill_diagonal_(0)   # exclude self-similarity

    # For numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # Exp similarities, zero out diagonal
    exp_sim = torch.exp(sim)
    exp_sim_no_diag = exp_sim * (1 - torch.eye(exp_sim.size(0), device=z.device))

    # Log probability for each positive pair
    log_prob = sim - torch.log(exp_sim_no_diag.sum(dim=1, keepdim=True) + 1e-8)

    # Average only over positive pairs
    n_positives = mask.sum(dim=1)
    loss = -(mask * log_prob).sum(dim=1) / (n_positives + 1e-8)

    # Only include samples that have at least one positive pair
    valid = n_positives > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z.device)

    return loss[valid].mean()


# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(CSV_PATH)

y_raw = df[LABEL_COL].values
X = df.drop(columns=[LABEL_COL])

le = LabelEncoder()
y_enc = le.fit_transform(y_raw)
n_classes = len(le.classes_)

print(f"\nClasses found ({n_classes} total):")
for i, name in enumerate(le.classes_):
    count = (y_enc == i).sum()
    print(f"  {i} → {name}  ({count} samples)")

X = X.select_dtypes(include=[np.number]).values
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_enc, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
input_dim = X_tensor.shape[1]
print(f"\nInput dim: {input_dim} | Samples: {len(X_tensor)}")


# -------------------------
# CLASS-WEIGHTED CE LOSS
# handles class imbalance (benign >> ransomware families)
# -------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_enc),
    y=y_enc
)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)


# -------------------------
# TRAIN
# -------------------------
model     = ContrastiveAutoencoder(input_dim, LATENT_DIM, n_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

mse_fn = nn.MSELoss()
ce_fn  = nn.CrossEntropyLoss(weight=weights_tensor)   # class-weighted

print("\nTraining...\n")
model.train()

for epoch in range(EPOCHS):
    total, r_tot, c_tot, con_tot = 0, 0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        z, recon, logits, z_proj = model(xb)

        recon_loss = mse_fn(recon, xb)
        cls_loss   = ce_fn(logits, yb)
        con_loss   = supervised_contrastive_loss(z_proj, yb, TEMPERATURE)

        # Total loss — matches formula: MSE + λ·CE + δ·SupCon
        loss = recon_loss + LAMBDA * cls_loss + DELTA * con_loss

        loss.backward()
        optimizer.step()

        total   += loss.item()
        r_tot   += recon_loss.item()
        c_tot   += cls_loss.item()
        con_tot += con_loss.item()

    scheduler.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Total {total:.3f} | "
              f"Recon {r_tot:.3f} | "
              f"CE {c_tot:.3f} | "
              f"Contrastive {con_tot:.3f}")


# -------------------------
# EXTRACT EMBEDDINGS
# -------------------------
model.eval()
with torch.no_grad():
    Z = model.encoder(X_tensor.to(device)).cpu().numpy()

print(f"\nEmbeddings shape: {Z.shape}")


# -------------------------
# t-SNE
# -------------------------
print("Running t-SNE...")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    max_iter=2000,
    learning_rate="auto",
    init="pca"
)
Z_2d = tsne.fit_transform(Z)


# -------------------------
# PLOT
# -------------------------
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(
    Z_2d[:, 0],
    Z_2d[:, 1],
    c=y_enc,
    cmap="viridis",
    s=8,
    alpha=0.8,
    vmin=0,
    vmax=n_classes - 1
)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Ransomware Family", rotation=270, labelpad=15)
cbar.set_ticks(np.arange(n_classes))
cbar.set_ticklabels(np.arange(n_classes))

ax.set_title("t-SNE — After Lantent Embeddings")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
plt.tight_layout()
plt.savefig("tsne_after_static.png", dpi=300)
plt.show()

print("\nSaved: tsne_after_static.png")
print("\nClass reference:")
for i, name in enumerate(le.classes_):
    print(f"  {i} → {name}")