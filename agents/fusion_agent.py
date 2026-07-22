
import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

class FusionAgent:
    def __init__(self, fusion_mode='upsample', embed_dir="embeddings", seed=42):
        self.fusion_mode = fusion_mode
        self.valid_families = {0, 1, 2, 3, 4, 5, 6}
        self.embed_dir = embed_dir
        os.makedirs(self.embed_dir, exist_ok=True)
        self.rng = np.random.default_rng(seed)

        # caches
        self.Z_fused = None
        self.y_bin = None
        self.y_fam = None
        self.groups_static = None  # <-- NEW: original IDs used to build static set

    # --- helpers ---
    def _path(self, name):
        return os.path.join(self.embed_dir, name)

    def _load_required(self, fname, allow_pickle=False):
        path = self._path(fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"[FusionAgent] Missing file: {path}")
        return np.load(path, allow_pickle=allow_pickle)

    @staticmethod
    def _to_2d(x):
        x = np.asarray(x)
        return x.reshape(-1, 1) if x.ndim == 1 else x

    def _upsample_to(self, x, target_len):
        """Repeat rows to reach target_len. Preserves 1D input as 1D output."""
        x = np.asarray(x)
        one_d = (x.ndim == 1)
        x2 = self._to_2d(x)
        reps = int(np.ceil(target_len / len(x2)))
        tiled = np.tile(x2, (reps, 1))[:target_len]
        return tiled.ravel() if one_d else tiled

    def filter_families(self, X, y_bin, y_fam):
        mask = np.isin(y_fam, list(self.valid_families))
        return X[mask], y_bin[mask], y_fam[mask]

    def print_distribution(self, modality, y_bin, y_fam):
        print(f"\n {modality} Modality Distribution:")
        print(" Benignware:", int((y_bin == 0).sum()))
        print(" Ransomware:", int((y_bin == 1).sum()))
        fam_ids, fam_counts = np.unique(y_fam[y_bin == 1], return_counts=True)
        for fam, count in zip(fam_ids, fam_counts):
            print(f"  Family {int(fam)}: {int(count)} samples")

    # --- per-modality processing ---
    def process_static(self):
        X = self._load_required("Z_static_supervised.npy")
        y_bin = self._load_required("y_static_binary.npy")
        y_fam = self._load_required("y_static_families.npy")

        X, y_bin, y_fam = self.filter_families(X, y_bin, y_fam)

        # Track original indices so we can build a groups vector
        orig_idx = np.arange(len(X))

        X_res, yb_res, yf_res, grp_res = [], [], [], []

        # Benign → 2671  (resample indices, then index arrays)
        ben_mask = (y_bin == 0)
        ben_idx = orig_idx[ben_mask]
        sel = resample(ben_idx, replace=True, n_samples=2671, random_state=42)
        X_res.append(X[sel]); yb_res.append(y_bin[sel]); yf_res.append(y_fam[sel]); grp_res.append(sel)

        # Ransom families → ~445 each
        ransom_fams = sorted(set(y_fam[y_bin == 1]))
        samples_per_family = [445] * len(ransom_fams)
        if samples_per_family:
            samples_per_family[0] = 446

        for fam, n in zip(ransom_fams, samples_per_family):
            fam_idx = np.where(y_fam == fam)[0]
            sel = resample(fam_idx, replace=True, n_samples=n, random_state=int(fam))
            X_res.append(X[sel]); yb_res.append(y_bin[sel]); yf_res.append(y_fam[sel]); grp_res.append(sel)

        Xf = np.vstack(X_res)
        ybf = np.concatenate(yb_res)
        yff = np.concatenate(yf_res)
        groups_final = np.concatenate(grp_res)             # <-- NEW: per-row original IDs

        order = self.rng.permutation(len(ybf))
        Xf, ybf, yff, groups_final = Xf[order], ybf[order], yff[order], groups_final[order]

        self.groups_static = groups_final                  # <-- cache for fuse()
        self.print_distribution("Static", ybf, yff)
        return Xf, ybf, yff

    def process_dynamic(self):
        Z = self._load_required("Z_dynamic_supervised.npy")
        y_bin = self._load_required("y_dynamic_binary.npy")
        y_fam = self._load_required("y_dynamic_families.npy")

        Z, y_bin, y_fam = self.filter_families(Z, y_bin, y_fam)
        Z_b = Z[y_bin == 0]
        Z_r = Z[y_bin == 1]; yf_r = y_fam[y_bin == 1]

        # Benign downsample to 2671
        b_idx = self.rng.choice(len(Z_b), size=2671, replace=False)
        Z_b_eq = Z_b[b_idx]
        yb_b_eq = np.zeros(2671, dtype=int)
        yf_b_eq = np.zeros(2671, dtype=int)

        # Ransom fams → ~445 each
        fams = sorted(set(yf_r))
        per = [445] * len(fams)
        if per: per[0] = 446
        Z_r_eq, yb_r_eq, yf_r_eq = [], [], []
        for fam, n in zip(fams, per):
            idx = np.where(yf_r == fam)[0]
            X_up = resample(Z_r[idx], replace=True, n_samples=n, random_state=int(fam))
            Z_r_eq.append(X_up); yb_r_eq.append(np.ones(n, dtype=int)); yf_r_eq.append(np.full(n, fam))

        Zf = np.vstack([Z_b_eq] + Z_r_eq)
        ybf = np.concatenate([yb_b_eq] + yb_r_eq)
        yff = np.concatenate([yf_b_eq] + yf_r_eq)
        order = self.rng.permutation(len(Zf))
        self.print_distribution("Dynamic", ybf, yff)
        return Zf[order], ybf[order], yff[order]

    def process_network(self):
        Z = self._load_required("Z_network_supervised.npy")
        y_bin = self._load_required("y_network_binary.npy")
        y_fam = self._load_required("y_network_families.npy", allow_pickle=True)

        m = min(len(Z), len(y_bin), len(y_fam))
        Z, y_bin, y_fam = Z[:m], y_bin[:m], np.array(y_fam[:m]).squeeze()
        mask = ~pd.isna(y_fam)
        Z, y_bin, y_fam = Z[mask], y_bin[mask], y_fam[mask]

        le = LabelEncoder()
        y_fam_enc = le.fit_transform(y_fam)
        Z, y_bin, y_fam_enc = self.filter_families(Z, y_bin, y_fam_enc)

        Z_b = Z[y_bin == 0]
        Z_r = Z[y_bin == 1]; yf_r = y_fam_enc[y_bin == 1]

        counts = Counter(yf_r)
        Z_r_eq, yb_r_eq, yf_r_eq = [], [], []
        TARGET, extra = 445, 1
        for fam in sorted(counts.keys()):
            idx = np.where(yf_r == fam)[0]
            n = TARGET + (1 if extra else 0); extra = 0
            up_idx = resample(idx, replace=True, n_samples=n, random_state=42)
            Z_r_eq.append(Z_r[up_idx]); yb_r_eq.append(np.ones(n, dtype=int)); yf_r_eq.append(np.full(n, fam))

        b_idx = resample(np.arange(len(Z_b)), replace=True, n_samples=2671, random_state=42)
        Z_b_eq = Z_b[b_idx]
        yb_b_eq = np.zeros(2671, dtype=int)
        yf_b_eq = np.zeros(2671, dtype=int)

        Zf = np.vstack(Z_r_eq + [Z_b_eq])
        ybf = np.concatenate(yb_r_eq + [yb_b_eq])
        yff = np.concatenate(yf_r_eq + [yf_b_eq])
        order = self.rng.permutation(len(ybf))
        self.print_distribution("Network", ybf, yff)
        return Zf[order], ybf[order], yff[order]

    # --- fusion ---
    def fuse(self):
        print("\nProcessing all modalities (static, dynamic, network)...")
        Zs, yb_s, yf_s = self.process_static()
        Zd, yb_d, yf_d = self.process_dynamic()
        Zn, yb_n, yf_n = self.process_network()

        print("\n Aligning sample counts and fusing embeddings...")
        L = max(len(Zs), len(Zd), len(Zn))
        Zs = self._upsample_to(Zs, L)
        Zd = self._upsample_to(Zd, L)
        Zn = self._upsample_to(Zn, L)

        Z_fused = np.concatenate([Zs, Zd, Zn], axis=1)
        y_bin   = self._upsample_to(yb_s, L).astype(int)    # labels from static
        y_fam   = self._upsample_to(yf_s, L).astype(int)

        # ---- NEW: upsample & save groups so you can do GroupShuffleSplit later
        if self.groups_static is None:
            raise RuntimeError("groups_static was not built in process_static()")
        groups = self._upsample_to(self.groups_static, L).astype(int)
        np.save(self._path("groups.npy"), groups)

        print("\n Fused shape:", Z_fused.shape)
        print(" Binary labels shape:", y_bin.shape)
        print(" Family labels shape:", y_fam.shape)

        np.save(self._path("Z_fused.npy"), Z_fused)
        np.save(self._path("y_binary.npy"), y_bin)
        np.save(self._path("y_families.npy"), y_fam)

        self.Z_fused, self.y_bin, self.y_fam = Z_fused, y_bin, y_fam
        return Z_fused, y_bin, y_fam

    # optional wrapper so you can call .run()
    def run(self):
        return self.fuse()
