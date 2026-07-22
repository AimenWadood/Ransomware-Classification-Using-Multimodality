# agents/classification_agent.py
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score


class SimpleTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=256, p_drop=0.20):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(p_drop)
        self.fc2   = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.drop(x)
        return self.fc2(x)  # RAW logits


class ClassificationAgent:
    def __init__(self, input_dim, num_classes, class_weighting="auto",
                 lr=1e-3, seed: int = 42, label_smoothing: float = 0.0,
                 weight_decay: float = 1e-2):
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SimpleTransformerClassifier(input_dim, num_classes).to(self.device)
        self.num_classes = num_classes
        self.class_weighting = class_weighting
        self.label_smoothing = float(label_smoothing)  # = 0.0 (no training-time flattening)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # cached validation tensors
        self.X_val: Optional[torch.Tensor] = None
        self.y_val: Optional[torch.Tensor] = None

        # extras
        self.class_weights: Optional[torch.Tensor] = None
        self.last_val_logits: Optional[torch.Tensor] = None
        self.last_val_probs: Optional[np.ndarray] = None
        self.last_preds: Optional[np.ndarray] = None

        # calibration state
        self.calib_type: str = "none"   # "none" | "temperature" | "vector"
        self.T: float = 1.0
        self.vec_a: Optional[torch.Tensor] = None
        self.vec_b: Optional[torch.Tensor] = None

        self.class_names: Optional[list[str]] = None

    # ----- convenience -----
    def eval(self):
        self.model.eval(); return self

    def train_mode(self):
        self.model.train(); return self

    def to(self, device: str | torch.device):
        device = torch.device(device)
        self.device = device
        self.model.to(device)
        if self.X_val is not None: self.X_val = self.X_val.to(device)
        if self.y_val is not None: self.y_val = self.y_val.to(device)
        if self.class_weights is not None: self.class_weights = self.class_weights.to(device)
        return self

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        return self.model.load_state_dict(state_dict, strict=strict)

    # ----- utilities -----
    def _compute_class_weights(self, y_np: np.ndarray) -> torch.Tensor:
        counts = np.bincount(y_np, minlength=self.num_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        w = counts.max() / counts
        w = w * (self.num_classes / w.sum())
        return torch.tensor(w, dtype=torch.float32, device=self.device)

    def _setup_loss(self, y_train_np: np.ndarray):
        ls = self.label_smoothing  # 0.0 by default
        if self.class_weighting == "auto":
            self.class_weights = self._compute_class_weights(y_train_np)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=ls)
        elif self.class_weighting is None:
            self.class_weights = None
            self.criterion = nn.CrossEntropyLoss(label_smoothing=ls)
        else:
            w = torch.tensor(self.class_weighting, dtype=torch.float32, device=self.device)
            assert len(w) == self.num_classes
            self.class_weights = w
            self.criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=ls)

    def _fit(self, X_train_np, y_train_np, X_val_np, y_val_np, *, epochs=50, batch_size=256, log_every=5):
        # to tensors on device
        X_train = torch.tensor(X_train_np, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train_np, dtype=torch.long, device=self.device)
        self.X_val = torch.tensor(X_val_np, dtype=torch.float32, device=self.device)
        self.y_val = torch.tensor(y_val_np, dtype=torch.long, device=self.device)

        n = X_train.shape[0]
        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0

            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = X_train[idx], y_train[idx]

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            if (epoch + 1) % log_every == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"[Epoch {epoch+1}/{epochs}] Loss: {epoch_loss / n:.4f}")

        # cache raw val logits and pure softmax probs (no clamps/mixing)
        self.model.eval()
        with torch.no_grad():
            self.last_val_logits = self.model(self.X_val)
            self.last_val_probs  = torch.softmax(self.last_val_logits, dim=1).cpu().numpy()

        self.evaluate()

    # ----- public training APIs -----
    def train(self, Z_fused, y_fam, epochs: int = 50, batch_size: int = 256, val_size: float = 0.2, seed: int = 42):
        X_train, X_val, y_train, y_val = train_test_split(
            Z_fused, y_fam, test_size=val_size, stratify=y_fam, random_state=seed
        )
        self._setup_loss(y_train)
        self._fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)

    def train_with_split(self, X_train, y_train, X_val, y_val, epochs: int = 50, batch_size: int = 256):
        self._setup_loss(np.asarray(y_train))
        self._fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)

    # ----- calibration helpers -----
    def forward_logits(self, X_t: torch.Tensor) -> torch.Tensor:
        return self.model(X_t.to(self.device))

    def fit_temperature(self) -> float:
        assert self.X_val is not None and self.y_val is not None, "No validation tensors cached"
        self.model.eval()
        with torch.no_grad():
            logits_val = self.forward_logits(self.X_val)

        T_param = torch.nn.Parameter(torch.ones(1, device=self.device))
        opt = torch.optim.LBFGS([T_param], lr=0.01, max_iter=60)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits_val / T_param, self.y_val)
            loss.backward()
            return loss

        opt.step(closure)
        self.T = float(T_param.detach().item())
        self.calib_type = "temperature"
        return self.T

    def apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        # allow true fitted T (could be < 1 or > 1); no artificial clamping
        return logits / self.T if self.calib_type == "temperature" else logits

    def fit_vector_scaler(self):
        assert self.X_val is not None and self.y_val is not None, "No validation tensors cached"
        self.model.eval()
        with torch.no_grad():
            logits_val = self.forward_logits(self.X_val)
        C = logits_val.shape[1]

        a = torch.nn.Parameter(torch.ones(C, device=self.device))
        b = torch.nn.Parameter(torch.zeros(C, device=self.device))
        opt = torch.optim.LBFGS([a, b], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits_val * a + b, self.y_val)
            loss.backward()
            return loss

        opt.step(closure)
        self.vec_a, self.vec_b = a.detach(), b.detach()
        self.calib_type = "vector"
        return self.vec_a.detach().cpu().numpy(), self.vec_b.detach().cpu().numpy()

    def apply_vector(self, logits: torch.Tensor) -> torch.Tensor:
        if self.calib_type == "vector" and self.vec_a is not None and self.vec_b is not None:
            return logits * self.vec_a + self.vec_b
        return logits

    def save_calibration(self, path: Path):
        data = {"type": self.calib_type, "T": self.T}
        if self.vec_a is not None and self.vec_b is not None:
            data["a"] = self.vec_a.detach().cpu().tolist()
            data["b"] = self.vec_b.detach().cpu().tolist()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def load_calibration(self, path: Path):
        if not path.exists(): return
        data = json.loads(path.read_text())
        self.calib_type = data.get("type", "none")
        self.T = float(data.get("T", 1.0))
        if "a" in data and "b" in data:
            self.vec_a = torch.tensor(data["a"], dtype=torch.float32, device=self.device)
            self.vec_b = torch.tensor(data["b"], dtype=torch.float32, device=self.device)

    # ----- evaluation / inference -----
    def evaluate(self, X_val=None, y_val=None, return_metrics: bool = False):
        self.model.eval()
        with torch.no_grad():
            Xv = self.X_val if X_val is None else torch.tensor(X_val, dtype=torch.float32, device=self.device)
            yv = self.y_val if y_val is None else torch.tensor(y_val, dtype=torch.long, device=self.device)

            logits = self.model(Xv)
            preds = logits.argmax(dim=1).cpu().numpy()
            self.last_preds = preds

            y_true = yv.cpu().numpy()
            print("\nClassification Report:")
            print(classification_report(y_true, preds))
            if self.class_weights is not None:
                print("Class weights used:", self.class_weights.detach().cpu().numpy())

            if return_metrics:
                return {
                    "f1_weighted": f1_score(y_true, preds, average="weighted"),
                    "accuracy": accuracy_score(y_true, preds),
                }

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            return self.model(X).argmax(dim=1).cpu().numpy()

    def predict_proba(self, X, calibrated: bool = False):
        """Return exact classifier probabilities (pure softmax)."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.forward_logits(X_t)
            if calibrated:
                if self.calib_type == "temperature":
                    logits = self.apply_temperature(logits)
                elif self.calib_type == "vector":
                    logits = self.apply_vector(logits)
            probs = torch.softmax(logits, dim=1)  # no clipping/mixing
            return probs.cpu().numpy()

    # convenience getters
    def get_predictions(self):
        return self.last_preds

    def get_val_data(self):
        if self.X_val is None or self.y_val is None:
            return None, None
        return self.X_val.detach().cpu().numpy(), self.y_val.detach().cpu().numpy()

    # ----- checkpoint helpers -----
    def save_checkpoint(self, path: Path, *, class_names: Optional[list[str]] = None, extra: Optional[dict] = None):
        payload = {
            "model_kind": "state_dict_agent",
            "model_payload": self.state_dict(),
            "num_classes": self.num_classes,
            "class_names": class_names or self.class_names,
            "calibration": {
                "type": self.calib_type,
                "T": float(self.T),
                "a": self.vec_a.detach().cpu().tolist() if self.vec_a is not None else None,
                "b": self.vec_b.detach().cpu().tolist() if self.vec_b is not None else None,
            },
            "extra": extra or {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(path))

    @staticmethod
    def load_checkpoint(path: Path, input_dim: int, num_classes: int):
        bundle = torch.load(str(path), map_location="cpu")
        agent = ClassificationAgent(input_dim=input_dim, num_classes=num_classes, class_weighting="auto", lr=1e-3, seed=42)
        sd = bundle.get("model_payload", None)
        if sd is not None:
            try:
                agent.load_state_dict(sd, strict=False)
            except Exception:
                if hasattr(agent, "model") and hasattr(agent.model, "load_state_dict"):
                    agent.model.load_state_dict(sd, strict=False)
        cal = bundle.get("calibration", {})
        agent.calib_type = cal.get("type", "none")
        agent.T = float(cal.get("T", 1.0))
        a, b = cal.get("a", None), cal.get("b", None)
        if a is not None and b is not None:
            agent.vec_a = torch.tensor(a, dtype=torch.float32, device=agent.device)
            agent.vec_b = torch.tensor(b, dtype=torch.float32, device=agent.device)
        agent.class_names = bundle.get("class_names", None)
        agent.eval()
        return agent, agent.class_names
