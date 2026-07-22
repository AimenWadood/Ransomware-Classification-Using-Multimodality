# agents/automl_manager_agent.py
from __future__ import annotations

class AutoMLManagerAgent:
    def __init__(
        self,
        threshold: float = 0.75,                 # legacy F1 threshold
        *,
        thresholds: dict | None = None,          # {"f1":0.75,"acc":0.80,"bleu":0.30}
        primary: str = "f1",                      # 'f1' | 'acc' | 'bleu'
        patience: int = 3,                        # epochs w/o improvement before suggest
        min_improve: float = 0.005,               # min delta on primary to count as improve
    ):
        # thresholds
        self.threshold = float(threshold)  # alias for legacy
        self.thresholds = {"f1": self.threshold, "acc": 0.80, "bleu": 0.30}
        if thresholds:
            self.thresholds.update({k.lower(): float(v) for k, v in thresholds.items()})

        self.primary = primary.lower()
        if self.primary not in {"f1", "acc", "bleu"}:
            raise ValueError("primary must be one of: 'f1', 'acc', 'bleu'")

        # plateau logic
        self.patience = int(patience)
        self.min_improve = float(min_improve)

        # state
        self.feedback_log: list[str] = []
        self.best_f1 = 0.0
        self.best_acc = 0.0
        self.best_bleu = 0.0
        self.best_epoch = -1
        self._no_improve = 0

    # ---------- Legacy batch API (kept for compatibility) ----------
    def evaluate_and_suggest(self, scores_dict: dict) -> dict:
        """
        Backward-compatible API:
          scores_dict = {agent_name: f1_score}  or
                        {agent_name: {"f1":..., "acc":..., "bleu":...}}
        Returns: {agent_name: 'retrain'|'retain'}
        """
        out = {}
        for agent, val in scores_dict.items():
            if isinstance(val, (int, float)):
                # treat as F1
                metric = "f1"
                value = float(val)
                thr = self.thresholds["f1"]
            else:
                # pick configured primary if provided, else fall back to F1
                vals = {k.lower(): float(v) for k, v in val.items()}
                metric = self.primary if self.primary in vals else "f1"
                value = vals.get(metric, 0.0)
                thr = self.thresholds.get(metric, self.thresholds["f1"])

            decision = "retrain" if value < thr else "retain"
            out[agent] = decision
            self.feedback_log.append(
                f" AutoML Decision → {agent}: {metric.upper()}={value:.3f} "
                f"(thr={thr:.3f}) → {decision.upper()}"
            )
        return out

    # ---------- Online API your runner uses ----------
    def evaluate(self, f1: float, acc: float, epoch: int, bleu: float | None = None) -> None:
        """
        Track metrics across epochs and print progress; BLEU is optional.
        Improvement is judged on the configured primary metric.
        """
        # compute current and best for the chosen primary (before updating bests)
        current_primary = self._primary_value(f1, acc, bleu)
        best_primary = self._primary_value(self.best_f1, self.best_acc, self.best_bleu if bleu is not None else None)

        improved = (current_primary - best_primary) >= self.min_improve or epoch == 0
        if improved:
            # update bests
            self.best_f1 = max(self.best_f1, f1)
            self.best_acc = max(self.best_acc, acc)
            if bleu is not None:
                self.best_bleu = max(self.best_bleu, bleu)

            self.best_epoch = epoch
            self._no_improve = 0
            msg = (f"[AutoML] New best @ epoch {epoch+1}: "
                   f"F1={f1:.4f}, Acc={acc:.4f}" +
                   (f", BLEU={bleu:.4f}" if bleu is not None else ""))
            print(msg)
        else:
            self._no_improve += 1
            msg = (f"[AutoML] No significant improvement "
                   f"({self._no_improve}/{self.patience}). "
                   f"Best so far → F1={self.best_f1:.4f}, Acc={self.best_acc:.4f}" +
                   (f", BLEU={self.best_bleu:.4f}" if bleu is not None else "") +
                   f" @ epoch {self.best_epoch+1}")
            print(msg)

    def adjust_if_needed(self) -> None:
        """Suggest actions after 'patience' stagnant epochs."""
        if self._no_improve < self.patience:
            return
        print("[AutoML] Triggering adjustments due to plateau:")
        print("  • Increase classifier epochs or reduce learning rate.")
        print("  • Rebalance classes more aggressively in FusionAgent.")
        print("  • Try switching AE architecture (Contrastive ↔ Stacked).")
        print("  • If BLEU is primary, adjust decoding (temperature/top-p/top-k) or max_new_tokens.")
        self._no_improve = 0

    # ---------- Logging helpers ----------
    def analyze_and_log(self, performance_log):
        for entry in performance_log:
            self.feedback_log.append(f"📈 Classifier Report: {entry}")

    # ---------- Utilities ----------
    def set_primary(self, primary: str):
        primary = primary.lower()
        if primary not in {"f1", "acc", "bleu"}:
            raise ValueError("primary must be one of: 'f1', 'acc', 'bleu'")
        self.primary = primary

    def set_threshold(self, metric: str, value: float):
        self.thresholds[metric.lower()] = float(value)

    def _primary_value(self, f1: float, acc: float, bleu: float | None):
        if self.primary == "acc":
            return acc
        if self.primary == "bleu" and bleu is not None:
            return bleu
        return f1
