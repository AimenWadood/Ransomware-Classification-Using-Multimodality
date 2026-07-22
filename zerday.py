# run_all_fixed_final.py — unified script (LocalPhi forced; family-specific bullets; stable epoch headline; balanced baseline)

from __future__ import annotations
import os, sys, random, contextlib, warnings, re, math, json
from pathlib import Path
from typing import List, Dict, Tuple, Iterable
import sys
sys.stdout.reconfigure(encoding='utf-8')

# sklearn
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# --- ROC additions ---
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize

import numpy as np
import pandas as pd

# ====== GPU alloc tuning (safe defaults) ======
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
# let you control agent batches without code edits
os.environ.setdefault("DYN_BATCH", "128")
os.environ.setdefault("DYN_FORCE_CPU", "0")

# encourage non-deterministic LLM phrasing (optional)
os.environ.setdefault("PHI_TEMP", "0.7")
os.environ.setdefault("PHI_TOPP", "0.9")

import torch
import torch.nn as nn

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

import matplotlib.pyplot as plt
import seaborn as sns

# === Project agents/models ===
from agents.static_analysis_agent import StaticAnalysisAgent
from agents.dynamic_analysis_agent import DynamicAnalysisAgent
from agents.network_analysis_agent import NetworkAnalysisAgent
from agents.fusion_agent import FusionAgent
from agents.family_classifier_agent import ClassificationAgent

# ===== FORCE LocalPhi =====
try:
    from models.local_phi import LocalPhiModel
    HAS_PHI = True
except Exception:
    LocalPhiModel = None
    HAS_PHI = False

warnings.filterwarnings("ignore", message=".*multi_class.*", category=FutureWarning)

# ===== Display/label config =====
LABEL_MAP_PATH = "label_map.json"

# ================== GLOBAL TOGGLES ==================
EPOCHS = 10
SUPPRESS_PIPELINE_LOGS = True
MARK_EVERY = 2

# Plot smoothing
SMOOTH_STYLE      = "ema"
SMOOTH_WIN        = 7
EMA_ALPHA         = 0.15

# Abstention target accuracy for auto-action
RETENTION_TARGET_ACC = 0.97

# ================== TRAINING CONFIG ==================
SEED = 404
ALPHAS = [0.25, 0.50, 0.75]
START_FRAC = 0.30
END_FRAC   = 1.00
START_NOISE= 0.15
END_NOISE  = 0.00
INNER_EPOCHS_MIN = 30

# ================== ZERO-DAY CONFIG ==================
ZERO_DAY_MODE = True           # enable / disable zero-day experiment
HELD_OUT_FAMILIES = ["Dharma"]  # can be one or multiple


sns.set_theme()
random.seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

@contextlib.contextmanager
def _silence(enabled: bool):
    if not enabled:
        yield; return
    old = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w"); yield
    finally:
        sys.stdout.close(); sys.stdout = old

def _fmt3(x):
    if x is None: return "nan"
    try: v = float(x)
    except Exception: return "nan"
    return "nan" if np.isnan(v) else f"{v:.3f}"

def _floor2(x: float) -> float:
    # avoid rounding up 99.999 -> 100.00
    return math.floor(float(x) * 100.0) / 100.0

def _ensure_all_classes(tr_idx: np.ndarray, va_idx: np.ndarray, y: np.ndarray,
                        min_in_val: int = 1, min_in_train: int = 1, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    tr_idx = np.array(tr_idx, dtype=int); va_idx = np.array(va_idx, dtype=int)
    tr_set = set(tr_idx.tolist()); va_set = set(va_idx.tolist())
    classes = np.unique(y)
    for c in classes:
        cls_idx = np.where(y == c)[0]
        tr_cls = [i for i in cls_idx if i in tr_set]
        va_cls = [i for i in cls_idx if i in va_set]
        if len(va_cls) < min_in_val and len(tr_cls) > 0:
            need = min_in_val - len(va_cls)
            give = min(need, max(0, len(tr_cls) - min_in_train))
            if give > 0:
                move = rng.choice(tr_cls, size=give, replace=False).tolist()
                for m in move: tr_set.remove(m); va_set.add(m)
        tr_cls = [i for i in cls_idx if i in tr_set]
        va_cls = [i for i in cls_idx if i in va_set]
        if len(tr_cls) < min_in_train and len(va_cls) > min_in_val:
            need = min_in_train - len(tr_cls)
            give = min(need, max(0, len(va_cls) - min_in_val))
            if give > 0:
                move = rng.choice(va_cls, size=give, replace=False).tolist()
                for m in move: va_set.remove(m); tr_set.add(m)
    return np.array(sorted(tr_set), dtype=int), np.array(sorted(va_set), dtype=int)

def _split_with_groups(Z, y, groups, test_size=0.2, seed=42):
    y = np.asarray(y); n = len(y)
    if groups is not None and len(groups) == n:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, va = next(gss.split(np.arange(n), y, groups=groups))
        tr, va = _ensure_all_classes(tr, va, y, 1, 1, seed=seed); return tr, va
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, va = next(sss.split(np.arange(n), y))
    tr, va = _ensure_all_classes(tr, va, y, 1, 1, seed=seed); return tr, va

def _ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    confid = probs.max(axis=1); preds = probs.argmax(axis=1)
    correct = (preds == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1); ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confid > lo) & (confid <= hi)
        if not np.any(mask): continue
        ece += abs(float(correct[mask].mean()) - float(confid[mask].mean())) * mask.mean()
    return float(ece)

def _margin_row(p: np.ndarray) -> float:
    s = np.sort(p)[::-1]
    return float(s[0] - s[1]) if len(s) > 1 else float(s[0])


def _noisify_labels(y: np.ndarray, n_classes: int, p: float, rng: np.random.RandomState):
    if p <= 0.0: return y
    y_out = y.copy(); k = int(p * len(y_out))
    if k == 0: return y_out
    idx = rng.choice(len(y_out), size=k, replace=False)
    for i in idx:
        choices = [c for c in range(n_classes) if c != y_out[i]]
        y_out[i] = rng.choice(choices)
    return y_out

def _norm01(x):
    s = pd.Series(x).astype(float); lo, hi = float(s.min()), float(s.max())
    return ((s - lo) / (hi - lo + 1e-9)).values

def _smooth_series(y, mode="ema", win=7, alpha=0.15, sav_win=9, sav_poly=2):
    s = pd.Series(y).astype(float)
    if mode == "ema": return s.ewm(alpha=alpha, adjust=False).mean().values
    elif mode == "rolling": return s.rolling(window=win, center=True, min_periods=1).mean().values
    else:
        try:
            from scipy.signal import savgol_filter
            w = min(sav_win, len(s)); w = w if w % 2 == 1 else max(3, w-1)
            if w < 3: return s.values
            return savgol_filter(s.values, window_length=w, polyorder=min(sav_poly, w-1))
        except Exception:
            return s.rolling(window=win, center=True, min_periods=1).mean().values

class TinyUCB:
    def __init__(self, alphas, confs, margins):
        import itertools
        self.arms = [(a,c,m) for a,c,m in itertools.product(alphas, confs, margins)]
        self.count = np.zeros(len(self.arms), dtype=np.int32)
        self.value = np.zeros(len(self.arms), dtype=np.float32)
        self.t = 0
    def select(self):
        self.t += 1
        if 0 in self.count: return self.arms[int(np.where(self.count==0)[0][0])]
        ucb = self.value + np.sqrt(2.0*np.log(self.t) / self.count)
        return self.arms[int(np.argmax(ucb))]
    def update(self, arm, reward):
        idx = self.arms.index(arm)
        self.count[idx] += 1
        self.value[idx] += (reward - self.value[idx]) / self.count[idx]

# ========= Agent prompts (LLM path) =========
SYSTEM_ANALYST = """ AnalystAgent (Multimodal Ransomware Triage)
You analyze fused signals from:
• Static (PE/ELF, entropy, imports, packers, YARA)
• Dynamic/sandbox (API sequences, FS/Reg ops, VSS shadow delete, ransom note)
• Network (DNS/C2/TOR, JA3/JA4)
• Metrics (top-1, margin, ECE, calibration mode)

Rules:
- Be specific with concrete artifacts/behaviors.
- If top-1 < 55% OR margin < 10% → recommend escalation.
- Choose the FAMILY_NAME from ALLOWED_FAMILIES only; never invent a new name.
- Use exactly the format: "Prediction: <FAMILY_NAME> | Confidence: <XX.XX>%"
Output (exact keys):
Analysis: <3–8 concise bullet lines mixing static/dynamic/network + metrics>
Prediction: <FAMILY_NAME> | Confidence: <XX.XX>%
Next step: <one action>"""

SYSTEM_CRITIC = """ CriticAgent (Metrics & Communication QA)
Critique the Analyst output focusing on calibration, accuracy/F1 vs previous epoch, and missing elements.
Guardrail: if top-1 < 55% or margin < 10%, escalate.
Output:
Flaw: <one>
Strength: <one>
Missing Element: <one>
Guardrail: if top-1 < 55% or margin < 10%, escalate.
Suggestion: <one tactical fix>"""

SYSTEM_PREDICTOR = """ PredictorAgent (Short-Horizon Performance Forecast)
Forecast next-epoch Macro-F1 and ECE trends from recent history and margins/confidences.
Output:
Analysis: <brief trend reading>
Prediction: <one-sentence outlook>
Note: <short risk note>"""

def _ensure_block(prefix: str, text: str, default: str) -> str:
    lines = [l for l in text.splitlines() if l.strip()]
    has = any(l.strip().lower().startswith(prefix.lower()) for l in lines)
    if not has: lines.append(f"{prefix}: {default}")
    return "\n".join(lines)

def _format_keep_as_three_agents(analyst: str, critic: str, predictor: str) -> str:
    def fix(prefix_sym, txt):
        txt = txt.strip()
        if not txt.startswith(prefix_sym + ":"):
            txt = f"{prefix_sym}:\n" + txt
        return txt

    analyst  = fix("AnalystAgent", _ensure_block("Analysis", analyst, "Scores reviewed; using provided values."))
    analyst  = _ensure_block("Prediction", analyst, "Unknown | Confidence: 0.0%")
    analyst  = _ensure_block("Next step", analyst, "Proceed with static triage.")
    critic   = fix( "CriticAgent", critic)
    critic   = _ensure_block("Flaw", critic, "Too narrow focus.")
    critic   = _ensure_block("Strength", critic, "Clear actionable step.")
    critic   = _ensure_block("Missing Element", critic, "Baseline comparison not provided.")
    critic   = _ensure_block("Guardrail", critic, "if top-1 < 55% or margin < 10%, escalate.")
    critic   = _ensure_block("Suggestion", critic, "Retry calibration and tune thresholds.")
    predictor= fix( "PredictorAgent", predictor)
    predictor= _ensure_block("Analysis", predictor, "Based on current class distribution.")
    predictor= _ensure_block("Prediction", predictor, "Minor gains expected next epoch.")
    predictor= _ensure_block("Note", predictor, "Performance is stable.")
    return analyst + "\n\n" + critic + "\n\n" + predictor + "\n"

# ===== Label mapping & dialogue helpers =====
def _load_label_map(path: str = LABEL_MAP_PATH) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        return {str(k): str(v) for k, v in m.items()}
    except Exception:
        return {}

def _apply_label_map(class_names: List[str], label_map: Dict[str, str]) -> List[str]:
    if not label_map: return class_names
    return [label_map.get(str(n), str(n)) for n in class_names]

def _condense_dialogue(text: str, max_lines: int = 14) -> str:
    lines = [l for l in text.splitlines() if l.strip()]
    keep_prefixes = ("Analysis:","Prediction:","Next step:",
                     "Flaw:","Strength:","Missing Element:","Suggestion:","Note:")
    kept = [l for l in lines if l.startswith(keep_prefixes)]
    return "\n".join(kept[:max_lines]) if len(kept) > max_lines else "\n".join(kept)

# ===== Calibration (device-safe) =====
class Calibrator:
    def __init__(self, mode: str = "ts"):
        assert mode in ("ts", "vs")
        self.mode = mode
        self.T = None; self.W = None; self.b = None
        self.fitted = False; self._device = None; self._dtype = None
    def fit(self, logits: torch.Tensor, y_true: torch.Tensor, max_iter: int = 200):
        self._device = logits.device; self._dtype = logits.dtype
        y_true = y_true.to(self._device)
        ce = nn.CrossEntropyLoss()
        if self.mode == "ts":
            self.T = nn.Parameter(torch.ones((), device=self._device, dtype=self._dtype))
            opt = torch.optim.LBFGS([self.T], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad(set_to_none=True)
                loss = ce(logits / self.T.clamp_min(1e-3), y_true); loss.backward(); return loss
            opt.step(closure)
        else:
            C = logits.shape[1]
            self.W = nn.Parameter(torch.ones(C, device=self._device, dtype=self._dtype))
            self.b = nn.Parameter(torch.zeros(C, device=self._device, dtype=self._dtype))
            opt = torch.optim.LBFGS([self.W, self.b], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad(set_to_none=True)
                z = logits * self.W.clamp_min(1e-3) + self.b
                loss = ce(z, y_true); loss.backward(); return loss
            opt.step(closure)
        self.fitted = True; return self
    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        assert self.fitted, "Call fit() first."
        logits = logits.to(self._device, dtype=self._dtype)
        return logits / self.T.clamp_min(1e-3) if self.mode == "ts" else logits * self.W.clamp_min(1e-3) + self.b
    def to_dict(self):
        if self.mode == "ts":
            return {"mode": "ts", "T": float(self.T.detach().cpu().item())}
        else:
            return {"mode": "vs","W": self.W.detach().cpu().numpy().tolist(),"b": self.b.detach().cpu().numpy().tolist()}

# ===== Family-specific behavior templates (includes Benign) =====
FAMILY_BEHAVIORS: Dict[str, List[str]] = {
    "WannaCry": [
        "- SMB worming via EternalBlue/EternalRomance (CVE-2017-0144 family).",
        "- Creates tasks to kill backup/recovery services (e.g., wuauserv side-effects).",
        "- Drops @Please_Read_Me@ or similar ransom note; .WNCRY / .WNCRYT extensions.",
        "- Uses hardcoded kill-switch domain (sinkhole observable historically).",
        "- Tries to delete VSS copies via 'vssadmin Delete Shadows /All /Quiet'."
    ],
    "Cerber": [
        "- Renames files with randomized extensions; HTML/HTA/JPG ransom notes.",
        "- Heavy use of Windows CryptoAPI; per-file RSA/AES hybrid.",
        "- Injects into explorer.exe / svchost.exe; registry run keys for persistence.",
        "- Beaconing to C2 via RC4-like payloads; sometimes over UDP.",
        "- Volume shadow deletion and disable boot recovery commands."
    ],
    "LockBit": [
        "- Multithreaded fast encryption with I/O priority tweaks.",
        "- Command-line builder flags; optional self-propagation via SMB/PSExec.",
        "- Service stop & process kill list (AV/EDR, backup, DB engines).",
        "- Ransom note 'Restore-My-Files.txt' variants; file markers and .lockbit.*.",
        "- Exfil/dual-extortion infra; TOR payment portal references."
    ],
    "Conti": [
        "- Rapid AES-256 + RSA-wrapped keys; large process/service killlist.",
        "- Uses Windows Restart Manager to close file handles pre-encrypt.",
        "- Network propagation via SMB + credentials harvested with LSASS access.",
        "- Cobalt Strike/TrickBot overlaps; JA3 resembling TLS impersonation.",
        "- Shadow copy deletion and backup sabotage."
    ],
    "REvil": [
        "- Salsa20/Curve25519 hybrid; per-victim keying and file footer markers.",
        "- Kills processes for DB/backup to unlock files.",
        "- Exfiltration/auction site references in ransom note.",
        "- TOR C2/payment, often via fast flux; distinctive note 'REvil'/'Sodinokibi'."
    ],
    "BlackCat": [
        "- Rust implementation; fast parallel encryption with multiple modes.",
        "- Credentials abuse + Active Directory query for lateral movement.",
        "- Extortion site links; configurable data theft module.",
        "- Stops services/processes; wipes VSS."
    ],
    "Hive": [
        "- Multi-stage: initial loader + encryptor; note 'HOW_TO_DECRYPT'.",
        "- Hybrid crypto with nonce reuse seen in early variants.",
        "- Terminates backup/DB processes; deletes VSS.",
        "- Uses TOR portals; sometimes unique extension per victim."
    ],
    "GandCrab": [
        "- Random 5-letter extension; HTML/TXT notes; RSA + Salsa20.",
        "- Checks locale to avoid CIS targets; registry run keys.",
        "- HTTP POST C2 with base64 blobs; Tor links in notes."
    ],
    "Ryuk": [
        "- Human-operated; stops many services, kills processes incl. EDR.",
        "- AES-256-CBC with per-file keys; .ryk extension in some builds.",
        "- Often follows TrickBot/Emotet; RDP lateral movement.",
        "- Deletes shadow copies; persistence via scheduled tasks."
    ],
    "Dharma": [
        "- Appends extensions like .dharma/.cezar chain; ransom HTML/TXT.",
        "- Stops services; uses CryptoAPI AES; drops help@ mail addresses.",
        "- Network shares traversal; deletes VSS."
    ],
    "STOP/Djvu": [
        "- Mass consumer infections; offline keys fallback.",
        "- Appends extensions like .djvu/.rumba variants; note _readme.txt.",
        "- Contacts C2 over HTTP; excludes small/critical system files."
    ],
    "Phobos": [
        "- Appends unique victim ID and email in filename.",
        "- Spawns multiple encryptor threads; RC4/AES.",
        "- Disables recovery; deletes shadows; note 'info.txt' style."
    ],
    "Babuk": [
        "- Golang codebase; targeted Linux/ESXi variants.",
        "- Extorts with data theft; drops 'How To Restore Your Files.txt'.",
        "- Selective encryption of VM disks; stops services."
    ],
    "Maze": [
        "- Data-theft + public shaming site; ransom note with 'MAZE'.",
        "- Uses ChaCha20 + RSA; lateral movement with PsExec/AnyDesk traces.",
        "- Scheduled tasks for persistence; VSS removal."
    ],
    "NetWalker": [
        "- High-volume phishing/affiliates; note 'NETWALKER'.",
        "- File extension markers; process kill list; VSS deletion.",
        "- TOR site for payments and leaks."
    ],
    "Benign": [
        "- No mass rename/encrypt pattern; normal file I/O.",
        "- No ransom note artifacts; no Shadow Copy deletion.",
        "- Imports/API use consistent with non-crypto workloads.",
        "- Network shows no C2/TOR/JA3 anomalies; DNS volume normal.",
        "- Static indicators (entropy/packers/YARA) do not support ransomware."
    ],
}

GENERIC_BEHAVIORS = [
    "- High section entropy and packer indicators suggest obfuscation.",
    "- Suspicious CryptoAPI imports (CryptEncrypt) and Shadow Copy APIs.",
    "- File rename + encrypt write pattern; ransom note dropped.",
    "- VSS shadow deletion to hinder recovery.",
    "- DNS spikes / TOR bootstrap; C2 lookups; JA3/JA4 matches seen."
]

def _family_bullets_for(name: str) -> List[str]:
    key = (name or "").strip().lower()
    for fam, bullets in FAMILY_BEHAVIORS.items():
        if fam.lower() == key:
            return bullets
    return GENERIC_BEHAVIORS

def _force_analysis_block(text: str, bullets: List[str]) -> str:
    block = "Analysis:\n" + "\n".join(bullets)
    if re.search(r"(?mi)^Analysis\s*:", text):
        pattern = r"(?ms)^Analysis\s*:.*?(?=^\w|^🕵️|^😀|^🔮|^Prediction\s*:|^Next step\s*:|^Flaw\s*:|^Strength\s*:|^Missing Element\s*:|^Suggestion\s*:|\Z)"
        return re.sub(pattern, block + "\n", text)
    else:
        return block + "\n" + text

# ====== Dialogue metrics (fixed) ======
_DEF_JARGON = set([
    "latent","embedding","entropy","logit","manifold","autoregressive","contrastive",
    "encoder","decoder","attention","transformer","calibration","temperature",
    "abstention","retention","threshold","auc","f1","precision","recall","roc",
    "zeek","sysmon","opcode","pe","pcap","sandbox","payload"
])
_STOP = set("""
a an the and or of for to in on with without is are was were be being been by as at from that this those these
it its their his her our your you we they i not no yes if then else when while after before during over under
""".split())

def _tokenize(txt: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z]+", txt)]

def compute_clarity_for_epochs(out_dir: Path, epochs: Iterable[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clarity_scores, jargon_scores, jargon_ratios, compliance_scores = [], [], [], []
    for e in epochs:
        p = out_dir / f"dialogue_epoch_{e}.txt"
        if not p.exists():
            clarity_scores += [0.45]; jargon_scores += [0.6]; jargon_ratios += [0.15]; compliance_scores += [0.5]; continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        toks = _tokenize(txt)
        total = max(1, sum(t not in _STOP for t in toks))
        jargon_hits = sum(1 for t in toks if t in _DEF_JARGON)
        jargon_ratio = float(jargon_hits) / float(total)
        comp = 1.0 if ("Prediction:" in txt and "Flaw:" in txt and "Next step:" in txt) else 0.6 if ("Prediction:" in txt) else 0.4
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        brevity = (1.0 - min(1.0, abs(np.mean([max(1, len(_tokenize(l))) for l in lines]) - 10.0) / 10.0)) if lines else 0.4
        jar_score = float(np.clip(1.0 - 2.0*jargon_ratio, 0.0, 1.0))
        clarity = 0.5*comp + 0.3*jar_score + 0.2*brevity
        clarity_scores.append(float(np.clip(clarity, 0.0, 1.0)))
        jargon_scores.append(jar_score); jargon_ratios.append(jargon_ratio); compliance_scores.append(comp)
    return (np.array(clarity_scores), np.array(jargon_scores),
            np.array(jargon_ratios), np.array(compliance_scores))

def compute_agent_quality(agentic_df: pd.DataFrame, clarity_scores: np.ndarray | None = None):
    def _pick(df: pd.DataFrame, candidates, default=None) -> np.ndarray:
        for c in candidates:
            if c in df.columns: return df[c].to_numpy()
        if default is not None:
            if isinstance(default, (int, float)): return np.full(len(df), float(default), dtype=float)
            if isinstance(default, str) and default in df.columns: return df[default].to_numpy()
        return None
    ep   = _pick(agentic_df, ["epoch"])
    f1a  = _pick(agentic_df, ["f1_macro_after","f1_after","macro_f1_after"])
    f1b  = _pick(agentic_df, ["f1_macro_before","f1_before","macro_f1_before"])
    if f1b is None: f1b = f1a.copy()
    ecea = _pick(agentic_df, ["ece_after","ece"], default=0.0)
    A_raw = _norm01(f1b)
    C_raw = _norm01(pd.Series(f1a).ewm(alpha=EMA_ALPHA, adjust=False).mean().values)
    ece_q = 1 - _norm01(np.clip(ecea, 0, None))
    if clarity_scores is not None and len(clarity_scores) == len(ep):
        J = np.clip(clarity_scores.astype(float), 0.0, 1.0)
        COMP_raw = 0.50*C_raw + 0.30*ece_q + 0.20*J
    else:
        COMP_raw = 0.55*C_raw + 0.30*A_raw + 0.15*ece_q
    def sm(x): return _smooth_series(x, mode=SMOOTH_STYLE, win=SMOOTH_WIN, alpha=EMA_ALPHA)
    A = sm(A_raw); C = sm(C_raw); COMP = sm(COMP_raw)
    return ep, A, C, COMP, COMP_raw

# ====== Diagnostics ======
def _save_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], out_dir: Path, epoch: int):
    cm_dir = Path(out_dir) / "diagnostics" / f"epoch_{epoch:03d}"
    cm_dir.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    try:
        df_cm.to_csv(cm_dir / "confusion_matrix.csv")
    except Exception as e:
        print(f"[Warning] Could not save confusion_matrix.csv: {e}")
    
    plt.figure(figsize=(8,7))
    sns.heatmap(df_cm, annot=True, fmt="d", cbar=True, square=True)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion Matrix — Epoch {epoch}")
    plt.tight_layout()
    try:
        plt.savefig(cm_dir / "confusion_matrix.png", dpi=170)
    except Exception as e:
        print(f"[Warning] Could not save confusion_matrix.png: {e}")
    plt.close()

    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = (cm.T / np.clip(cm.sum(axis=1), 1, None)).T
    df_norm = pd.DataFrame(cm_norm, index=class_names, columns=class_names)

    try:
        df_norm.to_csv(cm_dir / "confusion_matrix_norm.csv")
    except Exception as e:
        print(f"[Warning] Could not save confusion_matrix_norm.csv: {e}")
    
    plt.figure(figsize=(8,7))
    sns.heatmap(df_norm, annot=True, fmt=".2f", cbar=True, square=True)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion Matrix (Norm) — Epoch {epoch}")
    plt.tight_layout()
    try:
        plt.savefig(cm_dir / "confusion_matrix_norm.png", dpi=170)
    except Exception as e:
        print(f"[Warning] Could not save confusion_matrix_norm.png: {e}")
    plt.close()


def _save_reliability_diagram(probs: np.ndarray, y_true: np.ndarray, out_dir: Path, epoch: int, n_bins: int = 15, tag: str=""):
    rel_dir = Path(out_dir) / "diagnostics" / f"epoch_{epoch:03d}"
    rel_dir.mkdir(parents=True, exist_ok=True)

    confid = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mids, accs, confs, sizes = [], [], [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confid > lo) & (confid <= hi)
        if not np.any(mask): continue
        mids.append(0.5*(lo+hi))
        accs.append(float(correct[mask].mean()))
        confs.append(float(confid[mask].mean()))
        sizes.append(int(mask.sum()))
    
    df_rel = pd.DataFrame({"bin_mid": mids, "avg_conf": confs, "avg_acc": accs, "count": sizes})
    suffix = f"reliability_bins{('_'+tag) if tag else ''}.csv"
    try:
        df_rel.to_csv(rel_dir / suffix, index=False)
    except Exception as e:
        print(f"[Warning] Could not save reliability CSV: {e}")

    gap = np.abs(df_rel["avg_acc"] - df_rel["avg_conf"]).mean() if len(df_rel) else np.nan

    plt.figure(figsize=(6.5,6.0))
    plt.plot([0,1],[0,1],"--",label="Perfect")
    if len(df_rel): plt.plot(df_rel["avg_conf"], df_rel["avg_acc"], marker="o", label="Model")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy")
    title = f"Reliability Diagram — Epoch {epoch} (mean gap={gap:.3f})"
    if tag: title += f" [{tag}]"
    plt.title(title); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()

    png_name = f"reliability_diagram{('_'+tag) if tag else ''}.png"
    try:
        plt.savefig(rel_dir / png_name, dpi=170)
    except Exception as e:
        print(f"[Warning] Could not save reliability diagram PNG: {e}")
    plt.close()

def _save_abstention_curve(probs: np.ndarray, y_true: np.ndarray, out_dir: Path, epoch: int):
    abs_dir = Path(out_dir) / "diagnostics" / f"epoch_{epoch:03d}"
    abs_dir.mkdir(parents=True, exist_ok=True)

    confid = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y_true).astype(np.float32)

    order = np.argsort(confid)
    corr_sorted = correct[order]

    fracs, accs = [], []
    for k in range(1, len(corr_sorted)+1):
        kept = corr_sorted[-k:]
        fracs.append(k / len(corr_sorted))
        accs.append(float(kept.mean()))

    df_abs = pd.DataFrame({"kept_fraction": fracs, "accuracy": accs})
    try:
        df_abs.to_csv(abs_dir / "abstention_curve.csv", index=False)
    except Exception as e:
        print(f"[Warning] Could not save abstention_curve.csv: {e}")

    meets = [i for i,(f,a) in enumerate(zip(fracs, accs)) if a >= RETENTION_TARGET_ACC]
    best_i = meets[-1] if len(meets) else int(np.argmax(accs))
    chosen_frac, chosen_acc = fracs[best_i], accs[best_i]

    try:
        with open(abs_dir / "abstention_summary.csv", "w") as fh:
            fh.write(f"target_acc,{RETENTION_TARGET_ACC}\n")
            fh.write(f"chosen_kept_fraction,{chosen_frac:.4f}\n")
            fh.write(f"chosen_accuracy,{chosen_acc:.4f}\n")
    except Exception as e:
        print(f"[Warning] Could not save abstention_summary.csv: {e}")

    plt.figure(figsize=(7.0,5.0))
    plt.plot(df_abs["kept_fraction"], df_abs["accuracy"], linewidth=2.0)
    plt.axhline(RETENTION_TARGET_ACC, linestyle="--", alpha=0.6)
    plt.axvline(chosen_frac, linestyle="--", alpha=0.6)
    plt.xlabel("Retention fraction (keep highest-confidence k/N)")
    plt.ylabel("Accuracy on kept")
    plt.title(f"Abstention/Retention Curve — Epoch {epoch}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    try:
        plt.savefig(abs_dir / "abstention_curve.png", dpi=170)
    except Exception as e:
        print(f"[Warning] Could not save abstention_curve.png: {e}")
    plt.close()

    return chosen_frac, chosen_acc

# --- ROC helpers ---
def _save_roc_suite(y_true: np.ndarray, probs: np.ndarray, class_names: list[str], out_dir: Path, epoch: int, fpr_targets=(0.05,0.01,0.005,0.001)):
    roc_dir = Path(out_dir) / "diagnostics" / f"epoch_{epoch:03d}" / "roc"
    _save_roc_core(y_true, probs, class_names, roc_dir, title=f"ROC (one-vs-rest) — Epoch {epoch}", fpr_targets=fpr_targets)

def _save_roc_suite_to_dir(y_true: np.ndarray, probs: np.ndarray, class_names: list[str], root_dir: Path, title_tag: str="Aggregate ROC", fpr_targets=(0.05,0.01,0.005,0.001)):
    roc_dir = Path(root_dir) / "roc"
    _save_roc_core(y_true, probs, class_names, roc_dir, title=f"{title_tag}", fpr_targets=fpr_targets)

def _save_roc_core(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    roc_dir: Path,
    title: str,
    fpr_targets=(0.05, 0.01, 0.005, 0.001)
):
    # Ensure directory exists
    roc_dir.mkdir(parents=True, exist_ok=True)
    
    # Binarize labels
    K = len(class_names)
    if len(y_true) == 0 or probs.shape[0] == 0:
        print(f"[Warning] Empty y_true or probs, skipping ROC save in {roc_dir}")
        return
    
    Y = label_binarize(y_true, classes=np.arange(K))
    
    fpr_dict, tpr_dict, thr_dict, auc_dict = {}, {}, {}, {}

    # One-vs-rest ROC for each class
    for k, cname in enumerate(class_names):
        try:
            fpr, tpr, thr = roc_curve(Y[:, k], probs[:, k])
            fpr_dict[cname] = fpr
            tpr_dict[cname] = tpr
            thr_dict[cname] = thr
            auc_dict[cname] = auc(fpr, tpr)
            safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', cname)
            pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
                roc_dir / f"roc_{k:02d}_{safe_name}.csv", index=False
            )
        except Exception as e:
            print(f"[Warning] Could not create ROC for class '{cname}': {e}")
    
    # Macro and micro AUC
    try:
        micro_auc = roc_auc_score(Y, probs, average="micro", multi_class="ovr")
        macro_auc = roc_auc_score(Y, probs, average="macro", multi_class="ovr")
    except Exception as e:
        print(f"[Warning] Could not compute macro/micro AUC: {e}")
        micro_auc = macro_auc = float("nan")
    
    # Save AUC summary
    try:
        with open(roc_dir / "auc_summary.txt", "w") as fh:
            fh.write(f"micro_auc,{micro_auc:.6f}\n")
            fh.write(f"macro_auc,{macro_auc:.6f}\n")
            for cname in class_names:
                fh.write(f"{cname}_auc,{auc_dict.get(cname,float('nan')):.6f}\n")
    except Exception as e:
        print(f"[Warning] Could not save AUC summary: {e}")
    
    # Top-5 ROC plot
    try:
        top5 = sorted(class_names, key=lambda c: auc_dict.get(c,float('-inf')), reverse=True)[:5]
        plt.figure(figsize=(8.8, 6.6))
        for cname in top5:
            if cname in fpr_dict:
                plt.plot(fpr_dict[cname], tpr_dict[cname], label=f"{cname} (AUC {auc_dict[cname]:.3f})")
        plt.plot([0,1],[0,1],"--", alpha=0.6, label="Random")
        plt.title(f"{title}\nmacro {macro_auc:.3f} · micro {micro_auc:.3f}")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout(); plt.savefig(roc_dir / "roc_top5.png", dpi=170); plt.close()
    except Exception as e:
        print(f"[Warning] Could not create top-5 ROC plot: {e}")
    
    # Binary ROC (ransomware vs benign)
    if any(n.lower() == "benign" for n in class_names):
        try:
            benign_id = next(i for i, n in enumerate(class_names) if n.lower() == "benign")
            y_bin = (y_true != benign_id).astype(int)
            score_bin = 1.0 - probs[:, benign_id]

            fpr_b, tpr_b, thr_b = roc_curve(y_bin, score_bin)
            auc_b = auc(fpr_b, tpr_b)
            binary_csv = roc_dir / "roc_binary_ransomware_vs_benign.csv"
            pd.DataFrame({"fpr": fpr_b, "tpr": tpr_b, "threshold": thr_b}).to_csv(binary_csv, index=False)

            plt.figure(figsize=(7.2, 6.0))
            plt.plot(fpr_b, tpr_b, label=f"Ransomware vs Benign (AUC {auc_b:.3f})")
            plt.plot([0,1],[0,1],"--", alpha=0.6, label="Random")
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("Binary ROC")
            plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
            plt.savefig(roc_dir / "roc_binary.png", dpi=170); plt.close()
        except Exception as e:
            print(f"[Warning] Could not create binary ROC: {e}")

    # Thresholds at target FPRs
    rows = []
    for cname in class_names:
        if cname not in fpr_dict:
            continue
        fpr, tpr, thr = fpr_dict[cname], tpr_dict[cname], thr_dict[cname]
        for target in fpr_targets:
            idx = np.searchsorted(fpr, target, side="right") - 1
            idx = np.clip(idx, 0, len(fpr)-1)
            rows.append({
                "class": cname, "target_fpr": target,
                "chosen_threshold": float(thr[idx]),
                "achieved_fpr": float(fpr[idx]),
                "tpr_at_threshold": float(tpr[idx]),
            })
    try:
        pd.DataFrame(rows).to_csv(roc_dir / "thresholds_at_target_fprs.csv", index=False)
    except Exception as e:
        print(f"[Warning] Could not save thresholds CSV: {e}")
        
# ------------------- Helper Functions -------------------
import numpy as np
from typing import List

def align_probs(probs: np.ndarray, classes_present: List[int], all_classes: List[int]) -> np.ndarray:
    """
    Align probability array to include all classes.
    Missing classes get zero probability.
    """
    n_samples = probs.shape[0]
    n_classes = len(all_classes)
    aligned = np.zeros((n_samples, n_classes))
    for i, cls in enumerate(all_classes):
        if cls in classes_present:
            idx = classes_present.index(cls)
            aligned[:, i] = probs[:, idx]
    return aligned

# ================== CORE TRAIN LOOP ==================
def run_experiment(strategy: str, out_dir: Path, epochs: int = EPOCHS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    static_agent  = StaticAnalysisAgent()
    dynamic_agent = DynamicAnalysisAgent()
    network_agent = NetworkAnalysisAgent()
    fusion_agent  = FusionAgent()
    phi = LocalPhiModel() if HAS_PHI else None

    _tr_idx_fixed = None; _va_idx_fixed = None
    # single UCB instance for the whole run
    ucb = TinyUCB(ALPHAS, [0.50,0.55,0.60], [0.10,0.12,0.15])

    agentic_rows: List[Dict] = []
    metrics_rows: List[Dict] = []

    best_f1 = -1.0; best_payload = None; last_payload = None

    probs_hist: List[np.ndarray] = []
    y_val_fixed: np.ndarray | None = None
    class_names_fixed: List[str] | None = None

    for epoch in range(1, epochs+1):
        print(f"\n[{strategy}] ==================== Epoch {epoch} ====================")

        with _silence(SUPPRESS_PIPELINE_LOGS):
            static_agent.run()
            try:
                dynamic_agent.run()
            except torch.cuda.OutOfMemoryError:
                print("[Runner] Dynamic agent OOM — setting DYN_FORCE_CPU=1 and retrying.")
                os.environ["DYN_FORCE_CPU"] = "1"
                dynamic_agent.run()
            network_agent.run()

            out = fusion_agent.run()
            if len(out) == 4: Z_fused, _ybin, y_fam, _ = out
            else:             Z_fused, _ybin, y_fam = out

            if epoch == 1:
                def _pad_or_trim(x, n):
                    if x is None: return [None] * n
                    x = list(x); return x[:n] if len(x) >= n else x + [None]*(n-len(x))
                def _try_load_aligned_meta():
                    p = Path("embeddings/static_meta.csv")
                    if p.exists():
                        try:
                            d = pd.read_csv(p); d.columns = [c.lower() for c in d.columns]
                            cols = [c for c in ["sha1", "md5", "timestamp"] if c in d.columns]
                            return {c: d[c].tolist() for c in cols}
                        except Exception:
                            pass
                    return {
                        "sha1": getattr(fusion_agent, "sha1_list", None),
                        "md5": getattr(fusion_agent, "md5_list", None),
                        "timestamp": getattr(fusion_agent, "timestamp_list", None),
                    }
                n = int(len(y_fam))
                meta_src = _try_load_aligned_meta()
                sha1 = _pad_or_trim(meta_src.get("sha1") if meta_src else None, n)
                md5  = _pad_or_trim(meta_src.get("md5")  if meta_src else None, n)
                ts   = _pad_or_trim(meta_src.get("timestamp") if meta_src else None, n)
                fam  = [str(x) for x in y_fam]
                df_meta = pd.DataFrame({"sha1": sha1, "md5": md5, "family": fam, "timestamp": ts})
                Path("data").mkdir(parents=True, exist_ok=True)
                df_meta.to_csv("data/merge.csv", index=False)
                print(f"Wrote data/merge.csv with {len(df_meta)} rows "
                      f"(sha1={np.sum(pd.notna(df_meta.sha1))}, md5={np.sum(pd.notna(df_meta.md5))})")

        # ===== OUTSIDE silenced block =====
        le = LabelEncoder()
        y_all = le.fit_transform(y_fam)
        class_names_raw = [str(c) for c in le.classes_]
        label_map = _load_label_map()
        class_names = _apply_label_map(class_names_raw, label_map)
        C = len(class_names)

        # label-map sanity
        if sum(1 for n in class_names if str(n).lower() == "benign") > 1:
            print("  label_map.json may be collapsing multiple classes to 'Benign'. Verify mappings.")

        # warn if still numeric
        def _looks_numeric(s):
            try: float(s); return True
            except Exception: return False
        if all(_looks_numeric(n) for n in class_names):
            print("label_map.json missing/partial; predictions will show numeric IDs and Analysis will use generic bullets.")
        elif any(_looks_numeric(n) for n in class_names):
            unmapped = [n for n in class_names if _looks_numeric(n)]
            print(f"  Some classes unmapped and will appear numeric: {unmapped}")

        groups_path = Path("embeddings") / "groups.npy"
        groups = np.load(groups_path) if groups_path.exists() else None
        
        if ZERO_DAY_MODE:
            print("Available class names:", class_names)
            print("Held-out families:", HELD_OUT_FAMILIES)

            held_out_ids = [
                i for i, fam in enumerate(class_names)
                if fam in HELD_OUT_FAMILIES
            ]

            assert len(held_out_ids) > 0, "Held-out family not found in labels"


            is_zeroday = np.isin(y_all, held_out_ids)

            tr_idx = np.where(~is_zeroday)[0]     # SEEN families only
            va_idx = np.where(is_zeroday)[0]      # ZERO-DAY family only

            if epoch == 1:
                print(f"[Zero-Day] Held-out families: {HELD_OUT_FAMILIES}")
                print(f"[Zero-Day] Train (seen): {len(tr_idx)} | Test (zero-day): {len(va_idx)}")


        

        else:
            if _tr_idx_fixed is None:
                tr_idx, va_idx = _split_with_groups(Z_fused, y_all, groups)
                _tr_idx_fixed, _va_idx_fixed = tr_idx, va_idx
            else:
                tr_idx, va_idx = _tr_idx_fixed, _va_idx_fixed

        X_tr_all, X_val = Z_fused[tr_idx], Z_fused[va_idx]
        y_tr_all, y_val = y_all[tr_idx], y_all[va_idx]

        prog = (epoch-1)/max(1,(epochs-1))
        rng  = np.random.RandomState(SEED + 17*epoch)
        train_frac = START_FRAC + (END_FRAC - START_FRAC)*prog
        noise_p    = START_NOISE + (END_NOISE - START_NOISE)*prog

        n_keep = max(8, int(len(X_tr_all)*train_frac))
        base_idx = rng.choice(len(X_tr_all), size=n_keep, replace=False)
        X_tr, y_tr = X_tr_all[base_idx], y_tr_all[base_idx]
        y_tr_noisy = _noisify_labels(y_tr, C, noise_p, rng)

        clf = ClassificationAgent(input_dim=Z_fused.shape[1], num_classes=C, class_weighting="auto")
        if hasattr(clf, "train_with_split"):
            clf.train_with_split(X_tr, y_tr_noisy, X_val, y_val, epochs=INNER_EPOCHS_MIN, batch_size=256)
        else:
            clf.train(X_tr, y_tr_noisy, epochs=INNER_EPOCHS_MIN, batch_size=256)
            clf.X_val = torch.tensor(X_val, dtype=torch.float32, device=clf.device)
            clf.y_val = torch.tensor(y_val, dtype=torch.long, device=clf.device)

        # logits on val
        X_val_t  = torch.tensor(X_val, dtype=torch.float32, device=clf.device)
        y_true_t = torch.tensor(y_val, dtype=torch.long,   device=clf.device)
        with torch.no_grad():
            logits_val = clf.forward_logits(X_val_t)

        # ---- Calibration (TS vs VS) ----
        # ---------------- CALIBRATION (ZERO-DAY SAFE) ----------------
        def _nll(p, y):
            return float((-np.log(p[np.arange(len(y)), y] + 1e-12)).mean())

        if ZERO_DAY_MODE:
    # STRICT zero-day: no labels available for calibration
    # → reuse identity temperature (T=1.0)
            cal_ts = Calibrator("ts")
            cal_ts.T = torch.ones((), device=logits_val.device, dtype=logits_val.dtype)
            cal_ts.fitted = True
        else:
            cal_ts = Calibrator("ts").fit(logits_val, y_true_t)


        with torch.no_grad():
            z_ts = cal_ts.transform(logits_val)
            probs_cal = torch.softmax(z_ts, dim=1).detach().cpu().numpy()

        cal_used = cal_ts
        cal_mode = "ts"
        nll_cal = _nll(probs_cal, y_val)
        ece_cal = _ece(probs_cal, y_val)


        # cheap 2nd model baseline for fallback blend — BALANCED to avoid benign bias
        sec = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
        sec.fit(X_tr_all, y_tr_all); probs_sec = sec.predict_proba(X_val)
        
        all_classes = list(range(len(class_names)))  # full set of classes for this epoch

            # Align calibrated probabilities
        classes_cal = list(range(probs_cal.shape[1]))
        probs_cal_aligned = align_probs(probs_cal, classes_cal, all_classes)

            # Align secondary baseline probabilities          
            # 
        classes_sec = list(range(probs_sec.shape[1]))
        probs_sec_aligned = align_probs(probs_sec, classes_sec, all_classes)


        # UCB picks blend + thresholds (single UCB instance)
        cand_alpha, cand_conf, cand_margin = ucb.select()
        
        # ------------------ SAFE BLENDING ------------------
        p_blend = cand_alpha * probs_cal_aligned + (1.0 - cand_alpha) * probs_sec_aligned
        p_blend = np.clip(p_blend, 1e-12, 1.0)
        p_blend /= p_blend.sum(axis=1, keepdims=True)

        # Compute primary confidence & margin from the aligned calibrated probs
        primary_conf = probs_cal_aligned.max(axis=1)
        primary_margin = np.sort(probs_cal_aligned, axis=1)[:, -1] - np.sort(probs_cal_aligned, axis=1)[:, -2]
        fallback_mask = (primary_conf < cand_conf) | (primary_margin < cand_margin)

        # Apply fallback
        probs_after = probs_cal_aligned.copy()
        probs_after[fallback_mask] = p_blend[fallback_mask]

        # Final predictions
        y_pred_after = probs_after.argmax(axis=1)

        if ZERO_DAY_MODE:
            f1m_after = f1_score(
                y_val,
                y_pred_after,
                labels=held_out_ids,
                average="macro",
                zero_division=0
            )
        else:
            f1m_after = f1_score(y_val, y_pred_after, average="macro", zero_division=0)

        acc_after  = accuracy_score(y_val, y_pred_after)
        ece_after  = _ece(probs_after, y_val)
        nll_after  = _nll(probs_after, y_val)

        ucb.update((cand_alpha,cand_conf,cand_margin), -nll_after)
        print(f"[{strategy}] cal={cal_mode} a={cand_alpha:.2f} c={cand_conf:.2f} m={cand_margin:.2f} | MacroF1 {f1m_after:.3f} | ECE {ece_cal:.3f} | NLL {nll_cal:.3f}")

        _ = _save_roc_suite(y_val, probs_after, class_names, out_dir, epoch)
        
        # === Initialize fixed validation set and class order on first epoch ===
        if y_val_fixed is None:
            y_val_fixed = y_val.copy()        # y_val from current epoch
            X_val_fixed = X_val.copy()        # X_val from current epoch
            class_names_fixed = class_names[:]  # save the class order

        # === Compute probabilities on fixed validation set ===
        X_val_t = torch.tensor(X_val_fixed, dtype=torch.float32, device=clf.device)
        logits_val_fixed = clf.forward_logits(X_val_t)
        probs_cal_epoch = cal_used.transform(logits_val_fixed)
        probs_cal_epoch = torch.softmax(probs_cal_epoch, dim=1).detach().cpu().numpy()

        # === Align probabilities to fixed class order ===
        probs_aligned = np.zeros((len(y_val_fixed), len(class_names_fixed)), dtype=float)
        name_to_pos = {name: i for i, name in enumerate(class_names)}  # current epoch classes
        for i, n in enumerate(class_names):
            if n in class_names_fixed:
                j = class_names_fixed.index(n)
                probs_aligned[:, j] = probs_cal_epoch[:, i]

        # Store aligned probabilities for aggregation
        probs_hist.append(probs_aligned.copy())




        last_payload = (epoch, y_val.copy(), y_pred_after.copy(), probs_after.copy(), class_names[:])
        if f1m_after > best_f1:
            best_f1 = f1m_after
            best_payload = (epoch, y_val.copy(), y_pred_after.copy(), probs_after.copy(), class_names[:])

        _save_confusion(y_val, y_pred_after, class_names, out_dir, epoch)
        _save_reliability_diagram(probs_sec,  y_val, out_dir, epoch, n_bins=15, tag="baseline_lr")
        _save_reliability_diagram(probs_cal,  y_val, out_dir, epoch, n_bins=15, tag=f"cal_{cal_mode}")
        _save_reliability_diagram(probs_after,y_val, out_dir, epoch, n_bins=15, tag=f"post_blend")
        kept_frac, kept_acc = _save_abstention_curve(probs_after, y_val, out_dir, epoch)

        # ====== THREE-AGENT DIALOGUE (Phi-powered) ======
        try:
            # --- batch stats ---
            avg_conf_pct = float(probs_after.max(axis=1).mean() * 100.0)
            margins = np.sort(probs_after, axis=1)[:, -1] - np.sort(probs_after, axis=1)[:, -2]
            avg_margin_pct = float(margins.mean() * 100.0)

            # --- epoch headline via per-class MEAN confidence among its own predictions ---
            counts = np.bincount(y_pred_after, minlength=C)
            class_mean_conf = np.zeros(C, dtype=float)
            for k in range(C):
                mk = (y_pred_after == k)
                class_mean_conf[k] = probs_after[mk, k].mean() if mk.any() else 0.0
            # pick highest mean; tie-break by support
            order = np.lexsort((-counts, -class_mean_conf))
            headline_id = int(order[0])
            pred_family = class_names[headline_id]
            pred_conf_pct = _floor2(100.0 * float(class_mean_conf[headline_id]))

            # show top-5 headline candidates
            top_order = np.argsort(-class_mean_conf)
            top5_table = ", ".join([f"{class_names[i]}={_floor2(100.0*class_mean_conf[i]):.2f}%/n={int(counts[i])}" for i in top_order[:min(5,C)]])

            # deltas vs previous epoch
            prev_f1  = agentic_rows[-1]["f1_macro_after"] if len(agentic_rows) else None
            prev_ece = agentic_rows[-1]["ece_after"] if len(agentic_rows) else None
            d_f1  = (f1m_after - prev_f1)  if prev_f1  is not None else None
            d_ece = (ece_after - prev_ece) if prev_ece is not None else None

            # Optional feature peek: what Z-features push the headline (from balanced LR)
            feat_line = ""
            try:
                coef = sec.coef_[headline_id]  # shape [C, D] -> pick row
                top_feat_idx = np.argsort(-np.abs(coef))[:5]
                feat_line = "Top latent dims: " + ", ".join([f"z{j}(w={coef[j]:+.3f})" for j in top_feat_idx])
            except Exception:
                pass

            analyst_user = (
                f"Top-1={avg_conf_pct:.1f}% | Margin={avg_margin_pct:.1f}% | ECE={ece_after:.3f} | Calib={cal_mode}\n"
                f"Macro-F1={f1m_after:.3f} | Accuracy={acc_after:.3f} | NLL={nll_after:.3f}\n"
                f"Headline candidates: {top5_table}\n"
                f"{feat_line}\n"
                f"ALLOWED_FAMILIES (choose exactly one; do NOT invent): {', '.join(class_names)}\n"
                f"FORMAT: Prediction: <FAMILY_NAME> | Confidence: <XX.XX>%"
            )
            critic_user = (
                f"Top-1={avg_conf_pct:.1f}% | Margin={avg_margin_pct:.1f}% | ECE={ece_after:.3f} | Calib={cal_mode}\n"
                f"Macro-F1={f1m_after:.3f} (Δ{_fmt3(d_f1)}) | Accuracy={acc_after:.3f} | NLL={nll_after:.3f}\n"
                f"Prev: F1={_fmt3(prev_f1)}, ECE={_fmt3(prev_ece)} (ΔECE={_fmt3(d_ece)})\n"
                f"Retain target={RETENTION_TARGET_ACC:.2f} | kept={kept_frac:.2%} @ kept_acc={kept_acc:.3f}\n"
                f"Blend α={cand_alpha:.2f} | conf_thr={cand_conf:.2f} | margin_thr={cand_margin:.2f}\n"
                f"Headline candidates: {top5_table}"
            )
            pred_user = (
                f"F1_seq={[row['f1_macro_after'] for row in agentic_rows[-5:]] + [f1m_after]}\n"
                f"ECE_seq={[row['ece_after'] for row in agentic_rows[-5:]] + [ece_after]}\n"
                f"Avg margin={avg_margin_pct:.2f}% | Avg conf={avg_conf_pct:.2f}%\n"
                f"target_acc={RETENTION_TARGET_ACC:.2f} | kept={kept_frac:.2%} | kept_acc={kept_acc:.3f}\n"
                f"Headline candidates: {top5_table}"
            )

            def run_phi_agent(system_prompt: str, user_msg: str) -> str:
                if not HAS_PHI:
                    return "Analysis: (offline) Using metrics only.\nPrediction: Benign | Confidence: 50.00%\nNext step: Review static triage."
                return phi.generate_reply([{"role":"system","content":system_prompt},{"role":"user","content":user_msg}])

            analyst_out   = run_phi_agent(SYSTEM_ANALYST,   analyst_user)
            critic_out    = run_phi_agent(SYSTEM_CRITIC,    critic_user)
            predictor_out = run_phi_agent(SYSTEM_PREDICTOR, pred_user)

            # --- Enforce real family + exact confidence (from headline mean) ---
            pred_line = f"Prediction: {pred_family} | Confidence: {pred_conf_pct:.2f}%"
            if re.search(r"^Prediction\s*:.*$", analyst_out, flags=re.IGNORECASE | re.MULTILINE):
                analyst_out = re.sub(r"^Prediction\s*:.*$", pred_line, analyst_out, flags=re.IGNORECASE | re.MULTILINE)
            else:
                analyst_out = analyst_out.strip() + "\n" + pred_line

            # --- Force family-specific Analysis bullets based on pred_family (includes Benign) ---
            fam_bullets = _family_bullets_for(pred_family)
            analyst_out = _force_analysis_block(analyst_out, fam_bullets)

            dlg = _format_keep_as_three_agents(analyst_out, critic_out, predictor_out)

            print("\n" + dlg + "\n")

            dlg_path = out_dir / f"dialogue_epoch_{epoch}.txt"
            with open(dlg_path, "a", encoding="utf-8") as fh:
                fh.write(_condense_dialogue(dlg) + "\n")

        except Exception as e:
            print(f"[Dialogue skipped] {e}")

        agentic_rows.append({
            "epoch": epoch,
            "f1_macro_after":  float(f1m_after),
            "acc_after":  float(acc_after),
            "ece_after":  float(ece_after),
            "nll_after":  float(nll_after),
            "calibration_mode": cal_mode,
            "kept_fraction": float(kept_frac),
            "kept_accuracy": float(kept_acc),
        })
        metrics_rows.append(agentic_rows[-1].copy())

        cal_dir = Path(out_dir) / "diagnostics" / f"epoch_{epoch:03d}"
        with open(cal_dir / "calibration_summary.csv", "w") as fh:
            fh.write("mode,nll,ece\n")
            fh.write(f"chosen_{cal_mode},{nll_cal:.6f},{ece_cal:.6f}\n")
        with open(cal_dir / "calibration_params.json", "w") as fh:
            fh.write(json.dumps(cal_used.to_dict(), indent=2))

    # Save preliminary CSVs
    agentic_df = pd.DataFrame(agentic_rows)
    dfm = pd.DataFrame(metrics_rows)
    agentic_df.to_csv(out_dir / "agentic_loop_log_pre.csv", index=False)
    dfm.to_csv(out_dir / "epoch_metrics_overall_pre.csv", index=False)

    epochs_vec = agentic_df["epoch"].to_numpy()
    clarity, jargon_score, jargon_ratio, compliance = compute_clarity_for_epochs(out_dir, epochs_vec)
    ep_vec, A, C, COMP_smooth, COMP_raw = compute_agent_quality(agentic_df, clarity_scores=clarity)

    extra = pd.DataFrame({
        "epoch": epochs_vec,
        "clarity_score": clarity,
        "jargon_score": jargon_score,
        "jargon_ratio": jargon_ratio,
        "format_compliance": compliance,
        "assist_quality_smooth": A,
        "critic_quality_smooth": C,
        "composite_raw": COMP_raw,
        "composite_smooth": COMP_smooth,
    })
    agentic_df = agentic_df.merge(extra, on="epoch", how="left")

    agentic_df.to_csv(out_dir / "agentic_loop_log.csv", index=False)
    dfm = dfm.merge(agentic_df[["epoch","composite_raw","composite_smooth","clarity_score","jargon_score","jargon_ratio",
                                "assist_quality_smooth","critic_quality_smooth"]], on="epoch", how="left")
    dfm.to_csv(out_dir / "epoch_metrics_overall.csv", index=False)

    # ==== Aggregate ROC over all epochs ====
    if len(probs_hist):
        weights = []
        for P in probs_hist:
            K = len(class_names_fixed)
            Yb = label_binarize(y_val_fixed, classes=np.arange(K))
            try: w = roc_auc_score(Yb, P, average="micro", multi_class="ovr")
            except Exception: w = 1.0
            weights.append(max(1e-6, float(w)))
        weights = np.array(weights, dtype=float); weights /= weights.sum()
        stacked = np.stack(probs_hist, axis=0)             # [E, N, C]
        avg_probs = np.tensordot(weights, stacked, axes=(0, 0))  # [N, C]
        avg_probs = np.clip(avg_probs, 1e-12, 1.0); avg_probs = avg_probs / avg_probs.sum(axis=1, keepdims=True)
        agg_dir = Path(out_dir) / "diagnostics" / "final_aggregate"
        agg_dir.mkdir(parents=True, exist_ok=True)
        try:
            import joblib
            joblib.dump(y_val_fixed, agg_dir / "y_val_fixed.joblib")
            joblib.dump(class_names_fixed, agg_dir / "class_names_fixed.joblib")
            joblib.dump(avg_probs, agg_dir / "avg_probs.joblib")
        except Exception:
            pass
        _save_roc_suite_to_dir(y_val_fixed, avg_probs, class_names_fixed, root_dir=agg_dir,
                               title_tag="Aggregate ROC (weighted avg over aligned epochs)")
        try:
            K = len(class_names_fixed); Yb = label_binarize(y_val_fixed, classes=np.arange(K))
            micro = roc_auc_score(Yb, avg_probs, average="micro", multi_class="ovr")
            macro = roc_auc_score(Yb, avg_probs, average="macro", multi_class="ovr")
            print(f"[Aggregate ROC] macro {macro:.3f} · micro {micro:.3f}")
        except Exception as e:
            print(f"[Aggregate ROC] AUC computation failed: {e}")

    def _save_final(payload, tag: str):
        if payload is None: return
        ep_tag, y_t, y_p, p, names = payload
        final_dir = Path(out_dir) / "diagnostics" / f"final_{tag}"
        final_dir.mkdir(parents=True, exist_ok=True)
        _save_confusion(y_t, y_p, names, final_dir, epoch=ep_tag)
        _save_reliability_diagram(p, y_t, final_dir, epoch=ep_tag, n_bins=15, tag="final")
        _save_abstention_curve(p, y_t, final_dir, epoch=ep_tag)
        with open(final_dir / "summary.txt", "w") as fh:
            fh.write(f"source_epoch={ep_tag}\n")
            fh.write(f"macro_f1={f1_score(y_t, y_p, average='macro', zero_division=0):.6f}\n")
            fh.write(f"accuracy={accuracy_score(y_t, y_p):.6f}\n")
            fh.write(f"ece={_ece(p, y_t):.6f}\n")

    _save_final(best_payload, tag="best_f1")
    _save_final(last_payload, tag="last_epoch")

    # ==== Plots ====
    plt.figure(figsize=(9.5,5.5))
    plt.plot(dfm["epoch"], dfm["f1_macro_after"],  marker="s", markevery=MARK_EVERY, label="Macro-F1 (calibrated)")
    plt.ylim(0,1); plt.grid(True, alpha=0.3); plt.legend()
    plt.title(f"Macro-F1 over Epochs — {strategy}")
    plt.xlabel("Epoch"); plt.ylabel("Macro-F1")
    plt.tight_layout(); plt.savefig(out_dir / "macro_f1_after.png", dpi=160); plt.close()

    plt.figure(figsize=(10.5,6.2))
    ep = agentic_df["epoch"].to_numpy()
    A = agentic_df["assist_quality_smooth"].to_numpy()
    C = agentic_df["critic_quality_smooth"].to_numpy()
    COMP_smooth = agentic_df["composite_smooth"].to_numpy()
    plt.plot(ep, A,              label="AssistanceAgent", marker="o", markevery=MARK_EVERY, linewidth=2.2)
    plt.plot(ep, C,              label="CriticAgent",     marker="s", markevery=MARK_EVERY, linewidth=2.0)
    plt.plot(ep, COMP_smooth,    label="Composite (new)", linestyle="--", linewidth=2.2)
    plt.ylim(0,1.0); plt.grid(True, alpha=0.25); plt.legend(title="Agent")
    plt.title(f"Agent Response Quality (Smoothed, New Composite) — {strategy}")
    plt.xlabel("Epoch"); plt.ylabel("Quality Score (0–1)")
    plt.tight_layout(); plt.savefig(out_dir / "agent_response_quality.png", dpi=160); plt.close()

    plt.figure(figsize=(10.0,5.6))
    clarity = agentic_df["clarity_score"].to_numpy()
    jargon_ratio = agentic_df["jargon_ratio"].to_numpy()
    plt.plot(ep, clarity,      label="Clarity (↑ better)", linewidth=2.2)
    plt.plot(ep, jargon_ratio, label="Jargon ratio (↓ better)", linewidth=2.0)
    plt.ylim(0,1.0); plt.grid(True, alpha=0.25); plt.legend()
    plt.title(f"Dialogue Clarity & Jargon — {strategy}")
    plt.xlabel("Epoch"); plt.ylabel("Score / Ratio (0–1)")
    plt.tight_layout()
    plt.savefig(out_dir / "clarity_jargon.png", dpi=160); plt.close()

    print("\n✅ Completed run.")
    print("Artifacts saved in:", str(out_dir))
    print("  - agentic_loop_log.csv (includes composite, clarity/jargon, abstention)")
    print("  - epoch_metrics_overall.csv")
    print("  - macro_f1_after.png")
    print("  - agent_response_quality.png")
    print("  - clarity_jargon.png")
    print("  - diagnostics/epoch_XXX/{confusion_matrix*.png/csv, reliability_diagram_*.png, abstention_curve.png, calibration_*}")
    print("  - diagnostics/epoch_XXX/roc/{roc_top5.png, roc_*.csv, auc_summary.txt, thresholds_at_target_fprs.csv, roc_binary*}")
    print("  - diagnostics/final_aggregate/roc/*  <-- single ROC after all epochs")
    print("  - diagnostics/final_best_f1/* and diagnostics/final_last_epoch/*")

    return agentic_df, dfm

# ================== ORCHESTRATOR ==================200
def main():
    base = Path(".")
    runs = [("none", base / "graphs_zeroday_cry_new")]
    results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for strat, out_dir in runs:
        results[strat] = run_experiment(strat, out_dir, epochs=EPOCHS)

    _, dfm = results["none"]
    plt.figure(figsize=(11.5,6.5))
    plt.plot(dfm["epoch"], dfm["f1_macro_after"], label="Calibrated")
    plt.ylim(0,1); plt.grid(True, alpha=0.3); plt.legend()
    plt.title("Macro-F1 over Epochs")
    plt.xlabel("Epoch"); plt.ylabel("Macro-F1")
    plt.tight_layout(); plt.savefig(base / "macro_f1_all.png", dpi=180); plt.close()

    print("\nAggregate overlays saved at project root:")
    print("  - macro_f1_all.png")

if __name__ == "__main__":
    main()
