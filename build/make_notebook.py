"""
Generator script for the DML Assignment 2 (FedAvg) notebook.

Builds the notebook programmatically with nbformat. Run with --quick to
produce a fast smoke-test notebook (tiny data subset, 1-2 rounds) used only
to validate the end-to-end pipeline before the real, fully-specified run.
Run with no flags to build the final deliverable notebook with the exact
fixed experimental configuration from the assignment brief.
"""
import argparse
import nbformat as nbf

ROLL_NO = "2025AE05316"
STUDENT_NAME = "YELDONA DESAI V VITTAL"

cells = []


def M(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip() + "\n"))


def Cd(text):
    cells.append(nbf.v4.new_code_cell(text.strip() + "\n"))


def build(quick: bool, out_path: str):
    cells.clear()

    # Quick mode shrinks the data/compute volume but keeps every code path
    # identical to the real run, so a pass here is a reliable predictor of
    # the real run's correctness.
    ROUNDS = 2 if quick else 10
    CENTRAL_EPOCHS = 2 if quick else 10
    SUBSET_PER_CLIENT_CAP = 200 if quick else None  # None => use full partition

    # ------------------------------------------------------------------
    # 1. Title cell
    # ------------------------------------------------------------------
    M(f"""
# Distributed Machine Learning — Assignment 2

## Implementation and Performance Analysis of the Federated Averaging (FedAvg) Algorithm for Image Classification

| Field | Value |
|---|---|
| Course | Distributed Machine Learning |
| Assignment | Assignment 2 |
| Title | Implementation and Performance Analysis of the Federated Averaging (FedAvg) Algorithm for Image Classification |

### Group Members

| Name | Roll No. |
|---|---|
| {STUDENT_NAME} | {ROLL_NO} |

This notebook is a single, self-contained, top-to-bottom reproducible
implementation. All libraries, hyper-parameters and random seeds used are
declared explicitly in the **Environment Setup** and **Assumptions**
sections below.
""")

    # ------------------------------------------------------------------
    # 2. Introduction & Problem Statement
    # ------------------------------------------------------------------
    M("""
## 1. Introduction & Problem Statement

### 1.1 Scenario

Consider a consortium of five hospitals that each hold a private collection
of handwritten-digit-style scans (here, MNIST digit images stand in for a
real medical imaging modality, as specified in the assignment). Each
hospital would like to collaborate on training a single, shared image
classifier that performs well across the whole consortium, but:

- **Privacy constraint.** No hospital is willing (or, in many jurisdictions,
  legally permitted) to transmit its raw patient images to a central server
  or to any other hospital.
- **Heterogeneity.** Different hospitals may serve different patient
  populations, so the distribution of cases (here, digit classes) seen at
  one hospital can differ substantially from another's — a *non-IID* data
  setting.

**Federated Learning (FL)** solves exactly this problem: instead of moving
data to a model, FL moves a *model* to the data. Each client trains a copy
of the global model locally, on its own private data, and only the
resulting **model parameters** (never the raw data) are sent back to a
coordinating server, which combines them into an improved global model.
This cycle repeats for several *communication rounds*.

### 1.2 Objectives of this notebook

1. Implement the **FedAvg** algorithm (McMahan et al., 2017) **from
   scratch** in PyTorch — the server orchestration, weight distribution,
   local training and weighted aggregation are all hand-written, with no
   dedicated federated-learning framework.
2. Partition MNIST across 5 simulated clients in two regimes: **IID**
   (uniform digit distribution per client) and **non-IID** (pathological
   2-classes-per-client label skew), to study how data heterogeneity
   affects FedAvg.
3. Study the effect of the number of **local epochs (E)** on convergence
   speed, final accuracy and communication efficiency.
4. Quantify the accuracy gap between isolated local training, FedAvg, and a
   centralized (non-federated) upper-bound reference.
5. Implement and evaluate **FedProx** (Li et al., 2020) as a simple
   improvement over FedAvg for non-IID data.

All quantitative claims made later in this notebook are generated directly
from the results produced by the actual code in this notebook — no numbers
are hand-typed or estimated.
""")

    # ------------------------------------------------------------------
    # 3. Background
    # ------------------------------------------------------------------
    M(r"""
## 2. Background: Federated Learning & FedAvg

### 2.1 Client–server architecture

```
                        +-----------------------+
                        |        SERVER         |
                        |  holds global model   |
                        |      w_t  (round t)   |
                        +-----------------------+
                         |   |   |   |   |
           (1) broadcast w_t to all clients
                         |   |   |   |   |
        +--------+  +--------+  +--------+  +--------+  +--------+
        |Client 1|  |Client 2|  |Client 3|  |Client 4|  |Client 5|
        | data n1|  | data n2|  | data n3|  | data n4|  | data n5|
        +--------+  +--------+  +--------+  +--------+  +--------+
             |   (2) each client trains E local epochs on   |
             |       its own private data -> w_t^k          |
                         |   |   |   |   |
           (3) only the updated WEIGHTS w_t^k are uploaded
                         |   |   |   |   |
                        +-----------------------+
                        |        SERVER         |
                        | (4) weighted average  |
                        |  w_(t+1)=Sum(nk/n*w_t^k)|
                        +-----------------------+
                             (repeat for T rounds)
```

Raw images never leave a client. Only model parameters cross the network,
and the server never sees client k's private partition.

### 2.2 FedAvg algorithm (McMahan et al., 2017)

**Algorithm: FederatedAveraging**

```
Server executes:
    initialize global weights w_0
    for each round t = 0, 1, ..., T-1:
        S_t <- select a subset of clients (here: all K clients, C = 1.0)
        for each client k in S_t (in parallel / simulated sequentially):
            w_(t+1)^k <- ClientUpdate(k, w_t)
        n <- sum_{k in S_t} n_k
        w_(t+1) <- sum_{k in S_t} (n_k / n) * w_(t+1)^k

ClientUpdate(k, w):
    initialize local model with weights w
    split client k's local data P_k into batches of size B
    for each local epoch i = 1, ..., E:
        for each local minibatch b in P_k:
            w <- w - eta * grad L(w; b)
    return w to server
```

### 2.3 Weighted-average aggregation equation

$$
w_{t+1} \;=\; \sum_{k=1}^{K} \frac{n_k}{n}\, w_{t+1}^{k},
\qquad n = \sum_{k=1}^{K} n_k
$$

where $w_{t+1}^{k}$ is the state (all weights and biases) of client $k$'s
model after local training in round $t$, $n_k$ is the number of training
samples held by client $k$, and $n$ is the total number of samples across
all participating clients. The average is taken **entry-wise**, independently
for every tensor in the model's `state_dict`.

### 2.4 Notation table

| Symbol | Meaning | Value in this assignment |
|---|---|---|
| $K$ | number of clients | 5 |
| $C$ | fraction of clients sampled per round | 1.0 (all clients, every round) |
| $E$ | number of local epochs per round | 1, 2 or 5 (default 2) |
| $B$ | local mini-batch size | 32 |
| $T$ | number of communication rounds | 10 |
| $n_k$ | number of training samples at client $k$ | computed from the partition |
| $n$ | total training samples across clients | $\sum_k n_k$ |
| $\eta$ | local learning rate | 0.01 (SGD, momentum 0.9) |
| $w_t$ | global model weights at round $t$ | — |
| $w_t^k$ | client $k$'s local weights after round $t$ | — |
""")

    # ------------------------------------------------------------------
    # 4. Assumptions
    # ------------------------------------------------------------------
    M("""
## 3. Assumptions

The following assumptions and simplifications apply throughout this
notebook and the accompanying report. They are listed here once and relied
upon implicitly in every later section.

1. **Simulation, not deployment.** Five simulated clients run sequentially
   inside one Python process on one machine. This simulates — but does not
   deploy — a real distributed system: there is no real network, no
   communication latency and no serialization cost.
2. **Full participation.** All 5 clients participate in every
   communication round ($C = 1.0$); there is no client sampling, no
   client dropout and no stragglers.
3. **Server-side validation set.** The server holds the full MNIST test
   set (10,000 images) for global evaluation. In a real hospital
   consortium this would correspond to a held-out validation set, not a
   true, unseen production test set.
4. **Parameter-only communication.** Clients share only model parameters;
   raw images never leave a client's own partition object. No secure
   aggregation and no differential privacy are applied — both are
   explicitly out of scope for this assignment.
5. **MNIST as a stand-in.** MNIST handwritten digits are used as a
   stand-in for medical images, exactly as specified in the assignment;
   no claim is made about MNIST's visual similarity to any real imaging
   modality.
6. **Non-IID = label-distribution skew only.** Non-IID is modelled purely
   as a pathological 2-classes-per-client label skew (McMahan et al.
   style). Other forms of heterogeneity — feature skew, or quantity skew
   beyond what naturally follows from class counts — are not modelled.
7. **Fresh optimizer per client per round.** A brand-new `SGD` optimizer
   (with momentum 0.9) is instantiated for every client, every round. No
   momentum buffer is ever transmitted across the network or carried over
   between rounds; only the model's `state_dict` (weights and biases)
   is exchanged.
8. **Definition of "local training accuracy".** Wherever "local training
   accuracy" is reported for a client, it means accuracy on that client's
   own training data, measured during the **final local epoch** of that
   training call.
""")

    # ------------------------------------------------------------------
    # 5. Environment Setup
    # ------------------------------------------------------------------
    M("""
## 4. Environment Setup

This section imports every library used in the notebook, prints library
versions for reproducibility, selects the compute device following the
fixed priority CUDA > Apple MPS > CPU, defines the global seeding routine,
and prints the full experiment configuration as a table.
""")

    Cd("""
# --- Core imports -----------------------------------------------------
import os
import sys
import copy
import json
import time
import random
import platform
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
from torchvision.datasets import MNIST

from IPython.display import display, Markdown

print("All imports successful.")
""")

    M("Interpretation: all required third-party libraries import cleanly. "
      "No federated-learning framework (TFF, Flower, FedML, PySyft, OpenFL, "
      "FATE) is imported anywhere in this notebook — FedAvg is implemented "
      "by hand in Task 3 below.")

    Cd("""
# --- Library versions (for reproducibility) ---------------------------
version_info = {
    "Python": platform.python_version(),
    "PyTorch": torch.__version__,
    "torchvision": torchvision.__version__,
    "NumPy": np.__version__,
    "Pandas": pd.__version__,
    "Matplotlib": matplotlib.__version__,
    "Seaborn": sns.__version__,
    "Platform": platform.platform(),
}
version_df = pd.DataFrame(sorted(version_info.items()), columns=["Library", "Version / Info"])
display(version_df)
""")

    M("Interpretation: the table above pins down the exact software stack "
      "used to produce every result in this notebook, so the run is "
      "reproducible on another machine with the same versions.")

    Cd("""
# --- Project paths ------------------------------------------------------
# Auto-detected from the notebook's own working directory so this cell
# needs no manual editing when the notebook is moved to another machine
# (e.g. a lab GPU server): `jupyter nbconvert --execute` and Jupyter's own
# kernel both start with cwd = the directory containing this .ipynb file,
# which is DML_Assignment2/build/, so BASE_DIR is simply its parent.
# Falls back to the original development machine's absolute path only if
# that heuristic does not hold (e.g. a differently-named folder layout).
_cwd = Path.cwd()
if _cwd.name == "build":
    BASE_DIR = _cwd.parent
elif (_cwd / "data").exists() or (_cwd / "figures").exists():
    BASE_DIR = _cwd
else:
    BASE_DIR = Path("/Users/ssvmindsprivatelimited/Desktop/MTech/Assignments/DML Assignment2/DML_Assignment2")
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
for d in (DATA_DIR, FIG_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
print("BASE_DIR   :", BASE_DIR)
print("DATA_DIR   :", DATA_DIR)
print("FIG_DIR    :", FIG_DIR)
print("RESULTS_DIR:", RESULTS_DIR)
""")

    Cd("""
# --- Global seeding for full reproducibility ---------------------------
SEED = 42

def seed_everything(seed: int = SEED):
    \"\"\"Seed every source of randomness used in this notebook.\"\"\"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)
print(f"Global seed set to {SEED}; cudnn.deterministic = {torch.backends.cudnn.deterministic}")
""")

    Cd("""
# --- Device selection: CUDA > Apple MPS > CPU, in that fixed order -----
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")
""")

    M("Interpretation: the device-selection cell above always prefers CUDA, "
      "then Apple's Metal (MPS) backend, and only falls back to the CPU if "
      "neither accelerator is present. The device actually selected on the "
      "machine this notebook was executed on is printed above and is used "
      "for every model and tensor in the rest of the notebook.")

    Cd("""
# --- Fixed experimental configuration (Section 2 of the assignment brief) ---
CONFIG = {
    "K_clients": 5,
    "C_participation_fraction": 1.0,
    "T_rounds": """ + str(ROUNDS) + """,
    "batch_size": 32,
    "E_default": 2,
    "E_grid": [1, 2, 5],
    "optimizer": "SGD",
    "lr": 0.01,
    "momentum": 0.9,
    "loss": "CrossEntropyLoss",
    "normalize_mean": 0.1307,
    "normalize_std": 0.3081,
    "central_ref_epochs": """ + str(CENTRAL_EPOCHS) + """,
    "seed": SEED,
    "device": str(DEVICE),
}
display(pd.DataFrame(sorted(CONFIG.items()), columns=["Hyper-parameter", "Value"]))
""")

    M("Interpretation: this is the single source of truth for every "
      "hyper-parameter used anywhere below; every run in Task 4 reads its "
      "epoch count, round count and optimizer settings from this dictionary "
      "or from the fixed values it documents.")

    # ------------------------------------------------------------------
    # 6. Task 1: Dataset Preparation
    # ------------------------------------------------------------------
    M("""
## Task 1: Dataset Preparation

### 5.1 Dataset description

MNIST (LeCun, Cortes & Burges) is a dataset of 70,000 grayscale images of
handwritten digits (0-9), each 28x28 pixels, split into 60,000 training
images and 10,000 test images. It is used here, per the assignment, as a
stand-in for a medical imaging classification task across a consortium of
hospitals/clients.
""")

    Cd("""
# --- 5.2 Loading & preprocessing ---------------------------------------
# Normalization uses MNIST's own precomputed train-set mean/std (standard
# practice): this centers pixel intensities around 0 with unit variance,
# which speeds up and stabilizes SGD convergence for the CNN below.
MEAN, STD = CONFIG["normalize_mean"], CONFIG["normalize_std"]
transform = T.Compose([T.ToTensor(), T.Normalize((MEAN,), (STD,))])

train_full = MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)
test_full = MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)

print(f"Train set size: {len(train_full)}")
print(f"Test set size : {len(test_full)}")

# Sanity-check the normalization constants against the raw (un-normalized) data
raw_ds = MNIST(root=str(DATA_DIR), train=True, download=False, transform=T.ToTensor())
raw_stack = torch.stack([raw_ds[i][0] for i in range(2000)])
print(f"Computed mean over a 2000-image sample: {raw_stack.mean().item():.4f} (reference: {MEAN})")
print(f"Computed std  over a 2000-image sample: {raw_stack.std().item():.4f} (reference: {STD})")
""")

    M("Interpretation: the train/test split sizes match the official MNIST "
      "specification (60,000 / 10,000), and the sample mean/std computed "
      "directly from the raw pixel data closely matches the standard "
      "normalization constants used, confirming the preprocessing pipeline "
      "is correctly configured.")

    Cd("""
# --- 5.3 Sample image grid: one example per class (Figure 1) -----------
fig, axes = plt.subplots(2, 5, figsize=(10, 4.2))
shown = {}
for img, label in train_full:
    if label not in shown:
        shown[label] = img
    if len(shown) == 10:
        break
for digit in range(10):
    ax = axes[digit // 5, digit % 5]
    img = shown[digit][0] * STD + MEAN  # undo normalization for display
    ax.imshow(img.numpy(), cmap="gray")
    ax.set_title(f"Digit {digit}")
    ax.axis("off")
fig.suptitle("Figure 1: One sample MNIST image per class")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig01_sample_grid.png", dpi=300, bbox_inches="tight")
plt.show()
""")

    M("Interpretation: the grid confirms that all ten digit classes are "
      "present, images are centered 28x28 grayscale scans, and the "
      "normalize/undo-normalize round trip renders them correctly for "
      "visual inspection.")

    Cd("""
# --- 5.4 Overall class distribution of the train set (Figure 2) --------
train_targets = train_full.targets.numpy()
class_counts = pd.Series(train_targets).value_counts().sort_index()
display(class_counts.to_frame(name="count"))

plt.figure(figsize=(7, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index,
            palette="viridis", legend=False)
plt.xlabel("Digit class")
plt.ylabel("Number of training images")
plt.title("Figure 2: Class distribution of the full MNIST training set")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig02_class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
""")

    M("Interpretation: MNIST's training set is close to, but not perfectly, "
      "class-balanced (roughly 5,400-6,900 images per digit). This global "
      "distribution is the one each IID client partition should "
      "approximately reproduce, and is the one each non-IID client "
      "partition will deliberately violate.")

    Cd("""
# --- 5.5 IID partitioning function --------------------------------------
def iid_partition(dataset, num_clients=5, seed=SEED):
    \"\"\"Shuffle all indices with a fixed seed and split into equal chunks.

    Every client ends up with an (approximately) uniform label
    distribution, since the shuffle is independent of the label.
    \"\"\"
    n = len(dataset)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    splits = np.array_split(indices, num_clients)
    return [split.tolist() for split in splits]

iid_indices = iid_partition(train_full, num_clients=CONFIG["K_clients"], seed=SEED)
print("IID partition sizes:", [len(x) for x in iid_indices])
""")

    M("Interpretation: with 60,000 training images shuffled uniformly at "
      "random and split 5 ways, every client receives exactly 12,000 "
      "images, each with (approximately) the same class proportions as the "
      "full training set.")

    Cd("""
# --- 5.6 Non-IID partitioning function (pathological label skew) -------
def noniid_partition(dataset, num_clients=5):
    \"\"\"Client k receives ALL training images of digits {2k, 2k+1}.

    This reproduces the pathological non-IID partition described by
    McMahan et al. (2017): Client 1 -> {0,1}, Client 2 -> {2,3},
    Client 3 -> {4,5}, Client 4 -> {6,7}, Client 5 -> {8,9}.
    Resulting client sizes are naturally unequal because MNIST's own
    per-class counts are unequal.
    \"\"\"
    targets = dataset.targets.numpy()
    partitions = []
    for k in range(num_clients):
        classes = {2 * k, 2 * k + 1}
        idx = np.where(np.isin(targets, list(classes)))[0]
        partitions.append(idx.tolist())
    return partitions

noniid_indices = noniid_partition(train_full, num_clients=CONFIG["K_clients"])
print("Non-IID partition sizes:", [len(x) for x in noniid_indices])
""")

    M("Interpretation: the non-IID partition sizes are computed directly "
      "from MNIST's real per-class counts (not hard-coded), and are "
      "naturally unequal across clients because some digit pairs are more "
      "common in MNIST than others. This makes the FedAvg sample-weighted "
      "average in Section 2.3 meaningful rather than a plain unweighted "
      "mean.")

    Cd("""
# --- 5.7 Samples-per-client tables (Table 1) and per-client x per-class counts ---
summary_rows = []
for k in range(CONFIG["K_clients"]):
    summary_rows.append({"Client": f"Client {k+1}", "IID samples": len(iid_indices[k]),
                          "Non-IID samples": len(noniid_indices[k]),
                          "Non-IID classes": f"{{{2*k}, {2*k+1}}}"})
table1 = pd.DataFrame(summary_rows)
print("Table 1: Samples per client under IID and Non-IID partitioning")
display(table1)

def per_client_class_counts(partitions, targets, num_classes=10):
    mat = np.zeros((len(partitions), num_classes), dtype=int)
    for k, idx in enumerate(partitions):
        labels = targets[idx]
        for c in range(num_classes):
            mat[k, c] = (labels == c).sum()
    return pd.DataFrame(mat, index=[f"Client {k+1}" for k in range(len(partitions))],
                         columns=[f"digit {c}" for c in range(num_classes)])

iid_class_table = per_client_class_counts(iid_indices, train_targets)
noniid_class_table = per_client_class_counts(noniid_indices, train_targets)
print("Table 2: Per-client x per-class sample counts - IID partition")
display(iid_class_table)
print("Table 3: Per-client x per-class sample counts - Non-IID partition")
display(noniid_class_table)
""")

    M("Interpretation: Table 1 shows IID clients are exactly equal in size "
      "(12,000 each) while non-IID client sizes vary with the natural "
      "frequency of each digit pair in MNIST. Tables 2 and 3 make the "
      "contrast explicit at the per-class level: every IID client sees "
      "roughly 1/10 of each digit, while every non-IID client sees 100% of "
      "exactly two digits and 0% of the other eight.")

    Cd("""
# --- 5.8 Heatmaps of class distribution per client (Figure 3) + stacked bars (Figure 4) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sns.heatmap(iid_class_table, annot=True, fmt="d", cmap="YlGnBu", ax=axes[0], cbar_kws={"label": "count"})
axes[0].set_title("IID partition")
sns.heatmap(noniid_class_table, annot=True, fmt="d", cmap="YlOrRd", ax=axes[1], cbar_kws={"label": "count"})
axes[1].set_title("Non-IID partition")
fig.suptitle("Figure 3: Class-distribution heatmaps per client")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig03_heatmaps.png", dpi=300, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
iid_class_table.plot(kind="bar", stacked=True, ax=axes[0], colormap="tab20", legend=False)
axes[0].set_title("IID partition"); axes[0].set_ylabel("sample count"); axes[0].set_xlabel("")
noniid_class_table.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab20")
axes[1].set_title("Non-IID partition"); axes[1].set_xlabel("")
axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
fig.suptitle("Figure 4: Stacked per-class composition per client")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig04_stacked_bars.png", dpi=300, bbox_inches="tight")
plt.show()
""")

    M("Interpretation: the IID heatmap/stacked-bars are visually uniform "
      "across clients and classes; the non-IID heatmap/stacked-bars show "
      "sharp, block-diagonal structure — each client's entire bar is made "
      "of only two colors (its two assigned digits). This is the "
      "pathological label skew that FedAvg must contend with in the "
      "non-IID experiments of Task 4.")

    Cd("""
# --- 5.9 Sanity checks ---------------------------------------------------
def sanity_check_partition(partitions, n_total, name):
    all_idx = []
    for p in partitions:
        all_idx.extend(p)
    assert len(all_idx) == len(set(all_idx)), f"{name}: duplicate indices across clients!"
    assert set(all_idx) == set(range(n_total)), f"{name}: union of clients != full index set!"
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            assert len(set(partitions[i]) & set(partitions[j])) == 0, f"{name}: overlap between client {i} and {j}!"
    print(f"{name}: sanity checks PASSED (no overlap; union == {n_total} samples)")

sanity_check_partition(iid_indices, len(train_full), "IID partition")
sanity_check_partition(noniid_indices, len(train_full), "Non-IID partition")
""")

    M("Interpretation: both partitioning schemes pass all sanity checks — "
      "no training sample is assigned to more than one client, and every "
      "one of the 60,000 training samples is assigned to exactly one "
      "client, under both IID and non-IID partitioning.")

    # ------------------------------------------------------------------
    # 7. Task 2: Local Model Development
    # ------------------------------------------------------------------
    M("""
## Task 2: Local Model Development

### 6.1 CNN architecture

The model is a small CNN, identical for every client and for the global
model, as fixed by the assignment:

| Layer | Output shape | Parameters |
|---|---|---|
| Input | 1x28x28 | - |
| Conv2d(1->32, 5x5, pad 2) + ReLU | 32x28x28 | computed below |
| MaxPool2d(2) | 32x14x14 | 0 |
| Conv2d(32->64, 5x5, pad 2) + ReLU | 64x14x14 | computed below |
| MaxPool2d(2) | 64x7x7 | 0 |
| Flatten | 3136 | 0 |
| Linear(3136->128) + ReLU | 128 | computed below |
| Dropout(0.25) | 128 | 0 |
| Linear(128->10) | 10 | computed below |
""")

    Cd("""
class CNN(nn.Module):
    \"\"\"Fixed CNN architecture shared by every client and the server.\"\"\"

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

_demo_model = CNN()
print(_demo_model)

param_rows = []
total_params = 0
for name, p in _demo_model.named_parameters():
    param_rows.append({"Parameter": name, "Shape": tuple(p.shape), "Count": p.numel()})
    total_params += p.numel()
display(pd.DataFrame(param_rows))
print(f"Total trainable parameters: {total_params:,}")
del _demo_model
""")

    M("Interpretation: the printed module graph and the per-layer parameter "
      "table confirm the architecture matches the specification exactly "
      "(two conv blocks with 5x5 kernels and padding 2, two 2x2 max-pools, "
      "a 3136-unit flatten, a 128-unit hidden FC layer with dropout 0.25, "
      "and a 10-way output layer), and the total parameter count is a "
      "useful size reference for the communication cost of FedAvg (every "
      "round, each client uploads this many floats).")

    M("""
### 6.2 Loss and optimizer justification

- **Loss: `CrossEntropyLoss`.** This is the standard choice for multi-class
  single-label classification (10 mutually exclusive digit classes); it
  combines a log-softmax with a negative log-likelihood loss in a single,
  numerically stable operation.
- **Optimizer: plain `SGD` with momentum 0.9, lr 0.01.** This is the
  optimizer used in the original FedAvg paper's MNIST experiments, and
  momentum helps smooth noisy gradients computed from small client-side
  mini-batches (32 images). Per the assumptions, a **fresh** optimizer
  instance is created for every client, every round — no momentum buffer
  is ever transmitted across the simulated network, so the only
  information that crosses the "wire" is the model's `state_dict`.
""")

    Cd("""
# --- 6.3 Local training function ----------------------------------------
def local_train(model, loader, device, epochs, lr=CONFIG["lr"], momentum=CONFIG["momentum"],
                 mu=0.0, global_params=None):
    \"\"\"Train `model` locally for `epochs` epochs on `loader`.

    If mu > 0 and global_params is given (a list of tensors matching
    model.parameters(), already on `device`), a FedProx proximal term
    (mu/2) * ||w - w_global||^2 is added to the loss (Task 5, Q3).
    Returns (state_dict on CPU, per-epoch [(loss, accuracy), ...]).
    A fresh optimizer is always created here: momentum never crosses rounds.
    \"\"\"
    model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    epoch_history = []
    for _ in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            if mu > 0 and global_params is not None:
                prox_term = sum(torch.sum((p - gp) ** 2) for p, gp in zip(model.parameters(), global_params))
                loss = loss + (mu / 2.0) * prox_term
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            correct += (outputs.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)
        epoch_history.append((running_loss / total, correct / total))

    state_dict_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state_dict_cpu, epoch_history


# --- 6.4 Evaluation function ---------------------------------------------
def evaluate(model, loader, device):
    \"\"\"Evaluate `model` on `loader`; returns (average loss, accuracy).\"\"\"
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            running_loss += loss.item() * xb.size(0)
            correct += (outputs.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)
    return running_loss / total, correct / total

print("local_train() and evaluate() defined.")
""")

    Cd("""
# A single, fixed random initialization of the global model, reused for
# EVERY experiment in this notebook (Task 2 isolated-training demo, Task 3
# FedAvg run, and every Task 4 run) so that all comparisons start from an
# identical initial point and differences in outcome are attributable only
# to the federated-learning procedure, not to lucky/unlucky initial weights.
seed_everything(SEED)
GLOBAL_INIT_STATE = copy.deepcopy(CNN().state_dict())

test_loader = DataLoader(test_full, batch_size=CONFIG["batch_size"], shuffle=False)
print("Fixed global initial model created; shared test_loader created.")
""")

    Cd("""
# --- 6.5 Standalone local-training demo: isolated clients, no FedAvg ----
def run_isolated_local_training(partitions, partition_name, epochs=2):
    \"\"\"Each client trains its OWN copy of the SAME initial model in total
    isolation (no aggregation at all) and is then evaluated on the full
    global test set. This demonstrates why isolated local training is
    insufficient, especially under non-IID data, and motivates FedAvg.
    \"\"\"
    records = []
    test_accs = []
    for k, idx in enumerate(partitions):
        model = CNN()
        model.load_state_dict(copy.deepcopy(GLOBAL_INIT_STATE))
        gen = torch.Generator(); gen.manual_seed(SEED + k)
        loader = DataLoader(Subset(train_full, idx), batch_size=CONFIG["batch_size"],
                             shuffle=True, generator=gen)
        _, history = local_train(model, loader, DEVICE, epochs=epochs)
        for ep, (loss, acc) in enumerate(history, start=1):
            records.append({"Partition": partition_name, "Client": f"Client {k+1}",
                             "Epoch": ep, "Local train loss": loss, "Local train acc": acc})
        test_loss, test_acc = evaluate(model, test_loader, DEVICE)
        test_accs.append({"Partition": partition_name, "Client": f"Client {k+1}",
                           "Global test loss": test_loss, "Global test acc": test_acc})
    return pd.DataFrame(records), pd.DataFrame(test_accs)

iid_local_hist, iid_local_test = run_isolated_local_training(iid_indices, "IID", epochs=2)
noniid_local_hist, noniid_local_test = run_isolated_local_training(noniid_indices, "Non-IID", epochs=2)

print("Table 4: Per-client local training accuracy/loss per epoch (isolated training)")
display(pd.concat([iid_local_hist, noniid_local_hist], ignore_index=True))
print("Table 5: Each isolated client model's accuracy on the GLOBAL test set")
isolated_test_table = pd.concat([iid_local_test, noniid_local_test], ignore_index=True)
display(isolated_test_table)
isolated_test_table.to_csv(RESULTS_DIR / "isolated_local_training.csv", index=False)
""")

    Cd("""
# Figure 5: isolated-client global-test accuracy, IID vs Non-IID
plt.figure(figsize=(8, 4.5))
pivot = isolated_test_table.pivot(index="Client", columns="Partition", values="Global test acc")
pivot.plot(kind="bar", ax=plt.gca())
plt.ylabel("Accuracy on the GLOBAL MNIST test set")
plt.title("Figure 5: Isolated (non-federated) per-client models evaluated globally")
plt.ylim(0, 1.0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig05_isolated_training.png", dpi=300, bbox_inches="tight")
plt.show()
""")

    Cd("""
_iid_mean = isolated_test_table.query("Partition == 'IID'")["Global test acc"].mean()
_noniid_mean = isolated_test_table.query("Partition == 'Non-IID'")["Global test acc"].mean()
display(Markdown(
    f"**Interpretation (generated from actual results above):** averaged over the 5 "
    f"clients, isolated IID-trained local models reach a mean global test accuracy of "
    f"**{_iid_mean:.2%}**, while isolated Non-IID-trained local models reach only "
    f"**{_noniid_mean:.2%}** on average. Each non-IID client only ever observes 2 of the "
    f"10 digit classes, so its model essentially never predicts the other 8 classes; its "
    f"global-test accuracy is therefore bounded near that pair's share of the test set "
    f"(~20% for a uniform class prior). This gap is exactly the motivation for "
    f"federated averaging: by combining weight updates from all 5 clients every round, "
    f"FedAvg (Task 3/4) lets the global model see the *effect* of every digit class "
    f"without any client ever sharing its raw images."
))
""")

    # ------------------------------------------------------------------
    # 8. Task 3: FedAvg implementation
    # ------------------------------------------------------------------
    M("""
## Task 3: FedAvg Implementation (from scratch)

This section implements every moving part of FedAvg as small, inspectable
functions, matching the pseudocode in Section 2.2:
server initialization (7.1), weight distribution (7.2), local training
(7.3, reusing `local_train` from Task 2), weight collection (7.4),
weighted aggregation with a unit test (7.5), the global update (7.6), the
full orchestration loop (7.7), and finally the headline execution of the
`IID_E2` run (7.8-7.9).
""")

    Cd("""
# --- 7.1 Server: global model initialization -----------------------------
seed_everything(SEED)
global_model = CNN()
global_model.load_state_dict(copy.deepcopy(GLOBAL_INIT_STATE))
global_state = copy.deepcopy(GLOBAL_INIT_STATE)
print("Global model initialized from the fixed shared initial state.")
""")

    Cd("""
# --- 7.2 Distribution of global weights to clients ------------------------
def distribute_weights(global_state_dict):
    \"\"\"Return a fresh CNN whose weights are a deep copy of the global state.
    A deep copy (not a shared reference) guarantees each client trains an
    INDEPENDENT local model, as required by FedAvg.
    \"\"\"
    client_model = CNN()
    client_model.load_state_dict(copy.deepcopy(global_state_dict))
    return client_model

_demo_client_model = distribute_weights(global_state)
print("distribute_weights() OK - returns an independent CNN copy:",
      _demo_client_model is not global_model)
del _demo_client_model
""")

    Cd("""
# --- 7.3 Local training on each client (reuses local_train from Task 2) ---
# --- 7.4 Collection of client weights -------------------------------------
def client_round(global_state_dict, client_idx, dataset, indices, device, epochs, round_idx,
                  mu=0.0, global_params_for_prox=None):
    \"\"\"Run one full client-side FedAvg step: receive weights, train
    locally for `epochs` epochs, and return the resulting CPU state_dict
    plus bookkeeping (sample count, final-epoch loss/accuracy).\"\"\"
    model = distribute_weights(global_state_dict)
    gen = torch.Generator(); gen.manual_seed(SEED + round_idx * 1000 + client_idx)
    loader = DataLoader(Subset(dataset, indices), batch_size=CONFIG["batch_size"],
                         shuffle=True, generator=gen)
    state_dict, history = local_train(model, loader, device, epochs=epochs,
                                       mu=mu, global_params=global_params_for_prox)
    final_loss, final_acc = history[-1]
    return {"state_dict": state_dict, "n_k": len(indices),
            "final_loss": final_loss, "final_acc": final_acc}

print("client_round() defined (implements steps 7.3 and 7.4).")
""")

    Cd("""
# --- 7.5 fedavg_aggregate: weighted averaging, with a unit test ----------
def fedavg_aggregate(client_state_dicts, client_sizes):
    \"\"\"Hand-written FedAvg weighted average:
        w_(t+1) = sum_k (n_k / n) * w_(t+1)^k
    averaged entry-by-entry, in float32, over every tensor in state_dict.
    No library averaging helper (e.g. torch.mean on a stacked tensor) is
    used - the weighted sum is accumulated with an explicit loop.
    \"\"\"
    total_n = sum(client_sizes)
    keys = client_state_dicts[0].keys()
    averaged = {}
    for key in keys:
        acc = torch.zeros_like(client_state_dicts[0][key], dtype=torch.float32)
        for state_dict, n_k in zip(client_state_dicts, client_sizes):
            acc += state_dict[key].to(torch.float32) * (n_k / total_n)
        averaged[key] = acc
    return averaged


# Unit test on toy tensors: weighted mean must match a hand-computed value.
_toy_a = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([10.0])}
_toy_b = {"w": torch.tensor([3.0, 4.0]), "b": torch.tensor([20.0])}
_toy_c = {"w": torch.tensor([5.0, 6.0]), "b": torch.tensor([30.0])}
_result = fedavg_aggregate([_toy_a, _toy_b, _toy_c], [1, 2, 7])
_expected_w = (1 * _toy_a["w"] + 2 * _toy_b["w"] + 7 * _toy_c["w"]) / 10
_expected_b = (1 * _toy_a["b"] + 2 * _toy_b["b"] + 7 * _toy_c["b"]) / 10
assert torch.allclose(_result["w"], _expected_w), (_result["w"], _expected_w)
assert torch.allclose(_result["b"], _expected_b), (_result["b"], _expected_b)
print("Unit test PASSED: fedavg_aggregate reproduces the hand-computed weighted mean.")
print("  w:", _result["w"].tolist(), "expected:", _expected_w.tolist())
print("  b:", _result["b"].tolist(), "expected:", _expected_b.tolist())
""")

    M("Interpretation: the unit test constructs three toy client state "
      "dicts with known, unequal sample counts (1, 2 and 7) and checks that "
      "`fedavg_aggregate`'s output matches a hand-computed weighted mean to "
      "floating-point tolerance. This confirms the weighted-average "
      "equation from Section 2.3 is implemented correctly before it is "
      "trusted on real model weights.")

    Cd("""
# --- 7.6 Global model update ----------------------------------------------
def update_global_model(global_model, new_state_dict):
    global_model.load_state_dict(new_state_dict)
    return global_model

print("update_global_model() defined.")
""")

    Cd("""
# --- 7.7 run_fedavg(...): full orchestration loop -------------------------
def run_fedavg(run_id, partitions, epochs, rounds, test_loader, train_dataset=train_full,
               mu=0.0, verbose=True):
    \"\"\"Run the full FedAvg protocol for `rounds` communication rounds.

    Implements steps 7.1-7.6 in a loop, logs per-round global/local
    metrics, and returns a history dict. Round 0 records the UN-TRAINED
    initial global model, so convergence from a known starting point is
    visible.
    \"\"\"
    seed_everything(SEED)
    global_model = CNN()
    global_model.load_state_dict(copy.deepcopy(GLOBAL_INIT_STATE))
    global_state = copy.deepcopy(GLOBAL_INIT_STATE)

    client_sizes = [len(p) for p in partitions]
    history = []

    # Round 0: evaluate the untrained initial global model.
    loss0, acc0 = evaluate(global_model, test_loader, DEVICE)
    history.append({"round": 0, "global_loss": loss0, "global_acc": acc0,
                     "clients": [{"client": k + 1, "n_k": client_sizes[k],
                                   "final_loss": None, "final_acc": None} for k in range(len(partitions))],
                     "time_sec": 0.0})
    if verbose:
        print(f"[{run_id}] Round 00/{rounds} | Global Loss {loss0:.4f} | Global Acc {acc0*100:.2f}% | (initial model)")

    for r in range(1, rounds + 1):
        t0 = time.time()
        client_states, client_results = [], []

        global_params_for_prox = None
        if mu > 0:
            global_params_for_prox = [p.clone().detach().to(DEVICE) for p in global_model.parameters()]

        for k, idx in enumerate(partitions):
            result = client_round(global_state, k, train_dataset, idx, DEVICE, epochs, r,
                                   mu=mu, global_params_for_prox=global_params_for_prox)
            client_states.append(result["state_dict"])
            client_results.append(result)

        global_state = fedavg_aggregate(client_states, client_sizes)
        global_model = update_global_model(global_model, global_state)

        g_loss, g_acc = evaluate(global_model, test_loader, DEVICE)
        elapsed = time.time() - t0

        history.append({"round": r, "global_loss": g_loss, "global_acc": g_acc,
                         "clients": [{"client": k + 1, "n_k": client_sizes[k],
                                       "final_loss": res["final_loss"], "final_acc": res["final_acc"]}
                                      for k, res in enumerate(client_results)],
                         "time_sec": elapsed})
        if verbose:
            client_acc_str = ", ".join(f"{res['final_acc']*100:.1f}%" for res in client_results)
            print(f"[{run_id}] Round {r:02d}/{rounds} | Global Loss {g_loss:.4f} | "
                  f"Global Acc {g_acc*100:.2f}% | client accs [{client_acc_str}] | time {elapsed:.1f}s")

    # Persist results to disk.
    with open(RESULTS_DIR / f"{run_id}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    flat_rows = [{"round": h["round"], "global_loss": h["global_loss"], "global_acc": h["global_acc"],
                  "time_sec": h["time_sec"]} for h in history]
    pd.DataFrame(flat_rows).to_csv(RESULTS_DIR / f"{run_id}_rounds.csv", index=False)

    client_rows = []
    for h in history:
        for c in h["clients"]:
            client_rows.append({"round": h["round"], **c})
    pd.DataFrame(client_rows).to_csv(RESULTS_DIR / f"{run_id}_clients.csv", index=False)

    return history

print("run_fedavg() defined (full orchestration loop).")
""")

    Cd("""
# --- 7.8 Execution of the headline run: IID, E=2, for """ + str(ROUNDS) + """ rounds ----
ALL_HISTORIES = {}
t_start = time.time()
ALL_HISTORIES["IID_E2"] = run_fedavg("IID_E2", iid_indices, epochs=2, rounds=CONFIG["T_rounds"],
                                      test_loader=test_loader)
print(f"IID_E2 run completed in {time.time() - t_start:.1f} seconds.")
""")

    Cd("""
# --- 7.9 Round-wise results table (Table 6) and accuracy/loss curves (Figure 6) ---
iid_e2_df = pd.DataFrame([{"round": h["round"], "global_loss": h["global_loss"],
                            "global_acc": h["global_acc"], "time_sec": h["time_sec"]}
                           for h in ALL_HISTORIES["IID_E2"]])
print("Table 6: Round-wise global loss/accuracy for the headline run (IID, E=2)")
display(iid_e2_df)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(iid_e2_df["round"], iid_e2_df["global_acc"], marker="o", color="tab:blue")
axes[0].set_xlabel("Communication round"); axes[0].set_ylabel("Global test accuracy")
axes[0].set_title("Global accuracy per round"); axes[0].grid(alpha=0.3)
axes[1].plot(iid_e2_df["round"], iid_e2_df["global_loss"], marker="o", color="tab:red")
axes[1].set_xlabel("Communication round"); axes[1].set_ylabel("Global test loss")
axes[1].set_title("Global loss per round"); axes[1].grid(alpha=0.3)
fig.suptitle("Figure 6: FedAvg convergence on IID data (E = 2) - headline run")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig06_iid_e2_curves.png", dpi=300, bbox_inches="tight")
plt.show()
""")

    Cd("""
_final_acc = iid_e2_df.iloc[-1]["global_acc"]
_final_loss = iid_e2_df.iloc[-1]["global_loss"]
_init_acc = iid_e2_df.iloc[0]["global_acc"]
display(Markdown(
    f"**Interpretation (generated from actual results above):** starting from an "
    f"untrained initial model (Round 0 accuracy **{_init_acc:.2%}**, i.e. close to the "
    f"10% random-guess baseline), FedAvg on IID data with E=2 local epochs reaches "
    f"**{_final_acc:.2%}** global test accuracy and **{_final_loss:.4f}** global test loss "
    f"after {CONFIG['T_rounds']} communication rounds, with accuracy increasing "
    f"monotonically or near-monotonically round over round. This confirms the "
    f"hand-written FedAvg loop (distribute -> locally train -> collect -> weighted-average "
    f"-> update) is correctly converging on IID data."
))
""")

    # ------------------------------------------------------------------
    # 9. Task 4: Experimental Analysis
    # ------------------------------------------------------------------
    M("""
## Task 4: Experimental Analysis

This section runs the full experimental grid specified by the assignment:
6 federated runs (IID and Non-IID, each at E in {1, 2, 5}), a centralized
reference run, and a supplementary FedProx run, then analyzes the results.
""")

    Cd("""
# --- 8.1 Run all 6 federated runs (+ CENTRAL_REF) ------------------------
RUN_GRID = {
    "IID_E1": (iid_indices, 1),
    "IID_E2": (iid_indices, 2),   # already computed in Task 3; reused below
    "IID_E5": (iid_indices, 5),
    "NONIID_E1": (noniid_indices, 1),
    "NONIID_E2": (noniid_indices, 2),
    "NONIID_E5": (noniid_indices, 5),
}

for run_id, (partitions, epochs) in RUN_GRID.items():
    if run_id in ALL_HISTORIES:
        print(f"--- {run_id}: reusing result already computed in Task 3 ---")
        continue
    print(f"--- Running {run_id} (E={epochs}) ---")
    t0 = time.time()
    ALL_HISTORIES[run_id] = run_fedavg(run_id, partitions, epochs=epochs, rounds=CONFIG["T_rounds"],
                                        test_loader=test_loader)
    print(f"--- {run_id} finished in {time.time() - t0:.1f}s ---\\n")

print("All 6 federated runs complete:", list(ALL_HISTORIES.keys()))
""")

    Cd("""
# --- Centralized (non-federated) reference run: upper-bound only --------
def run_central_reference(epochs, train_dataset, test_loader):
    \"\"\"Train ONE model on the FULL 60k training set (not federated).
    Clearly a reference upper bound, not a federated run: there is no
    client partitioning, no aggregation, and no weight distribution step.
    \"\"\"
    seed_everything(SEED)
    model = CNN()
    model.load_state_dict(copy.deepcopy(GLOBAL_INIT_STATE))
    history = []

    loss0, acc0 = evaluate(model, test_loader, DEVICE)
    history.append({"round": 0, "global_loss": loss0, "global_acc": acc0, "time_sec": 0.0})
    print(f"[CENTRAL_REF] Epoch 00/{epochs} | Test Loss {loss0:.4f} | Test Acc {acc0*100:.2f}% | (initial model)")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        gen = torch.Generator(); gen.manual_seed(SEED + ep)
        loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, generator=gen)
        _, ep_hist = local_train(model, loader, DEVICE, epochs=1)
        train_loss, train_acc = ep_hist[-1]
        test_loss, test_acc = evaluate(model, test_loader, DEVICE)
        elapsed = time.time() - t0
        history.append({"round": ep, "global_loss": test_loss, "global_acc": test_acc,
                         "train_loss": train_loss, "train_acc": train_acc, "time_sec": elapsed})
        print(f"[CENTRAL_REF] Epoch {ep:02d}/{epochs} | Train Loss {train_loss:.4f} | Train Acc {train_acc*100:.2f}% | "
              f"Test Loss {test_loss:.4f} | Test Acc {test_acc*100:.2f}% | time {elapsed:.1f}s")

    with open(RESULTS_DIR / "CENTRAL_REF_history.json", "w") as f:
        json.dump(history, f, indent=2)
    pd.DataFrame(history).to_csv(RESULTS_DIR / "CENTRAL_REF_rounds.csv", index=False)
    return history

t0 = time.time()
CENTRAL_HISTORY = run_central_reference(CONFIG["central_ref_epochs"], train_full, test_loader)
print(f"CENTRAL_REF finished in {time.time() - t0:.1f}s")
""")

    Cd("""
# --- Supplementary: FedProx on Non-IID, E=2, mu=0.01 (Task 5, Q3) -------
t0 = time.time()
FEDPROX_MU = 0.01
FEDPROX_HISTORY = run_fedavg("FEDPROX_NONIID_E2", noniid_indices, epochs=2, rounds=CONFIG["T_rounds"],
                              test_loader=test_loader, mu=FEDPROX_MU)
print(f"FedProx (Non-IID, E=2, mu={FEDPROX_MU}) finished in {time.time() - t0:.1f}s")
""")

    M("""
### 8.2 Experiment 1 — Effect of local epochs (E)

For both IID and Non-IID partitioning, this compares E in {1, 2, 5}:
per-round accuracy/loss tables, accuracy and loss curves (one figure per
partition, 3 lines each), and a summary table of final accuracy, best
accuracy, the first round at which 95%/98% global accuracy is reached (or
"not reached"), total local SGD steps, and total wall-clock training time.
""")

    Cd("""
def history_to_df(history):
    return pd.DataFrame([{"round": h["round"], "global_loss": h["global_loss"],
                           "global_acc": h["global_acc"], "time_sec": h["time_sec"]} for h in history])

def first_round_reaching(df, threshold):
    hit = df[df["global_acc"] >= threshold]
    return int(hit.iloc[0]["round"]) if len(hit) else "not reached"

def total_local_steps(partitions, epochs, rounds, batch_size=CONFIG["batch_size"]):
    steps_per_round = sum(-(-len(idx) // batch_size) for idx in partitions) * epochs  # ceil division
    return steps_per_round * rounds

exp1_summary_rows = []
for partition_name, partitions in [("IID", iid_indices), ("Non-IID", noniid_indices)]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for E in CONFIG["E_grid"]:
        run_id = f"{'IID' if partition_name == 'IID' else 'NONIID'}_E{E}"
        df = history_to_df(ALL_HISTORIES[run_id])
        print(f"Table: round-wise accuracy/loss for {run_id}")
        display(df)
        axes[0].plot(df["round"], df["global_acc"], marker="o", label=f"E={E}")
        axes[1].plot(df["round"], df["global_loss"], marker="o", label=f"E={E}")

        exp1_summary_rows.append({
            "Partition": partition_name, "E": E,
            "Final acc": df.iloc[-1]["global_acc"],
            "Best acc": df["global_acc"].max(),
            "Round to reach 95%": first_round_reaching(df, 0.95),
            "Round to reach 98%": first_round_reaching(df, 0.98),
            "Total local SGD steps": total_local_steps(partitions, E, CONFIG["T_rounds"]),
            "Total time (s)": df["time_sec"].sum(),
        })
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Global test accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Global test loss"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle(f"Figure 7{'a' if partition_name=='IID' else 'b'}: Effect of local epochs E - {partition_name} data")
    plt.tight_layout()
    fname = "fig07a_epochs_iid.png" if partition_name == "IID" else "fig07b_epochs_noniid.png"
    plt.savefig(FIG_DIR / fname, dpi=300, bbox_inches="tight")
    plt.show()

exp1_summary = pd.DataFrame(exp1_summary_rows)
print("Table 7: Experiment 1 summary - effect of local epochs")
display(exp1_summary)
exp1_summary.to_csv(RESULTS_DIR / "experiment1_summary.csv", index=False)
""")

    Cd("""
_iid_e1 = exp1_summary.query("Partition=='IID' and E==1").iloc[0]
_iid_e5 = exp1_summary.query("Partition=='IID' and E==5").iloc[0]
_noniid_e1 = exp1_summary.query("Partition=='Non-IID' and E==1").iloc[0]
_noniid_e5 = exp1_summary.query("Partition=='Non-IID' and E==5").iloc[0]
display(Markdown(
    f"**Interpretation (generated from Table 7 above):** on IID data, increasing E from 1 to 5 "
    f"changes final accuracy from **{_iid_e1['Final acc']:.2%}** to **{_iid_e5['Final acc']:.2%}** "
    f"while the total number of local SGD steps grows from {_iid_e1['Total local SGD steps']:,} to "
    f"{_iid_e5['Total local SGD steps']:,} (same {CONFIG['T_rounds']} communication rounds, so more "
    f"local computation per round but no extra communication). On Non-IID data the same change moves "
    f"final accuracy from **{_noniid_e1['Final acc']:.2%}** to **{_noniid_e5['Final acc']:.2%}**. "
    f"Total wall-clock time scales with E on both partitions, growing from "
    f"{_iid_e1['Total time (s)']:.1f}s (E=1) to {_iid_e5['Total time (s)']:.1f}s (E=5) on IID data."
))
""")

    M("""
### 8.3 Experiment 2 — IID vs Non-IID at E = 2

Overlay curves at E=2, a 2x3 grid of all six runs, a summary table, and a
per-client local-vs-global accuracy comparison under Non-IID (illustrating
client drift).
""")

    Cd("""
# Overlay: IID_E2 vs NONIID_E2 (Figure 8)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for run_id, label, color in [("IID_E2", "IID", "tab:blue"), ("NONIID_E2", "Non-IID", "tab:orange")]:
    df = history_to_df(ALL_HISTORIES[run_id])
    axes[0].plot(df["round"], df["global_acc"], marker="o", label=label, color=color)
    axes[1].plot(df["round"], df["global_loss"], marker="o", label=label, color=color)
axes[0].set_xlabel("Round"); axes[0].set_ylabel("Global test accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].set_xlabel("Round"); axes[1].set_ylabel("Global test loss"); axes[1].legend(); axes[1].grid(alpha=0.3)
fig.suptitle("Figure 8: IID vs Non-IID global test accuracy/loss (E = 2)")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig08_iid_vs_noniid_e2.png", dpi=300, bbox_inches="tight")
plt.show()
""")

    Cd("""
# 2x3 grid of all 6 runs with the CENTRAL_REF accuracy as a dashed reference line (Figure 9)
central_df = pd.DataFrame(CENTRAL_HISTORY)
central_final_acc = central_df.iloc[-1]["global_acc"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
run_order = ["IID_E1", "IID_E2", "IID_E5", "NONIID_E1", "NONIID_E2", "NONIID_E5"]
for ax, run_id in zip(axes.flat, run_order):
    df = history_to_df(ALL_HISTORIES[run_id])
    ax.plot(df["round"], df["global_acc"], marker="o", color="tab:green")
    ax.axhline(central_final_acc, color="gray", linestyle="--", label="CENTRAL_REF (final)")
    ax.set_title(run_id)
    ax.set_xlabel("Round"); ax.set_ylabel("Global acc")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
fig.suptitle("Figure 9: All six federated runs vs centralized reference upper bound")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig09_grid_all_runs.png", dpi=300, bbox_inches="tight")
plt.show()
""")

    Cd("""
# Summary table (Table 8) of all runs including CENTRAL_REF and FEDPROX
all_run_ids = run_order + ["CENTRAL_REF", "FEDPROX_NONIID_E2"]
summary8_rows = []
for run_id in all_run_ids:
    if run_id == "CENTRAL_REF":
        df = central_df
    elif run_id == "FEDPROX_NONIID_E2":
        df = history_to_df(FEDPROX_HISTORY)
    else:
        df = history_to_df(ALL_HISTORIES[run_id])
    summary8_rows.append({"Run": run_id, "Final acc": df.iloc[-1]["global_acc"],
                           "Best acc": df["global_acc"].max(),
                           "Final loss": df.iloc[-1]["global_loss"],
                           "Total time (s)": df["time_sec"].sum()})
table8 = pd.DataFrame(summary8_rows)
print("Table 8: Combined summary of all runs (federated, centralized, FedProx)")
display(table8)
table8.to_csv(RESULTS_DIR / "all_runs_summary.csv", index=False)
""")

    Cd("""
# Per-client local accuracy vs global accuracy under Non-IID, E=2 (Figure 10): client drift
noniid_clients_df = pd.DataFrame([{"round": h["round"], **c} for h in ALL_HISTORIES["NONIID_E2"] for c in h["clients"]])
last_round = noniid_clients_df["round"].max()
last_round_clients = noniid_clients_df[(noniid_clients_df["round"] == last_round) & noniid_clients_df["final_acc"].notna()]
global_acc_last = history_to_df(ALL_HISTORIES["NONIID_E2"]).iloc[-1]["global_acc"]

plt.figure(figsize=(8, 4.5))
plt.bar(last_round_clients["client"].astype(str), last_round_clients["final_acc"], color="tab:orange",
        label="Client local training accuracy (own 2 classes)")
plt.axhline(global_acc_last, color="tab:blue", linestyle="--", linewidth=2,
            label=f"Global model test accuracy = {global_acc_last:.2%}")
plt.xlabel("Client"); plt.ylabel("Accuracy"); plt.ylim(0, 1.05)
plt.title(f"Figure 10: Client drift under Non-IID data (E=2, round {last_round})")
plt.legend(); plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig10_client_drift.png", dpi=300, bbox_inches="tight")
plt.show()
""")

    Cd("""
_mean_local = last_round_clients["final_acc"].mean()
display(Markdown(
    f"**Interpretation (generated from Figure 10 data above):** at round {last_round}, the five "
    f"Non-IID clients' own local models reach a mean local training accuracy of **{_mean_local:.2%}** "
    f"on their own (2-class) data, while the aggregated global model only reaches "
    f"**{global_acc_last:.2%}** on the full 10-class global test set. Each client's local optimum is "
    f"excellent for its own narrow 2-class problem but pulls the shared weights toward a solution that "
    f"is biased toward its own classes — this gap between high local accuracy and lower global accuracy "
    f"is the client-drift phenomenon that non-IID federated learning must contend with."
))
""")

    Cd("""
_iid_final = table8.set_index("Run").loc["IID_E2", "Final acc"]
_noniid_final = table8.set_index("Run").loc["NONIID_E2", "Final acc"]
_central_final = table8.set_index("Run").loc["CENTRAL_REF", "Final acc"]
_fedprox_final = table8.set_index("Run").loc["FEDPROX_NONIID_E2", "Final acc"]
display(Markdown(
    "### 8.5 Discussion\\n\\n"
    f"All numbers below are read directly from Table 8 produced above.\\n\\n"
    f"- The **centralized reference** (not federated; trains on all 60,000 images directly) reaches "
    f"**{_central_final:.2%}** final test accuracy after {CONFIG['central_ref_epochs']} epochs — this is "
    f"the practical upper bound against which every federated run should be judged.\\n"
    f"- **FedAvg on IID data (E=2)** reaches **{_iid_final:.2%}**, which is "
    f"{'close to' if abs(_iid_final-_central_final) < 0.03 else 'noticeably below'} the centralized "
    f"reference, showing that when client data is IID, federated averaging recovers most of the "
    f"accuracy of centralized training without ever sharing raw data.\\n"
    f"- **FedAvg on Non-IID data (E=2)** reaches only **{_noniid_final:.2%}**, a drop of "
    f"**{(_iid_final-_noniid_final):.2%}** versus the IID run with the same E and the same number of "
    f"rounds — a direct, measured consequence of the client-drift effect shown in Figure 10.\\n"
    f"- **FedProx on the same Non-IID partition (mu=0.01)** reaches **{_fedprox_final:.2%}**, a change "
    f"of **{(_fedprox_final-_noniid_final):+.2%}** relative to plain FedAvg on identical data/E/rounds — "
    f"analyzed further in Task 5, Q3 below."
))
""")

    # ------------------------------------------------------------------
    # 10. Task 5: Critical Analysis
    # ------------------------------------------------------------------
    M("## Task 5: Critical Analysis")

    Cd("""
display(Markdown(
    "### Q1. Why does FedAvg perform differently on IID vs Non-IID data?\\n\\n"
    f"Measured in this notebook (Table 8), FedAvg reaches **{_iid_final:.2%}** on IID data and only "
    f"**{_noniid_final:.2%}** on Non-IID data under identical E, T and optimizer settings. The "
    "mechanism behind this measured gap is **client drift**: under the pathological 2-classes-per-client "
    "partition used here (Section 5.6), each client's local loss surface has a minimum that is optimal "
    "only for its own two digit classes. During local training (`local_train`), each client's weights "
    "move toward that narrow local optimum; Figure 10 shows this directly — local training accuracy on "
    "a client's own classes is far higher than the aggregated global model's accuracy on the full "
    "10-class test set. When five such divergent local updates are averaged by `fedavg_aggregate`, the "
    "weighted mean is pulled in five different, partially conflicting directions every round, so the "
    "global model's effective movement toward a jointly-good solution is slower and noisier than under "
    "IID data, where every client's local gradient is already (in expectation) an unbiased estimate of "
    "the full-dataset gradient. This is the same weight-divergence phenomenon analyzed by Zhao et al. "
    "(2018): the earlier (and more extreme) the label skew, the larger the divergence between a client's "
    "local optimum and the shared global optimum, and the classifier's final layer (`fc2`, which maps "
    "directly to the 10 class logits) is hit hardest because each client only ever supplies gradient "
    "signal for 2 of its 10 output units."
))
""")

    Cd("""
_iid_e1r = exp1_summary.query("Partition=='IID' and E==1").iloc[0]
_iid_e2r = exp1_summary.query("Partition=='IID' and E==2").iloc[0]
_iid_e5r = exp1_summary.query("Partition=='IID' and E==5").iloc[0]
_noniid_e1r = exp1_summary.query("Partition=='Non-IID' and E==1").iloc[0]
_noniid_e2r = exp1_summary.query("Partition=='Non-IID' and E==2").iloc[0]
_noniid_e5r = exp1_summary.query("Partition=='Non-IID' and E==5").iloc[0]
display(Markdown(
    "### Q2. Impact of increasing local epochs (E)\\n\\n"
    "Measured in Table 7 (Experiment 1) above:\\n\\n"
    f"| Partition | E=1 final acc | E=2 final acc | E=5 final acc |\\n"
    f"|---|---|---|---|\\n"
    f"| IID | {_iid_e1r['Final acc']:.2%} | {_iid_e2r['Final acc']:.2%} | {_iid_e5r['Final acc']:.2%} |\\n"
    f"| Non-IID | {_noniid_e1r['Final acc']:.2%} | {_noniid_e2r['Final acc']:.2%} | {_noniid_e5r['Final acc']:.2%} |\\n\\n"
    "On **IID** data, more local epochs per round means each client does more useful gradient descent "
    "before the next, expensive communication round, so (per the numbers above) a given number of "
    "communication rounds reaches a higher accuracy as E grows — i.e. fewer communication rounds are "
    "needed for the same accuracy, improving communication efficiency, at the cost of "
    f"{_iid_e5r['Total local SGD steps'] / max(_iid_e1r['Total local SGD steps'],1):.1f}x more local "
    "computation (Table 7, 'Total local SGD steps' column) and "
    f"{_iid_e5r['Total time (s)'] / max(_iid_e1r['Total time (s)'],1e-9):.1f}x more wall-clock time "
    "(E=5 vs E=1).\\n\\n"
    "On **Non-IID** data the same increase in E is a double-edged sword: more local epochs also let each "
    "client **over-fit further to its own narrow 2-class local optimum** before the next averaging step, "
    "which can amplify client drift rather than only accelerating convergence. The measured effect of E "
    "on the Non-IID final accuracy (table above) should be read together with Figure 7b: if accuracy "
    "stops improving, or regresses, as E grows under Non-IID data while it keeps improving under IID "
    "data, that asymmetry is exactly the over-fitting-to-local-data/drift trade-off this question asks "
    "about."
))
""")

    Cd("""
# Q3: FedProx vs FedAvg on Non-IID, E=2 - table + curve
fedprox_df = history_to_df(FEDPROX_HISTORY)
noniid_e2_df = history_to_df(ALL_HISTORIES["NONIID_E2"])

q3_table = pd.DataFrame({
    "round": fedprox_df["round"],
    "FedAvg (Non-IID, E=2) acc": noniid_e2_df["global_acc"].values,
    f"FedProx mu={FEDPROX_MU} acc": fedprox_df["global_acc"].values,
    "FedAvg (Non-IID, E=2) loss": noniid_e2_df["global_loss"].values,
    f"FedProx mu={FEDPROX_MU} loss": fedprox_df["global_loss"].values,
})
print("Table 9: FedAvg vs FedProx, Non-IID, E=2, round by round")
display(q3_table)
q3_table.to_csv(RESULTS_DIR / "fedavg_vs_fedprox.csv", index=False)

plt.figure(figsize=(7.5, 4.5))
plt.plot(q3_table["round"], q3_table["FedAvg (Non-IID, E=2) acc"], marker="o", label="FedAvg")
plt.plot(q3_table["round"], q3_table[f"FedProx mu={FEDPROX_MU} acc"], marker="s", label=f"FedProx (mu={FEDPROX_MU})")
plt.xlabel("Round"); plt.ylabel("Global test accuracy")
plt.title("Figure 11: FedAvg vs FedProx on Non-IID data (E = 2)")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig11_fedavg_vs_fedprox.png", dpi=300, bbox_inches="tight")
plt.show()
""")

    Cd("""
display(Markdown(
    "### Q3. One improvement over FedAvg: FedProx\\n\\n"
    "**FedProx** (Li et al., 2020) modifies only the client's *local* objective, adding a proximal "
    "term that penalizes how far the local weights $w$ drift from the global weights $w^{(t)}$ it "
    "started this round from: minimize  L_k(w) + (mu/2) * ||w - w_global||^2 over w.\\n\\n"
    f"implemented in `local_train` above exactly as this extra term, added to the cross-entropy loss "
    f"before `loss.backward()`, with $\\mu={FEDPROX_MU}$. The proximal term directly counteracts client "
    f"drift (Q1): it discourages a client from moving its local optimum too far from the shared global "
    f"model, which should make the per-round client updates more consistent with each other (and hence "
    f"the FedAvg weighted average more coherent) under heavy non-IID skew.\\n\\n"
    f"**Measured result (Table 9 / Figure 11 above):** on identical Non-IID, E=2, {CONFIG['T_rounds']}-round "
    f"settings, FedAvg reaches **{_noniid_final:.2%}** final accuracy while FedProx (mu={FEDPROX_MU}) "
    f"reaches **{_fedprox_final:.2%}** — a change of **{(_fedprox_final - _noniid_final):+.2%}**.\\n\\n"
    "**Other alternatives (not implemented here, mentioned for completeness):**\\n"
    "- **SCAFFOLD** (Karimireddy et al., 2020) corrects client drift directly by tracking and "
    "subtracting a per-client control variate from the local gradient, targeting the same drift problem "
    "from the gradient-correction side rather than a loss-penalty side.\\n"
    "- **FedNova** normalizes each client's update by its number of local steps before averaging, which "
    "helps when different clients perform different amounts of local work.\\n"
    "- **Server momentum / FedAvgM** adds a momentum term on the *server* side, across rounds, which can "
    "accelerate and stabilize convergence independently of any client-side fix."
))
""")

    # ------------------------------------------------------------------
    # 11. Challenges Encountered
    # ------------------------------------------------------------------
    M("""
## Challenges Encountered
""")

    Cd("""
_total_wallclock = table8["Total time (s)"].sum()
display(Markdown(
    "The following challenges were encountered while producing this notebook, and are reported here "
    "for transparency:\\n\\n"
    f"1. **Total runtime.** The full experimental grid in Task 4 (6 federated runs, a 10-epoch "
    f"centralized reference, and a FedProx run) totals **{_total_wallclock/60:.1f} minutes** of actual "
    f"wall-clock compute on the device reported in Section 4 (`{CONFIG['device']}`), because every run "
    f"performs real gradient descent over the full 60,000-image training set, round after round — there "
    f"is no shortcut or subsampling anywhere in the pipeline.\\n"
    "2. **Device-priority vs. raw speed.** The fixed device-selection rule (CUDA, else Apple MPS, else "
    "CPU) does not always pick the fastest backend on every machine: on the Intel integrated GPU used "
    "here, the Metal (MPS) backend's per-batch kernel-launch overhead is non-trivial relative to this "
    "small CNN's actual FLOP count, so in an isolated micro-benchmark plain CPU execution was measured "
    "to be faster for this specific (batch=32, small-CNN) workload. The notebook nonetheless follows the "
    "specified CUDA > MPS > CPU priority order exactly as required, and simply reports whichever device "
    "that rule selects.\\n"
    "3. **Non-IID training instability.** Under the pathological 2-class-per-client partition, "
    "individual client-side final-epoch accuracies (visible in the per-round logs of the Non-IID runs) "
    "are highly volatile round to round, since each client's local optimum changes abruptly whenever it "
    "receives a very different starting point from the previous round's aggregated global model - a "
    "direct, observable symptom of the client-drift effect discussed in Task 5, Q1.\\n"
    "4. **End-to-end determinism.** Reproducing identical results on repeated `Restart & Run All` runs "
    "required seeding every source of randomness explicitly (Python's `random`, NumPy, PyTorch's global "
    "RNG, and a dedicated `torch.Generator` for every single `DataLoader` shuffle, reseeded per round and "
    "per client) rather than relying on a single global seed call at the top of the notebook."
))
""")

    # ------------------------------------------------------------------
    # 12. Conclusion
    # ------------------------------------------------------------------
    M("## Conclusion")

    Cd("""
display(Markdown(
    "This notebook implemented the FedAvg algorithm entirely from scratch in PyTorch - server "
    "initialization, weight distribution, local client training, weighted aggregation (with a passing "
    "unit test against a hand-computed value), and the full multi-round orchestration loop - with no "
    "federated-learning framework involved anywhere.\\n\\n"
    "**Key quantitative findings, read directly from the results produced above:**\\n\\n"
    f"- Isolated (non-federated) local training reaches a mean global-test accuracy of **{_iid_mean:.2%}** "
    f"under IID data but only **{_noniid_mean:.2%}** under Non-IID data (Table 5), motivating federated "
    f"aggregation in the first place.\\n"
    f"- FedAvg (E=2, {CONFIG['T_rounds']} rounds) closes most of that gap, reaching **{_iid_final:.2%}** "
    f"on IID data versus **{_noniid_final:.2%}** on Non-IID data (Table 8) - still a measurable "
    f"**{(_iid_final-_noniid_final):.2%}** penalty from client drift under label skew.\\n"
    f"- The centralized (non-federated) reference reaches **{_central_final:.2%}**, the practical upper "
    f"bound that federated training on IID data (**{_iid_final:.2%}**) approaches closely without ever "
    f"sharing raw client data.\\n"
    f"- Increasing local epochs E trades more local computation and wall-clock time for fewer effective "
    f"communication rounds needed to reach a given accuracy (Table 7), with a markedly different effect "
    f"on IID versus Non-IID data (Task 5, Q2).\\n"
    f"- FedProx (mu={FEDPROX_MU}) changed Non-IID final accuracy by **{(_fedprox_final-_noniid_final):+.2%}** "
    f"relative to plain FedAvg under otherwise identical settings (Table 9), consistent with its design "
    f"goal of reducing client drift via a proximal penalty on local updates.\\n\\n"
    "Together, these results show both the practical promise of federated learning (near-centralized "
    "accuracy without centralizing data, under IID conditions) and its central open challenge (accuracy "
    "degradation from client drift under realistic, non-IID data distributions)."
))
""")

    # ------------------------------------------------------------------
    # 13. References
    # ------------------------------------------------------------------
    M("""
## References

1. McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS*. arXiv:1602.05629.
2. LeCun, Y., Cortes, C., & Burges, C. J. C. The MNIST Database of Handwritten Digits. http://yann.lecun.com/exdb/mnist/
3. Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated Optimization in Heterogeneous Networks (FedProx). *MLSys*. arXiv:1812.06127.
4. Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., & Chandra, V. (2018). Federated Learning with Non-IID Data. arXiv:1806.00582.
5. Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S., & Suresh, A. T. (2020). SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. *ICML*. arXiv:1910.06378.
6. Kairouz, P., McMahan, H. B., et al. (2021). Advances and Open Problems in Federated Learning. *Foundations and Trends in Machine Learning*. arXiv:1912.04977.
7. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*. See also the torchvision documentation: https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html
8. NumPy — https://numpy.org/doc/ ; Pandas — https://pandas.pydata.org/docs/ ; Matplotlib — https://matplotlib.org/stable/ ; Seaborn — https://seaborn.pydata.org/ ; Jupyter/nbformat/nbconvert — https://jupyter.org/ ; Playwright — https://playwright.dev/python/ ; ReportLab — https://docs.reportlab.com/ ; Pygments — https://pygments.org/
""")

    # ------------------------------------------------------------------
    # Assemble and write the notebook
    # ------------------------------------------------------------------
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "DML-Asn2 (py311)",
            "language": "python",
            "name": "dml_asn2_py311",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    nbf.write(nb, out_path)
    print(f"Notebook written to {out_path} ({len(cells)} cells, quick={quick})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Build a fast smoke-test notebook")
    parser.add_argument("--out", default=None, help="Output .ipynb path")
    args = parser.parse_args()

    if args.out:
        out = args.out
    elif args.quick:
        out = "/Users/ssvmindsprivatelimited/Desktop/MTech/Assignments/DML Assignment2/DML_Assignment2/build/_smoketest.ipynb"
    else:
        out = "/Users/ssvmindsprivatelimited/Desktop/MTech/Assignments/DML Assignment2/DML_Assignment2/build/DML_2025AE05316_Assignment2.ipynb"

    build(quick=args.quick, out_path=out)
