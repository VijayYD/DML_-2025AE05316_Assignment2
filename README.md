# DML Assignment 2 — FedAvg on MNIST

Implementation and performance analysis of the Federated Averaging (FedAvg)
algorithm for image classification, with a manual (no framework) PyTorch
implementation, an IID vs. non-IID study, a local-epoch sweep, and a
FedProx comparison.

- **Roll No.:** 2025AE05316
- **Name:** Yeldona Desai V Vittal

## Folder structure

```
DML_Assignment2/
├── build/
│   ├── make_notebook.py   # generates the notebook (no training happens here)
│   ├── make_report.py     # builds the PDF report from an EXECUTED notebook's results
│   └── DML_2025AE05316_Assignment2.ipynb   # generated notebook (unexecuted until run)
├── data/                  # MNIST, auto-downloaded on first run (not committed)
├── figures/                # PNG figures saved by the notebook (not committed until a full run finishes)
├── results/                # per-run JSON/CSV metrics saved by the notebook (not committed until a full run finishes)
├── screenshots/            # cell screenshots captured by make_report.py for the PDF
└── submission/              # final two deliverables land here
```

## Step 1 — Set up the environment

You need a Python environment with PyTorch + torchvision. **PyTorch currently has no
wheels for Python 3.13 on Intel Macs** — if `pip install torch` fails with
"no matching distribution", use a Python 3.9–3.12 environment instead (e.g. a
conda env, or whatever default Python your lab machine has).

```bash
pip install torch torchvision numpy pandas matplotlib seaborn jupyter nbformat nbconvert ipykernel
```

Only needed if you'll also build the PDF report (Step 4) on this machine:

```bash
pip install pygments playwright pypdf
python3 -m playwright install chromium
```

Verify no prohibited federated-learning library is installed (the assignment
requires FedAvg to be hand-written):

```bash
pip list | grep -iE "flwr|fedml|tensorflow|pysyft|syft|openfl|fate"   # must print nothing
```

## Step 2 — Generate the notebook

The notebook in `build/DML_2025AE05316_Assignment2.ipynb` is already generated and
committed. Only regenerate it if you changed `make_notebook.py`:

```bash
cd DML_Assignment2/build
python3 make_notebook.py              # full run: T=10 rounds, E in {1,2,5}, ~2-3 hours on CPU/MPS
python3 make_notebook.py --quick      # fast smoke test (2 rounds) to sanity-check the pipeline
```

`BASE_DIR` inside the generated notebook auto-detects the project folder from
the notebook's own working directory, so no manual path edits are needed on a
new machine.

## Step 3 — Execute the notebook (this is the actual training run)

```bash
cd DML_Assignment2/build
jupyter kernelspec list     # find the name of a kernel with torch+torchvision
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=<your_kernel_name> \
    "DML_2025AE05316_Assignment2.ipynb" \
    --output "DML_2025AE05316_Assignment2.ipynb"
```

- Runs top-to-bottom with no manual edits; all outputs get embedded in the notebook.
- Device is auto-selected (CUDA > Apple MPS > CPU) and printed in the Environment Setup section.
- On a CUDA GPU this should take well under 30 minutes; on CPU/Apple MPS, budget 2-3 hours.
- Produces `figures/*.png` and `results/*.json` + `results/*.csv` — these are what Step 4 reads.

## Step 4 — Build the PDF report

Only after Step 3 has fully completed (this step does **not** retrain anything —
it only reads the saved `results/` and `figures/` files and the executed notebook):

```bash
cd DML_Assignment2/build
python3 make_report.py
```

This exports the notebook to HTML, captures a few cell screenshots with
Playwright, assembles `build/report.html`, and prints it to
`build/DML_2025AE05316_Assignment2.pdf` (A4, 2 cm margins, page numbers,
verified to be ≤ 20 pages).

## Step 5 — Assemble the submission

Copy the two final deliverables into `submission/` (nothing else should go there):

```bash
cp "DML_Assignment2/build/DML_2025AE05316_Assignment2.ipynb" "DML_Assignment2/submission/"
cp "DML_Assignment2/build/DML_2025AE05316_Assignment2.pdf"   "DML_Assignment2/submission/"
```

## Running on a different machine (e.g. a lab GPU server)

1. Copy over `DML_Assignment2/build/DML_2025AE05316_Assignment2.ipynb` (or the whole `build/` folder).
2. Install the dependencies from Step 1 in whatever Python/kernel is available there.
3. Run Step 3's `nbconvert` command, pointing `--ExecutePreprocessor.kernel_name` at that
   machine's kernel.
4. Bring back the executed `.ipynb` plus the `results/` and `figures/` folders it produced,
   so the PDF report (Step 4) can be built from real results.

## Notes

- No federated-learning framework (TensorFlow Federated, Flower, FedML, PySyft, OpenFL, FATE)
  is used anywhere — FedAvg's server/client orchestration and weighted aggregation are
  hand-written in `make_notebook.py`.
- Every random source (Python `random`, NumPy, PyTorch, and a per-round/per-client
  `torch.Generator` for each `DataLoader`) is seeded with `SEED = 42` for reproducibility.
