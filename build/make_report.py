"""
Builds the PDF report (Section 5 of the assignment brief) from the already
EXECUTED notebook's saved results/figures. Does NOT re-run any training.

Pipeline:
  1. Read the executed notebook (nbformat) to pull out code-cell source
     (for the Source Code listings) and markdown section headers.
  2. Load all results/*.csv tables and figures/*.png produced by the
     notebook run.
  3. Render the executed notebook to standalone HTML (nbconvert) purely so
     we can screenshot a handful of key executed cells with Playwright.
  4. Assemble one long report HTML page (cover, TOC, problem statement,
     methodology, source-code listings via Pygments, execution results,
     analysis & discussion, conclusion, references).
  5. Print that HTML to a paginated A4 PDF with Playwright
     (page numbers via displayHeaderFooter), verify page count <= 20.
"""
import json
import re
import sys
from pathlib import Path

import nbformat
import pandas as pd
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

BASE_DIR = Path("/Users/ssvmindsprivatelimited/Desktop/MTech/Assignments/DML Assignment2/DML_Assignment2")
BUILD_DIR = BASE_DIR / "build"
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
SUBMISSION_DIR = BASE_DIR / "submission"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

ROLL_NO = "2025AE05316"
STUDENT_NAME = "YELDONA DESAI V VITTAL"

NOTEBOOK_PATH = BUILD_DIR / f"DML_{ROLL_NO}_Assignment2.ipynb"
REPORT_HTML_PATH = BUILD_DIR / "report.html"
REPORT_PDF_PATH = BUILD_DIR / f"DML_{ROLL_NO}_Assignment2.pdf"
NOTEBOOK_HTML_PATH = BUILD_DIR / "notebook_export.html"


# ---------------------------------------------------------------------
# 1. Pull code-cell source (grouped by the most recent markdown heading)
#    out of the executed notebook, for the Source Code listings section.
# ---------------------------------------------------------------------
def extract_code_by_task(nb):
    groups = []  # list of (heading, [code_strings])
    current_heading = "Setup"
    current_code = []

    def flush():
        if current_code:
            groups.append((current_heading, current_code[:]))
            current_code.clear()

    for cell in nb.cells:
        if cell.cell_type == "markdown":
            m = re.search(r"^#+\s*(.+)$", cell.source.strip().splitlines()[0] if cell.source.strip() else "")
            if m and ("Task" in m.group(1) or "Setup" in m.group(1) or "Background" in m.group(1)
                      or "Assumptions" in m.group(1) or "Critical Analysis" in m.group(1)
                      or "Challenges" in m.group(1) or "Conclusion" in m.group(1)
                      or "References" in m.group(1) or "Introduction" in m.group(1)):
                flush()
                current_heading = m.group(1).strip()
        elif cell.cell_type == "code":
            src = cell.source.strip()
            if src:
                current_code.append(src)
    flush()
    return groups


# ---------------------------------------------------------------------
# 2. Load result tables
# ---------------------------------------------------------------------
def load_csv(name):
    p = RESULTS_DIR / name
    if p.exists():
        return pd.read_csv(p)
    return None


def df_to_html_table(df, max_rows=None):
    if df is None:
        return "<p><em>Result file not found.</em></p>"
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_html(index=False, classes="rtable", border=0, float_format=lambda x: f"{x:.4f}")


# ---------------------------------------------------------------------
# 3. nbconvert -> HTML (for screenshots only)
# ---------------------------------------------------------------------
def export_notebook_html(nb):
    from nbconvert import HTMLExporter
    exporter = HTMLExporter(template_name="classic")
    body, _ = exporter.from_notebook_node(nb)
    NOTEBOOK_HTML_PATH.write_text(body, encoding="utf-8")
    return NOTEBOOK_HTML_PATH


# ---------------------------------------------------------------------
# 4. Playwright screenshots of key cells, located by unique output text
# ---------------------------------------------------------------------
SCREENSHOT_TARGETS = [
    ("Python", "screenshot01_environment.png", "Screenshot 1: Environment setup — library versions and device selection"),
    ("Table 1: Samples per client", "screenshot02_samples_per_client.png", "Screenshot 2: Samples-per-client table (IID vs Non-IID)"),
    ("Total trainable parameters", "screenshot03_model_summary.png", "Screenshot 3: CNN model summary and parameter count"),
    ("[IID_E2] Round", "screenshot04_fedavg_log.png", "Screenshot 4: FedAvg per-round execution log (IID, E = 2)"),
    ("Table 7: Experiment 1 summary", "screenshot05_exp1_summary.png", "Screenshot 5: Experiment 1 summary table (effect of local epochs)"),
    ("Table 8: Combined summary", "screenshot06_exp2_summary.png", "Screenshot 6: Combined summary table of all runs"),
]


def capture_screenshots(html_path):
    from playwright.sync_api import sync_playwright

    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000}, device_scale_factor=2)
        page.goto(f"file://{html_path}")
        page.wait_for_timeout(500)

        for marker, fname, caption in SCREENSHOT_TARGETS:
            try:
                loc = page.locator(f"text={marker}").first
                loc.wait_for(timeout=3000)
                box_js = """(el) => {
                    let node = el;
                    while (node && node !== document.body) {
                        if (node.className && /cell/i.test(node.className)) {
                            const r = node.getBoundingClientRect();
                            return {x: r.x, y: r.y, width: r.width, height: r.height};
                        }
                        node = node.parentElement;
                    }
                    const r = el.getBoundingClientRect();
                    return {x: r.x, y: r.y, width: r.width, height: r.height};
                }"""
                handle = loc.element_handle()
                rect = page.evaluate(box_js, handle)
                out_path = SCREENSHOTS_DIR / fname
                clip = {"x": max(rect["x"], 0), "y": max(rect["y"], 0),
                        "width": max(rect["width"], 50), "height": min(max(rect["height"], 50), 900)}
                page.screenshot(path=str(out_path), clip=clip)
                results.append((fname, caption, True))
                print(f"Captured {fname}")
            except Exception as e:
                results.append((fname, caption, False))
                print(f"WARNING: could not capture screenshot for marker '{marker}': {e}")
        browser.close()
    return results


# ---------------------------------------------------------------------
# 5. HTML report assembly
# ---------------------------------------------------------------------
CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: 'Helvetica Neue', Arial, sans-serif; color: #1a1a1a; font-size: 10.5pt; line-height: 1.35; }
h1 { font-size: 20pt; margin-top: 0; }
h2 { font-size: 14.5pt; border-bottom: 1.5pt solid #333; padding-bottom: 3px; margin-top: 26px; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 18px; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin-top: 14px; page-break-after: avoid; }
p  { margin: 6px 0; }
.cover { text-align: center; padding-top: 18%; page-break-after: always; }
.cover h1 { font-size: 24pt; margin-bottom: 4px; }
.cover h2 { border: none; font-size: 15pt; font-weight: normal; margin-top: 2px;}
.cover table { margin: 40px auto; border-collapse: collapse; font-size: 11pt;}
.cover table td { padding: 6px 14px; border-bottom: 0.5pt solid #ccc; text-align: left;}
.toc { page-break-after: always; }
.toc ol { font-size: 11.5pt; line-height: 2.0; }
.section { page-break-before: always; }
table.rtable { border-collapse: collapse; width: 100%; margin: 8px 0 14px 0; font-size: 8.3pt; }
table.rtable th, table.rtable td { border: 0.5pt solid #999; padding: 3px 5px; text-align: center; }
table.rtable th { background: #eef1f5; }
.hparam-table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; }
.hparam-table th, .hparam-table td { border: 0.5pt solid #999; padding: 4px 8px; text-align: left; }
.hparam-table th { background: #eef1f5; }
figure { text-align: center; margin: 14px 0; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 9.5cm; border: 0.5pt solid #ccc; }
figcaption { font-size: 9pt; color: #333; margin-top: 4px; }
.listing-wrap { page-break-inside: auto; margin: 10px 0 18px 0; }
.listing-title { font-size: 9.5pt; font-weight: bold; margin-bottom: 3px; }
.listing-wrap .highlight pre {
  font-family: 'SF Mono', Menlo, Consolas, monospace;
  font-size: 7.6pt;
  line-height: 1.22;
  white-space: pre-wrap;
  word-break: break-word;
  border: 0.5pt solid #ccc;
  background: #fafafa;
  padding: 6px 8px;
  margin: 0;
}
.refs ol { font-size: 9.8pt; line-height: 1.7; }
.assumptions ol { font-size: 10pt; line-height: 1.55; }
code.inline { font-family: Menlo, Consolas, monospace; font-size: 9pt; background: #f1f1f1; padding: 1px 3px;}
"""


def build_hparam_table(rows):
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f'<table class="hparam-table"><tr><th>Item</th><th>Value</th></tr>{body}</table>'


def fig(path_name, number, caption):
    p = FIG_DIR / path_name
    if not p.exists():
        return f"<p><em>[Figure {number} missing: {path_name}]</em></p>"
    return f'<figure><img src="file://{p}"><figcaption>Figure {number}: {caption}</figcaption></figure>'


def screenshot_fig(fname, number, caption):
    p = SCREENSHOTS_DIR / fname
    if not p.exists():
        return f"<p><em>[Screenshot {number} missing: {fname}]</em></p>"
    return f'<figure><img src="file://{p}"><figcaption>Screenshot {number}: {caption}</figcaption></figure>'


def build_report_html(nb, code_groups, screenshot_results):
    pygments_css = HtmlFormatter(style="default").get_style_defs(".highlight")

    # ---- Source code listings, grouped by task ----
    listing_no = 0
    listing_html_parts = []
    for heading, snippets in code_groups:
        listing_html_parts.append(f"<h3>{heading}</h3>")
        for snippet in snippets:
            listing_no += 1
            highlighted = highlight(snippet, PythonLexer(), HtmlFormatter(style="default"))
            listing_html_parts.append(
                f'<div class="listing-wrap"><div class="listing-title">Listing {listing_no} — {heading}</div>{highlighted}</div>'
            )
    listings_html = "\n".join(listing_html_parts)

    # ---- Result tables ----
    table1 = load_csv("isolated_local_training.csv")
    exp1 = load_csv("experiment1_summary.csv")
    all_summary = load_csv("all_runs_summary.csv")
    fedprox_tbl = load_csv("fedavg_vs_fedprox.csv")
    iid_e2_rounds = load_csv("IID_E2_rounds.csv")
    central_rounds = load_csv("CENTRAL_REF_rounds.csv")

    screenshots_html = "\n".join(
        screenshot_fig(fname, i + 1, caption) for i, (fname, caption, ok) in enumerate(screenshot_results) if ok
    )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DML Assignment 2 Report</title>
<style>{CSS}\n{pygments_css}</style></head>
<body>

<div class="cover">
  <h1>Distributed Machine Learning</h1>
  <h2>Assignment 2</h2>
  <h2>Implementation and Performance Analysis of the Federated Averaging (FedAvg)<br>Algorithm for Image Classification</h2>
  <table>
    <tr><td>Course</td><td>Distributed Machine Learning</td></tr>
    <tr><td>Group Number</td><td>Individual submission</td></tr>
    <tr><td>Name</td><td>{STUDENT_NAME}</td></tr>
    <tr><td>Roll No.</td><td>{ROLL_NO}</td></tr>
    <tr><td>Date</td><td>{pd.Timestamp.now().strftime('%B %d, %Y')}</td></tr>
  </table>
</div>

<div class="toc">
<h2>Table of Contents</h2>
<ol>
<li>Problem Statement</li>
<li>Methodology</li>
<li>Source Code</li>
<li>Execution Results</li>
<li>Analysis and Discussion</li>
<li>Conclusion</li>
<li>References</li>
</ol>
</div>

<div class="section">
<h2>1. Problem Statement</h2>
<p>A consortium of five simulated hospital clients each hold a private partition of MNIST digit
images (used, per the assignment brief, as a stand-in for a real medical-imaging modality) and
wish to train one shared image classifier without ever transmitting raw images to a central
server or to each other. This assignment implements and empirically evaluates the Federated
Averaging (FedAvg) algorithm (McMahan et al., 2017) — written entirely by hand in PyTorch,
with no federated-learning framework — as a solution: each client trains a local copy of a
shared CNN on its own data for a fixed number of local epochs; only model weights are
communicated to a server, which combines them via a sample-size-weighted average, over
repeated communication rounds. The study covers both an IID data partition and a
pathological non-IID (2-classes-per-client) partition, measures the effect of the number of
local epochs, and evaluates FedProx as one concrete improvement over plain FedAvg for
non-IID data.</p>
</div>

<div class="section">
<h2>2. Methodology</h2>
<h3>2.1 Overall approach</h3>
<p>Five clients are simulated sequentially in a single Python process. Every round, the server
broadcasts its current global weights to all 5 clients (full participation, C = 1.0); each
client trains locally for E epochs using a freshly-created SGD optimizer (momentum 0.9,
learning rate 0.01, no momentum state ever crosses the network); the server collects the 5
resulting state dictionaries and combines them with a hand-written, sample-size-weighted
average; the updated global model is evaluated on the server-held MNIST test set before the
next round begins. Round 0 always records the untrained initial model, so convergence from a
known starting point is visible in every curve and table below.</p>

<h3>2.2 FedAvg algorithm and aggregation equation</h3>
<p>See the FedAvg pseudocode and notation table reproduced in the notebook's Background
section (Listings under "Background"). The server-side weighted-average update rule
implemented by <code class="inline">fedavg_aggregate()</code> is:</p>
<p style="text-align:center; font-size: 12pt;">w<sub>t+1</sub> = &sum;<sub>k=1..K</sub> (n<sub>k</sub> / n) &middot; w<sub>t+1</sub><sup>k</sup>,  n = &sum;<sub>k</sub> n<sub>k</sub></p>

<h3>2.3 Partitioning strategy</h3>
<p><strong>IID:</strong> all 60,000 training indices are shuffled with a fixed seed (42) and
split into 5 equal chunks of 12,000 images each, so every client's label distribution
approximates the global distribution.</p>
<p><strong>Non-IID (pathological label skew):</strong> client k receives every training image
whose label is in {{2k, 2k+1}}: Client 1 &rarr; {{0,1}}, Client 2 &rarr; {{2,3}}, Client 3 &rarr; {{4,5}},
Client 4 &rarr; {{6,7}}, Client 5 &rarr; {{8,9}}. Resulting client sizes are naturally unequal,
computed directly from MNIST's real per-class counts (see Execution Results, Table 1).</p>

<h3>2.4 Model architecture</h3>
{build_hparam_table([
    ("Conv1", "1 -> 32 channels, 5x5 kernel, padding 2, ReLU"),
    ("Pool1", "MaxPool2d(2)"),
    ("Conv2", "32 -> 64 channels, 5x5 kernel, padding 2, ReLU"),
    ("Pool2", "MaxPool2d(2)"),
    ("Flatten", "64*7*7 = 3136 features"),
    ("FC1", "3136 -> 128, ReLU"),
    ("Dropout", "p = 0.25"),
    ("FC2", "128 -> 10 (output logits)"),
])}

<h3>2.5 Hyper-parameters</h3>
{build_hparam_table([
    ("Clients (K)", "5, full participation every round (C = 1.0)"),
    ("Communication rounds (T)", "10"),
    ("Batch size (B)", "32"),
    ("Local epochs (E)", "1, 2, 5 (default 2)"),
    ("Optimizer", "SGD, lr = 0.01, momentum = 0.9 (fresh instance per client per round)"),
    ("Loss", "CrossEntropyLoss"),
    ("Normalization", "mean 0.1307, std 0.3081"),
    ("Seed", "42 (Python random, NumPy, PyTorch, DataLoader generators, cudnn.deterministic=True)"),
    ("FedProx &mu;", "0.01 (supplementary run on Non-IID, E = 2)"),
])}

<h3>2.6 Assumptions</h3>
<div class="assumptions"><ol>
<li>Five clients run sequentially in one process — this simulates, not deploys, a real distributed system.</li>
<li>All clients participate every round; no sampling, no dropout, no stragglers.</li>
<li>The server holds the MNIST test set for global evaluation (would be a held-out validation set in a real deployment).</li>
<li>Only model parameters are ever communicated; no secure aggregation and no differential privacy are applied.</li>
<li>MNIST digits stand in for medical images, as specified by the assignment.</li>
<li>Non-IID is modelled purely as 2-class-per-client label skew; other heterogeneity types are out of scope.</li>
<li>A fresh optimizer (and momentum buffer) is created for every client, every round.</li>
<li>"Local training accuracy" means accuracy on a client's own training data during its final local epoch.</li>
</ol></div>

<h3>2.7 Tools, libraries and hardware</h3>
<p>PyTorch 2.2.2 + torchvision 0.17.2 (model, local training, evaluation only — FedAvg itself
is hand-written); NumPy, Pandas, Matplotlib, Seaborn (analysis and plotting); Jupyter /
nbformat / nbconvert (notebook authoring and execution); Playwright (headless Chromium, report
screenshots and PDF rendering); Pygments (source-code syntax highlighting in this report).
Executed on the device reported by the notebook's Environment Setup section (see Screenshot 1
and the Execution Results section below).</p>
</div>

<div class="section">
<h2>3. Source Code</h2>
<p>The complete source code from the executed notebook, grouped by task, exactly as it was run.</p>
{listings_html}
</div>

<div class="section">
<h2>4. Execution Results</h2>
<h3>4.1 Environment and dataset preparation</h3>
{screenshot_fig("screenshot01_environment.png", 1, "Environment setup output (library versions and device selection)") if any(f=="screenshot01_environment.png" and ok for f,_,ok in screenshot_results) else ""}
{screenshot_fig("screenshot02_samples_per_client.png", 2, "Samples-per-client table (IID vs Non-IID)") if any(f=="screenshot02_samples_per_client.png" and ok for f,_,ok in screenshot_results) else ""}
{fig("fig03_heatmaps.png", 3, "Class-distribution heatmaps per client (IID vs Non-IID)")}

<h3>4.2 Model summary</h3>
{screenshot_fig("screenshot03_model_summary.png", 3, "CNN model summary and parameter count") if any(f=="screenshot03_model_summary.png" and ok for f,_,ok in screenshot_results) else ""}

<h3>4.3 Isolated local training (motivation for FedAvg)</h3>
{fig("fig05_isolated_training.png", 5, "Isolated per-client models evaluated on the global test set")}
{df_to_html_table(table1)}

<h3>4.4 Headline FedAvg run: IID, E = 2</h3>
{screenshot_fig("screenshot04_fedavg_log.png", 4, "FedAvg per-round execution log (IID, E = 2)") if any(f=="screenshot04_fedavg_log.png" and ok for f,_,ok in screenshot_results) else ""}
{df_to_html_table(iid_e2_rounds)}
{fig("fig06_iid_e2_curves.png", 6, "FedAvg convergence on IID data (E = 2) — headline run")}

<h3>4.5 Experiment 1 — effect of local epochs</h3>
{fig("fig07a_epochs_iid.png", 7, "Effect of local epochs E — IID data")}
{fig("fig07b_epochs_noniid.png", 8, "Effect of local epochs E — Non-IID data")}
{screenshot_fig("screenshot05_exp1_summary.png", 5, "Experiment 1 summary table (effect of local epochs)") if any(f=="screenshot05_exp1_summary.png" and ok for f,_,ok in screenshot_results) else ""}
{df_to_html_table(exp1)}

<h3>4.6 Experiment 2 — IID vs Non-IID at E = 2</h3>
{fig("fig08_iid_vs_noniid_e2.png", 9, "IID vs Non-IID global test accuracy/loss (E = 2)")}
{fig("fig09_grid_all_runs.png", 10, "All six federated runs vs. centralized reference")}
{fig("fig10_client_drift.png", 11, "Client drift under Non-IID data")}
{screenshot_fig("screenshot06_exp2_summary.png", 6, "Combined summary table of all runs") if any(f=="screenshot06_exp2_summary.png" and ok for f,_,ok in screenshot_results) else ""}
{df_to_html_table(all_summary)}

<h3>4.7 FedProx supplementary run</h3>
{fig("fig11_fedavg_vs_fedprox.png", 12, "FedAvg vs. FedProx on Non-IID data (E = 2)")}
{df_to_html_table(fedprox_tbl)}

<h3>4.8 Centralized reference run</h3>
{df_to_html_table(central_rounds)}
</div>

<div class="section">
<h2>5. Analysis and Discussion</h2>
<p>{get_markdown_text(nb, "Discussion")}</p>
<p>{get_markdown_text(nb, "Q1.")}</p>
<p>{get_markdown_text(nb, "Q2.")}</p>
<p>{get_markdown_text(nb, "Q3.")}</p>
<p>{get_markdown_text(nb, "Challenges")}</p>
</div>

<div class="section">
<h2>6. Conclusion</h2>
<p>{get_markdown_text(nb, "This notebook implemented")}</p>
</div>

<div class="section refs">
<h2>7. References</h2>
<ol>
<li>McMahan, B., Moore, E., Ramage, D., Hampson, S., &amp; Ag&uuml;era y Arcas, B. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. <em>AISTATS</em>. arXiv:1602.05629.</li>
<li>LeCun, Y., Cortes, C., &amp; Burges, C. J. C. The MNIST Database of Handwritten Digits. http://yann.lecun.com/exdb/mnist/</li>
<li>Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., &amp; Smith, V. (2020). Federated Optimization in Heterogeneous Networks (FedProx). <em>MLSys</em>. arXiv:1812.06127.</li>
<li>Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., &amp; Chandra, V. (2018). Federated Learning with Non-IID Data. arXiv:1806.00582.</li>
<li>Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S., &amp; Suresh, A. T. (2020). SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. <em>ICML</em>. arXiv:1910.06378.</li>
<li>Kairouz, P., McMahan, H. B., et al. (2021). Advances and Open Problems in Federated Learning. <em>Foundations and Trends in Machine Learning</em>. arXiv:1912.04977.</li>
<li>Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. <em>NeurIPS</em>; torchvision docs: https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html</li>
<li>NumPy — https://numpy.org/doc/ ; Pandas — https://pandas.pydata.org/docs/ ; Matplotlib — https://matplotlib.org/stable/ ; Seaborn — https://seaborn.pydata.org/ ; Jupyter/nbformat/nbconvert — https://jupyter.org/ ; Playwright — https://playwright.dev/python/ ; ReportLab — https://docs.reportlab.com/ ; Pygments — https://pygments.org/</li>
</ol>
</div>

</body></html>"""
    return html


def get_markdown_text(nb, marker):
    """Pull the markdown-rendered discussion text generated dynamically by
    a code cell's display(Markdown(...)) output, identified by a marker
    substring, from the executed notebook's stored outputs."""
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            text = data.get("text/markdown") or data.get("text/plain")
            if text and marker in (text if isinstance(text, str) else "".join(text)):
                full = text if isinstance(text, str) else "".join(text)
                # Convert a little bit of markdown to HTML for the report (bold, headers already stripped)
                full = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", full)
                full = full.replace("\\n\\n", "<br><br>").replace("\n\n", "<br><br>").replace("\n", " ")
                return full
    return f"[Discussion text matching '{marker}' not found in notebook outputs.]"


def main():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_groups = extract_code_by_task(nb)
    total_code_lines = sum(len(s.splitlines()) for _, snippets in code_groups for s in snippets)
    print(f"Extracted {sum(len(s) for _, s in code_groups)} code cells, {total_code_lines} total lines, "
          f"grouped into {len(code_groups)} task sections.")

    html_path = export_notebook_html(nb)
    print(f"Notebook exported to HTML: {html_path}")

    screenshot_results = capture_screenshots(html_path)

    report_html = build_report_html(nb, code_groups, screenshot_results)
    REPORT_HTML_PATH.write_text(report_html, encoding="utf-8")
    print(f"Report HTML written: {REPORT_HTML_PATH}")

    # Render to PDF with Playwright, page numbers via displayHeaderFooter
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{REPORT_HTML_PATH}")
        page.wait_for_timeout(300)
        page.pdf(
            path=str(REPORT_PDF_PATH),
            format="A4",
            margin={"top": "2cm", "bottom": "2cm", "left": "2cm", "right": "2cm"},
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=(
                '<div style="font-size:9px; width:100%; text-align:center; color:#555;">'
                'Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>'
            ),
            print_background=True,
        )
        browser.close()
    print(f"PDF written: {REPORT_PDF_PATH}")

    # Verify page count
    from pypdf import PdfReader
    reader = PdfReader(str(REPORT_PDF_PATH))
    n_pages = len(reader.pages)
    print(f"PDF page count: {n_pages}")
    if n_pages > 20:
        print("WARNING: PDF exceeds 20 pages - needs font/figure-size reduction.")
        sys.exit(2)


if __name__ == "__main__":
    main()
