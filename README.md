# Reviewer run instructions

Manuscript: "A shared dynamics representation for prediction- and control-relevant computations"

This repository contains the code and data used to generate the manuscript results. The files should be kept intact for review: do not rename, move, or remove the packaged data archives, saved tensors, model checkpoints, notebooks, or scripts before running the instructions below.

## Repository contents

- `SUBMISSION/PAPER/` contains the submitted manuscript and supplementary information PDFs.
- `SUBMISSION/FIGURES/PDF AND PNG/` contains the submitted rendered figure files.
- `SUBMISSION/FIGURES/DATA AND SCRIPTS/` contains per-figure notebooks and the matching packaged data archives:
  - `FIG2.ipynb` with `FIG2.zip`
  - `FIG3.ipynb` with `FIG3.zip`
  - `FIG4.ipynb` with `FIG4.zip`
  - `FIG5.ipynb` with `FIG5.zip`
  - `FIGS1.ipynb` with `FIGS1.zip`
  - `FIGS2.ipynb` with `FIGS2.zip`
- `SIMULATIONS/MAIN_RUN_ME.ipynb` is the full end-to-end notebook for derivations, simulation data generation, model training, inverse-dynamics evaluation, forward-dynamics evaluation, optimizer sweeps, and hardware reporting.
- `SIMULATIONS/HYPERPARAMETERS_RUN_ME.ipynb` contains the hyperparameter-search workflow.
- `SIMULATIONS/*.pt`, `SIMULATIONS/*.pth`, `SIMULATIONS/*.csv`, `SIMULATIONS/*.json`, and `SIMULATIONS/*.pkl` are the saved result, timing, model, normalization, training-history, and hyperparameter-search artifacts used by the analyses.

Figure 1 is a schematic/conceptual figure and is provided as rendered PDF/PNG output in `SUBMISSION/FIGURES/PDF AND PNG/`.

## Quick path: regenerate figures from packaged data

This is the recommended reviewer workflow. It regenerates the manuscript figures from the provided figure data archives without rerunning model training or long timing benchmarks.

From the repository root:

```bash
cd /path/to/ms-annet
python3 -m venv .venv-review
source .venv-review/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyter nbconvert numpy pandas scipy matplotlib
```

Then execute the figure notebooks from the directory that contains the ZIP archives:

```bash
cd "SUBMISSION/FIGURES/DATA AND SCRIPTS"

for nb in FIG2 FIG3 FIG4 FIG5 FIGS1 FIGS2; do
  jupyter nbconvert --to notebook --execute "${nb}.ipynb" --output "executed-${nb}"
done
```

Expected regenerated outputs in `SUBMISSION/FIGURES/DATA AND SCRIPTS/`:

- `FIG2.svg`, `FIG2.pdf`, `FIG2.png`
- `FIG3.svg`, `FIG3.pdf`, `FIG3.png`
- `FIG4.svg`, `FIG4.pdf`, `FIG4.png`
- `FIG5.svg`, `FIG5.pdf`, `FIG5.png`
- `FIGS1.svg`, `FIGS1.pdf`, `FIGS1.png`
- `FIGS2.svg`, `FIGS2.pdf`, `FIGS2.png`

The regenerated files should match the scientific content of the submitted rendered figures in `SUBMISSION/FIGURES/PDF AND PNG/`. Small visual differences in fonts, PDF metadata, or raster antialiasing can occur across operating systems.

## Inspect packaged figure data

The ZIP archives can be inspected directly:

```bash
cd /path/to/ms-annet/SUBMISSION/FIGURES/DATA\ AND\ SCRIPTS

for archive in FIG2.zip FIG3.zip FIG4.zip FIG5.zip FIGS1.zip FIGS2.zip; do
  echo "$archive"
  unzip -l "$archive"
done
```

Each archive contains the CSV, JSON, and/or NPZ files used by the matching figure notebook.

## Optional full-analysis setup

The full simulation notebooks require additional packages:

```bash
cd /path/to/ms-annet
source .venv-review/bin/activate
python -m pip install psutil sympy statsmodels torch seaborn holoviews mne
```

For GPU execution, install the PyTorch build appropriate for the reviewer machine. CPU execution is sufficient for the packaged figure notebooks, but a full training/timing rerun can be slow on CPU-only machines.

## Optional full notebook rerun

To rerun the full analysis:

```bash
cd /path/to/ms-annet/SIMULATIONS
jupyter notebook MAIN_RUN_ME.ipynb
```

Run the notebook from top to bottom. It performs the symbolic derivations, generates the synthetic double-pendulum datasets, trains the neural models across dataset sizes, evaluates inverse dynamics, evaluates closed-loop forward dynamics, benchmarks timing, performs optimizer sweeps, and records hardware information.

Reproducibility notes:

- The main notebook seed is `SEED = 27`.
- The optimizer sweep seed is `SWEEP_SEED = 20260418`.
- The notebook uses CUDA when available and otherwise uses CPU.
- Timing measurements are hardware-dependent and may differ across reviewer machines.

## Optional hyperparameter-search rerun

To rerun the hyperparameter-search workflow:

```bash
cd /path/to/ms-annet/SIMULATIONS
jupyter notebook HYPERPARAMETERS_RUN_ME.ipynb
```

The saved hyperparameter-search outputs used by the figure notebooks are already included in `SIMULATIONS/` and in the packaged figure archives.

## Submission reminder

For journal review, include this repository or a repository archive with the manuscript files, and upload the completed Nature Machine Learning Checklist PDF requested by the editor.
