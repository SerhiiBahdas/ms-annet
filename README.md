# Appellian Neural Networks

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="SUBMISSION/FIGURES/FEATURED%20IMAGE/Featured%20Image%20-%20dark%20background.png">
  <img src="SUBMISSION/FIGURES/FEATURED%20IMAGE/Featured%20Image%20-%20light%20background.png" alt="A simulated double pendulum above the objective that the forward-dynamics solver builds from the learned Appell acceleration energy" width="100%">
</picture>

Code and data for the manuscript "Appellian Neural Networks" by Serhii Bahdasariants, Lauren Parola, Kriti Kacker, Ariel K. Feldman, Zachary Zdobinski, Inseung Kang and Douglas J. Weber.

The Appellian Neural Network (ANNet) is a physics-informed network that learns the Appell acceleration energy of a simulated double pendulum. The gradient of the learned energy with respect to the accelerations gives the joint torques (inverse dynamics), and the minimization of an objective built on the same energy gives the accelerations (forward dynamics). This repository holds the notebooks that derive the equations, generate the data, train the networks and run every analysis of the manuscript, together with the saved results and one notebook and one data archive for each figure.

The image above is the featured image of the manuscript. It shows one forward simulation of the pendulum above the objective that the solver minimizes at one instant of that simulation (see "Featured image").

The code is released under the MIT License (see `LICENSE`). `CITATION.cff` gives the citation of this repository.

## Repository layout

| Path | Content |
|---|---|
| `SIMULATIONS/MAIN_RUN_ME_NEW.ipynb` | Full analysis notebook, Sections 1 to 13 (see below) |
| `SIMULATIONS/HYPERPARAMETERS_RUN_ME.ipynb` | Hyperparameter search that selected the network architecture and the learning rate |
| `SIMULATIONS/*.pt`, `*.pth`, `*.csv`, `*.json`, `*.pkl`, `*.txt` | Saved results of both notebooks (see "Saved results") |
| `SUBMISSION/FIGURES/DATA AND SCRIPTS/` | One notebook `FIGn.ipynb` and one data archive `FIGn.zip` for each of Figs. 2 to 9 and Supplementary Figs. 1 to 3 (`FIGS1` to `FIGS3`) |
| `SUBMISSION/FIGURES/PDF AND PNG/` | Figures of the manuscript as PDF and PNG (`FIG1` to `FIG9`, `FIGS1` to `FIGS3`) |
| `SUBMISSION/FIGURES/UPLOAD/` | The nine main figures as `Figure 1.pdf` to `Figure 9.pdf`, copies of the PDF files above |
| `SUBMISSION/FIGURES/FEATURED IMAGE/` | Featured image of the manuscript with a light and with a dark background (SVG and PNG), its description and the script that draws it |
| `SUBMISSION/SOURCE DATA/` | Source data of the figures as Excel workbooks (`Supplementary Data 1.xlsx` to `Supplementary Data 11.xlsx`) and the script that builds them |
| `SUBMISSION/PAPER/` | Manuscript (`Article File.pdf`) and Supplementary Information (`Supplementary Information.pdf`) |
| `requirements.txt` | Python packages with the versions used for the analyses |

Fig. 1 is a schematic and has no data archive.

## Requirements

The analyses ran in Python 3.13.11 on an Apple M4 computer (CPU only). `requirements.txt` pins the package versions.

```bash
cd /path/to/ms-annet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Regenerate the figures from the packaged data

Each figure notebook reads only its own archive and redraws the figure within seconds. Run the notebooks from the folder that holds the archives.

```bash
cd "SUBMISSION/FIGURES/DATA AND SCRIPTS"
for nb in FIG2 FIG3 FIG4 FIG5 FIG6 FIG7 FIG8 FIG9 FIGS1 FIGS2 FIGS3; do
  jupyter nbconvert --to notebook --execute "${nb}.ipynb" --output "executed-${nb}"
done
```

Each notebook writes `FIGn.svg`, `FIGn.pdf` and `FIGn.png` next to itself. The files in `SUBMISSION/FIGURES/PDF AND PNG/` are the versions of the manuscript, whose labels, keys and layout were finished in a vector graphics editor. They show the same data as the regenerated files.

Each archive holds the tables that its figure draws (CSV, and NPZ for Fig. 6), a metadata file (JSON) and a short description (`FIGn_README.txt`). `FIG2.zip` also holds the saved Optuna study (`optuna_study.pkl`) and the best configuration as a JSON file, and `FIGS1.zip` holds the best trial and the axis definitions as JSON files. To list the content of the archives:

```bash
cd "SUBMISSION/FIGURES/DATA AND SCRIPTS"
for archive in FIG*.zip; do unzip -l "$archive"; done
```

## Source data

`SUBMISSION/SOURCE DATA/` holds one Excel workbook for each figure with data. The first sheet of a workbook (README) lists the sheets with their panels and sample sizes and defines every column with its unit. The other sheets hold the plotted values and the statistics quoted in the legend, copied from the figure archives without change.

| Workbook | Figure |
|---|---|
| `Supplementary Data 1.xlsx` to `Supplementary Data 8.xlsx` | Figs. 2 to 9 |
| `Supplementary Data 9.xlsx` to `Supplementary Data 11.xlsx` | Supplementary Figs. 1 to 3 |

To rebuild the workbooks from the archives:

```bash
cd "SUBMISSION/SOURCE DATA"
python build_source_data.py
```

The script compares every sheet with its archive table after writing and reports the result.

## Featured image

`SUBMISSION/FIGURES/FEATURED IMAGE/` holds the image at the top of this page with a light and with a dark background, each as an SVG file and as a PNG file of 1200 × 675 pixels. The image shows a simulated double pendulum above the objective that the forward-dynamics solver builds from the learned energy at one instant of a closed-loop simulation, drawn over the accelerations of the two joints. The disc marks the minimum of this objective, which is the acceleration that the solver returns for that instant. The bright pendulum is the state at the same instant, and the fainter poses and the trails show the preceding 1.04 s of the same simulation. Blue marks joint 1 and red marks joint 2. `Featured Image - description.txt` gives the full description, and `Featured Image - preview in a home page banner.png` shows both versions next to the title and the authors, as in a banner of a journal home page.

`draw_featured_image.py` draws both SVG files from the saved results of the manuscript (`simulation_results_50trials_N16_B15_A10.pt`, `stats_16000.pt` and `model_16000.pth` in `SIMULATIONS/`). The PNG files hold the same images.

```bash
cd "SUBMISSION/FIGURES/FEATURED IMAGE"
python draw_featured_image.py
```

## Full analysis

```bash
cd SIMULATIONS
jupyter notebook MAIN_RUN_ME_NEW.ipynb
```

Run the notebook from top to bottom with `SIMULATIONS` as the working directory. Training, the closed-loop simulations, the timing benchmarks, the solver grid search and the noise and transfer analyses load their saved results from `SIMULATIONS/` when these are present, so a run with the files of this repository does not repeat them. Without the saved files the notebook recomputes everything, which takes several hours on the hardware named above (training alone takes 2.2 hours for the largest dataset, the forward-noise simulations about 40 minutes and the transfer simulations about two and a half hours).

| Section | Content | Figure |
|---|---|---|
| 1 | Symbolic derivation of the Appell acceleration energy and of the equations of motion of the double pendulum | |
| 2 | Physical parameters, sampling ranges and data generation | |
| 3 | Physics-informed training of ANNet on nine dataset sizes (1,000 to 256,000 samples) | Fig. 3 |
| 4 | Inverse dynamics on 50 test trajectories, inference time and accuracy | Fig. 3, Fig. 6, Supplementary Fig. 2 |
| 5 | Direct torque-regression baseline with the architecture of ANNet | Fig. 4 |
| 6 | Forward dynamics by optimization, closed-loop simulation and timing | Fig. 6 |
| 7 | Closed-loop simulation with the analytical acceleration energy in place of the learned energy | Supplementary Fig. 3 |
| 8 | Share of the evaluation states outside the training sampling ranges and the inverse-dynamics error inside and outside them | |
| 9 | Grid search over the settings of the forward solver | Fig. 5 |
| 10 | Hardware and software versions | |
| 11 | Inverse dynamics under Gaussian noise on the inputs, error scaling, noise on single input groups and low-pass filtering | Fig. 7 |
| 12 | Forward dynamics under Gaussian noise on the gravitational torques, with the learned and with the analytical energy | Fig. 8 |
| 13 | Transfer of inverse-dynamics learning to forward dynamics after a change of the masses, the link lengths or both, with an unchanged pendulum as control | Fig. 9 |

Sections 5, 7, 11, 12 and 13 also draw their figure and write it to `SUBMISSION/FIGURES/PDF AND PNG/` (`FIG4`, `FIGS3`, `FIG7`, `FIG8`, `FIG9`, as PDF and PNG), where it replaces the file of the same name, and they write the matching data archive to `SUBMISSION/FIGURES/DATA AND SCRIPTS/`. Keep a copy of the figure files if you want to preserve the versions of the manuscript. Sections 3, 4, 6 and 9 save working versions of their plots as SVG, PDF or PNG files in `SIMULATIONS/`, which are not part of the repository.

Random seeds are fixed in the notebook: 27 for data generation and training, 23 for the test trajectories and the closed-loop simulations, 20260418 for the solver grid search, 31 and 37 for the noise realizations of Sections 11 and 12, 41 and 43 for the adaptation data and the mini-batch order of Section 13. The notebook uses CUDA when it is available and the CPU otherwise. Timing results depend on the hardware.

## Hyperparameter search

```bash
cd SIMULATIONS
jupyter notebook HYPERPARAMETERS_RUN_ME.ipynb
```

The search used an independent dataset of 10,000 samples (seed 22, 80% for training and 20% for validation), 500 Optuna trials of up to 100 epochs with the tree-structured Parzen estimator sampler and a median pruner. Its results are `SIMULATIONS/optuna_study.pkl`, `optuna_search_results.csv` and `best_hyperparameters.json`. The notebook loads the saved study when `optuna_study.pkl` is present and then reports the trial statistics, the best configuration and the sensitivity of the search to the hyperparameters that every trial has (learning rate, depth, width of the first hidden layer and activation function), and it draws its figures into `SIMULATIONS/hyperparameter_figures/`. Without the saved study it runs the search, which takes about 35 minutes. A new search can return slightly different trial values on another system, because training is not bit-reproducible across numerical environments.

## Saved results

| Files in `SIMULATIONS/` | Content |
|---|---|
| `model_<n>.pth`, `stats_<n>.pt`, `history_<n>.pt`, `time_<n>.txt` | Trained ANNet, normalization statistics, training history and training time for each dataset size `<n>` |
| `results_model_<n>.pt` | Inverse-dynamics predictions of each model on the test trajectories |
| `test_set_50_traj_5s.pt` | The 50 test trajectories |
| `*_mlp_16000.*`, `baseline_torque_rmse_statistics_mlp_16000.csv` | Direct torque-regression baseline (Section 5) |
| `timing_benchmark_interleaved.pt`, `timing_stats_N16_B15_A10.pt` | Timing of the inverse model and of the forward solver |
| `simulation_results_50trials_N16_B15_A10.pt` | Closed-loop forward simulations (Section 6) |
| `simulation_results_50trials_analyticalS_N16_B15_A10.pt`, `forward_solver_analyticalS_statistics_N16_B15_A10.csv` | Simulations with the analytical energy (Section 7) |
| `coverage_*.csv` | Coverage of the sampling ranges (Section 8). In the rows of `coverage_error_split_model_16000.csv` that describe trajectories, `num_states` holds the number of trajectories, and `median_abs_error_nm` and `p95_abs_error_nm` hold the median and the maximum of the trajectory-wise RMSE |
| `optimizer_grid_search_angle_mae_*` | Grid search of the forward solver (Section 9) |
| `noise_robustness_*` | Noise analyses of the inverse dynamics (Section 11) |
| `forward_noise_*` | Noise analyses of the forward dynamics (Section 12) |
| `transfer_*` | Transfer analysis (Section 13) |
| `optuna_study.pkl`, `optuna_search_results.csv`, `best_hyperparameters.json` | Hyperparameter search |
