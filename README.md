# MMLD

![Version](https://img.shields.io/badge/Version-1.0.0-4c1?style=flat)![Release date](https://img.shields.io/badge/Release%20date-Oct.%2012%2C%202026-4c1?style=flat)

## Content

- [Introduction](#introduction)
- [Installation](#installation)
- [Input example](#input-tables)
- [Usage example](#usage-example)
- [Command arguments](#command-arguments)
- [Output files](#output-files)
- [Datasets](#datasets)
- [Citation](#citation)

## Introduction

MMLD is a toolkit for modelling longitudinal microbial community dynamics by incorporating edge-specific intermicrobial interactions with higher-order temporal dependencies. It supports a range of intermicrobial-interaction analyses and machine-learning tasks, including causal graph construction, trajectory prediction, classification, minimum driver-node identification, model predictive control (MPC) intervention, keystone-species analysis and time-series change-point detection.

## Installation

Required packages:

```
Package        Version
----------------------
Python         3.9.23
NumPy          2.0.2
pandas         2.3.1
openpyxl       3.1.5
SciPy          1.13.1
scikit-learn   1.6.1
Matplotlib     3.9.4
seaborn        0.13.2
NetworkX       3.2.1
PyTorch        2.8.0
statsmodels    0.14.6
Tigramite      5.2.10.1
LiNGAM         1.12.2
hybridmetrics  0.2.0
causal-learn   0.1.4.8
Julia          1.12.6
Hungarian      0.7.0
MatrixNetworks 1.0.4
```

One-click installation:

```bash
cd /your_path/MMLD
bash scripts/install_fresh_conda_env.sh
conda activate MMLD
MMLD --help
```

TUNA mirrors are enabled by default. Disable them when needed:

```bash
export MMLD_USE_TUNA_MIRROR=0
bash scripts/install_fresh_conda_env.sh
```

## Input example

Abundance table:

| SubjectID | Time | Microbe1 | Microbe2 | Group   |
| --------- | :--: | -------: | -------: | ------- |
| S01       |  1   |     1200 |      340 | IBD     |
| S01       |  2   |     1180 |      420 | IBD     |
| S02       |  1   |      530 |      180 | Control |
| S02       |  2   |      500 |      280 | Control |

`Group`, `label`, or `class` is detected automatically. Classification requires it. MPC requires labels in its initial and
target state tables. Binary and multiclass classification are supported. Use `--no-group` only when a nonstandard metadata column could otherwise be mistaken for a label.

For cross-sectional data, provide one row per `SubjectID` and a placeholder `Time`, such as `1`. Static causal discovery supports `lingam` and `pc`. 

Interaction network:

|  source   |  target   | lag  | p_value | test_statistic |
| :-------: | :-------: | :--: | :-----: | :------------: |
| Species_A | Species_B |  1   |  0.012  |      0.3       |
| Species_B | Species_C |  1   |  0.021  |      -0.2      |
| Species_C | Species_A |  2   |  0.034  |      0.43      |
| Species_A | Species_D |  1   |  0.047  |     -0.23      |

Algorithms without a lag use `lag=1`. Algorithms without native p-values use `p_value=0.05`. `test_statistic` keeps the signed effect size.

| `--alg`     | Data type       | Implementation                     |
| ----------- | --------------- | ---------------------------------- |
| `pcmci`     | longitudinal    | Tigramite PCMCI [1]                |
| `granger`   | longitudinal    | OLS Granger regression [2]         |
| `cmlp`      | longitudinal    | cMLP [3]                           |
| `clstm`     | longitudinal    | cLSTM [3]                          |
| `lasso_dbn` | longitudinal    | LASSO dynamic Bayesian network [4] |
| `lingam`    | cross-sectional | LiNGAM [5]                         |
| `pc`        | cross-sectional | causal-learn PC [6]                |

## Usage example

Run `MMLD prepare` first. An abundance workbook is required; a causal graph can
either be supplied or inferred with `--alg`. The supplied abundance workbook and,
when present, causal graph are copied into the prepare output directory as
`input_abundance.xlsx` and `input_causal_graph.xlsx`.
If `initial.xlsx` and `target.xlsx` are located beside the abundance workbook,
they are automatically copied as `input_mpc_initial.xlsx` and
`input_mpc_target.xlsx`. Custom MPC paths can instead be supplied with
`--mpc-initial` and `--mpc-target`.
When `--output-dir` is omitted, the directory name includes the abundance-file
stem, for example `outputs/00_prepare_time_series_example2`.
All downstream commands explicitly use the standardized files saved in this prepare directory. 

The `example` directory contains two types of input files: `example1.xlsx`, which includes `Group` labels, and `example2.xlsx`, which does not contain group labels.

```bash
MMLD prepare --abundance example/example2.xlsx
```

Subsequently, the task commands for MMLD can be implemented:

| Task | Minimal command                                                                                                                                                                                                                     |
|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Predict | `MMLD predict --abundance outputs/00_prepare_time_series_example1/input_abundance.xlsx --graph outputs/00_prepare_time_series_example2/MMLD_processed_causal_graph.xlsx` |
| Classify | `MMLD classify --abundance outputs/00_prepare_time_series_example1/input_abundance.xlsx --graph outputs/00_prepare_time_series_example1/MMLD_processed_causal_graph.xlsx` |
| Keystone analysis | `MMLD keystone --abundance outputs/00_prepare_time_series_example1/input_abundance.xlsx --graph outputs/00_prepare_time_series_example2/MMLD_processed_causal_graph.xlsx` |
| Driver microbes | `MMLD driver --abundance outputs/00_prepare_time_series_example1/input_abundance.xlsx --graph outputs/00_prepare_time_series_example2/MMLD_processed_causal_graph.xlsx` |
| MPC intervention | `MMLD mpc --graph outputs/00_prepare_time_series_example1/MMLD_processed_causal_graph.xlsx --model outputs/00_prepare_time_series_example1/MMLD_model.pt --initial outputs/00_prepare_time_series_example1/input_mpc_initial.xlsx --target outputs/00_prepare_time_series_example1/input_mpc_target.xlsx --driver-species "Alistipes_putredinis,Bacteroides_uniformis,Eubacterium_rectale,Faecalibacterium_prausnitzii,Prevotella_copri,others"` |
| Change-point detection | `MMLD changepoint --abundance outputs/00_prepare_time_series_example1/input_abundance.xlsx --graph outputs/00_prepare_time_series_example2/MMLD_processed_causal_graph.xlsx` |
| One-click complete workflow | `MMLD all --abundance example/example1.xlsx --graph example/example1_graph.xlsx` |

## Command arguments

Use the command-specific help to see required arguments, defaults, choices, and input formats:

```bash
MMLD --help
MMLD prepare --help
MMLD predict --help
MMLD classify --help
MMLD keystone --help
MMLD driver --help
MMLD mpc --help
MMLD changepoint --help
MMLD all --help
```

Tips:

- The default split is a subject-level 80/20 train-test split (`--split 82`); `custom` uses xlsx files with a `SubjectID` column.
- `sequence` recursively predicts future states after the observed prefix.
- `step1` performs one-step prediction and `stepN` performs an N-step rollout.
## Output files

Detailed output file descriptions are maintained in the [outputs](outputs/README.md).

## Datasets

The preprocessing of the public datasets used by MMLD follows the steps described in the original paper, without modification.

See the [datasets](<datasets>) for the dataset table and source details.

## Citation

[1] Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., and Sejdinovic, D. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. *Science Advances*, 5(11), eaau4996. https://doi.org/10.1126/sciadv.aau4996

[2] Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424-438. https://doi.org/10.2307/1912791

[3] Tank, A., Covert, I., Foti, N., Shojaie, A., and Fox, E. B. (2021). Neural Granger causality. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. https://doi.org/10.1109/TPAMI.2021.3065601

[4] Song, L., Kolar, M., and Xing, E. P. (2009). Time-varying dynamic Bayesian networks. In *Advances in Neural Information Processing Systems* (Vol. 22, pp. 1732-1740). https://dl.acm.org/doi/10.5555/2984093.2984287

[5] Shimizu, S., Hoyer, P. O., Hyvarinen, A., and Kerminen, A. (2006). A linear non-Gaussian acyclic model for causal discovery. *Journal of Machine Learning Research*, 7, 2003-2030. https://jmlr.org/papers/v7/shimizu06a.html

[6] Spirtes, P., Glymour, C., and Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press. https://doi.org/10.7551/mitpress/1754.003.0021
