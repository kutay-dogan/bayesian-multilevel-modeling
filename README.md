# Bayesian Multilevel Modeling

A Bayesian statistics term project exploring hierarchical models, partial pooling, shrinkage estimation, and Bayesian multilevel regression.

The project combines a LaTeX report with an interactive [marimo](https://marimo.io/) notebook to explain the theoretical foundations and demonstrate the models through practical examples.

## Project Structure

- `report/main.tex` — Main report containing theoretical explanations, mathematical derivations, and results.
- `notebooks/presentation.py` — Interactive presentation following the structure of the report.
- `pixi.toml` — Project environment, dependencies, and task definitions.

## Getting Started

The project uses [Pixi](https://pixi.sh/) for environment management.

First, install the dependencies from the project root:

```bash
pixi install
```

### Run the Interactive Presentation

Launch the notebook in presentation (app) mode:

```bash
pixi run marimo run notebooks/presentation.py
```

To open the notebook in edit mode:

```bash
pixi run marimo edit notebooks/presentation.py
```

### Alternative: Using the Pixi Shell

You can also activate the project environment:

```bash
pixi shell
```

Once inside the environment, run marimo directly:

```bash
marimo run notebooks/presentation.py
```

Or open the notebook in edit mode:

```bash
marimo edit notebooks/presentation.py
```

## Interactive Presentation

The marimo notebook follows the sections of `report/main.tex`, combining theoretical explanations with interactive statistical examples.

Topics include:

- Bayesian inference and hierarchical modeling
- Complete pooling, no pooling, and partial pooling
- Shrinkage estimation and the role of group-level variance
- Bayesian multilevel regression
- Posterior inference using PyMC

### Hierarchical Model Sampling

In **Section 6.4**, click **Fit hierarchical model with PyMC** to run posterior sampling.

Select **Report** mode to reproduce Table 2 from the written report.

Sampling may take some time depending on the hardware and selected configuration.

## Report

The accompanying LaTeX report, `report/main.tex`, provides the mathematical derivations, modeling assumptions, and statistical results underlying the interactive presentation.
