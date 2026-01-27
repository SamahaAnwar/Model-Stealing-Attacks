# Model Stealing Attacks – Unified CLI

This repository contains experiments for **model stealing attacks** against **Logistic Regression (LR)** and **Multilayer Perceptron (MLP)** models using a unified command-line interface (CLI).

The implemented attack strategies include:

* **Uniform Retraining**
* **Equation-Solving (LR only)**
* **Line Search (MLP only)**

The CLI allows consistent and reproducible evaluation across datasets and attack types.

---

## 1. Environment Setup

### 1.1 Create a Python virtual environment

```bash
python -m venv venv
```

Activate it:

* **Linux / macOS**

```bash
source venv/bin/activate
```

* **Windows (PowerShell)**

```powershell
venv\Scripts\Activate.ps1
```

---

### 1.2 Install dependencies

```bash
pip install -r requirements.txt
```

---

### 1.3 Verify environment

The CLI prints library versions at runtime to ensure reproducibility:

```bash
python main.py --help
```

---

## 2. CLI Overview

The project exposes a **single unified CLI** supporting both LR and MLP experiments.

### 2.1 General arguments

| Argument         | Description                           | Required                     |
| ---------------- | ------------------------------------- | ---------------------------- |
| `--model`        | Model type: `lr` or `mlp`             | Yes                          |
| `--dataset`      | Dataset: `iris` or `wine`             | Yes                          |
| `--seed`         | Random seed                           | No (default: 42)             |
| `--test-size`    | Test split proportion                 | No (default: 0.25)           |
| `--val-size`     | Validation split proportion (LR only) | No (default: 0.25)           |
| `--query-size`   | Number of attacker queries            | No (default: 200)            |
| `--save-results` | Output Excel file                     | No (default: `results.xlsx`) |

---

## 3. Logistic Regression Experiments

### 3.1 Logistic Regression – Uniform Retraining Attack

```bash
python main.py \
  --model lr \
  --dataset iris \
  --attack-type uniform \
  --query-size 200
```

Optional LR-specific parameters:

* `--C` (inverse regularization strength, default: 1.0)
* `--max-iter` (default: 2000)

---

### 3.2 Logistic Regression – Equation-Solving Attack

```bash
python main.py \
  --model lr \
  --dataset wine \
  --attack-type equation \
  --query-size 300
```

> ⚠️ The equation-solving attack is only available for Logistic Regression.

---

## 4. MLP Experiments

### 4.1 MLP – Uniform Retraining Attack

```bash
python main.py \
  --model mlp \
  --dataset iris \
  --mlp-attack-type uniform \
  --query-size 200
```

In this setting:

* `query-size` corresponds to the **initial seed set size**
* Additional queries are generated uniformly during retraining rounds

---

### 4.2 MLP – Line Search Attack

```bash
python main.py \
  --model mlp \
  --dataset wine \
  --mlp-attack-type line_search
```

> The line-search attack adaptively selects queries based on decision boundary exploration.

---

## 5. Output

* Results are saved as an **Excel file (`.xlsx`)**
* Each row corresponds to a substitute model
* Reported metrics typically include:

  * Test accuracy
  * Fidelity to victim model
  * Query budget used

Example output location:

```text
results.xlsx
```

---

## 6. Reproducibility Notes

* All experiments are seeded using `--seed`
* Library versions are printed at runtime
* Uniform and adaptive attacks follow identical evaluation pipelines for fair comparison

---

## 7. Troubleshooting

* **Argument errors**: Run `python main_cli.py --help`
* **Missing packages**: Re-run `pip install -r requirements.txt`
* **Excel file open error**: Close the file before re-running experiments

---


