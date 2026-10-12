# Application and Evaluation of the SISA Framework for Machine Unlearning in Bank Credit Scoring Models

**การประยุกต์ใช้และประเมินผลสถาปัตยกรรม SISA สำหรับ Machine Unlearning ในโมเดลการให้คะแนนเครดิตของธนาคาร**

## Overview

This Senior Project applies and evaluates the SISA (Sharded, Isolated, Sliced, and Aggregated) framework for machine unlearning in bank credit-scoring models. The study examines predictive utility, exact retraining fidelity, deletion locality, retraining workload, and selected privacy-related diagnostics under a controlled experimental protocol.

## Dataset

The project uses the **Home Credit Default Risk** dataset. The dataset is not redistributed in this repository. Download it from its original source and place the following files in the repository root, beside the notebook:

- `application_train.csv`
- `bureau.csv`
- `previous_application.csv`
- `installments_payments.csv`
- `credit_card_balance.csv`

The release preserves the original relative data-loading logic, which expects these filenames in the repository root. Raw data files are intentionally ignored by Git.

## Models

- Logistic Regression
- Random Forest
- LightGBM
- XGBoost

## Experimental Scope

- Baseline versus SISA evaluation
- Cross-model shard-count sensitivity
- Controlled Sharding Strategy comparison
- Exact SISA versus protocol-compatible Gold retraining
- Incremental Slicing
- Approximate Continued Boosting
- Thirty-round cumulative deletion stress testing
- Focused Membership Inference Attack diagnostics

The project does not claim guaranteed privacy, exact removal of all privacy risk, production readiness, or regulatory compliance.

## Repository Contents

- `home_credit_sisa_final.ipynb` — executed final research notebook, copied from the verified `home_credit_sisa_restored.ipynb` and renamed for release.
- `final_pipeline.py` — core data, preprocessing, baseline, SISA, unlearning, and evaluation pipeline.
- `expanded_experiments.py` — expanded sharding, slicing, locality, and model-comparison experiments.
- `policy_aware_extension.py` — supplementary policy-aware deletion experiments.
- `scalability_finalization.py` — extended shard-count and finalization experiments.
- `thesis_notebook_support.py` — supporting thesis-oriented audits and result tables.
- `restored_experiments.py` — additive final restoration experiments and verification gates.
- `results/` — selected small final summary tables only; no predictions, checkpoints, caches, or model binaries.
- `requirements.txt` — packages used by the notebook and project modules.

## Setup

Python 3.12 was used for the verified experiment environment.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running

1. Download the Home Credit Default Risk data and place the five CSV files listed above in the repository root.
2. Start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `home_credit_sisa_final.ipynb`.
4. Review the configuration cell and run the notebook from top to bottom if a full recomputation is required.

A full run is computationally expensive and creates local `outputs_final/` and `outputs_restored/` directories. The committed notebook retains the outputs from the verified final run for inspection.

### Public-release provenance note

The final restoration appendix contains strict provenance checks against the author's immutable original/legacy notebooks and large local checkpoint inventories. Those provenance-only files are deliberately not redistributed because this release contains one authoritative notebook and excludes obsolete notebooks and large artifacts. The scientific code, final executed outputs, configuration, selected result summaries, and final source modules are included; provenance cells that require the omitted private artifact inventory should be treated as verification records unless that local provenance state is reconstructed.

## Reproducibility Notes

- Deterministic random seed: `20260920`
- Preprocessing is learned from the Training Set only.
- Calibration, Validation, and Test roles are kept separate.
- Threshold selection uses Validation data only.
- The Test Set is used for final reporting, not post-hoc configuration selection.
- The released notebook contains no retained error outputs.
- This repository is the research implementation accompanying a Senior Project.

## Data Availability

The Home Credit Default Risk dataset is not included and must be obtained from the original data provider. No raw applicant data, trained model binaries, caches, or deletion-request membership files are redistributed in this release.

