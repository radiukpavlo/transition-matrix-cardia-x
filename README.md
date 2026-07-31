# CARDIA-X

[![Continuous integration](https://github.com/radiukpavlo/transition-matrix-cardia-x/actions/workflows/ci.yml/badge.svg)](https://github.com/radiukpavlo/transition-matrix-cardia-x/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducible research software for auditable post-hoc analysis of electrocardiographic (ECG) representations.

CARDIA-X maps features from a frozen ECG encoder to waveform-derived semantic measurements and specifies a conflict-aware symbolic audit layer. The repository also contains a separate structured compatibility classifier, signal-processing and fiducial-validation pipelines, clinical-validation contracts, and a curated aggregate snapshot of the reported results.

This is a research project. It is not a medical device, does not provide medical advice, and must not be used for diagnosis, treatment, triage, autonomous deferral, or clinical decision-making.

## Study and manuscript

The active manuscript associated with this repository is the root `manuscript/main.tex` in the local manuscript workspace. That workspace is intentionally excluded from the public Git repository; the citation below is the authoritative public reference.

> Pavlo Radiuk, Oleksander Barmak, Liliana Klymenko, and Iurii Krak. “CARDIA-X: Global Semantic Transition and Rough-Set Rules for Auditable Post-hoc Electrocardiographic Explainability.” Unpublished manuscript, 2026.

BibTeX:

```bibtex
@unpublished{Radiuk2026CardiaX,
  author = {Radiuk, Pavlo and Barmak, Oleksander and Klymenko, Liliana and Krak, Iurii},
  title = {CARDIA-X: Global Semantic Transition and Rough-Set Rules for Auditable Post-hoc Electrocardiographic Explainability},
  year = {2026},
  note = {Unpublished manuscript}
}
```

The machine-readable software and manuscript citation is in [`CITATION.cff`](CITATION.cff). No DOI, journal publication, or preprint identifier is claimed by this repository.

## Scientific status and evidence boundary

The public repository contains:

- the implementation of the ECG ingestion, preprocessing, fiducial, semantic-transition, compatibility, explanation, DSS, and clinical-validation components;
- versioned ontology, label, acceptance, benchmark, and reporting contracts;
- unit and regression tests;
- the curated aggregate result snapshot in [`results/reported_metrics.json`](results/reported_metrics.json); and
- deterministic verifiers for the public file manifest and reported aggregate quantities.

It deliberately excludes:

- manuscript source files, PDFs, figures, tables, and publication build records;
- raw PhysioNet archives and derived ECG tables;
- trained model binaries and local run artifacts;
- reader workbooks, case-level responses, patient identifiers, and free-text clinical comments; and
- exploratory comparisons or unpublished intermediate outputs.

The manuscript describes a symbolic audit-system specification. In the reported evaluation, the strict decision-system eligibility gate failed; consequently, no executable WEDD discretization or authorized rough-set rulebook is presented as a validated clinical decision system. Legacy candidate rule definitions and their unfavorable review findings remain evidence about the research prototype, not evidence of clinical readiness.

## Reported findings

The versioned result snapshot preserves both positive and negative findings:

- On 2,198 patient-disjoint PTB-XL test records, exact multilabel subset accuracy was 0.741128, micro-F1 was 0.813226, and per-label bitwise accuracy was 0.952027.
- Performance was heterogeneous: AF F1 was 0.875000, while PVC, APB, and AFL F1 values were 0.170543, 0.240000, and 0.250000.
- Strict observed-target masked B1 semantic-reconstruction MAE was 0.571700 on the test split.
- At a 100 ms tolerance, LUDB R-peak detection produced 1,761 true positives, 267 false positives, and 56 false negatives, with F1 of 0.915995.
- The historical formula-linked endpoint in the retrospective 100-case, single-physician audit yielded Cohen’s kappa of 0.683067.
- No assisted diagnosis changed among 99 eligible cases. None of 40 structural abstentions was endorsed as clinically appropriate, six of ten reviewed rule artifacts received a medical-soundness score of 3/5, and reconstructed QRS morphology was judged plausible in 0/40 B2 cases.

These results support CARDIA-X as a research audit layer. They do not establish prospective clinical benefit, multicenter generalizability, safe autonomous deferral, robustness across encoders, or deployment readiness.

## Reproducibility levels

The repository supports three reproducibility levels:

1. **Offline public-snapshot verification** checks the tracked-file manifest, recalculates deterministic quantities in the bundled result snapshot, and runs synthetic/unit regression tests. It requires no ECG dataset.
2. **Environment verification** checks the Python runtime, required packages, exact dependency lock, dataset archive presence, and generated-artifact readiness.
3. **Full computational reproduction** downloads the licensed public datasets locally and runs the B1 and B2 workflows. It is computationally intensive and is not run by continuous integration.

## Reference environment and installation

Run all commands from the repository root. The reference environment is Python 3.13.6; Python 3.12 and 3.13 are supported. The package metadata intentionally rejects Python 3.14 and newer because the locked scientific dependencies are bounded below 3.14.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the exact lock and the local package:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

The lock contains the CPU reference environment, including the development tools, Invoke, and the optional training dependency used by the full workflows. The editable installation exposes the `tm-ecg`, `tm-ecg-verify-reported`, and `tm-ecg-verify-manifest` commands.

## Offline verification

After installation, run:

```bash
tm-ecg-verify-manifest
tm-ecg-verify-reported
python -m tm_ecg.cli doctor
python -m pytest -q
```

Equivalent module forms are available when console-script resolution is inconvenient:

```bash
python -m tm_ecg.reproducibility
python -m tm_ecg.reported_results
```

The expected clean-snapshot result is:

```text
Verified public manifest:  ... public files and ... manifest entries.
Verified CARDIA-X reported results: 2,198 PTB-XL test records, 100 reader-audit cases, 4 transition estimates, and 9 LUDB landmarks.
```

Without local datasets, `tm-ecg doctor` should report that the required packages and exact lock are valid while dataset archives and materialized full-run artifacts are missing. That is an expected clean-clone state, not evidence that the full analyses have been rerun.

Run the combined offline check with Invoke:

```bash
invoke verify
```

After changing tracked public files, maintainers regenerate the ledger and then verify it:

```bash
python -m tm_ecg.reproducibility --write-manifest
python -m tm_ecg.reproducibility
```

For development-quality checks, run:

```bash
python -m ruff check src tests tasks.py
python -m mypy src/tm_ecg/reproducibility.py
```

## Dataset acquisition and placement

The full workflows use unmodified public archives obtained under the original PhysioNet terms. Download the following exact releases from their authoritative pages:

- [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)
- [PTB-XL+ v1.0.1](https://physionet.org/content/ptb-xl-plus/1.0.1/)
- [LUDB v1.0.1](https://physionet.org/content/ludb/1.0.1/)

Place the archives, without renaming or modifying them, in `data/archives/`:

```text
data/archives/
├── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
├── ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1.zip
└── lobachevsky-university-electrocardiography-database-1.0.1.zip
```

The pipeline extracts and indexes the archives locally. Dataset files are never committed to GitHub. PTB-XL uses patient-disjoint folds 1–8 for training, 9 for validation, and 10 for testing. The exact paths, dataset versions, seed, thresholds, and output locations are governed by [`configs/defaults.toml`](configs/defaults.toml).

## Reproducing the computational analyses

The complete workflows are launched through [`tasks.py`](tasks.py). They write derived data and run products below `data/` and `artifacts/`, all of which are ignored by Git.

### B1: PTB-XL and PTB-XL+

Run the compatibility and semantic-transition branch:

```bash
invoke pipeline --dataset=b1
```

The B1 workflow performs:

1. environment bootstrap and archive ingestion;
2. dataset indexing and patient-aware split construction;
3. ECG preprocessing, pacing detection, R-peak detection, and fiducial delineation;
4. triad and waveform-feature extraction;
5. PTB-XL compatibility-classifier training and evaluation;
6. latent feature extraction and semantic matrix construction;
7. transition fitting and train/validation/test explanation generation;
8. DSS eligibility and reporting.

### B2: LUDB fiducial validation

Run the LUDB validation branch:

```bash
invoke pipeline --dataset=b2
```

The B2 workflow performs the shared preprocessing and fiducial stages, LUDB fiducial validation, semantic feature construction, transition fitting, explanation generation, DSS eligibility, and reporting.

### Gate behavior

The DSS and reporting stages preserve negative scientific findings. A workflow may write its evidence-producing report and still return a nonzero status when a registered eligibility or reporting gate fails. Do not convert that status into success or discard the generated report; inspect the gate output and record the failure as part of the result.

To run individual stages, use the installed CLI from the repository root, for example:

```bash
tm-ecg doctor
tm-ecg bootstrap-env
tm-ecg ingest
tm-ecg index
tm-ecg splits --dataset ptbxl
tm-ecg report --experiment b1
```

The task pipeline is the authoritative ordering for a complete B1 or B2 reproduction.

## Outputs and provenance

The public repository distinguishes immutable inputs, derived run products, and curated reported evidence:

- `data/archives/`: locally downloaded public archives;
- `data/raw/`: extracted dataset files;
- `data/interim/`: indexed and intermediate tables;
- `data/processed/`: feature and latent tables;
- `artifacts/models/`: locally trained model files;
- `artifacts/transition/`: fitted transition operators and signatures;
- `artifacts/reports/`: metrics, audit outputs, and release reports;
- `artifacts/manifests/` and `artifacts/logs/`: stage manifests and logs;
- `results/`: the only curated aggregate snapshot committed to Git; and
- `MANIFEST.sha256`: hashes for all tracked public files except the manifest itself.

Generated artifacts contain provenance fields for configuration, code state, data source, and SHA-256 inputs where the stage supports them. The full-workflow bootstrap confidence intervals require row-level predictions or timing errors; they cannot be reconstructed from the aggregate snapshot alone.

## Docker

The Docker image packages the source, locked environment, public configuration contracts, schemas, and reported snapshot. It does not contain datasets or manuscript files.

```bash
docker build -t cardia-x .
docker run --rm cardia-x doctor
docker run --rm --entrypoint tm-ecg-verify-reported cardia-x
```

The image is an offline verification/runtime container. Dataset-backed reproduction still requires licensed archives mounted or supplied through a separate local workflow.

## Repository layout

```text
clinical_validation/config/  Versioned clinical-audit and scenario contracts
configs/                     Dataset, ontology, label, and transition contracts
data/                        Local dataset and derived-output boundary
results/                     Curated aggregate result snapshot
schemas/                     Machine-readable output schemas
src/tm_ecg/                  ECG processing, modeling, transition, DSS, and audit code
tests/                       Unit, regression, and contract tests
tasks.py                     Invoke reproduction entry points
Dockerfile                   Locked offline runtime image
MANIFEST.sha256              Public tracked-file integrity ledger
manuscript/                  Local ignored manuscript workspace; never committed
```

## Contributing and citation

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development checks, scientific-change policy, public-data restrictions, and release checklist. The repository is released under the [MIT License](LICENSE); dataset files remain subject to their original PhysioNet terms.

When discussing or building on the study, cite both the manuscript and this software repository when appropriate. The manuscript citation is repeated above and is encoded in [`CITATION.cff`](CITATION.cff); the repository citation should use the software metadata in that file.

## Limitations

CARDIA-X was evaluated retrospectively using public datasets and a single-physician audit. The study did not evaluate prospective workflow performance, patient outcomes, treatment effects, multicenter generalizability, robust transfer across encoders, or autonomous clinical use. A successful software verification run confirms implementation and snapshot integrity; it does not establish medical validity or safety.
