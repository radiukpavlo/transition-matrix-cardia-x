# Contributing

Contributions that improve reproducibility, numerical correctness, documentation, portability, or test coverage are welcome. CARDIA-X is research software; changes must preserve the distinction between computational evidence, clinical-audit evidence, and unsupported clinical claims.

## Development setup

Use Python 3.12 or 3.13. The locked reference environment is Python 3.13.6. From the repository root, create an isolated environment and install the exact public lock:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

Before opening a pull request, run the same checks used by continuous integration:

```bash
python -m tm_ecg.reproducibility
tm-ecg-verify-reported
python -m tm_ecg.cli doctor
python -m pytest -q
python -m ruff check src tests tasks.py
python -m mypy src/tm_ecg/reproducibility.py
```

The combined offline check is also available through Invoke:

```bash
invoke verify
```

The mypy check is intentionally scoped to the new public-manifest and environment utility. The numerical pipeline uses dynamic tabular records and remains protected by its comprehensive regression suite and Ruff checks rather than an assertion of full-project static typing.

## Public manifest

`MANIFEST.sha256` covers every Git-tracked public file except itself. After changing any tracked file, regenerate and verify it from the repository root:

```bash
python -m tm_ecg.reproducibility --write-manifest
python -m tm_ecg.reproducibility
```

The manifest must never include the ignored `manuscript/` workspace, datasets, derived data, models, or local artifacts.

## Scientific-change policy

Changes that affect an estimand, cohort, label contract, threshold, split, seed, or numerical result must include:

- a concise rationale;
- an updated or new regression test;
- an explicit statement of whether the change affects a reported study result; and
- updated result provenance when applicable.

Do not replace an unfavorable result with a more favorable endpoint, silently recode labels, tune on a held-out test partition, or convert a non-estimable quantity into a point estimate. The strict DSS eligibility gate may legitimately return a nonzero status after writing its evidence-producing report; that negative finding must remain visible.

## Data, manuscript, and privacy boundaries

Do not commit PhysioNet archives, derived patient identifiers, reader workbooks, case-level reader responses, credentials, manuscript source/PDFs, or local filesystem paths. The local `manuscript/` directory is intentionally ignored. Test fixtures must be synthetic or irreversibly de-identified.

Dataset files remain governed by the original PhysioNet licenses and terms. Do not upload them to GitHub or place them in generated release artifacts.

## Release checklist

Before creating a public release or updating the reported snapshot:

1. Confirm that the manuscript and all local data/artifact directories remain ignored and untracked.
2. Run the offline verification, full tests, Ruff, mypy, and the CLI doctor check.
3. Verify that `results/reported_metrics.json` still recalculates to the registered summary.
4. Regenerate `MANIFEST.sha256` last and verify it again.
5. Describe any changed endpoint, contract, environment, or evidence boundary in the release notes.
