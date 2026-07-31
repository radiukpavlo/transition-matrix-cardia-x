# Contributing

Contributions that improve reproducibility, numerical correctness, documentation, portability, or test coverage are welcome.

## Development setup

Create an isolated Python 3.12 or 3.13 environment, then install the package and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Before opening a pull request, run:

```bash
python -m ruff check src tests
python -m mypy src
python -m pytest
tm-ecg-verify-reported
```

## Scientific-change policy

Changes that affect an estimand, cohort, label contract, threshold, split, seed, or numerical result must include:

- a concise rationale;
- an updated or new regression test;
- an explicit statement of whether the change affects a reported study result; and
- updated result provenance when applicable.

Do not replace an unfavorable result with a more favorable endpoint, silently recode labels, tune on a held-out test partition, or convert a non-estimable quantity into a point estimate.

## Data and privacy

Do not commit PhysioNet archives, derived patient identifiers, reader workbooks, case-level reader responses, credentials, or local filesystem paths. Test fixtures must be synthetic or irreversibly de-identified.
