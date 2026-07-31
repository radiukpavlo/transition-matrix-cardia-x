"""Reproduction tasks for the public CARDIA-X repository."""

from invoke import Exit, task


@task
def test(c, verbose=False):
    """Run the regression suite."""
    command = "python -m pytest"
    if verbose:
        command += " -v"
    c.run(command, env={"PYTHONPATH": "src"})


@task(name="verify-reported")
def verify_reported(c):
    """Recalculate deterministic quantities in the reported-result snapshot."""
    c.run("python -m tm_ecg.reported_results", env={"PYTHONPATH": "src"})


@task
def verify(c):
    """Run the offline public-snapshot verification suite."""
    c.run("python -m tm_ecg.reproducibility", env={"PYTHONPATH": "src"})
    c.run("python -m tm_ecg.reported_results", env={"PYTHONPATH": "src"})
    c.run("python -m pytest -q", env={"PYTHONPATH": "src"})


@task
def pipeline(c, dataset="b1", verbose=False):
    """Run the reported PTB-XL (b1) or LUDB (b2) computational workflow."""
    if dataset not in {"b1", "b2"}:
        raise Exit("dataset must be 'b1' or 'b2'", code=2)

    dataset_name = "ptbxl" if dataset == "b1" else "ludb"
    verbosity = "-v " if verbose else ""
    commands = [
        f"python -m tm_ecg.cli {verbosity}bootstrap-env",
        f"python -m tm_ecg.cli {verbosity}ingest",
        f"python -m tm_ecg.cli {verbosity}index",
        f"python -m tm_ecg.cli {verbosity}splits --dataset {dataset_name}",
        f"python -m tm_ecg.cli {verbosity}preprocess --dataset {dataset_name}",
        f"python -m tm_ecg.cli {verbosity}pace --dataset {dataset_name}",
        f"python -m tm_ecg.cli {verbosity}rpeaks --dataset {dataset_name}",
        f"python -m tm_ecg.cli {verbosity}delineate --dataset {dataset_name}",
        f"python -m tm_ecg.cli {verbosity}triads --dataset {dataset_name}",
    ]
    if dataset == "b1":
        commands.extend(
            [
                f"python -m tm_ecg.cli {verbosity}train-classifier --dataset ptbxl",
                f"python -m tm_ecg.cli {verbosity}extract-a --dataset ptbxl",
                f"python -m tm_ecg.cli {verbosity}build-b --dataset b1",
                f"python -m tm_ecg.cli {verbosity}train-signatures --dataset b1",
            ]
        )
    else:
        commands.extend(
            [
                f"python -m tm_ecg.cli {verbosity}validate-fiducials",
                f"python -m tm_ecg.cli {verbosity}extract-a --dataset ludb",
                f"python -m tm_ecg.cli {verbosity}build-b --dataset b2",
            ]
        )
    commands.extend(
        [
            f"python -m tm_ecg.cli {verbosity}fit-transition --dataset {dataset}",
            f"python -m tm_ecg.cli {verbosity}explain --dataset {dataset} --split train",
            f"python -m tm_ecg.cli {verbosity}explain --dataset {dataset} --split val",
            f"python -m tm_ecg.cli {verbosity}explain --dataset {dataset} --split test",
        ]
    )

    for command in commands:
        c.run(command, env={"PYTHONPATH": "src"})

    # The rule-eligibility gate may fail while still producing a scientifically
    # necessary negative result. Always write the final report, then propagate
    # a nonzero gate status to prevent accidental release as a passing system.
    dss_command = (
        f"python -m tm_ecg.cli {verbosity}dss --dataset {dataset} --min-support 10"
    )
    dss_result = c.run(dss_command, env={"PYTHONPATH": "src"}, warn=True)
    report_command = f"python -m tm_ecg.cli {verbosity}report --experiment {dataset}"
    report_result = c.run(report_command, env={"PYTHONPATH": "src"}, warn=True)
    if not dss_result.ok or not report_result.ok:
        raise Exit(
            "The evidence-producing workflow completed, but a registered "
            "eligibility or reporting gate returned a nonzero status.",
            code=int(dss_result.exited or report_result.exited or 1),
        )
