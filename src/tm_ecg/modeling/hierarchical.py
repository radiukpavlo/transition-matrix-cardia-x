"""Cross-fitted Normal/residual/specific hierarchy for exact-set decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tm_ecg.constants import PROJECT_LABELS


STATE_NAMES = ("normal", "residual", "specific_abnormal")
NORMAL_INDEX = PROJECT_LABELS.index("Normal")
RESIDUAL_INDEX = PROJECT_LABELS.index("Other / unmapped")
SPECIFIC_INDICES = tuple(
    index
    for index in range(len(PROJECT_LABELS))
    if index not in {NORMAL_INDEX, RESIDUAL_INDEX}
)


def _gate_features(probabilities: object) -> object:
    import numpy as np  # type: ignore

    matrix = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    if matrix.ndim != 2 or matrix.shape[1] != len(PROJECT_LABELS):
        raise ValueError("Hierarchical gate expects the frozen nine-label order")
    specific = matrix[:, SPECIFIC_INDICES]
    entropy = -np.sum(
        matrix * np.log(matrix) + (1 - matrix) * np.log(1 - matrix),
        axis=1,
    )
    return np.column_stack(
        (
            matrix,
            np.log(matrix / (1 - matrix)),
            specific.max(axis=1),
            specific.sum(axis=1),
            1.0 - np.prod(1.0 - specific, axis=1),
            entropy,
        )
    )


def _gate_states(targets: object) -> object:
    import numpy as np  # type: ignore

    matrix = np.asarray(targets, dtype=int)
    if matrix.ndim != 2 or matrix.shape[1] != len(PROJECT_LABELS):
        raise ValueError("Hierarchical gate expects the frozen nine-label order")
    states = np.full(len(matrix), 2, dtype=int)
    states[matrix[:, NORMAL_INDEX] == 1] = 0
    states[matrix[:, RESIDUAL_INDEX] == 1] = 1
    return states


@dataclass(slots=True)
class HierarchicalGate:
    model: object | None = None
    constant_state: int | None = None

    def fit(self, probabilities: object, targets: object) -> "HierarchicalGate":
        import numpy as np  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.pipeline import make_pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore

        states = _gate_states(targets)
        unique = np.unique(states)
        if len(unique) == 1:
            self.constant_state = int(unique[0])
            self.model = None
            return self
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.2,
                class_weight="balanced",
                max_iter=2000,
                random_state=23,
                solver="lbfgs",
            ),
        ).fit(_gate_features(probabilities), states)
        self.constant_state = None
        return self

    def state_probabilities(self, probabilities: object) -> object:
        import numpy as np  # type: ignore

        rows = len(np.asarray(probabilities))
        output = np.zeros((rows, 3), dtype=float)
        if self.constant_state is not None:
            output[:, self.constant_state] = 1.0
            return output
        if self.model is None:
            raise RuntimeError("Hierarchical gate is not fitted")
        raw = np.asarray(
            self.model.predict_proba(_gate_features(probabilities)),  # type: ignore[attr-defined]
            dtype=float,
        )
        for column, state in enumerate(self.model.classes_):  # type: ignore[attr-defined]
            output[:, int(state)] = raw[:, column]
        return np.clip(output, 1e-6, 1 - 1e-6)

    def transform(self, probabilities: object) -> object:
        import numpy as np  # type: ignore

        matrix = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        states = self.state_probabilities(matrix)
        output = matrix.copy()
        output[:, NORMAL_INDEX] = states[:, 0]
        output[:, RESIDUAL_INDEX] = states[:, 1]
        base_specific_any = np.clip(
            1.0 - np.prod(1.0 - matrix[:, SPECIFIC_INDICES], axis=1),
            1e-6,
            1.0,
        )
        scale = states[:, 2] / base_specific_any
        output[:, SPECIFIC_INDICES] = np.clip(
            matrix[:, SPECIFIC_INDICES] * scale[:, None],
            1e-6,
            1 - 1e-6,
        )
        return output


def crossfit_hierarchical_probabilities(
    train_probabilities: object,
    train_targets: object,
    groups: Sequence[str],
    validation_probabilities: object,
    *,
    n_splits: int = 5,
    seed: int = 23,
) -> tuple[object, object, dict[str, object]]:
    """Fit the hierarchy out-of-fold, then refit once for validation."""

    import numpy as np  # type: ignore
    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore

    train = np.asarray(train_probabilities, dtype=float)
    targets = np.asarray(train_targets, dtype=int)
    validation = np.asarray(validation_probabilities, dtype=float)
    if len(train) != len(targets) or len(groups) != len(train):
        raise ValueError("Hierarchical crossfit inputs must align")
    states = _gate_states(targets)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    oof = np.full_like(train, np.nan, dtype=float)
    fold_assignments = np.full(len(train), -1, dtype=int)
    for fold, (fit_indices, held_indices) in enumerate(
        splitter.split(train, states, groups)
    ):
        gate = HierarchicalGate().fit(
            train[fit_indices],
            targets[fit_indices],
        )
        oof[held_indices] = gate.transform(train[held_indices])
        fold_assignments[held_indices] = fold
    if not np.isfinite(oof).all() or (fold_assignments < 0).any():
        raise RuntimeError("Hierarchical crossfit did not cover every row once")
    full = HierarchicalGate().fit(train, targets)
    transformed_validation = full.transform(validation)
    return oof, transformed_validation, {
        "version": 1,
        "states": list(STATE_NAMES),
        "n_splits": n_splits,
        "patient_group_count": len(set(groups)),
        "all_training_rows_transformed_out_of_fold": True,
        "fold_counts": {
            str(fold): int((fold_assignments == fold).sum())
            for fold in range(n_splits)
        },
        "normal_residual_specific_exclusive": True,
    }
