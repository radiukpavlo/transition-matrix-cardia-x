"""Regularized log-linear decoder for complete compatibility label sets."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import itertools
import json
from pathlib import Path
from typing import Sequence

from tm_ecg.modeling.label_contract import (
    CompatibilityLabelContractV4,
    DEFAULT_COMPATIBILITY_CONTRACT_V4,
    LabelContractError,
)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _matrix_hash(value: object) -> str:
    import numpy as np  # type: ignore

    matrix = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(matrix.dtype).encode("ascii"))
    digest.update(json.dumps(matrix.shape).encode("ascii"))
    digest.update(matrix.tobytes())
    return digest.hexdigest()


@dataclass(slots=True)
class StructuredLabelSetDecoder:
    """Decode calibrated probabilities under a versioned set contract."""

    contract: CompatibilityLabelContractV4 = (
        DEFAULT_COMPATIBILITY_CONTRACT_V4
    )
    regularization: float = 5.0
    pairwise_regularization: float = 20.0
    source_permits_af_afl: bool = False
    maximum_iterations: int = 200
    candidate_sets: tuple[tuple[str, ...], ...] = ()
    parameters: object | None = None
    set_log_priors: object | None = None
    fit_metadata: dict[str, object] = field(default_factory=dict)

    def _valid(self, labels: Sequence[str]) -> bool:
        values = set(labels)
        if not values:
            return False
        if self.contract.normal_label in values and len(values) != 1:
            return False
        if self.contract.residual_label in values and len(values) != 1:
            return False
        if (
            self.contract.af_afl_mutually_exclusive
            and not self.source_permits_af_afl
            and {"AF", "AFL"} <= values
        ):
            return False
        if (
            not self.contract.mixed_ectopy_allowed
            and {"APB", "PVC"} <= values
        ):
            return False
        conduction = {"RBBB spectrum", "LBBB spectrum"}
        if (
            not self.contract.pacing_conduction_cooccurrence
            and "Paced" in values
            and values.intersection(conduction)
        ):
            return False
        return values <= set(self.contract.label_order)

    def _labels_from_row(self, row: object) -> tuple[str, ...]:
        import numpy as np  # type: ignore

        values = np.asarray(row, dtype=int).reshape(-1)
        if len(values) != len(self.contract.label_order):
            raise ValueError("Target row does not match the label contract")
        labels = tuple(
            label
            for index, label in enumerate(self.contract.label_order)
            if values[index]
        )
        if not self._valid(labels):
            raise LabelContractError(f"Invalid structured target set: {labels}")
        return labels

    def build_candidate_sets(self, targets: object) -> tuple[tuple[str, ...], ...]:
        import numpy as np  # type: ignore

        y = np.asarray(targets, dtype=int)
        if y.ndim != 2 or y.shape[1] != len(self.contract.label_order):
            raise ValueError("Candidate targets must match the label contract")
        observed = {self._labels_from_row(row) for row in y}
        candidates = set(observed)
        candidates.add((self.contract.normal_label,))
        candidates.add((self.contract.residual_label,))
        for label in self.contract.specific_labels:
            candidates.add((label,))
        for labels in tuple(observed):
            values = set(labels)
            for label in self.contract.specific_labels:
                added = values | {label}
                removed = values - {label}
                for neighbor in (added, removed):
                    ordered = tuple(
                        item
                        for item in self.contract.label_order
                        if item in neighbor
                    )
                    if self._valid(ordered):
                        candidates.add(ordered)
        ordered_candidates = tuple(
            sorted(
                (candidate for candidate in candidates if self._valid(candidate)),
                key=lambda labels: (
                    len(labels),
                    tuple(
                        self.contract.label_order.index(label)
                        for label in labels
                    ),
                ),
            )
        )
        if not observed <= set(ordered_candidates):
            raise RuntimeError("Candidate generation dropped an observed target set")
        return ordered_candidates

    def _candidate_matrix(self) -> object:
        import numpy as np  # type: ignore

        return np.asarray(
            [
                [int(label in candidate) for label in self.contract.label_order]
                for candidate in self.candidate_sets
            ],
            dtype=float,
        )

    def _score_components(
        self,
        probabilities: object,
        specialist_probabilities: object | None,
    ) -> tuple[object, object, object, int]:
        import numpy as np  # type: ignore

        p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        if p.ndim != 2 or p.shape[1] != len(self.contract.label_order):
            raise ValueError("Decoder probabilities must match the label contract")
        candidates = self._candidate_matrix()
        base = (
            np.log(p)[:, None, :] * candidates[None, :, :]
            + np.log1p(-p)[:, None, :] * (1.0 - candidates[None, :, :])
        ).sum(axis=2)
        logits = np.log(p / (1.0 - p))
        signs = 2.0 * candidates - 1.0
        unary = logits[:, None, :] * signs[None, :, :]

        specialist_columns = 0
        if specialist_probabilities is not None:
            specialist = np.clip(
                np.asarray(specialist_probabilities, dtype=float),
                1e-6,
                1 - 1e-6,
            )
            if specialist.shape != p.shape:
                raise ValueError(
                    "Specialist probabilities must align with base probabilities"
                )
            specialist_logits = np.log(specialist / (1.0 - specialist))
            unary = np.concatenate(
                (
                    unary,
                    specialist_logits[:, None, :] * signs[None, :, :],
                ),
                axis=2,
            )
            specialist_columns = p.shape[1]

        pairs = list(itertools.combinations(range(p.shape[1]), 2))
        pair_features = np.asarray(
            [
                [
                    candidates[candidate, left] * candidates[candidate, right]
                    for left, right in pairs
                ]
                for candidate in range(len(candidates))
            ],
            dtype=float,
        )
        priors = np.asarray(self.set_log_priors, dtype=float).reshape(-1, 1)
        static = np.concatenate((pair_features, priors), axis=1)
        return base, unary, static, specialist_columns

    def fit(
        self,
        probabilities: object,
        targets: object,
        *,
        specialist_probabilities: object | None = None,
    ) -> "StructuredLabelSetDecoder":
        import numpy as np  # type: ignore
        from scipy.optimize import minimize  # type: ignore
        from scipy.special import logsumexp  # type: ignore

        p = np.asarray(probabilities, dtype=float)
        y = np.asarray(targets, dtype=int)
        if p.shape != y.shape or p.ndim != 2:
            raise ValueError("Decoder probabilities and targets must align")
        self.candidate_sets = self.build_candidate_sets(y)
        lookup = {
            candidate: index
            for index, candidate in enumerate(self.candidate_sets)
        }
        target_indices = np.asarray(
            [lookup[self._labels_from_row(row)] for row in y],
            dtype=int,
        )
        counts = np.bincount(
            target_indices,
            minlength=len(self.candidate_sets),
        ).astype(float)
        self.set_log_priors = np.log(
            (counts + 1.0) / (len(y) + len(self.candidate_sets))
        )
        base, unary, static, specialist_columns = self._score_components(
            p,
            specialist_probabilities,
        )
        unary = np.asarray(unary, dtype=float)
        static = np.asarray(static, dtype=float)
        unary_columns = unary.shape[2]
        static_columns = static.shape[1]
        initial = np.zeros(unary_columns + static_columns, dtype=float)

        def objective(parameters: object) -> tuple[float, object]:
            values = np.asarray(parameters, dtype=float)
            unary_parameters = values[:unary_columns]
            static_parameters = values[unary_columns:]
            scores = (
                base
                + np.einsum("ncu,u->nc", unary, unary_parameters)
                + static.dot(static_parameters)[None, :]
            )
            log_partition = logsumexp(scores, axis=1)
            loss = float(
                np.mean(log_partition - scores[np.arange(len(y)), target_indices])
            )
            probabilities_over_sets = np.exp(
                scores - log_partition[:, None]
            )
            probabilities_over_sets[np.arange(len(y)), target_indices] -= 1.0
            unary_gradient = np.einsum(
                "nc,ncu->u",
                probabilities_over_sets,
                unary,
            ) / len(y)
            static_gradient = (
                probabilities_over_sets.sum(axis=0).dot(static) / len(y)
            )
            regularization_weights = np.full_like(values, self.regularization)
            pair_start = unary_columns
            pair_stop = unary_columns + static_columns - 1
            regularization_weights[pair_start:pair_stop] = (
                self.pairwise_regularization
            )
            loss += 0.5 * float(
                np.sum(regularization_weights * values * values)
            )
            gradient = np.concatenate((unary_gradient, static_gradient))
            gradient += regularization_weights * values
            return loss, gradient

        optimization = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self.maximum_iterations,
                "ftol": 1e-10,
                "gtol": 1e-7,
            },
        )
        if not optimization.success:
            raise RuntimeError(
                f"Structured decoder optimization failed: {optimization.message}"
            )
        self.parameters = np.asarray(optimization.x, dtype=float)
        predictions = self.predict(
            p,
            specialist_probabilities=specialist_probabilities,
        )
        self.fit_metadata = {
            "version": 1,
            "objective": "regularized_exact_set_conditional_log_likelihood",
            "training_rows": len(y),
            "label_order": list(self.contract.label_order),
            "candidate_count": len(self.candidate_sets),
            "candidate_sets_hash": _canonical_hash(self.candidate_sets),
            "probabilities_hash": _matrix_hash(p),
            "targets_hash": _matrix_hash(y),
            "parameters_hash": _matrix_hash(self.parameters),
            "source_permits_af_afl": self.source_permits_af_afl,
            "specialist_probability_columns": specialist_columns,
            "regularization": self.regularization,
            "pairwise_regularization": self.pairwise_regularization,
            "optimization_iterations": int(optimization.nit),
            "optimization_loss": float(optimization.fun),
            "training_exact_subset_accuracy": float(
                np.all(predictions == y, axis=1).mean()
            ),
            "constraints": {
                "normal_exclusive": True,
                "residual_exclusive": True,
                "nonempty": True,
                "af_afl_mutually_exclusive": (
                    self.contract.af_afl_mutually_exclusive
                    and not self.source_permits_af_afl
                ),
                "mixed_ectopy_allowed": self.contract.mixed_ectopy_allowed,
                "pacing_conduction_cooccurrence": (
                    self.contract.pacing_conduction_cooccurrence
                ),
            },
        }
        return self

    def candidate_scores(
        self,
        probabilities: object,
        *,
        specialist_probabilities: object | None = None,
    ) -> object:
        import numpy as np  # type: ignore

        if self.parameters is None or self.set_log_priors is None:
            raise RuntimeError("Structured decoder has not been fitted")
        base, unary, static, _ = self._score_components(
            probabilities,
            specialist_probabilities,
        )
        parameters = np.asarray(self.parameters, dtype=float)
        unary_columns = unary.shape[2]
        return (
            base
            + np.einsum("ncu,u->nc", unary, parameters[:unary_columns])
            + static.dot(parameters[unary_columns:])[None, :]
        )

    def predict(
        self,
        probabilities: object,
        *,
        specialist_probabilities: object | None = None,
    ) -> object:
        import numpy as np  # type: ignore

        scores = self.candidate_scores(
            probabilities,
            specialist_probabilities=specialist_probabilities,
        )
        selected = np.asarray(scores).argmax(axis=1)
        candidates = self._candidate_matrix().astype(int)
        predictions = candidates[selected]
        self.contract.validate_prediction_matrix(predictions)
        return predictions

    def to_artifact(self) -> dict[str, object]:
        import numpy as np  # type: ignore

        if self.parameters is None or self.set_log_priors is None:
            raise RuntimeError("Structured decoder has not been fitted")
        return {
            "version": 1,
            "decoder": "regularized_log_linear_label_set_v1",
            "candidate_sets": [list(item) for item in self.candidate_sets],
            "parameters": np.asarray(self.parameters).tolist(),
            "set_log_priors": np.asarray(self.set_log_priors).tolist(),
            "fit_metadata": self.fit_metadata,
        }

    def write_artifact(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                self.to_artifact(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return destination

