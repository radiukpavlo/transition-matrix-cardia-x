"""Multilead triad classifier definition."""

from __future__ import annotations

from typing import Mapping


try:
    import torch  # type: ignore
    from torch import nn  # type: ignore
except ImportError:  # pragma: no cover - exercised only without optional torch
    torch = None  # type: ignore
    nn = None  # type: ignore


if nn is not None:
    class TriadCNN(nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            in_leads: int = 12,
            triad_length: int = 3,
            samples_per_beat: int = 256,
            latent_dim: int = 512,
            num_classes: int = 9,
            axis_classes: Mapping[str, int] | None = None,
        ) -> None:
            super().__init__()
            channels = in_leads * triad_length
            self.features = nn.Sequential(
                nn.Conv1d(channels, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.penultimate = nn.Linear(256, latent_dim)
            self.classifier = nn.Linear(latent_dim, num_classes)
            self.axis_heads = nn.ModuleDict(
                {
                    str(axis): nn.Linear(latent_dim, int(class_count))
                    for axis, class_count in dict(axis_classes or {}).items()
                }
            )
            self.samples_per_beat = samples_per_beat
            self.latent_dim = latent_dim
            self.num_classes = num_classes

        def forward(self, x):  # type: ignore[no-untyped-def]
            x = self.features(x).squeeze(-1)
            preactivation = self.penultimate(x)
            logits = self.classifier(torch.relu(preactivation))
            return logits, preactivation

        def forward_multiaxial(self, x):  # type: ignore[no-untyped-def]
            x = self.features(x).squeeze(-1)
            preactivation = self.penultimate(x)
            activated = torch.relu(preactivation)
            compatibility_logits = self.classifier(activated)
            axis_logits = {
                axis: head(activated) for axis, head in self.axis_heads.items()
            }
            return compatibility_logits, axis_logits, preactivation

else:
    class TriadCNN:  # pragma: no cover
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("TriadCNN requires torch. Install the train optional dependencies.")


def build_model(
    in_leads: int = 12,
    triad_length: int = 3,
    samples_per_beat: int = 256,
    latent_dim: int = 512,
    num_classes: int = 9,
    axis_classes: Mapping[str, int] | None = None,
):
    return TriadCNN(
        in_leads=in_leads,
        triad_length=triad_length,
        samples_per_beat=samples_per_beat,
        latent_dim=latent_dim,
        num_classes=num_classes,
        axis_classes=axis_classes,
    )
