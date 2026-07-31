"""Triad construction utilities."""

from __future__ import annotations

from tm_ecg.types import BeatAcceptance, TriadMembership


def accepted_beat_ids(beat_acceptances: list[BeatAcceptance]) -> list[str]:
    return [item.beat_id for item in beat_acceptances if item.accepted]


def build_triad_memberships(record_id: str, beat_acceptances: list[BeatAcceptance]) -> list[TriadMembership]:
    accepted = accepted_beat_ids(beat_acceptances)
    triads = []
    for idx in range(1, len(accepted) - 1):
        triads.append(
            TriadMembership(
                triad_id=f"{record_id}-triad-{idx:04d}",
                record_id=record_id,
                previous_beat_id=accepted[idx - 1],
                current_beat_id=accepted[idx],
                next_beat_id=accepted[idx + 1],
            )
        )
    return triads
