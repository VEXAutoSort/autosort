"""The VEX part table, and matching a measured shape against it.

Every VEX fastener is #8-32 and every gear is 24 diametral pitch, so part
dimensions land on a known lattice — identifying a piece is snapping a
measurement to the nearest entry rather than recognising an image.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .vision import Shape

DEFAULT_PATH = Path(__file__).with_name("vex_parts.yaml")

# how far a field may stray before it counts as one unit of distance
TOLERANCE = {
    "length_mm": 1.0,
    "width_mm": 1.0,
    "extent": 0.08,
    "circularity": 0.08,
    "solidity": 0.08,
    "holes": 0.5,
}


@dataclass
class Entry:
    label: str
    name: str
    values: dict[str, float]

    def distance(self, shape: Shape) -> float:
        """RMS tolerance-units away, over the fields this entry specifies."""
        total = sum(
            ((getattr(shape, field) - want) / TOLERANCE[field]) ** 2
            for field, want in self.values.items()
        )
        return (total / len(self.values)) ** 0.5


@dataclass
class Match:
    entry: Entry
    distance: float
    runner_up: float   # nearest entry carrying a *different* label

    @property
    def margin(self) -> float:
        """0 when two labels fit equally well, approaching 1 when one clearly wins."""
        if self.runner_up == float("inf"):
            return 1.0
        if self.runner_up <= 0.0:
            return 0.0
        return max(0.0, 1.0 - self.distance / self.runner_up)


class Catalog:
    def __init__(self, entries: list[Entry]):
        self.entries = entries

    @property
    def labels(self) -> set[str]:
        return {e.label for e in self.entries}

    @staticmethod
    def load(path: str | Path | None = None) -> "Catalog":
        import yaml

        p = Path(path) if path else DEFAULT_PATH
        rows = yaml.safe_load(p.read_text()) or []
        entries = []
        for row in rows:
            row = dict(row)
            label, name = row.pop("label"), row.pop("name")
            bad = set(row) - set(TOLERANCE)
            if bad:
                raise ValueError(f"{p}: entry {name!r} has unknown field(s) {sorted(bad)}")
            if not row:
                raise ValueError(f"{p}: entry {name!r} specifies no dimensions")
            entries.append(Entry(label, name, {k: float(v) for k, v in row.items()}))
        if not entries:
            raise ValueError(f"{p}: no entries")
        return Catalog(entries)

    def match(self, shape: Shape) -> Match:
        ranked = sorted(self.entries, key=lambda e: e.distance(shape))
        best = ranked[0]
        runner_up = next(
            (e.distance(shape) for e in ranked[1:] if e.label != best.label), float("inf")
        )
        return Match(best, best.distance(shape), runner_up)
