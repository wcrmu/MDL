from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    mode: str = "min"
    best: float | None = None
    bad_epochs: int = 0

    def step(self, value: float) -> bool:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        improved = (
            self.best is None
            or (self.mode == "min" and value < self.best)
            or (self.mode == "max" and value > self.best)
        )
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
