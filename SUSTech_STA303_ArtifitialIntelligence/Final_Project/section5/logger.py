"""
Simple CSV + TensorBoard logger for Section 5 experiments.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Optional

from section5.common import ensure_dir

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


class MetricLogger:
    def __init__(self, log_dir: str):
        ensure_dir(log_dir)
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, "metrics.csv")
        self.writer = SummaryWriter(log_dir=log_dir) if SummaryWriter is not None else None
        self.fieldnames = None

    def _ensure_header(self, metrics: Dict[str, float]):
        if self.fieldnames is None:
            if os.path.exists(self.csv_path):
                with open(self.csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    self.fieldnames = reader.fieldnames
            if not self.fieldnames:
                self.fieldnames = ["step"] + list(metrics.keys())
                with open(self.csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()

        new_keys = [k for k in metrics.keys() if k not in self.fieldnames]
        if new_keys:
            self.fieldnames.extend(new_keys)
            existing_rows = []
            if os.path.exists(self.csv_path):
                with open(self.csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    existing_rows = list(reader)
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow(row)

    def log(self, step: int, metrics: Dict[str, float]):
        self._ensure_header(metrics)
        row = {"step": step, **metrics}
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()
