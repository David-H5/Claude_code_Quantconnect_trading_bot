"""
CSV Metrics Exporter

Exports metrics to CSV format for spreadsheet analysis.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from pathlib import Path

from observability.metrics.base import MetricRegistry, MetricSnapshot, get_default_registry


class CsvExporter:
    """Export metrics to CSV format."""

    def __init__(
        self,
        registry: MetricRegistry | None = None,
        include_header: bool = True,
        timestamp_column: bool = True,
    ):
        """
        Initialize CSV exporter.

        Args:
            registry: Metric registry (uses default if not provided)
            include_header: Include header row
            timestamp_column: Include timestamp column
        """
        self.registry = registry or get_default_registry()
        self.include_header = include_header
        self.timestamp_column = timestamp_column

    def export(self) -> str:
        """Export all metrics to CSV string."""
        snapshots = self.registry.get_all_snapshots()
        return self._snapshots_to_csv(snapshots)

    def export_to_file(self, filepath: str | Path) -> None:
        """Export metrics to a CSV file."""
        content = self.export()
        Path(filepath).write_text(content)

    def _snapshots_to_csv(self, snapshots: list[MetricSnapshot]) -> str:
        """Convert snapshots to CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Build header
        if self.include_header:
            header = ["metric_name", "metric_type", "unit", "labels", "value"]
            if self.timestamp_column:
                header.append("timestamp")
            writer.writerow(header)

        timestamp = datetime.now(timezone.utc).isoformat()

        # Write rows
        for snapshot in snapshots:
            for label_key, value in snapshot.values.items():
                # Format labels
                labels_str = ""
                if label_key and any(label_key):
                    labels_dict = dict(zip(snapshot.definition.labels, label_key))
                    labels_str = ",".join(f"{k}={v}" for k, v in labels_dict.items())

                row = [
                    snapshot.definition.name,
                    snapshot.definition.metric_type.value,
                    snapshot.definition.unit.value,
                    labels_str,
                    value,
                ]
                if self.timestamp_column:
                    row.append(timestamp)
                writer.writerow(row)

        return output.getvalue()


def export_to_csv(
    registry: MetricRegistry | None = None,
    include_header: bool = True,
) -> str:
    """
    Export metrics to CSV string.

    Args:
        registry: Metric registry (uses default if not provided)
        include_header: Include header row

    Returns:
        CSV string
    """
    exporter = CsvExporter(registry, include_header)
    return exporter.export()
