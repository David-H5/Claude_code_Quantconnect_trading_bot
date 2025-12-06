"""
JSON Metrics Exporter

Exports metrics to JSON format for API responses and file storage.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from observability.metrics.base import MetricRegistry, MetricSnapshot, get_default_registry


class JsonExporter:
    """Export metrics to JSON format."""

    def __init__(
        self,
        registry: MetricRegistry | None = None,
        pretty: bool = True,
        include_metadata: bool = True,
    ):
        """
        Initialize JSON exporter.

        Args:
            registry: Metric registry (uses default if not provided)
            pretty: Pretty-print JSON output
            include_metadata: Include export metadata
        """
        self.registry = registry or get_default_registry()
        self.pretty = pretty
        self.include_metadata = include_metadata

    def export(self) -> str:
        """Export all metrics to JSON string."""
        snapshots = self.registry.get_all_snapshots()
        return self._snapshots_to_json(snapshots)

    def export_to_file(self, filepath: str | Path) -> None:
        """Export metrics to a JSON file."""
        content = self.export()
        Path(filepath).write_text(content)

    def _snapshots_to_json(self, snapshots: list[MetricSnapshot]) -> str:
        """Convert snapshots to JSON string."""
        data = self._build_export_data(snapshots)
        indent = 2 if self.pretty else None
        return json.dumps(data, indent=indent, default=str)

    def _build_export_data(self, snapshots: list[MetricSnapshot]) -> dict[str, Any]:
        """Build export data structure."""
        metrics_data = []
        for snapshot in snapshots:
            metric_data = {
                "name": snapshot.definition.name,
                "type": snapshot.definition.metric_type.value,
                "description": snapshot.definition.description,
                "unit": snapshot.definition.unit.value,
                "values": [],
            }

            # Convert values
            for label_key, value in snapshot.values.items():
                entry = {"value": value}
                if label_key and any(label_key):
                    entry["labels"] = dict(zip(snapshot.definition.labels, label_key))
                metric_data["values"].append(entry)

            # Add statistics if present
            if snapshot.statistics:
                metric_data["statistics"] = snapshot.statistics

            metrics_data.append(metric_data)

        result: dict[str, Any] = {"metrics": metrics_data}

        if self.include_metadata:
            result["metadata"] = {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "metric_count": len(snapshots),
                "format": "json",
                "version": "1.0",
            }

        return result


def export_to_json(
    registry: MetricRegistry | None = None,
    pretty: bool = True,
) -> str:
    """
    Export metrics to JSON string.

    Args:
        registry: Metric registry (uses default if not provided)
        pretty: Pretty-print output

    Returns:
        JSON string
    """
    exporter = JsonExporter(registry, pretty)
    return exporter.export()
