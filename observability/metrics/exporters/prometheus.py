"""
Prometheus Metrics Exporter

Exports metrics in Prometheus text exposition format.
Compatible with Prometheus scraping.
"""

from __future__ import annotations

from pathlib import Path

from observability.metrics.base import (
    MetricRegistry,
    MetricSnapshot,
    MetricType,
    get_default_registry,
)


class PrometheusExporter:
    """Export metrics in Prometheus text format."""

    def __init__(
        self,
        registry: MetricRegistry | None = None,
        prefix: str = "",
        include_help: bool = True,
        include_type: bool = True,
    ):
        """
        Initialize Prometheus exporter.

        Args:
            registry: Metric registry (uses default if not provided)
            prefix: Prefix for all metric names
            include_help: Include HELP comments
            include_type: Include TYPE comments
        """
        self.registry = registry or get_default_registry()
        self.prefix = prefix
        self.include_help = include_help
        self.include_type = include_type

    def export(self) -> str:
        """Export all metrics to Prometheus format."""
        snapshots = self.registry.get_all_snapshots()
        return self._snapshots_to_prometheus(snapshots)

    def export_to_file(self, filepath: str | Path) -> None:
        """Export metrics to a file."""
        content = self.export()
        Path(filepath).write_text(content)

    def _snapshots_to_prometheus(self, snapshots: list[MetricSnapshot]) -> str:
        """Convert snapshots to Prometheus text format."""
        lines: list[str] = []

        for snapshot in snapshots:
            metric_name = self._format_metric_name(snapshot.definition.name)

            # Add HELP comment
            if self.include_help and snapshot.definition.description:
                lines.append(f"# HELP {metric_name} {snapshot.definition.description}")

            # Add TYPE comment
            if self.include_type:
                prom_type = self._get_prometheus_type(snapshot.definition.metric_type)
                lines.append(f"# TYPE {metric_name} {prom_type}")

            # Add metric values
            for label_key, value in snapshot.values.items():
                labels_str = self._format_labels(snapshot.definition.labels, label_key)
                if labels_str:
                    lines.append(f"{metric_name}{{{labels_str}}} {value}")
                else:
                    lines.append(f"{metric_name} {value}")

            # For histograms, add statistics as separate metrics
            if snapshot.statistics and snapshot.definition.metric_type in (MetricType.HISTOGRAM, MetricType.TIMER):
                stats = snapshot.statistics
                if "count" in stats:
                    lines.append(f"{metric_name}_count {stats['count']}")
                if "sum" in stats:
                    lines.append(f"{metric_name}_sum {stats['sum']}")

            lines.append("")  # Empty line between metrics

        return "\n".join(lines)

    def _format_metric_name(self, name: str) -> str:
        """Format metric name for Prometheus."""
        # Replace invalid characters with underscore
        formatted = name.replace(".", "_").replace("-", "_")
        if self.prefix:
            formatted = f"{self.prefix}_{formatted}"
        return formatted

    def _format_labels(self, label_names: list[str], label_values: tuple) -> str:
        """Format labels for Prometheus."""
        if not label_names or not label_values or not any(label_values):
            return ""

        pairs = []
        for name, value in zip(label_names, label_values):
            if value:
                # Escape special characters in value
                escaped_value = str(value).replace("\\", "\\\\").replace('"', '\\"')
                pairs.append(f'{name}="{escaped_value}"')

        return ",".join(pairs)

    def _get_prometheus_type(self, metric_type: MetricType) -> str:
        """Convert MetricType to Prometheus type."""
        type_map = {
            MetricType.COUNTER: "counter",
            MetricType.GAUGE: "gauge",
            MetricType.HISTOGRAM: "histogram",
            MetricType.TIMER: "histogram",
            MetricType.RATE: "gauge",
        }
        return type_map.get(metric_type, "untyped")


def export_to_prometheus(
    registry: MetricRegistry | None = None,
    prefix: str = "",
) -> str:
    """
    Export metrics to Prometheus format.

    Args:
        registry: Metric registry (uses default if not provided)
        prefix: Prefix for metric names

    Returns:
        Prometheus text format string
    """
    exporter = PrometheusExporter(registry, prefix)
    return exporter.export()
