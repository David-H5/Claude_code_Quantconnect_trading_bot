"""
Attention Heatmap Chart Widget (UPGRADE-010 Sprint 2 P1)

Visualizes attention weights from multi-head attention layers.
Supports per-head selection and matrix export.

QuantConnect Compatible: N/A (UI component)
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Any

from .base_chart import (
    MATPLOTLIB_AVAILABLE,
    BaseChartWidget,
)


logger = logging.getLogger(__name__)


class AttentionHeatmapChart(BaseChartWidget):
    """
    Chart widget for visualizing attention weights as heatmaps.

    Features:
    - Heatmap display of attention matrices
    - Per-head selection
    - Asset labels on axes
    - Export to PNG/CSV
    """

    def __init__(
        self,
        parent: Any | None = None,
        title: str = "Attention Weights",
        figsize: tuple[float, float] = (8, 8),
    ):
        """
        Initialize attention heatmap chart.

        Args:
            parent: Parent widget
            title: Chart title
            figsize: Figure size (width, height)
        """
        super().__init__(parent=parent, title=title, figsize=figsize)
        self._weights: list[list[float]] = []
        self._head_weights: list[list[list[float]]] = []  # Per-head weights
        self._labels: list[str] = []
        self._selected_head: int | None = None
        self._colormap: str = "Blues"

    def set_weights(
        self,
        weights: list[list[float]],
        labels: list[str] | None = None,
    ) -> None:
        """
        Set attention weights matrix.

        Args:
            weights: 2D attention weights (query_len, key_len)
            labels: Optional labels for axes (asset names)
        """
        self._weights = weights
        if labels:
            self._labels = labels
        elif weights:
            self._labels = [f"Asset {i}" for i in range(len(weights))]
        self.update_chart()

    def set_head_weights(
        self,
        head_weights: list[list[list[float]]],
        labels: list[str] | None = None,
    ) -> None:
        """
        Set per-head attention weights.

        Args:
            head_weights: Per-head weights (num_heads, query_len, key_len)
            labels: Optional labels for axes
        """
        self._head_weights = head_weights
        if labels:
            self._labels = labels
        elif head_weights and head_weights[0]:
            self._labels = [f"Asset {i}" for i in range(len(head_weights[0]))]

        # Compute average weights if no specific head selected
        if head_weights and self._selected_head is None:
            num_heads = len(head_weights)
            if num_heads > 0 and head_weights[0]:
                seq_len = len(head_weights[0])
                key_len = len(head_weights[0][0]) if head_weights[0] else 0

                avg_weights = [[0.0] * key_len for _ in range(seq_len)]
                for h in range(num_heads):
                    for i in range(seq_len):
                        for j in range(key_len):
                            avg_weights[i][j] += head_weights[h][i][j] / num_heads
                self._weights = avg_weights
        elif head_weights and self._selected_head is not None:
            if 0 <= self._selected_head < len(head_weights):
                self._weights = head_weights[self._selected_head]

        self.update_chart()

    def select_head(self, head_index: int | None) -> None:
        """
        Select a specific attention head to display.

        Args:
            head_index: Head index (0-based), or None for averaged weights
        """
        self._selected_head = head_index

        if head_index is None and self._head_weights:
            # Recompute average
            self.set_head_weights(self._head_weights, self._labels)
        elif head_index is not None and self._head_weights:
            if 0 <= head_index < len(self._head_weights):
                self._weights = self._head_weights[head_index]
                self.update_chart()

    def get_head_count(self) -> int:
        """
        Get number of attention heads.

        Returns:
            Number of heads available
        """
        return len(self._head_weights)

    def set_colormap(self, colormap: str) -> None:
        """
        Set colormap for heatmap.

        Args:
            colormap: Matplotlib colormap name (e.g., 'Blues', 'Reds', 'viridis')
        """
        self._colormap = colormap
        self.update_chart()

    def update_chart(self) -> None:
        """Redraw the chart with current data."""
        if not MATPLOTLIB_AVAILABLE:
            self._show_placeholder("matplotlib not available")
            return

        if not self._weights:
            self._show_placeholder("No attention weights")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Create heatmap
        import numpy as np

        weights_array = np.array(self._weights)

        im = ax.imshow(
            weights_array,
            cmap=self._colormap,
            aspect="auto",
            vmin=0,
            vmax=1,
        )

        # Add colorbar
        self.figure.colorbar(im, ax=ax, label="Attention Weight")

        # Set labels
        if self._labels:
            ax.set_xticks(range(len(self._labels)))
            ax.set_xticklabels(self._labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(self._labels)))
            ax.set_yticklabels(self._labels, fontsize=8)

        # Add title with head info
        title = self.title
        if self._selected_head is not None:
            title += f" (Head {self._selected_head})"
        elif self._head_weights:
            title += f" (Avg of {len(self._head_weights)} heads)"
        ax.set_title(title)

        ax.set_xlabel("Key (Assets)")
        ax.set_ylabel("Query (Assets)")

        # Add values as text annotations
        for i in range(len(self._weights)):
            for j in range(len(self._weights[i])):
                value = self._weights[i][j]
                color = "white" if value > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=6,
                )

        self.figure.tight_layout()
        self.canvas.draw()

    def export_csv(self) -> str:
        """
        Export attention weights to CSV format.

        Returns:
            CSV string
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Header row with labels
        header = [""] + self._labels if self._labels else [""]
        writer.writerow(header)

        # Data rows
        for i, row in enumerate(self._weights):
            label = self._labels[i] if i < len(self._labels) else f"Row {i}"
            writer.writerow([label] + [f"{v:.4f}" for v in row])

        return output.getvalue()

    def export_png(self, filepath: str) -> bool:
        """
        Export chart to PNG file.

        Args:
            filepath: Output file path

        Returns:
            True if successful
        """
        if not MATPLOTLIB_AVAILABLE:
            return False

        try:
            self.figure.savefig(filepath, dpi=150, bbox_inches="tight")
            return True
        except Exception as e:
            logger.error(f"Failed to export PNG: {e}")
            return False

    def get_attention_summary(self) -> dict[str, Any]:
        """
        Get summary statistics for attention weights.

        Returns:
            Dictionary with attention statistics
        """
        if not self._weights:
            return {"error": "No weights available"}

        # Flatten weights
        all_weights = [w for row in self._weights for w in row]

        # Calculate statistics
        avg_weight = sum(all_weights) / len(all_weights) if all_weights else 0
        max_weight = max(all_weights) if all_weights else 0
        min_weight = min(all_weights) if all_weights else 0

        # Find strongest attention pairs
        strongest_pairs = []
        for i, row in enumerate(self._weights):
            for j, weight in enumerate(row):
                if i != j:  # Skip self-attention
                    src = self._labels[i] if i < len(self._labels) else f"Asset {i}"
                    dst = self._labels[j] if j < len(self._labels) else f"Asset {j}"
                    strongest_pairs.append((src, dst, weight))

        strongest_pairs.sort(key=lambda x: x[2], reverse=True)

        return {
            "avg_weight": avg_weight,
            "max_weight": max_weight,
            "min_weight": min_weight,
            "num_assets": len(self._weights),
            "num_heads": len(self._head_weights),
            "selected_head": self._selected_head,
            "top_5_pairs": strongest_pairs[:5],
        }


def create_attention_heatmap_chart(
    parent: Any | None = None,
    title: str = "Attention Weights",
    figsize: tuple[float, float] = (8, 8),
) -> AttentionHeatmapChart:
    """
    Factory function to create an attention heatmap chart.

    Args:
        parent: Parent widget
        title: Chart title
        figsize: Figure size

    Returns:
        AttentionHeatmapChart instance
    """
    return AttentionHeatmapChart(parent=parent, title=title, figsize=figsize)
