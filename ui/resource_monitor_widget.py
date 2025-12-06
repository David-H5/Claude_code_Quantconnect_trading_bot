#!/usr/bin/env python3
"""
Resource Monitor Widget for Trading Dashboard

Real-time display of QuantConnect compute node resource usage.
Shows memory, CPU, latency, and alerts.

Author: QuantConnect Trading Bot
Date: 2025-11-30
"""

from __future__ import annotations


try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QProgressBar,
        QVBoxLayout,
        QWidget,
    )

    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False

    class QWidget:
        pass


class ResourceMonitorWidget(QWidget if PYSIDE_AVAILABLE else object):
    """
    Widget for displaying real-time resource metrics.

    Shows:
    - Memory usage (with progress bar)
    - CPU usage (with progress bar)
    - Broker latency
    - Active securities count
    - Recent alerts

    Example usage:
        widget = ResourceMonitorWidget(resource_monitor=monitor)
        widget.set_node_info("L2-4 (2 cores, 4GB RAM)")
        dashboard.add_widget(widget)
    """

    def __init__(
        self,
        resource_monitor: object | None = None,
        parent: QWidget | None = None,
    ):
        """
        Initialize resource monitor widget.

        Args:
            resource_monitor: ResourceMonitor instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.resource_monitor = resource_monitor

        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Node info section
        node_group = QGroupBox("Compute Node")
        node_layout = QVBoxLayout()
        self.node_label = QLabel("Unknown Node")
        self.node_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        node_layout.addWidget(self.node_label)
        node_group.setLayout(node_layout)
        layout.addWidget(node_group)

        # Resource usage section
        resource_group = QGroupBox("Resource Usage")
        resource_layout = QGridLayout()

        # Memory
        resource_layout.addWidget(QLabel("Memory:"), 0, 0)
        self.memory_bar = QProgressBar()
        self.memory_bar.setRange(0, 100)
        self.memory_bar.setTextVisible(True)
        self.memory_bar.setFormat("%v%")
        resource_layout.addWidget(self.memory_bar, 0, 1)
        self.memory_value = QLabel("0.0 / 0.0 GB")
        resource_layout.addWidget(self.memory_value, 0, 2)

        # CPU
        resource_layout.addWidget(QLabel("CPU:"), 1, 0)
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        self.cpu_bar.setTextVisible(True)
        self.cpu_bar.setFormat("%v%")
        resource_layout.addWidget(self.cpu_bar, 1, 1)
        self.cpu_cores = QLabel("0 cores")
        resource_layout.addWidget(self.cpu_cores, 1, 2)

        # Latency
        resource_layout.addWidget(QLabel("Broker Latency:"), 2, 0)
        self.latency_label = QLabel("-- ms")
        resource_layout.addWidget(self.latency_label, 2, 1, 1, 2)

        # Securities
        resource_layout.addWidget(QLabel("Active Securities:"), 3, 0)
        self.securities_label = QLabel("0")
        resource_layout.addWidget(self.securities_label, 3, 1, 1, 2)

        # Positions
        resource_layout.addWidget(QLabel("Active Positions:"), 4, 0)
        self.positions_label = QLabel("0")
        resource_layout.addWidget(self.positions_label, 4, 1, 1, 2)

        resource_group.setLayout(resource_layout)
        layout.addWidget(resource_group)

        # Health status
        health_group = QGroupBox("System Health")
        health_layout = QVBoxLayout()
        self.health_label = QLabel("Healthy")
        self.health_label.setAlignment(Qt.AlignCenter)
        self.health_label.setStyleSheet("font-weight: bold; font-size: 14px; color: green;")
        health_layout.addWidget(self.health_label)
        health_group.setLayout(health_layout)
        layout.addWidget(health_group)

        # Recent alerts
        alerts_group = QGroupBox("Recent Alerts")
        alerts_layout = QVBoxLayout()
        self.alerts_list = QListWidget()
        self.alerts_list.setMaximumHeight(150)
        alerts_layout.addWidget(self.alerts_list)
        alerts_group.setLayout(alerts_layout)
        layout.addWidget(alerts_group)

        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout()

        stats_layout.addWidget(QLabel("Avg Memory:"), 0, 0)
        self.avg_memory_label = QLabel("--")
        stats_layout.addWidget(self.avg_memory_label, 0, 1)

        stats_layout.addWidget(QLabel("Max Memory:"), 1, 0)
        self.max_memory_label = QLabel("--")
        stats_layout.addWidget(self.max_memory_label, 1, 1)

        stats_layout.addWidget(QLabel("Avg CPU:"), 0, 2)
        self.avg_cpu_label = QLabel("--")
        stats_layout.addWidget(self.avg_cpu_label, 0, 3)

        stats_layout.addWidget(QLabel("Max CPU:"), 1, 2)
        self.max_cpu_label = QLabel("--")
        stats_layout.addWidget(self.max_cpu_label, 1, 3)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        layout.addStretch()

    def _setup_timer(self) -> None:
        """Set up refresh timer."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(1000)  # Update every second

    def set_node_info(self, node_info: str) -> None:
        """
        Set compute node information.

        Args:
            node_info: Node description (e.g., "L2-4 (2 cores, 4GB RAM)")
        """
        self.node_label.setText(node_info)

        # Parse cores info if available
        if "cores" in node_info.lower():
            try:
                cores = node_info.split("cores")[0].split()[-1]
                self.cpu_cores.setText(f"{cores} cores")
            except (IndexError, ValueError):
                pass

    def update_display(self) -> None:
        """Update display with latest metrics."""
        if not self.resource_monitor:
            return

        # Get current metrics
        metrics = self.resource_monitor.get_current_metrics()
        if not metrics:
            return

        # Update memory
        memory_pct = int(metrics.memory_pct)
        self.memory_bar.setValue(memory_pct)
        self.memory_value.setText(f"{metrics.memory_used_mb / 1024:.2f} / {metrics.memory_total_mb / 1024:.2f} GB")

        # Color code based on usage
        if memory_pct >= 90:
            self.memory_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        elif memory_pct >= 80:
            self.memory_bar.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
        else:
            self.memory_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")

        # Update CPU
        cpu_pct = int(metrics.cpu_pct)
        self.cpu_bar.setValue(cpu_pct)

        if cpu_pct >= 85:
            self.cpu_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        elif cpu_pct >= 75:
            self.cpu_bar.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
        else:
            self.cpu_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")

        # Update latency
        if metrics.broker_latency_ms is not None:
            latency_color = "red" if metrics.broker_latency_ms > 100 else "green"
            self.latency_label.setText(f"{metrics.broker_latency_ms:.1f} ms")
            self.latency_label.setStyleSheet(f"color: {latency_color}; font-weight: bold;")
        else:
            self.latency_label.setText("-- ms")

        # Update counts
        self.securities_label.setText(str(metrics.active_securities))
        self.positions_label.setText(str(metrics.active_positions))

        # Update health status
        is_healthy = self.resource_monitor.is_healthy()
        if is_healthy:
            self.health_label.setText("✓ Healthy")
            self.health_label.setStyleSheet("font-weight: bold; font-size: 14px; color: green;")
        else:
            self.health_label.setText("⚠ Warning")
            self.health_label.setStyleSheet("font-weight: bold; font-size: 14px; color: red;")

        # Update alerts
        alerts = self.resource_monitor.get_recent_alerts(limit=5)
        if len(alerts) != self.alerts_list.count():
            self.alerts_list.clear()
            for alert in reversed(alerts):  # Most recent first
                item = QListWidgetItem(f"[{alert.severity.upper()}] {alert.message}")
                if alert.severity == "critical":
                    item.setForeground(QColor("red"))
                else:
                    item.setForeground(QColor("orange"))
                self.alerts_list.addItem(item)

        # Update statistics
        stats = self.resource_monitor.get_statistics()
        if "memory" in stats:
            self.avg_memory_label.setText(f"{stats['memory']['avg_pct']:.1f}%")
            self.max_memory_label.setText(f"{stats['memory']['max_pct']:.1f}%")
        if "cpu" in stats:
            self.avg_cpu_label.setText(f"{stats['cpu']['avg_pct']:.1f}%")
            self.max_cpu_label.setText(f"{stats['cpu']['max_pct']:.1f}%")

    def set_resource_monitor(self, monitor: object) -> None:
        """
        Set or update the resource monitor.

        Args:
            monitor: ResourceMonitor instance
        """
        self.resource_monitor = monitor
        self.update_display()


def create_resource_widget(
    resource_monitor: object,
    node_info: str = "Unknown Node",
) -> ResourceMonitorWidget:
    """
    Create a configured resource monitor widget.

    Args:
        resource_monitor: ResourceMonitor instance
        node_info: Node description string

    Returns:
        Configured ResourceMonitorWidget
    """
    widget = ResourceMonitorWidget(resource_monitor=resource_monitor)
    widget.set_node_info(node_info)
    return widget
