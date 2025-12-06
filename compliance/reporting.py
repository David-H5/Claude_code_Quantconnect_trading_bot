"""
Compliance Reporting Module

UPGRADE-015 Phase 11: Compliance and Audit Logging

Generates compliance reports for regulatory requirements:
- Daily trading summaries
- Risk exposure reports
- Audit trail reports
- Exception reports
- Regulatory filings

Supports multiple output formats:
- JSON
- CSV
- HTML
- PDF (requires external library)
"""

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


class ReportFormat(Enum):
    """Report output formats."""

    JSON = "json"
    CSV = "csv"
    HTML = "html"
    TEXT = "text"


class ReportPeriod(Enum):
    """Report time periods."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    CUSTOM = "custom"


class ReportType(Enum):
    """Types of compliance reports."""

    TRADING_SUMMARY = "trading_summary"
    RISK_EXPOSURE = "risk_exposure"
    AUDIT_TRAIL = "audit_trail"
    EXCEPTION = "exception"
    POSITION_RECONCILIATION = "position_reconciliation"
    ORDER_ACTIVITY = "order_activity"
    MANIPULATION_ALERTS = "manipulation_alerts"


@dataclass
class ReportSection:
    """Section within a compliance report."""

    title: str
    content: Any  # Can be dict, list, or str
    section_type: str = "data"  # data, summary, chart


@dataclass
class ComplianceReport:
    """Generated compliance report."""

    report_id: str
    report_type: ReportType
    period: ReportPeriod
    generated_at: datetime
    start_date: datetime
    end_date: datetime

    # Content
    title: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    sections: list[ReportSection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Status
    status: str = "generated"  # generated, reviewed, approved, submitted

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "period": self.period.value,
            "generated_at": self.generated_at.isoformat(),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "title": self.title,
            "summary": self.summary,
            "sections": [{"title": s.title, "content": s.content, "type": s.section_type} for s in self.sections],
            "metadata": self.metadata,
            "status": self.status,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_csv(self) -> str:
        """Export data sections to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header info
        writer.writerow(["Report ID", self.report_id])
        writer.writerow(["Report Type", self.report_type.value])
        writer.writerow(["Period", f"{self.start_date.date()} to {self.end_date.date()}"])
        writer.writerow([])

        # Write each data section
        for section in self.sections:
            if section.section_type == "data" and isinstance(section.content, list):
                writer.writerow([section.title])
                if section.content and isinstance(section.content[0], dict):
                    # Write headers
                    headers = list(section.content[0].keys())
                    writer.writerow(headers)
                    # Write data
                    for row in section.content:
                        writer.writerow([row.get(h, "") for h in headers])
                writer.writerow([])

        return output.getvalue()

    def to_html(self) -> str:
        """Export to HTML."""
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{self.title}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #4CAF50; color: white; }",
            "tr:nth-child(even) { background-color: #f2f2f2; }",
            ".summary { background-color: #e7f3ff; padding: 15px; margin: 10px 0; }",
            ".section { margin: 20px 0; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{self.title}</h1>",
            f"<p>Report ID: {self.report_id}</p>",
            f"<p>Period: {self.start_date.date()} to {self.end_date.date()}</p>",
            f"<p>Generated: {self.generated_at.isoformat()}</p>",
        ]

        # Summary section
        if self.summary:
            html.append('<div class="summary">')
            html.append("<h2>Summary</h2>")
            html.append("<ul>")
            for key, value in self.summary.items():
                html.append(f"<li><strong>{key}:</strong> {value}</li>")
            html.append("</ul>")
            html.append("</div>")

        # Content sections
        for section in self.sections:
            html.append('<div class="section">')
            html.append(f"<h2>{section.title}</h2>")

            if isinstance(section.content, list) and section.content:
                if isinstance(section.content[0], dict):
                    # Table format
                    html.append("<table>")
                    headers = list(section.content[0].keys())
                    html.append("<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>")
                    for row in section.content:
                        html.append("<tr>" + "".join(f"<td>{row.get(h, '')}</td>" for h in headers) + "</tr>")
                    html.append("</table>")
                else:
                    # List format
                    html.append("<ul>")
                    for item in section.content:
                        html.append(f"<li>{item}</li>")
                    html.append("</ul>")
            elif isinstance(section.content, dict):
                html.append("<ul>")
                for key, value in section.content.items():
                    html.append(f"<li><strong>{key}:</strong> {value}</li>")
                html.append("</ul>")
            else:
                html.append(f"<p>{section.content}</p>")

            html.append("</div>")

        html.extend(["</body>", "</html>"])
        return "\n".join(html)


@dataclass
class ReporterConfig:
    """Configuration for compliance reporter."""

    output_dir: Path = field(default_factory=lambda: Path("compliance_reports"))
    default_format: ReportFormat = ReportFormat.JSON
    include_raw_data: bool = True
    anonymize_accounts: bool = False


class ComplianceReporter:
    """Generate compliance reports from audit and trading data."""

    def __init__(
        self,
        config: ReporterConfig | None = None,
    ):
        """
        Initialize compliance reporter.

        Args:
            config: Reporter configuration
        """
        self.config = config or ReporterConfig()
        self._report_counter = 0
        self._generated_reports: list[ComplianceReport] = []

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Report Generation
    # ==========================================================================

    def generate_trading_summary(
        self,
        trades: list[dict[str, Any]],
        orders: list[dict[str, Any]],
        period: ReportPeriod = ReportPeriod.DAILY,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> ComplianceReport:
        """
        Generate trading summary report.

        Args:
            trades: List of trade records
            orders: List of order records
            period: Report period
            start_date: Start date
            end_date: End date

        Returns:
            Generated report
        """
        end_date = end_date or datetime.utcnow()
        start_date = start_date or self._get_period_start(end_date, period)

        # Calculate statistics
        total_trades = len(trades)
        total_orders = len(orders)
        total_volume = sum(t.get("quantity", 0) for t in trades)
        total_notional = sum(t.get("quantity", 0) * t.get("price", 0) for t in trades)
        total_commission = sum(t.get("commission", 0) for t in trades)

        # Order statistics
        filled_orders = len([o for o in orders if o.get("status") == "filled"])
        cancelled_orders = len([o for o in orders if o.get("status") == "cancelled"])
        fill_rate = filled_orders / total_orders if total_orders > 0 else 0

        # Group by symbol
        symbols: dict[str, int] = {}
        for t in trades:
            sym = t.get("symbol", "UNKNOWN")
            symbols[sym] = symbols.get(sym, 0) + t.get("quantity", 0)

        report = self._create_report(
            report_type=ReportType.TRADING_SUMMARY,
            period=period,
            start_date=start_date,
            end_date=end_date,
            title=f"Trading Summary Report - {period.value.capitalize()}",
        )

        report.summary = {
            "total_trades": total_trades,
            "total_orders": total_orders,
            "total_volume": total_volume,
            "total_notional": f"${total_notional:,.2f}",
            "total_commission": f"${total_commission:,.2f}",
            "fill_rate": f"{fill_rate:.1%}",
            "cancelled_orders": cancelled_orders,
            "unique_symbols": len(symbols),
        }

        # Add sections
        report.sections.append(
            ReportSection(
                title="Volume by Symbol",
                content=[{"symbol": k, "volume": v} for k, v in sorted(symbols.items(), key=lambda x: -x[1])[:20]],
                section_type="data",
            )
        )

        if self.config.include_raw_data:
            report.sections.append(
                ReportSection(
                    title="Trade Details",
                    content=trades[:100],  # Limit to 100 trades
                    section_type="data",
                )
            )

        self._generated_reports.append(report)
        return report

    def generate_risk_exposure(
        self,
        positions: list[dict[str, Any]],
        portfolio_value: float,
        risk_metrics: dict[str, Any] | None = None,
        period: ReportPeriod = ReportPeriod.DAILY,
    ) -> ComplianceReport:
        """
        Generate risk exposure report.

        Args:
            positions: Current positions
            portfolio_value: Total portfolio value
            risk_metrics: Additional risk metrics
            period: Report period

        Returns:
            Generated report
        """
        now = datetime.utcnow()
        report = self._create_report(
            report_type=ReportType.RISK_EXPOSURE,
            period=period,
            start_date=now,
            end_date=now,
            title=f"Risk Exposure Report - {now.date()}",
        )

        # Calculate exposures
        long_exposure = sum(p.get("market_value", 0) for p in positions if p.get("quantity", 0) > 0)
        short_exposure = abs(sum(p.get("market_value", 0) for p in positions if p.get("quantity", 0) < 0))
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure

        # Concentration
        position_values = [abs(p.get("market_value", 0)) for p in positions]
        max_position = max(position_values) if position_values else 0
        concentration = max_position / portfolio_value if portfolio_value > 0 else 0

        report.summary = {
            "portfolio_value": f"${portfolio_value:,.2f}",
            "long_exposure": f"${long_exposure:,.2f}",
            "short_exposure": f"${short_exposure:,.2f}",
            "gross_exposure": f"${gross_exposure:,.2f}",
            "net_exposure": f"${net_exposure:,.2f}",
            "gross_leverage": f"{gross_exposure / portfolio_value:.2f}x" if portfolio_value > 0 else "N/A",
            "max_concentration": f"{concentration:.1%}",
            "position_count": len(positions),
        }

        if risk_metrics:
            report.summary.update(risk_metrics)

        # Position details
        position_data = []
        for p in positions:
            position_data.append(
                {
                    "symbol": p.get("symbol", ""),
                    "quantity": p.get("quantity", 0),
                    "market_value": f"${p.get('market_value', 0):,.2f}",
                    "pct_of_portfolio": f"{p.get('market_value', 0) / portfolio_value * 100:.1f}%"
                    if portfolio_value > 0
                    else "0%",
                    "unrealized_pnl": f"${p.get('unrealized_pnl', 0):,.2f}",
                }
            )

        report.sections.append(ReportSection(title="Position Details", content=position_data, section_type="data"))

        self._generated_reports.append(report)
        return report

    def generate_audit_trail(
        self,
        audit_entries: list[dict[str, Any]],
        period: ReportPeriod = ReportPeriod.DAILY,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> ComplianceReport:
        """
        Generate audit trail report.

        Args:
            audit_entries: Audit log entries
            period: Report period
            start_date: Start date
            end_date: End date

        Returns:
            Generated report
        """
        end_date = end_date or datetime.utcnow()
        start_date = start_date or self._get_period_start(end_date, period)

        report = self._create_report(
            report_type=ReportType.AUDIT_TRAIL,
            period=period,
            start_date=start_date,
            end_date=end_date,
            title=f"Audit Trail Report - {period.value.capitalize()}",
        )

        # Categorize entries
        categories: dict[str, int] = {}
        levels: dict[str, int] = {}
        actors: dict[str, int] = {}

        for entry in audit_entries:
            cat = entry.get("category", "system")
            categories[cat] = categories.get(cat, 0) + 1

            lvl = entry.get("level", "info")
            levels[lvl] = levels.get(lvl, 0) + 1

            actor = entry.get("actor", "unknown")
            actors[actor] = actors.get(actor, 0) + 1

        report.summary = {
            "total_entries": len(audit_entries),
            "categories": categories,
            "levels": levels,
            "unique_actors": len(actors),
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
        }

        # Entries by category
        report.sections.append(
            ReportSection(
                title="Entries by Category",
                content=[{"category": k, "count": v} for k, v in sorted(categories.items(), key=lambda x: -x[1])],
                section_type="data",
            )
        )

        # High severity entries
        high_severity = [e for e in audit_entries if e.get("level") in ["error", "critical", "warning"]]
        if high_severity:
            report.sections.append(
                ReportSection(
                    title="High Severity Events",
                    content=high_severity[:50],
                    section_type="data",
                )
            )

        self._generated_reports.append(report)
        return report

    def generate_exception_report(
        self,
        exceptions: list[dict[str, Any]],
        period: ReportPeriod = ReportPeriod.DAILY,
    ) -> ComplianceReport:
        """
        Generate exception/violation report.

        Args:
            exceptions: List of exceptions/violations
            period: Report period

        Returns:
            Generated report
        """
        now = datetime.utcnow()
        report = self._create_report(
            report_type=ReportType.EXCEPTION,
            period=period,
            start_date=self._get_period_start(now, period),
            end_date=now,
            title=f"Exception Report - {period.value.capitalize()}",
        )

        # Categorize exceptions
        by_type: dict[str, list[dict]] = {}
        by_severity: dict[str, int] = {}

        for exc in exceptions:
            exc_type = exc.get("type", "unknown")
            if exc_type not in by_type:
                by_type[exc_type] = []
            by_type[exc_type].append(exc)

            severity = exc.get("severity", "low")
            by_severity[severity] = by_severity.get(severity, 0) + 1

        report.summary = {
            "total_exceptions": len(exceptions),
            "by_severity": by_severity,
            "exception_types": len(by_type),
            "critical_count": by_severity.get("critical", 0),
            "high_count": by_severity.get("high", 0),
        }

        # Add sections by type
        for exc_type, exc_list in by_type.items():
            report.sections.append(
                ReportSection(
                    title=f"{exc_type.replace('_', ' ').title()} Exceptions",
                    content=exc_list,
                    section_type="data",
                )
            )

        self._generated_reports.append(report)
        return report

    def generate_manipulation_report(
        self,
        alerts: list[dict[str, Any]],
        period: ReportPeriod = ReportPeriod.DAILY,
    ) -> ComplianceReport:
        """
        Generate manipulation alerts report.

        Args:
            alerts: Manipulation alerts
            period: Report period

        Returns:
            Generated report
        """
        now = datetime.utcnow()
        report = self._create_report(
            report_type=ReportType.MANIPULATION_ALERTS,
            period=period,
            start_date=self._get_period_start(now, period),
            end_date=now,
            title=f"Market Manipulation Alerts - {period.value.capitalize()}",
        )

        # Categorize alerts
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_symbol: dict[str, int] = {}

        for alert in alerts:
            alert_type = alert.get("manipulation_type", "unknown")
            by_type[alert_type] = by_type.get(alert_type, 0) + 1

            severity = alert.get("severity", "low")
            by_severity[severity] = by_severity.get(severity, 0) + 1

            symbol = alert.get("symbol", "unknown")
            by_symbol[symbol] = by_symbol.get(symbol, 0) + 1

        report.summary = {
            "total_alerts": len(alerts),
            "by_type": by_type,
            "by_severity": by_severity,
            "symbols_affected": len(by_symbol),
            "critical_alerts": by_severity.get("critical", 0),
        }

        # Alert details
        report.sections.append(
            ReportSection(
                title="Alert Details",
                content=alerts,
                section_type="data",
            )
        )

        # Alerts by symbol
        report.sections.append(
            ReportSection(
                title="Alerts by Symbol",
                content=[{"symbol": k, "alert_count": v} for k, v in sorted(by_symbol.items(), key=lambda x: -x[1])],
                section_type="data",
            )
        )

        self._generated_reports.append(report)
        return report

    # ==========================================================================
    # Export and Persistence
    # ==========================================================================

    def export_report(
        self,
        report: ComplianceReport,
        format: ReportFormat | None = None,
        filename: str | None = None,
    ) -> Path:
        """
        Export report to file.

        Args:
            report: Report to export
            format: Output format
            filename: Custom filename

        Returns:
            Path to exported file
        """
        format = format or self.config.default_format

        if not filename:
            date_str = report.generated_at.strftime("%Y%m%d_%H%M%S")
            filename = f"{report.report_type.value}_{date_str}"

        # Add extension
        ext = format.value
        if not filename.endswith(f".{ext}"):
            filename = f"{filename}.{ext}"

        filepath = self.config.output_dir / filename

        # Export based on format
        if format == ReportFormat.JSON:
            content = report.to_json()
        elif format == ReportFormat.CSV:
            content = report.to_csv()
        elif format == ReportFormat.HTML:
            content = report.to_html()
        else:
            content = str(report.to_dict())

        with open(filepath, "w") as f:
            f.write(content)

        return filepath

    def export_all_reports(
        self,
        format: ReportFormat | None = None,
    ) -> list[Path]:
        """Export all generated reports."""
        paths = []
        for report in self._generated_reports:
            path = self.export_report(report, format)
            paths.append(path)
        return paths

    # ==========================================================================
    # Helpers
    # ==========================================================================

    def _create_report(
        self,
        report_type: ReportType,
        period: ReportPeriod,
        start_date: datetime,
        end_date: datetime,
        title: str,
    ) -> ComplianceReport:
        """Create a new report."""
        self._report_counter += 1
        report_id = f"RPT-{datetime.utcnow().strftime('%Y%m%d')}-{self._report_counter:05d}"

        return ComplianceReport(
            report_id=report_id,
            report_type=report_type,
            period=period,
            generated_at=datetime.utcnow(),
            start_date=start_date,
            end_date=end_date,
            title=title,
        )

    def _get_period_start(self, end_date: datetime, period: ReportPeriod) -> datetime:
        """Calculate period start date."""
        if period == ReportPeriod.DAILY:
            return end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == ReportPeriod.WEEKLY:
            return end_date - timedelta(days=7)
        elif period == ReportPeriod.MONTHLY:
            return end_date - timedelta(days=30)
        elif period == ReportPeriod.QUARTERLY:
            return end_date - timedelta(days=90)
        elif period == ReportPeriod.ANNUAL:
            return end_date - timedelta(days=365)
        return end_date

    def get_reports(
        self,
        report_type: ReportType | None = None,
    ) -> list[ComplianceReport]:
        """Get generated reports."""
        if report_type:
            return [r for r in self._generated_reports if r.report_type == report_type]
        return self._generated_reports.copy()


def create_compliance_reporter(
    output_dir: str | Path | None = None,
    default_format: ReportFormat = ReportFormat.JSON,
) -> ComplianceReporter:
    """
    Factory function to create a compliance reporter.

    Args:
        output_dir: Output directory for reports
        default_format: Default export format

    Returns:
        Configured ComplianceReporter
    """
    config = ReporterConfig(
        output_dir=Path(output_dir) if output_dir else Path("compliance_reports"),
        default_format=default_format,
    )
    return ComplianceReporter(config)
