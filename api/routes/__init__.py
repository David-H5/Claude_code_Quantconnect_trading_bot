"""
API Route Handlers

Modular route handlers for the REST API:
- orders: Order submission and management
- positions: Position and P&L queries
- templates: Recurring order templates
- health: Health checks and system status
- decision_audit: Agent decision audit trail

UPGRADE-008: REST API Server (December 2025)
"""

from . import decision_audit, health, orders, positions, templates


__all__ = [
    "decision_audit",
    "health",
    "orders",
    "positions",
    "templates",
]
