#!/usr/bin/env python3
"""
Health Check HTTP Server for Overnight Sessions

Provides /health and /status endpoints for monitoring.
Part of OVERNIGHT-002 refactoring.

Usage:
    python3 scripts/health_check.py [port]

Endpoints:
    GET /health - Simple health check (200 if running)
    GET /status - Detailed session status with progress info
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from utils.overnight_state import OvernightStateManager
from utils.progress_parser import ProgressParser


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoints."""

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/health":
            self.send_health()
        elif self.path == "/status":
            self.send_status()
        else:
            self.send_error(404, "Not Found")

    def send_health(self) -> None:
        """Simple health check - returns 200 if server is running."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.wfile.write(json.dumps(response).encode())

    def send_status(self) -> None:
        """Detailed status with session info."""
        try:
            # Load session state
            state_mgr = OvernightStateManager()
            state = state_mgr.load()

            # Parse progress file
            progress_file = _project_root / "claude-progress.txt"
            parser = ProgressParser(progress_file)
            progress = parser.parse()

            # Build status response
            status = {
                "status": "running",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session": {
                    "id": state.session_id,
                    "goal": state.goal,
                    "started_at": state.started_at,
                    "restart_count": state.restart_count,
                    "continuation_count": state.continuation_count,
                },
                "progress": {
                    "total_tasks": progress.total_count,
                    "completed": progress.completed_count,
                    "pending": len(progress.pending_tasks),
                    "completion_pct": round(progress.completion_pct, 1),
                    "p0_pending": len(progress.get_pending_by_priority("P0")),
                    "p1_pending": len(progress.get_pending_by_priority("P1")),
                    "p2_pending": len(progress.get_pending_by_priority("P2")),
                },
                "ric": {
                    "active": state.ric_active,
                    "iteration": state.ric_iteration,
                    "phase": state.ric_phase,
                    "can_exit": state.ric_can_exit,
                },
                "hook_activity": {
                    "events": state.hook_events,
                    "errors": state.hook_errors,
                    "last_activity": state.last_hook_activity,
                },
            }

            # Add next task if available
            next_task = progress.get_next_task()
            if next_task:
                status["next_task"] = {
                    "description": next_task.description,
                    "priority": next_task.priority,
                    "category": next_task.category,
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status, indent=2).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ).encode()
            )

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default logging to stderr."""
        pass


def main() -> None:
    """Run the health check server."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"Health check server running on port {port}")
    print(f"  GET /health - Simple health check")
    print(f"  GET /status - Detailed session status")
    print(f"\nPress Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
