"""
Base MCP Server Implementation

Provides a base class for building Model Context Protocol (MCP) servers
that can be extended for specific use cases like market data, broker
operations, portfolio management, and backtesting.

UPGRADE-015 Phase 1: MCP Server Foundation

Usage:
    class MyServer(BaseMCPServer):
        def __init__(self):
            super().__init__(name="my-server", version="1.0.0")
            self.register_tool(my_tool_func, schema)

        def initialize(self) -> None:
            # Custom initialization
            pass
"""

import asyncio
import json
import logging
import sys
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class ServerState(Enum):
    """Server lifecycle states."""

    STOPPED = "stopped"
    INITIALIZING = "initializing"
    RUNNING = "running"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class ToolCategory(Enum):
    """Categories for organizing tools."""

    MARKET_DATA = "market_data"
    PORTFOLIO = "portfolio"
    BROKER = "broker"
    BACKTEST = "backtest"
    RISK = "risk"
    ANALYSIS = "analysis"
    EXECUTION = "execution"
    SENTIMENT = "sentiment"
    SYSTEM = "system"


@dataclass
class ToolSchema:
    """
    Schema definition for an MCP tool.

    Attributes:
        name: Unique tool identifier
        description: Human-readable description
        category: Tool category for organization
        parameters: JSON Schema for input parameters
        returns: Description of return value
        requires_auth: Whether tool requires authentication
        is_dangerous: Whether tool can modify state/execute trades
    """

    name: str
    description: str
    category: ToolCategory
    parameters: dict[str, Any]
    returns: str
    requires_auth: bool = False
    is_dangerous: bool = False

    def to_mcp_schema(self) -> dict[str, Any]:
        """Convert to MCP-compatible tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.parameters.get("properties", {}),
                "required": self.parameters.get("required", []),
            },
        }


@dataclass
class ToolResult:
    """
    Result from a tool invocation.

    Attributes:
        success: Whether the tool call succeeded
        data: Result data if successful
        error: Error message if failed
        error_code: Machine-readable error code
        execution_time_ms: Time taken in milliseconds
        metadata: Additional metadata about the call
    """

    success: bool
    data: Any | None = None
    error: str | None = None
    error_code: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "error_code": self.error_code,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }

    def to_mcp_content(self) -> list[dict[str, Any]]:
        """Convert to MCP content format."""
        if self.success:
            return [
                {
                    "type": "text",
                    "text": json.dumps(self.data, indent=2, default=str),
                }
            ]
        else:
            return [
                {
                    "type": "text",
                    "text": f"Error: {self.error}",
                }
            ]


@dataclass
class ServerConfig:
    """
    Configuration for an MCP server.

    Attributes:
        name: Server name
        version: Server version
        description: Server description
        max_concurrent_calls: Maximum concurrent tool calls
        timeout_seconds: Default timeout for tool calls
        enable_logging: Whether to log tool calls
        mock_mode: Whether to use mock data
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    max_concurrent_calls: int = 10
    timeout_seconds: float = 30.0
    enable_logging: bool = True
    mock_mode: bool = False


class BaseMCPServer(ABC):
    """
    Abstract base class for MCP servers.

    Provides common functionality for tool registration, invocation,
    health checks, and lifecycle management.

    Subclasses must implement:
        - initialize(): Setup server-specific resources
        - cleanup(): Clean up resources on shutdown
    """

    def __init__(self, config: ServerConfig):
        """
        Initialize the MCP server.

        Args:
            config: Server configuration
        """
        self.config = config
        self.state = ServerState.STOPPED
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, ToolSchema] = {}
        self._call_count: int = 0
        self._error_count: int = 0
        self._start_time: datetime | None = None
        self._semaphore: asyncio.Semaphore | None = None

        logger.info(f"Created MCP server: {config.name} v{config.version}")

    @property
    def name(self) -> str:
        """Server name."""
        return self.config.name

    @property
    def version(self) -> str:
        """Server version."""
        return self.config.version

    @property
    def is_running(self) -> bool:
        """Whether server is running."""
        return self.state == ServerState.RUNNING

    def register_tool(
        self,
        func: Callable,
        schema: ToolSchema,
    ) -> None:
        """
        Register a tool with the server.

        Args:
            func: Function to call when tool is invoked
            schema: Tool schema definition
        """
        if schema.name in self._tools:
            logger.warning(f"Overwriting existing tool: {schema.name}")

        self._tools[schema.name] = func
        self._schemas[schema.name] = schema

        logger.debug(f"Registered tool: {schema.name} ({schema.category.value})")

    def get_tool_schemas(self) -> list[ToolSchema]:
        """Get all registered tool schemas."""
        return list(self._schemas.values())

    def get_tools_by_category(self, category: ToolCategory) -> list[ToolSchema]:
        """Get tools filtered by category."""
        return [s for s in self._schemas.values() if s.category == category]

    def list_tools(self) -> list[dict[str, Any]]:
        """List all tools in MCP format."""
        return [schema.to_mcp_schema() for schema in self._schemas.values()]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolResult:
        """
        Call a registered tool.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            ToolResult with success/failure and data
        """
        start_time = time.time()
        arguments = arguments or {}

        # Check if tool exists
        if name not in self._tools:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {name}",
                error_code="TOOL_NOT_FOUND",
            )

        # Check server state
        if not self.is_running:
            return ToolResult(
                success=False,
                error=f"Server not running (state: {self.state.value})",
                error_code="SERVER_NOT_RUNNING",
            )

        # Rate limiting via semaphore
        if self._semaphore:
            await self._semaphore.acquire()

        try:
            self._call_count += 1
            func = self._tools[name]
            schema = self._schemas[name]

            if self.config.enable_logging:
                logger.info(f"Calling tool: {name} with args: {list(arguments.keys())}")

            # Check for dangerous operations
            if schema.is_dangerous:
                logger.warning(f"Executing dangerous tool: {name}")

            # Execute the tool
            if asyncio.iscoroutinefunction(func):
                result_data = await asyncio.wait_for(
                    func(**arguments),
                    timeout=self.config.timeout_seconds,
                )
            else:
                result_data = func(**arguments)

            execution_time = (time.time() - start_time) * 1000

            return ToolResult(
                success=True,
                data=result_data,
                execution_time_ms=execution_time,
                metadata={
                    "tool": name,
                    "category": schema.category.value,
                },
            )

        except asyncio.TimeoutError:
            self._error_count += 1
            return ToolResult(
                success=False,
                error=f"Tool timed out after {self.config.timeout_seconds}s",
                error_code="TIMEOUT",
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            self._error_count += 1
            logger.error(f"Tool {name} failed: {e}")
            logger.debug(traceback.format_exc())
            return ToolResult(
                success=False,
                error=str(e),
                error_code="EXECUTION_ERROR",
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        finally:
            if self._semaphore:
                self._semaphore.release()

    def call_tool_sync(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolResult:
        """
        Synchronous wrapper for call_tool.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            ToolResult with success/failure and data
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.call_tool(name, arguments))

    async def start(self) -> None:
        """Start the server."""
        if self.state != ServerState.STOPPED:
            logger.warning(f"Cannot start server in state: {self.state.value}")
            return

        self.state = ServerState.INITIALIZING
        logger.info(f"Starting server: {self.name}")

        try:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent_calls)
            self._start_time = datetime.utcnow()

            await self.initialize()

            self.state = ServerState.RUNNING
            logger.info(f"Server started: {self.name}")

        except Exception as e:
            self.state = ServerState.ERROR
            logger.error(f"Failed to start server: {e}")
            raise

    async def stop(self) -> None:
        """Stop the server."""
        if self.state == ServerState.STOPPED:
            return

        self.state = ServerState.SHUTTING_DOWN
        logger.info(f"Stopping server: {self.name}")

        try:
            await self.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            self.state = ServerState.STOPPED
            logger.info(f"Server stopped: {self.name}")

    def health_check(self) -> dict[str, Any]:
        """
        Get server health status.

        Returns:
            Dictionary with health information
        """
        uptime_seconds = None
        if self._start_time:
            uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()

        return {
            "server": self.name,
            "version": self.version,
            "state": self.state.value,
            "is_healthy": self.state == ServerState.RUNNING,
            "uptime_seconds": uptime_seconds,
            "tools_registered": len(self._tools),
            "total_calls": self._call_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._call_count),
            "timestamp": datetime.utcnow().isoformat(),
        }

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize server-specific resources.

        Called during server startup. Subclasses should override
        to set up connections, load data, etc.
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up server-specific resources.

        Called during server shutdown. Subclasses should override
        to close connections, save state, etc.
        """
        pass

    async def run_stdio(self) -> None:
        """
        Run server in stdio mode for MCP integration.

        Reads JSON-RPC messages from stdin and writes responses to stdout.
        """
        await self.start()

        try:
            while True:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)

                if not line:
                    break

                try:
                    message = json.loads(line)
                    response = await self._handle_message(message)
                    if response:
                        print(json.dumps(response), flush=True)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON: {line}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")

        finally:
            await self.stop()

    async def _handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Handle an incoming MCP message."""
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": True},
                    },
                    "serverInfo": {
                        "name": self.name,
                        "version": self.version,
                    },
                },
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": self.list_tools()},
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            result = await self.call_tool(tool_name, arguments)

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": result.to_mcp_content(),
                    "isError": not result.success,
                },
            }

        elif method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {},
            }

        return None


def create_base_server(
    name: str,
    version: str = "1.0.0",
    **kwargs,
) -> ServerConfig:
    """
    Create a server configuration.

    Args:
        name: Server name
        version: Server version
        **kwargs: Additional config options

    Returns:
        ServerConfig instance
    """
    return ServerConfig(
        name=name,
        version=version,
        **kwargs,
    )
