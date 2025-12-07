"""
Custom Log Handlers for Trading Bot

Provides:
- Rotating file handler with gzip compression
- Object Store handler for persistence
- Async handler wrapper for non-blocking logging
- Memory buffer handler for batch processing
- LogPrefix constants for standardized debug messages

UPGRADE-009: Structured Logging (December 2025)
"""

from __future__ import annotations


# ============================================================================
# Log Prefix Constants (for QuantConnect Debug messages)
# ============================================================================


class LogPrefix:
    """
    Standardized log prefixes for consistent debug message formatting.

    Usage in QuantConnect algorithms:
        from utils.log_handlers import LogPrefix

        self.Debug(f"{LogPrefix.OK} Configuration loaded")
        self.Debug(f"{LogPrefix.WARN} Config file not found")
        self.Debug(f"{LogPrefix.ERR} Failed to initialize")

    These replace inconsistent emoji usage with standardized prefixes.
    """

    OK = "OK:"
    WARN = "WARN:"
    ERR = "ERR:"
    INFO = "INFO:"

    # Additional prefixes for common scenarios
    ALERT = "ALERT:"
    TRADE = "TRADE:"
    CIRCUIT = "CIRCUIT:"
    RESOURCE = "RESOURCE:"
    SENTIMENT = "SENTIMENT:"

import gzip
import json
import logging
import os
import queue
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


# ============================================================================
# Compressed Rotating File Handler
# ============================================================================


class CompressedRotatingFileHandler(RotatingFileHandler):
    """
    Rotating file handler with gzip compression for rotated files.

    Compresses old log files to save space while keeping recent logs
    accessible for debugging.

    Example:
        >>> handler = CompressedRotatingFileHandler(
        ...     "logs/trading.log",
        ...     maxBytes=50*1024*1024,  # 50MB
        ...     backupCount=10,
        ...     compress=True
        ... )
    """

    def __init__(
        self,
        filename: str,
        mode: str = "a",
        maxBytes: int = 50 * 1024 * 1024,  # 50MB default
        backupCount: int = 10,
        encoding: str | None = "utf-8",
        delay: bool = False,
        compress: bool = True,
    ):
        """Initialize handler.

        Args:
            filename: Log file path
            mode: File mode
            maxBytes: Max file size before rotation
            backupCount: Number of backup files to keep
            encoding: File encoding
            delay: Delay file opening until first write
            compress: Whether to compress rotated files
        """
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            filename,
            mode=mode,
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
        )
        self.compress = compress

    def doRollover(self):
        """Roll over and compress old file."""
        # Close current file
        if self.stream:
            self.stream.close()
            self.stream = None

        # Rotate files
        for i in range(self.backupCount - 1, 0, -1):
            sfn = self.rotation_filename(f"{self.baseFilename}.{i}")
            dfn = self.rotation_filename(f"{self.baseFilename}.{i + 1}")

            # Handle both compressed and uncompressed
            sfn_gz = f"{sfn}.gz"
            dfn_gz = f"{dfn}.gz"

            if os.path.exists(sfn_gz):
                if os.path.exists(dfn_gz):
                    os.remove(dfn_gz)
                os.rename(sfn_gz, dfn_gz)
            elif os.path.exists(sfn):
                if os.path.exists(dfn):
                    os.remove(dfn)
                os.rename(sfn, dfn)

        # Rename current to .1
        dfn = self.rotation_filename(f"{self.baseFilename}.1")
        if os.path.exists(dfn):
            os.remove(dfn)
        self.rotate(self.baseFilename, dfn)

        # Compress the new .1 file if enabled
        if self.compress and os.path.exists(dfn):
            self._compress_file(dfn)

        # Open new file
        if not self.delay:
            self.stream = self._open()

    def _compress_file(self, filepath: str) -> None:
        """Compress a file with gzip.

        Args:
            filepath: Path to file to compress
        """
        try:
            with open(filepath, "rb") as f_in, gzip.open(f"{filepath}.gz", "wb") as f_out:
                f_out.writelines(f_in)
            os.remove(filepath)
        except Exception as e:
            # Log to stderr if compression fails
            import sys

            print(f"Failed to compress {filepath}: {e}", file=sys.stderr)


# ============================================================================
# Object Store Handler
# ============================================================================


class ObjectStoreHandler(logging.Handler):
    """
    Handler that writes logs to QuantConnect Object Store.

    Buffers logs in memory and periodically flushes to Object Store
    for persistence across algorithm restarts.

    Example:
        >>> handler = ObjectStoreHandler(
        ...     object_store_manager=manager,
        ...     buffer_size=100,
        ...     flush_interval_seconds=60
        ... )
    """

    def __init__(
        self,
        object_store_manager: Any,
        buffer_size: int = 100,
        flush_interval_seconds: int = 60,
        category: str = "monitoring_data",
    ):
        """Initialize handler.

        Args:
            object_store_manager: ObjectStoreManager instance
            buffer_size: Number of records before auto-flush
            flush_interval_seconds: Time between auto-flushes
            category: Storage category for logs
        """
        super().__init__()
        self.manager = object_store_manager
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval_seconds
        self.category = category
        self._buffer: list[str] = []
        self._lock = threading.Lock()
        self._last_flush = datetime.now(timezone.utc)

    def emit(self, record: logging.LogRecord) -> None:
        """Buffer log record for batch write.

        Args:
            record: Log record to buffer
        """
        try:
            msg = self.format(record)
            with self._lock:
                self._buffer.append(msg)

                # Check if we should flush
                if len(self._buffer) >= self.buffer_size or self._should_time_flush():
                    self._do_flush()

        except Exception:
            self.handleError(record)

    def _should_time_flush(self) -> bool:
        """Check if enough time has passed for a flush."""
        elapsed = (datetime.now(timezone.utc) - self._last_flush).total_seconds()
        return elapsed >= self.flush_interval

    def _do_flush(self) -> None:
        """Write buffered logs to Object Store (internal)."""
        if not self._buffer:
            return

        now = datetime.now(timezone.utc)
        key = f"logs/{now.strftime('%Y/%m/%d/%H%M%S')}_{len(self._buffer)}.jsonl"

        try:
            content = "\n".join(self._buffer)
            self.manager.save(
                key=key,
                data=content,
                category=self.category,
            )
            self._buffer.clear()
            self._last_flush = now
        except Exception as e:
            import sys

            print(f"Failed to write logs to Object Store: {e}", file=sys.stderr)

    def flush(self) -> None:
        """Write buffered logs to Object Store."""
        with self._lock:
            self._do_flush()

    def close(self) -> None:
        """Flush remaining logs and close handler."""
        self.flush()
        super().close()


# ============================================================================
# Async Handler Wrapper
# ============================================================================


class AsyncHandler(logging.Handler):
    """
    Async wrapper for any logging handler.

    Processes log records in a background thread to avoid blocking
    the main trading loop.

    Example:
        >>> file_handler = CompressedRotatingFileHandler("logs/trading.log")
        >>> async_handler = AsyncHandler(file_handler)
    """

    def __init__(
        self,
        handler: logging.Handler,
        queue_size: int = 10000,
    ):
        """Initialize async handler.

        Args:
            handler: Underlying handler to wrap
            queue_size: Maximum queue size (0 = unlimited)
        """
        super().__init__()
        self.handler = handler
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._process_queue, daemon=True)
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        """Queue log record for async processing.

        Args:
            record: Log record to queue
        """
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            # Drop oldest if queue is full
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(record)
            except (queue.Empty, queue.Full):
                pass

    def _process_queue(self) -> None:
        """Background thread to process queued records."""
        while not self._shutdown.is_set():
            try:
                record = self._queue.get(timeout=1.0)
                self.handler.emit(record)
            except queue.Empty:
                continue
            except Exception:
                pass

    def flush(self) -> None:
        """Flush the underlying handler."""
        # Process remaining items
        while not self._queue.empty():
            try:
                record = self._queue.get_nowait()
                self.handler.emit(record)
            except queue.Empty:
                break
        self.handler.flush()

    def close(self) -> None:
        """Shutdown and close handler."""
        self._shutdown.set()
        self._thread.join(timeout=5.0)
        self.flush()
        self.handler.close()
        super().close()


# ============================================================================
# Callback Handler
# ============================================================================


class CallbackHandler(logging.Handler):
    """
    Handler that calls a callback function for each log record.

    Useful for real-time log streaming to WebSocket clients or
    custom processing.

    Example:
        >>> def on_log(event: dict):
        ...     websocket.broadcast(event)
        >>> handler = CallbackHandler(on_log)
    """

    def __init__(
        self,
        callback: Callable[[dict], None],
        include_levels: list[str] | None = None,
    ):
        """Initialize callback handler.

        Args:
            callback: Function to call with log event dict
            include_levels: Only process these log levels (None = all)
        """
        super().__init__()
        self.callback = callback
        self.include_levels = include_levels

    def emit(self, record: logging.LogRecord) -> None:
        """Process log record through callback.

        Args:
            record: Log record to process
        """
        try:
            # Check level filter
            if self.include_levels:
                if record.levelname.lower() not in self.include_levels:
                    return

            # Try to parse JSON message (structured log)
            try:
                event = json.loads(record.getMessage())
            except json.JSONDecodeError:
                event = {
                    "level": record.levelname.lower(),
                    "message": record.getMessage(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            self.callback(event)

        except Exception:
            self.handleError(record)


# ============================================================================
# Factory Functions
# ============================================================================


def create_rotating_file_handler(
    filename: str,
    max_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 10,
    compress: bool = True,
) -> logging.Handler:
    """Create a rotating file handler with compression.

    Args:
        filename: Log file path
        max_bytes: Max file size before rotation (default 50MB)
        backup_count: Number of backup files to keep
        compress: Whether to compress rotated files

    Returns:
        Configured CompressedRotatingFileHandler
    """
    handler = CompressedRotatingFileHandler(
        filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        compress=compress,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


def create_async_file_handler(
    filename: str,
    max_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 10,
    compress: bool = True,
    queue_size: int = 10000,
) -> logging.Handler:
    """Create an async file handler.

    Args:
        filename: Log file path
        max_bytes: Max file size before rotation
        backup_count: Number of backup files
        compress: Whether to compress rotated files
        queue_size: Async queue size

    Returns:
        AsyncHandler wrapping a CompressedRotatingFileHandler
    """
    file_handler = create_rotating_file_handler(filename, max_bytes, backup_count, compress)
    return AsyncHandler(file_handler, queue_size=queue_size)


def create_object_store_handler(
    object_store_manager: Any,
    buffer_size: int = 100,
    flush_interval: int = 60,
) -> logging.Handler:
    """Create an Object Store handler.

    Args:
        object_store_manager: ObjectStoreManager instance
        buffer_size: Records before auto-flush
        flush_interval: Seconds between auto-flushes

    Returns:
        Configured ObjectStoreHandler
    """
    handler = ObjectStoreHandler(
        object_store_manager,
        buffer_size=buffer_size,
        flush_interval_seconds=flush_interval,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


def create_callback_handler(
    callback: Callable[[dict], None],
    include_levels: list[str] | None = None,
) -> logging.Handler:
    """Create a callback handler.

    Args:
        callback: Function to call with log events
        include_levels: Only process these levels

    Returns:
        Configured CallbackHandler
    """
    return CallbackHandler(callback, include_levels=include_levels)
