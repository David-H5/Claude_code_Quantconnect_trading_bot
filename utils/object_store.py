#!/usr/bin/env python3
"""
Object Store Manager for QuantConnect

Manages persistent storage in QuantConnect's Object Store (5GB tier).
Handles compression, expiration, size limits, and data persistence.

Author: QuantConnect Trading Bot
Date: 2025-11-30
"""

import base64
import gzip
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class StorageCategory(Enum):
    """Storage categories for quota management."""

    TRADING_STATE = "trading_state"
    OPTIONS_GREEKS = "options_greeks"
    LLM_CACHE = "llm_cache"
    BACKTEST_RESULTS = "backtest_results"
    ML_MODELS = "ml_models"
    MONITORING_DATA = "monitoring_data"
    RESEARCH_DATA = "research_data"
    SENTIMENT_DATA = "sentiment_data"  # UPGRADE-014


@dataclass
class StoredObject:
    """Metadata for a stored object."""

    key: str
    category: StorageCategory
    size_bytes: int
    compressed: bool
    created_at: datetime
    expires_at: datetime | None = None
    metadata: dict = field(default_factory=dict)


class ObjectStoreManager:
    """
    Manager for QuantConnect Object Store with 5GB tier.

    Features:
    - Automatic compression for large objects
    - Expiration and automatic cleanup
    - Size limit enforcement (<50MB per file)
    - Category-based storage quotas
    - Metadata tracking
    - Safe serialization/deserialization

    Example usage:
        manager = ObjectStoreManager(
            algorithm=self,
            max_size_mb=45,
            compression_threshold_kb=100,
        )

        # Save data with expiration
        manager.save(
            key="greeks_snapshot",
            data=greeks_data,
            category=StorageCategory.OPTIONS_GREEKS,
            expire_days=30,
        )

        # Load data
        greeks = manager.load("greeks_snapshot")

        # Cleanup expired entries
        manager.cleanup_expired()
    """

    def __init__(
        self,
        algorithm: object,
        max_size_mb: float = 45,
        compression_threshold_kb: float = 100,
        auto_compress: bool = True,
    ):
        """
        Initialize Object Store manager.

        Args:
            algorithm: QCAlgorithm instance with ObjectStore
            max_size_mb: Maximum file size in MB (must be <50MB)
            compression_threshold_kb: Compress objects larger than this
            auto_compress: Automatically compress large objects
        """
        self.algorithm = algorithm
        self.store = algorithm.ObjectStore
        self.max_size_mb = min(max_size_mb, 49)  # Safety margin
        self.compression_threshold_kb = compression_threshold_kb
        self.auto_compress = auto_compress

        # Track stored objects (persist in ObjectStore)
        self._object_registry_key = "__object_registry__"
        self._registry: dict[str, StoredObject] = {}
        self._load_registry()

        # Category quotas (GB)
        self._category_quotas: dict[StorageCategory, float] = {}

    def set_category_quotas(self, quotas: dict[StorageCategory, float]) -> None:
        """
        Set storage quotas per category in GB.

        Args:
            quotas: Dictionary mapping categories to GB limits
        """
        self._category_quotas = quotas

    def save(
        self,
        key: str,
        data: Any,
        category: StorageCategory,
        expire_days: int | None = None,
        force_compress: bool = False,
        metadata: dict | None = None,
    ) -> bool:
        """
        Save data to Object Store.

        Args:
            key: Storage key (unique identifier)
            data: Data to store (must be JSON serializable)
            category: Storage category for quota management
            expire_days: Days until expiration (None = never)
            force_compress: Force compression regardless of size
            metadata: Additional metadata to store

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Serialize to JSON
            json_data = json.dumps(data)
            size_bytes = len(json_data.encode("utf-8"))
            size_mb = size_bytes / (1024 * 1024)
            size_kb = size_bytes / 1024

            # Check size limit before compression
            if size_mb > self.max_size_mb and not self.auto_compress:
                logger.error(f"Object too large: {size_mb:.2f}MB > {self.max_size_mb}MB")
                return False

            # Compression decision
            should_compress = force_compress or (self.auto_compress and size_kb > self.compression_threshold_kb)

            if should_compress:
                # Compress and encode
                compressed = gzip.compress(json_data.encode("utf-8"))
                encoded = base64.b64encode(compressed).decode("utf-8")
                final_data = encoded
                final_size = len(compressed)
                compressed_flag = True

                logger.debug(
                    f"Compressed {key}: {size_kb:.1f}KB → {final_size/1024:.1f}KB "
                    f"({final_size/size_bytes*100:.1f}%)"
                )
            else:
                final_data = json_data
                final_size = size_bytes
                compressed_flag = False

            # Final size check
            final_size_mb = final_size / (1024 * 1024)
            if final_size_mb > self.max_size_mb:
                logger.error(f"Compressed object still too large: {final_size_mb:.2f}MB")
                return False

            # Check category quota
            if not self._check_category_quota(category, final_size):
                logger.warning(f"Category {category.value} quota exceeded, cleaning up old files")
                self._cleanup_category(category)

            # Create object metadata
            expires_at = None
            if expire_days:
                expires_at = datetime.now() + timedelta(days=expire_days)

            obj = StoredObject(
                key=key,
                category=category,
                size_bytes=final_size,
                compressed=compressed_flag,
                created_at=datetime.now(),
                expires_at=expires_at,
                metadata=metadata or {},
            )

            # Wrap with metadata
            wrapped_data = {
                "data": final_data,
                "compressed": compressed_flag,
                "created_at": obj.created_at.isoformat(),
                "expires_at": obj.expires_at.isoformat() if obj.expires_at else None,
                "category": category.value,
                "size_bytes": final_size,
                "metadata": metadata or {},
            }

            # Save to Object Store
            self.store.Save(key, json.dumps(wrapped_data))

            # Update registry
            self._registry[key] = obj
            self._save_registry()

            logger.info(f"Saved {key} ({final_size_mb:.2f}MB, compressed={compressed_flag})")
            return True

        except Exception as e:
            logger.error(f"Failed to save {key}: {e}")
            return False

    def load(self, key: str) -> Any | None:
        """
        Load data from Object Store.

        Args:
            key: Storage key

        Returns:
            Stored data, or None if not found or expired
        """
        try:
            # Check if key exists
            if not self.store.ContainsKey(key):
                return None

            # Load wrapped data
            wrapped_json = self.store.Read(key)
            wrapped_data = json.loads(wrapped_json)

            # Check expiration
            expires_at_str = wrapped_data.get("expires_at")
            if expires_at_str:
                expires_at = datetime.fromisoformat(expires_at_str)
                if datetime.now() > expires_at:
                    logger.info(f"Object {key} expired, deleting")
                    self.delete(key)
                    return None

            # Extract data
            data_str = wrapped_data["data"]
            compressed = wrapped_data.get("compressed", False)

            if compressed:
                # Decompress
                compressed_bytes = base64.b64decode(data_str.encode("utf-8"))
                json_data = gzip.decompress(compressed_bytes).decode("utf-8")
            else:
                json_data = data_str

            # Deserialize
            data = json.loads(json_data)

            return data

        except Exception as e:
            logger.error(f"Failed to load {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """
        Delete object from Object Store.

        Args:
            key: Storage key

        Returns:
            True if deleted successfully
        """
        try:
            if self.store.ContainsKey(key):
                self.store.Delete(key)

            # Remove from registry
            if key in self._registry:
                del self._registry[key]
                self._save_registry()

            logger.info(f"Deleted {key}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete {key}: {e}")
            return False

    def list_keys(self, category: StorageCategory | None = None) -> list[str]:
        """
        List all stored keys, optionally filtered by category.

        Args:
            category: Filter by category (None = all)

        Returns:
            List of keys
        """
        if category:
            return [key for key, obj in self._registry.items() if obj.category == category]
        return list(self._registry.keys())

    def get_metadata(self, key: str) -> StoredObject | None:
        """
        Get metadata for a stored object.

        Args:
            key: Storage key

        Returns:
            StoredObject metadata or None
        """
        return self._registry.get(key)

    def get_storage_stats(self) -> dict:
        """
        Get storage usage statistics.

        Returns:
            Dictionary with storage stats by category
        """
        stats = {
            "total_objects": len(self._registry),
            "total_size_mb": 0,
            "by_category": {},
        }

        # Calculate per-category stats
        for category in StorageCategory:
            category_objects = [obj for obj in self._registry.values() if obj.category == category]
            total_size = sum(obj.size_bytes for obj in category_objects)

            stats["by_category"][category.value] = {
                "count": len(category_objects),
                "size_mb": total_size / (1024 * 1024),
                "quota_mb": self._category_quotas.get(category, 0) * 1024,
            }

            stats["total_size_mb"] += total_size / (1024 * 1024)

        return stats

    def cleanup_expired(self) -> int:
        """
        Remove expired objects.

        Returns:
            Number of objects deleted
        """
        deleted_count = 0
        now = datetime.now()

        for key, obj in list(self._registry.items()):
            if obj.expires_at and now > obj.expires_at:
                if self.delete(key):
                    deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} expired objects")
        return deleted_count

    def _check_category_quota(self, category: StorageCategory, new_size: int) -> bool:
        """Check if adding new_size bytes would exceed category quota."""
        if category not in self._category_quotas:
            return True

        quota_bytes = self._category_quotas[category] * 1024 * 1024 * 1024
        current_bytes = sum(obj.size_bytes for obj in self._registry.values() if obj.category == category)

        return (current_bytes + new_size) <= quota_bytes

    def _cleanup_category(self, category: StorageCategory) -> None:
        """Remove oldest objects in category to free space."""
        category_objects = [(key, obj) for key, obj in self._registry.items() if obj.category == category]

        # Sort by creation date (oldest first)
        category_objects.sort(key=lambda x: x[1].created_at)

        # Delete oldest 20%
        delete_count = max(1, len(category_objects) // 5)
        for key, _ in category_objects[:delete_count]:
            self.delete(key)

        logger.info(f"Cleaned up {delete_count} old {category.value} objects")

    def _load_registry(self) -> None:
        """Load object registry from Object Store."""
        try:
            if self.store.ContainsKey(self._object_registry_key):
                registry_json = self.store.Read(self._object_registry_key)
                registry_data = json.loads(registry_json)

                for key, obj_data in registry_data.items():
                    self._registry[key] = StoredObject(
                        key=key,
                        category=StorageCategory(obj_data["category"]),
                        size_bytes=obj_data["size_bytes"],
                        compressed=obj_data["compressed"],
                        created_at=datetime.fromisoformat(obj_data["created_at"]),
                        expires_at=datetime.fromisoformat(obj_data["expires_at"])
                        if obj_data.get("expires_at")
                        else None,
                        metadata=obj_data.get("metadata", {}),
                    )

                logger.info(f"Loaded {len(self._registry)} objects from registry")
        except Exception as e:
            logger.warning(f"Failed to load registry: {e}, starting fresh")

    def _save_registry(self) -> None:
        """Save object registry to Object Store."""
        try:
            registry_data = {}
            for key, obj in self._registry.items():
                registry_data[key] = {
                    "category": obj.category.value,
                    "size_bytes": obj.size_bytes,
                    "compressed": obj.compressed,
                    "created_at": obj.created_at.isoformat(),
                    "expires_at": obj.expires_at.isoformat() if obj.expires_at else None,
                    "metadata": obj.metadata,
                }

            self.store.Save(self._object_registry_key, json.dumps(registry_data))
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")


def create_object_store_manager(algorithm: object, config: dict | None = None) -> ObjectStoreManager:
    """
    Create a configured Object Store manager.

    Args:
        algorithm: QCAlgorithm instance
        config: Configuration dictionary

    Returns:
        Configured ObjectStoreManager
    """
    if config is None:
        config = {}

    manager = ObjectStoreManager(
        algorithm=algorithm,
        max_size_mb=config.get("max_file_size_mb", 45),
        compression_threshold_kb=config.get("compression_threshold_kb", 100),
        auto_compress=config.get("compression_enabled", True),
    )

    # Set category quotas
    allocations = config.get("storage_allocation", {})
    quotas = {
        StorageCategory.TRADING_STATE: allocations.get("trading_state", 0.5),
        StorageCategory.OPTIONS_GREEKS: allocations.get("options_greeks", 1.5),
        StorageCategory.LLM_CACHE: allocations.get("llm_cache", 0.3),
        StorageCategory.BACKTEST_RESULTS: allocations.get("backtest_results", 0.5),
        StorageCategory.ML_MODELS: allocations.get("ml_models", 1.5),
        StorageCategory.MONITORING_DATA: allocations.get("monitoring_data", 0.3),
        StorageCategory.RESEARCH_DATA: allocations.get("research_data", 0.4),
        StorageCategory.SENTIMENT_DATA: allocations.get("sentiment_data", 0.2),  # UPGRADE-014
    }
    manager.set_category_quotas(quotas)

    return manager


class SentimentPersistence:
    """
    Helper class for persisting sentiment data to Object Store.

    UPGRADE-014: LLM Sentiment Integration persistence layer.

    Stores:
    - Sentiment history per symbol
    - Ensemble prediction records for feedback loop
    - Provider performance metrics
    - Filter decisions for audit trail

    Example usage:
        persistence = SentimentPersistence(object_store_manager)

        # Save sentiment history
        persistence.save_sentiment_history("AAPL", [0.5, 0.3, -0.2])

        # Load sentiment history
        history = persistence.load_sentiment_history("AAPL")

        # Save ensemble predictions for feedback
        persistence.save_prediction(prediction_id, prediction_data)
    """

    def __init__(self, object_store: ObjectStoreManager):
        """
        Initialize sentiment persistence.

        Args:
            object_store: ObjectStoreManager instance
        """
        self.store = object_store
        self._history_key_prefix = "sentiment_history_"
        self._prediction_key_prefix = "sentiment_prediction_"
        self._filter_key_prefix = "sentiment_filter_"
        self._provider_key = "sentiment_provider_performance"

    def save_sentiment_history(
        self,
        symbol: str,
        scores: list[float],
        max_history: int = 100,
        expire_days: int = 30,
    ) -> bool:
        """
        Save sentiment history for a symbol.

        Args:
            symbol: Ticker symbol
            scores: List of sentiment scores
            max_history: Maximum history items to keep
            expire_days: Days until expiration

        Returns:
            True if saved successfully
        """
        # Trim to max history
        trimmed_scores = scores[-max_history:] if len(scores) > max_history else scores

        return self.store.save(
            key=f"{self._history_key_prefix}{symbol}",
            data={
                "symbol": symbol,
                "scores": trimmed_scores,
                "updated_at": datetime.now().isoformat(),
                "count": len(trimmed_scores),
            },
            category=StorageCategory.SENTIMENT_DATA,
            expire_days=expire_days,
            metadata={"symbol": symbol, "type": "history"},
        )

    def load_sentiment_history(self, symbol: str) -> list[float] | None:
        """
        Load sentiment history for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            List of sentiment scores, or None if not found
        """
        data = self.store.load(f"{self._history_key_prefix}{symbol}")
        if data:
            return data.get("scores", [])
        return None

    def save_prediction(
        self,
        prediction_id: str,
        prediction_data: dict,
        expire_days: int = 7,
    ) -> bool:
        """
        Save ensemble prediction for feedback loop.

        Args:
            prediction_id: Unique prediction ID
            prediction_data: Prediction details
            expire_days: Days until expiration

        Returns:
            True if saved successfully
        """
        return self.store.save(
            key=f"{self._prediction_key_prefix}{prediction_id}",
            data={
                "id": prediction_id,
                **prediction_data,
                "stored_at": datetime.now().isoformat(),
            },
            category=StorageCategory.SENTIMENT_DATA,
            expire_days=expire_days,
            metadata={"type": "prediction"},
        )

    def load_prediction(self, prediction_id: str) -> dict | None:
        """
        Load a prediction for feedback.

        Args:
            prediction_id: Prediction ID

        Returns:
            Prediction data, or None if not found
        """
        return self.store.load(f"{self._prediction_key_prefix}{prediction_id}")

    def save_provider_performance(
        self,
        performance_data: dict[str, dict],
        expire_days: int = 90,
    ) -> bool:
        """
        Save provider performance metrics.

        Args:
            performance_data: Provider name -> performance metrics
            expire_days: Days until expiration

        Returns:
            True if saved successfully
        """
        return self.store.save(
            key=self._provider_key,
            data={
                "providers": performance_data,
                "updated_at": datetime.now().isoformat(),
            },
            category=StorageCategory.SENTIMENT_DATA,
            expire_days=expire_days,
            metadata={"type": "provider_performance"},
        )

    def load_provider_performance(self) -> dict[str, dict] | None:
        """
        Load provider performance metrics.

        Returns:
            Provider performance data, or None if not found
        """
        data = self.store.load(self._provider_key)
        if data:
            return data.get("providers", {})
        return None

    def save_filter_decision(
        self,
        symbol: str,
        decision_data: dict,
        expire_days: int = 30,
    ) -> bool:
        """
        Save filter decision for audit trail.

        Args:
            symbol: Ticker symbol
            decision_data: Filter decision details
            expire_days: Days until expiration

        Returns:
            True if saved successfully
        """
        # Create unique key with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        key = f"{self._filter_key_prefix}{symbol}_{timestamp}"

        return self.store.save(
            key=key,
            data={
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                **decision_data,
            },
            category=StorageCategory.SENTIMENT_DATA,
            expire_days=expire_days,
            metadata={"symbol": symbol, "type": "filter_decision"},
        )

    def get_filter_decisions(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Get recent filter decisions.

        Args:
            symbol: Filter by symbol (None = all)
            limit: Maximum number of decisions to return

        Returns:
            List of filter decisions (newest first)
        """
        keys = self.store.list_keys(StorageCategory.SENTIMENT_DATA)
        filter_keys = [k for k in keys if k.startswith(self._filter_key_prefix)]

        if symbol:
            filter_keys = [k for k in filter_keys if symbol in k]

        # Sort by key (which contains timestamp)
        filter_keys.sort(reverse=True)

        decisions = []
        for key in filter_keys[:limit]:
            data = self.store.load(key)
            if data:
                decisions.append(data)

        return decisions

    def get_all_sentiment_history(self) -> dict[str, list[float]]:
        """
        Load all sentiment history across all symbols.

        Returns:
            Dictionary mapping symbol -> sentiment scores
        """
        keys = self.store.list_keys(StorageCategory.SENTIMENT_DATA)
        history_keys = [k for k in keys if k.startswith(self._history_key_prefix)]

        all_history = {}
        for key in history_keys:
            symbol = key.replace(self._history_key_prefix, "")
            history = self.load_sentiment_history(symbol)
            if history:
                all_history[symbol] = history

        return all_history

    def get_sentiment_stats(self) -> dict:
        """
        Get statistics about stored sentiment data.

        Returns:
            Dictionary with sentiment storage statistics
        """
        keys = self.store.list_keys(StorageCategory.SENTIMENT_DATA)

        stats = {
            "total_objects": len(keys),
            "history_count": len([k for k in keys if k.startswith(self._history_key_prefix)]),
            "prediction_count": len([k for k in keys if k.startswith(self._prediction_key_prefix)]),
            "filter_decision_count": len([k for k in keys if k.startswith(self._filter_key_prefix)]),
            "has_provider_performance": self._provider_key in keys,
        }

        # Get storage usage
        storage_stats = self.store.get_storage_stats()
        sentiment_stats = storage_stats.get("by_category", {}).get("sentiment_data", {})
        stats["size_mb"] = sentiment_stats.get("size_mb", 0)
        stats["quota_mb"] = sentiment_stats.get("quota_mb", 0)

        return stats


def create_sentiment_persistence(
    object_store: ObjectStoreManager,
) -> SentimentPersistence:
    """
    Create a sentiment persistence helper.

    Args:
        object_store: ObjectStoreManager instance

    Returns:
        SentimentPersistence instance
    """
    return SentimentPersistence(object_store)
