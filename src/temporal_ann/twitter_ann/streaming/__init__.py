"""Kafka ingestion and deterministic event-time ordering for RT-HNSW."""

from .ordering import EventTimeOrderer, OrdererBatchResult, validate_embedded_event

__all__ = ["EventTimeOrderer", "OrdererBatchResult", "validate_embedded_event"]
