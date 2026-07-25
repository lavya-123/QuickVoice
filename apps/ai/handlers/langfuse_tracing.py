"""Langfuse observability/evaluation layer for the LiveKit voice agent.

Wires LiveKit Agents' built-in OpenTelemetry tracing to Langfuse via the
OTLP endpoint, so every call session (LLM generations, STT/TTS spans,
tool calls, latency metrics) shows up as a trace in Langfuse for
inspection and scoring.

Enabled only when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
are set, so local dev without a Langfuse account keeps working unchanged.
"""

import base64
import os
from typing import Any, Optional

from loguru import logger
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util.types import AttributeValue

from livekit.agents.telemetry import set_tracer_provider


def langfuse_enabled() -> bool:
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
        and os.getenv("LANGFUSE_HOST")
    )


def setup_langfuse_tracing(
    metadata: Optional[dict[str, AttributeValue]] = None,
) -> Optional[TracerProvider]:
    """Configure a TracerProvider that exports spans to Langfuse.

    Returns None (and logs a warning) if Langfuse env vars aren't
    configured, so callers can no-op instead of crashing the call.
    """
    if not langfuse_enabled():
        logger.debug("Langfuse tracing not configured; skipping")
        return None

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    host = os.environ["LANGFUSE_HOST"].rstrip("/")

    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth}"

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    set_tracer_provider(trace_provider, metadata=metadata)

    logger.info("Langfuse tracing enabled for this session")
    return trace_provider


def call_trace_metadata(call_context: dict[str, Any], room_name: str) -> dict[str, AttributeValue]:
    """Build the Langfuse session/user metadata attached to every span."""
    metadata: dict[str, AttributeValue] = {"langfuse.session.id": room_name}
    if call_context.get("agent_id"):
        metadata["langfuse.tags"] = [f"agent:{call_context['agent_id']}"]
    if call_context.get("direction"):
        metadata["call.direction"] = call_context["direction"]
    return metadata
