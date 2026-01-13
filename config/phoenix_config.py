"""
Phoenix observability configuration для LLM tracing та моніторингу.
Instrumentation для LangChain, OpenAI та custom metrics.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def setup_phoenix_instrumentation(
    collector_endpoint: Optional[str] = None,
    project_name: Optional[str] = None
) -> bool:
    """
    Setup Phoenix instrumentation для tracing LLM calls та retrieval.

    Args:
        collector_endpoint: Phoenix collector URL (default: from env)
        project_name: Project name для Phoenix (default: from env)

    Returns:
        True if successfully configured, False otherwise
    """
    try:
        # OPTIMIZATION: Check if Phoenix is enabled
        enable_phoenix = os.getenv("ENABLE_PHOENIX", "true").lower() == "true"
        if not enable_phoenix:
            logger.info("Phoenix instrumentation disabled via ENABLE_PHOENIX=false")
            return False

        # Get configuration from environment
        # NOTE: Phoenix register() automatically adds /v1/traces, so use base URL only
        collector_endpoint = collector_endpoint or os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT",
            "http://phoenix:6006"  # Base URL without /v1/traces
        )
        project_name = project_name or os.getenv(
            "PHOENIX_PROJECT_NAME",
            "graphiti-lapa-agent"
        )

        logger.info(f"Setting up Phoenix instrumentation: {collector_endpoint}")

        # Import Phoenix OTEL components
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource

        # Create resource with project name
        resource = Resource.create({
            "service.name": project_name
        })

        # Create TracerProvider with resource
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Create OTLP exporter with correct endpoint
        # Phoenix expects traces at http://phoenix:6006/v1/traces
        otlp_endpoint = f"{collector_endpoint}/v1/traces"
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)

        # Add BatchSpanProcessor
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)

        # Instrument LangChain (для LangGraph nodes)
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

        # Instrument OpenAI (для LLM calls)
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

        logger.info("✅ Phoenix instrumentation configured successfully")
        logger.info(f"   - OTLP Endpoint: {otlp_endpoint}")
        logger.info(f"   - Project: {project_name}")
        logger.info(f"   - Phoenix UI: http://localhost:6006")
        logger.info(f"   - Instrumentation: LangChain, OpenAI")

        return True

    except ImportError as e:
        logger.warning(f"Phoenix dependencies not installed: {e}")
        logger.warning("Install with: pip install arize-phoenix arize-phoenix-otel openinference-instrumentation-langchain openinference-instrumentation-openai")
        return False

    except Exception as e:
        logger.error(f"Failed to setup Phoenix instrumentation: {e}", exc_info=True)
        return False


def add_phoenix_metadata(span_name: str, metadata: dict):
    """
    Add custom metadata to current Phoenix span.

    Args:
        span_name: Name of the span
        metadata: Dictionary of metadata to add
    """
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(span_name) as span:
            for key, value in metadata.items():
                span.set_attribute(key, str(value))

    except Exception as e:
        logger.debug(f"Could not add Phoenix metadata: {e}")


def get_phoenix_url() -> str:
    """
    Get Phoenix UI URL.

    Returns:
        Phoenix UI URL
    """
    collector_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    # Extract base URL
    if "phoenix:" in collector_endpoint:
        # In Docker, convert service name to localhost
        return "http://localhost:6006"
    return collector_endpoint
