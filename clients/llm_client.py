"""
LLM client wrapper for vLLM server with OpenAI-compatible API.
Supports structured outputs and fallback to OpenAI API.
"""

from openai import AsyncOpenAI, OpenAI
from typing import Optional, Type, TypeVar, Any, Dict, List
from pydantic import BaseModel
import json
import logging

from config.settings import settings

# Import LangSmith for tracing
try:
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
except ImportError:
    # Fallback if langsmith not installed
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    get_current_run_tree = None

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class LLMClient:
    """
    Wrapper for LLM interactions with vLLM or OpenAI.
    Provides both async and sync interfaces.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        use_openai: bool = False
    ):
        """
        Initialize LLM client.

        Args:
            base_url: Base URL for vLLM server (e.g., http://localhost:8000/v1)
            api_key: API key (use "EMPTY" for vLLM, actual key for OpenAI)
            model_name: Model name to use
            use_openai: If True, use OpenAI API instead of vLLM
        """
        self.base_url = base_url or settings.vllm_base_url
        self.api_key = api_key or settings.vllm_api_key
        self.model_name = model_name or settings.vllm_model_name
        self.use_openai = use_openai or settings.use_openai_fallback

        if self.use_openai and settings.openai_api_key:
            self.api_key = settings.openai_api_key
            self.base_url = "https://api.openai.com/v1"
            self.model_name = "gpt-4o-mini"  # or another OpenAI model
            logger.info("Using OpenAI API fallback")
        else:
            logger.info(f"Using vLLM at {self.base_url} with model {self.model_name}")

        # Initialize async and sync clients з timeout для швидкості
        import httpx

        # HTTP client з оптимізованим timeout для швидкості
        # ВАЖЛИВО: для hosted API потрібен більший timeout для knowledge extraction
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=120.0,  # 120s total timeout
                connect=10.0,   # 10s to establish connection
                read=120.0,     # 120s to read response (для knowledge extraction з великими промптами)
                write=10.0      # 10s to write request
            ),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )

        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=http_client,
            max_retries=0  # Без retry для швидкості
        )

        sync_http_client = httpx.Client(
            timeout=httpx.Timeout(
                timeout=120.0,
                connect=10.0,
                read=120.0,
                write=10.0
            )
        )

        self.sync_client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=sync_http_client,
            max_retries=0
        )

    @traceable(name="llm_generate")
    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Type[T]] = None,
        **kwargs
    ) -> str | T:
        """
        Generate a response asynchronously.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional parameters for the API

        Returns:
            Either a string response or a Pydantic model instance
        """
        temperature = temperature or settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens

        request_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        # Add structured output if Pydantic model provided
        if response_format is not None:
            if self.use_openai:
                # OpenAI native structured outputs
                request_params["response_format"] = response_format
            else:
                # vLLM structured outputs via JSON schema
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": response_format.model_json_schema(),
                        "strict": True
                    }
                }

        try:
            response = await self.async_client.chat.completions.create(**request_params)
            print(f"Response: {response}")
            content = response.choices[0].message.content

            # Clean content from invalid UTF-8 surrogate characters
            if content:
                try:
                    content = content.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                except Exception:
                    # Fallback: replace known problematic characters
                    content = content.replace('\udcd1', '').replace('\udcd0', '')

            # Add input/output to OpenTelemetry span for Phoenix
            try:
                from opentelemetry import trace
                current_span = trace.get_current_span()
                if current_span and current_span.is_recording():
                    # Set input messages
                    import json as json_module
                    current_span.set_attribute("llm.input_messages", json_module.dumps(messages))
                    # Set output
                    current_span.set_attribute("llm.output_messages", json_module.dumps([{"role": "assistant", "content": content}]))
                    # Set model info
                    current_span.set_attribute("llm.model", self.model_name)
                    current_span.set_attribute("llm.temperature", temperature)
                    current_span.set_attribute("llm.max_tokens", max_tokens)
            except Exception as e:
                logger.debug(f"Could not add span attributes: {e}")

            # Log token usage for debugging/monitoring
            if hasattr(response, 'usage') and response.usage:
                usage_info = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                logger.info(f"Token usage: {usage_info}")

                # Add token usage to OpenTelemetry span
                try:
                    from opentelemetry import trace
                    current_span = trace.get_current_span()
                    if current_span and current_span.is_recording():
                        current_span.set_attribute("llm.usage.prompt_tokens", response.usage.prompt_tokens)
                        current_span.set_attribute("llm.usage.completion_tokens", response.usage.completion_tokens)
                        current_span.set_attribute("llm.usage.total_tokens", response.usage.total_tokens)
                except Exception as e:
                    logger.debug(f"Could not add token usage to span: {e}")

                # Add cost tracking to Phoenix span
                try:
                    from config.cost_tracking import add_cost_to_span
                    add_cost_to_span(
                        model=self.model_name,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        is_embedding=False
                    )
                except Exception as e:
                    logger.debug(f"Could not add cost to span: {e}")

                # Add token metadata to LangSmith trace if available
                if get_current_run_tree:
                    try:
                        run_tree = get_current_run_tree()
                        if run_tree:
                            run_tree.extra = run_tree.extra or {}
                            run_tree.extra["usage_metadata"] = usage_info
                    except Exception as e:
                        logger.debug(f"Could not add usage metadata to LangSmith: {e}")

            if response_format is not None:
                # Parse JSON response into Pydantic model
                try:
                    parsed = json.loads(content)
                    return response_format.model_validate(parsed)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse structured output: {e}")
                    logger.error(f"Raw content: {content}")
                    # Fallback: try to extract JSON from content
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group())
                            return response_format.model_validate(parsed)
                        except Exception:
                            pass
                    raise ValueError(f"Could not parse response into {response_format.__name__}: {content}")

            return content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_sync(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Type[T]] = None,
        **kwargs
    ) -> str | T:
        """
        Generate a response synchronously.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional parameters for the API

        Returns:
            Either a string response or a Pydantic model instance
        """
        temperature = temperature or settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens

        request_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        # Add structured output if Pydantic model provided
        if response_format is not None:
            if self.use_openai:
                request_params["response_format"] = response_format
            else:
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": response_format.model_json_schema(),
                        "strict": True
                    }
                }

        try:
            response = self.sync_client.chat.completions.create(**request_params)
            content = response.choices[0].message.content

            # Clean content from invalid UTF-8 surrogate characters
            if content:
                try:
                    content = content.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                except Exception:
                    # Fallback: replace known problematic characters
                    content = content.replace('\udcd1', '').replace('\udcd0', '')

            if response_format is not None:
                try:
                    parsed = json.loads(content)
                    return response_format.model_validate(parsed)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse structured output: {e}")
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group())
                            return response_format.model_validate(parsed)
                        except Exception:
                            pass
                    raise ValueError(f"Could not parse response into {response_format.__name__}: {content}")

            return content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def get_async_client(self) -> AsyncOpenAI:
        """Get the underlying async OpenAI client."""
        return self.async_client

    def get_sync_client(self) -> OpenAI:
        """Get the underlying sync OpenAI client."""
        return self.sync_client


# Global client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def create_llm_client(**kwargs) -> LLMClient:
    """Create a new LLM client with custom parameters."""
    return LLMClient(**kwargs)
