"""
LLM client wrapper for vLLM server with OpenAI-compatible API.
Supports structured outputs and fallback to OpenAI API.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import Optional, Type, TypeVar, Any, Dict, List
from pydantic import BaseModel
import logging

from config.settings import settings

# Import LangSmith for tracing
from langsmith import traceable


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
    ):
        """
        Initialize LLM client.

        Args:
            base_url: Base URL for vLLM server (e.g., http://localhost:8000/v1)
            api_key: API key (use "EMPTY" for vLLM, actual key for OpenAI)
            model_name: Model name to use
        """
        self.base_url = base_url or settings.lapa_url
        self.api_key = api_key or settings.api_key
        self.model_name = model_name or settings.model_name

        # LangChain ChatOpenAI with proper timeout and retry
        self.llm = ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model_name,
            timeout=120.0,  # 120s timeout for hosted API
            max_retries=2,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
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
        temperature = temperature or settings.temperature
        max_tokens = max_tokens or settings.max_tokens

        # Convert dict messages to LangChain messages
        lc_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))

        try:
            # Configure LLM for this request
            llm = self.llm.bind(
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Handle structured output
            if response_format is not None:
                llm = llm.with_structured_output(response_format)

            # Invoke LLM
            response = await llm.ainvoke(lc_messages)

            # If structured output, return directly
            if response_format is not None:
                return response

            # Otherwise extract content
            if isinstance(response, AIMessage):
                content = response.content
            else:
                content = str(response)

            # Clean content
            if content:
                try:
                    content = content.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                except Exception:
                    content = content.replace('\udcd1', '').replace('\udcd0', '')

            return content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def get_langchain_llm(self) -> ChatOpenAI:
        """Get the underlying LangChain ChatOpenAI instance."""
        return self.llm


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
