"""Wrapper for Graphiti OpenAI client to fix message role alternation bug.

Graphiti's retry logic can append consecutive 'user' messages, which violates
OpenAI/LiteLLM's requirement that roles must alternate user/assistant/user/assistant.

This wrapper ensures proper alternation by intercepting at the lowest level
before messages are sent to the OpenAI API.
"""

from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message
from pydantic import BaseModel
import logging
import typing

logger = logging.getLogger(__name__)


class FixedOpenAIGenericClient(OpenAIGenericClient):
    """OpenAI client wrapper that fixes message role alternation.

    Intercepts generate_response to fix messages BEFORE retry logic,
    preventing retry logic from breaking alternation.
    """

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: typing.Any = None,
    ) -> dict[str, typing.Any]:
        """Intercept at EVERY call to fix messages before API call.
        
        This is called by parent's generate_response() on each retry attempt.
        By fixing here, we ensure messages are correct even after parent adds error messages.
        """
        # Fix message alternation on EVERY attempt
        original_count = len(messages)
        fixed_messages = self._ensure_alternating_roles(messages)
        
        # Log only if we made changes
        if len(fixed_messages) != original_count:
            logger.warning(f"Fixed message alternation: {original_count} -> {len(fixed_messages)} messages")
            logger.debug("Roles: " + " → ".join([m.role for m in fixed_messages]))
        
        # Call parent with fixed messages
        return await super()._generate_response(
            fixed_messages,
            response_model,
            max_tokens,
            model_size,
        )

    def _ensure_alternating_roles(self, messages: list[Message]) -> list[Message]:
        """Ensure messages alternate between user and assistant roles.

        When two consecutive 'user' messages are detected, inserts a placeholder
        'assistant' message between them to maintain proper alternation.

        Args:
            messages: Original list of Message objects

        Returns:
            New list with proper role alternation (original is not modified)
        """
        if not messages:
            return messages

        fixed = []

        for i, curr_msg in enumerate(messages):
            # Always add system messages without checks
            if curr_msg.role == "system":
                fixed.append(curr_msg)
                continue

            # For non-system messages, check previous non-system message
            prev_non_system = None
            for prev_msg in reversed(fixed):
                if prev_msg.role != "system":
                    prev_non_system = prev_msg
                    break

            # If we have a previous non-system message, check alternation
            if prev_non_system:
                # If two consecutive user messages, insert assistant placeholder
                if curr_msg.role == "user" and prev_non_system.role == "user":
                    logger.debug(
                        f"Inserting assistant placeholder between two user messages at index {i}"
                    )
                    placeholder = Message(
                        role="assistant",
                        content="Acknowledged.",
                    )
                    fixed.append(placeholder)

                # If two consecutive assistant messages, insert user placeholder
                elif curr_msg.role == "assistant" and prev_non_system.role == "assistant":
                    logger.debug(
                        f"Inserting user placeholder between two assistant messages at index {i}"
                    )
                    placeholder = Message(
                        role="user",
                        content="Continue.",
                    )
                    fixed.append(placeholder)

            fixed.append(curr_msg)

        # Ensure the conversation starts with a user message when possible
        for idx, msg in enumerate(fixed):
            if msg.role != "system":
                if msg.role == "assistant":
                    user_placeholder = Message(
                        role="user",
                        content="Please provide the next step."
                    )
                    fixed.insert(idx, user_placeholder)
                    logger.debug("Inserted leading user placeholder to satisfy alternation")
                break

        return fixed