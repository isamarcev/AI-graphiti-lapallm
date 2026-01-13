"""
Cost Tracking для LLM calls - підрахунок вартості токенів для Phoenix.
"""

import logging
from typing import Dict, Optional
from opentelemetry import trace

logger = logging.getLogger(__name__)


# Ціни за 1M токенів (у доларах) для різних моделів
# Оновіть ці значення відповідно до вашого pricing
TOKEN_PRICES = {
    # Lapa/Mamay (hosted Lapathon API) - приблизні ціни
    "lapa": {
        "input": 0.0,  # Безкоштовно для хакатону, або встановіть реальну ціну
        "output": 0.0
    },
    "mamay": {
        "input": 0.0,
        "output": 0.0
    },

    # OpenAI pricing (для fallback)
    "gpt-4": {
        "input": 30.0,  # $30 per 1M input tokens
        "output": 60.0  # $60 per 1M output tokens
    },
    "gpt-4-turbo": {
        "input": 10.0,
        "output": 30.0
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60
    },
    "gpt-3.5-turbo": {
        "input": 0.5,
        "output": 1.5
    },

    # Embeddings pricing
    "text-embedding-qwen": {
        "input": 0.0,  # Безкоштовно для хакатону
        "output": 0.0
    },
    "paraphrase-multilingual-mpnet-base-v2": {
        "input": 0.0,  # Local model
        "output": 0.0
    },
    "text-embedding-3-small": {
        "input": 0.02,  # $0.02 per 1M tokens
        "output": 0.0
    }
}


def calculate_llm_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int
) -> float:
    """
    Обчислити вартість LLM виклику.

    Args:
        model: Назва моделі (lapa, gpt-4, etc.)
        prompt_tokens: Кількість input токенів
        completion_tokens: Кількість output токенів

    Returns:
        Вартість у доларах
    """
    # Отримати pricing для моделі
    pricing = TOKEN_PRICES.get(model, TOKEN_PRICES.get("lapa", {}))

    if not pricing:
        logger.warning(f"No pricing info for model {model}, cost set to 0")
        return 0.0

    input_price = pricing.get("input", 0.0)
    output_price = pricing.get("output", 0.0)

    # Обчислити вартість (ціна за 1M токенів)
    input_cost = (prompt_tokens / 1_000_000) * input_price
    output_cost = (completion_tokens / 1_000_000) * output_price

    total_cost = input_cost + output_cost

    logger.debug(
        f"Cost calculation: {model} - "
        f"input={prompt_tokens} tokens (${input_cost:.6f}), "
        f"output={completion_tokens} tokens (${output_cost:.6f}), "
        f"total=${total_cost:.6f}"
    )

    return total_cost


def calculate_embedding_cost(
    model: str,
    tokens: int
) -> float:
    """
    Обчислити вартість embedding виклику.

    Args:
        model: Назва embedding моделі
        tokens: Кількість токенів (або приблизна оцінка: chars / 4)

    Returns:
        Вартість у доларах
    """
    pricing = TOKEN_PRICES.get(model, {})

    if not pricing:
        logger.warning(f"No pricing info for embedding model {model}, cost set to 0")
        return 0.0

    input_price = pricing.get("input", 0.0)
    cost = (tokens / 1_000_000) * input_price

    logger.debug(f"Embedding cost: {model} - {tokens} tokens (${cost:.6f})")

    return cost


def add_cost_to_span(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    is_embedding: bool = False
):
    """
    Додати інформацію про вартість до поточного OpenTelemetry span.

    Це дозволить Phoenix відображати cumulative cost у UI.

    Args:
        model: Назва моделі
        prompt_tokens: Input tokens
        completion_tokens: Output tokens (0 для embeddings)
        is_embedding: True якщо це embedding call
    """
    try:
        # Обчислити вартість
        if is_embedding:
            # Для embeddings використовуємо prompt_tokens
            cost = calculate_embedding_cost(model, prompt_tokens)
            total_tokens = prompt_tokens
        else:
            # Для LLM calls
            cost = calculate_llm_cost(model, prompt_tokens, completion_tokens)
            total_tokens = prompt_tokens + completion_tokens

        # Отримати поточний span
        current_span = trace.get_current_span()

        if current_span and current_span.is_recording():
            # Додати атрибути які Phoenix розуміє
            current_span.set_attribute("llm.token_count.prompt", prompt_tokens)
            current_span.set_attribute("llm.token_count.completion", completion_tokens)
            current_span.set_attribute("llm.token_count.total", total_tokens)

            # Додати інформацію про вартість
            current_span.set_attribute("llm.cost.input", round((prompt_tokens / 1_000_000) * TOKEN_PRICES.get(model, {}).get("input", 0.0), 6))
            current_span.set_attribute("llm.cost.output", round((completion_tokens / 1_000_000) * TOKEN_PRICES.get(model, {}).get("output", 0.0), 6))
            current_span.set_attribute("llm.cost.total", round(cost, 6))
            current_span.set_attribute("llm.cost.currency", "USD")

            # Додати pricing info
            current_span.set_attribute("llm.pricing.input_per_million", TOKEN_PRICES.get(model, {}).get("input", 0.0))
            current_span.set_attribute("llm.pricing.output_per_million", TOKEN_PRICES.get(model, {}).get("output", 0.0))

            logger.debug(f"Added cost tracking to span: ${cost:.6f}")
        else:
            logger.debug("No active span to add cost info")

    except Exception as e:
        logger.error(f"Failed to add cost to span: {e}", exc_info=True)


def get_session_cost_summary(spans_data: list) -> Dict[str, float]:
    """
    Обчислити сумарну вартість для всіх spans у сесії.

    Args:
        spans_data: Список span dictionaries з Phoenix

    Returns:
        Dict з breakdown по моделях та загальна вартість
    """
    summary = {
        "total_cost": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "by_model": {}
    }

    for span in spans_data:
        attrs = span.get("attributes", {})

        model = attrs.get("llm.model_name") or attrs.get("gen_ai.request.model", "unknown")
        input_tokens = attrs.get("llm.token_count.prompt", 0)
        output_tokens = attrs.get("llm.token_count.completion", 0)

        if input_tokens > 0 or output_tokens > 0:
            cost = calculate_llm_cost(model, input_tokens, output_tokens)

            summary["total_cost"] += cost
            summary["total_input_tokens"] += input_tokens
            summary["total_output_tokens"] += output_tokens

            # Breakdown по моделях
            if model not in summary["by_model"]:
                summary["by_model"][model] = {
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "calls": 0
                }

            summary["by_model"][model]["cost"] += cost
            summary["by_model"][model]["input_tokens"] += input_tokens
            summary["by_model"][model]["output_tokens"] += output_tokens
            summary["by_model"][model]["calls"] += 1

    return summary


def update_token_prices(model: str, input_price: float, output_price: float):
    """
    Оновити ціни токенів для моделі.

    Використовуйте це якщо ціни змінилися або додається нова модель.

    Args:
        model: Назва моделі
        input_price: Ціна за 1M input tokens (USD)
        output_price: Ціна за 1M output tokens (USD)
    """
    TOKEN_PRICES[model] = {
        "input": input_price,
        "output": output_price
    }
    logger.info(f"Updated pricing for {model}: input=${input_price}, output=${output_price} per 1M tokens")


def log_cost_summary():
    """
    Виведе summary pricing info у логи.
    Корисно для debugging та підтвердження налаштувань.
    """
    logger.info("=== LLM Cost Tracking Configuration ===")
    for model, pricing in TOKEN_PRICES.items():
        if pricing.get("input", 0) > 0 or pricing.get("output", 0) > 0:
            logger.info(
                f"  {model}: "
                f"${pricing['input']:.2f}/1M input, "
                f"${pricing['output']:.2f}/1M output"
            )
        else:
            logger.info(f"  {model}: FREE")
    logger.info("=" * 40)
