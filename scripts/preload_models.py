#!/usr/bin/env python3
"""
Preload ML models during Docker build.

Завантажує моделі заздалегідь, щоб вони були в кеші HuggingFace
і не завантажувались при першому запиті.

Використання:
    python scripts/preload_models.py

Викликається в Dockerfile на етапі build.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def preload_reranker():
    """Preload cross-encoder reranker model."""
    model_name = os.environ.get(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    logger.info(f"Preloading reranker model: {model_name}")

    try:
        from sentence_transformers import CrossEncoder

        # Завантаження моделі (збережеться в HF_HOME cache)
        model = CrossEncoder(model_name, max_length=512)

        # Тестовий прогін для перевірки
        test_score = model.predict([("test query", "test document")])
        logger.info(f"Reranker loaded successfully. Test score: {test_score[0]:.4f}")

        return True

    except ImportError:
        logger.warning("sentence-transformers not installed, skipping reranker preload")
        return False
    except Exception as e:
        logger.error(f"Failed to preload reranker: {e}")
        return False


def main():
    """Preload all configured models."""
    logger.info("=" * 50)
    logger.info("Preloading ML models for Tabula Rasa Agent")
    logger.info("=" * 50)

    results = {}

    # Preload reranker
    results["reranker"] = preload_reranker()

    # Summary
    logger.info("=" * 50)
    logger.info("Preload Summary:")
    for model, success in results.items():
        status = "OK" if success else "SKIPPED/FAILED"
        logger.info(f"  {model}: {status}")
    logger.info("=" * 50)

    # Return exit code based on critical models
    # Reranker is optional, so we don't fail the build
    return 0


if __name__ == "__main__":
    sys.exit(main())
