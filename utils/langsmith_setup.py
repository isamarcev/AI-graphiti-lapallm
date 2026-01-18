"""
LangSmith tracing setup.
Automatically configures LangSmith based on settings.
"""

import os
from config.settings import settings


def setup_langsmith():
    """
    Setup LangSmith environment variables from settings.

    Call this at the start of your application to enable tracing.
    """
    if settings.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if settings.langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        print(f"✅ LangSmith tracing enabled for project: {settings.langchain_project}")
    else:
        # Ensure tracing is disabled
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
