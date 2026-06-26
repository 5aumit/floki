import os

from langchain_google_genai import ChatGoogleGenerativeAI


def _get_gemini_api_key(llm_config: dict) -> str:
    api_key = llm_config.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY for Gemini provider.")
    return api_key


def get_llm_from_config(llm_config: dict):
    provider = llm_config.get("provider", "gemini")
    if provider != "gemini":
        raise ValueError(f"Unsupported LLM provider: {provider}. Only 'gemini' is supported.")
    return ChatGoogleGenerativeAI(
        google_api_key=_get_gemini_api_key(llm_config),
        model=llm_config.get("gemini_model", "gemini-2.5-flash"),
        **llm_config.get("gemini_params", {}),
    )


def get_formatter_llm_from_config(llm_config: dict):
    provider = llm_config.get("provider", "gemini")
    if provider != "gemini":
        raise ValueError(f"Unsupported LLM provider: {provider}. Only 'gemini' is supported.")
    return ChatGoogleGenerativeAI(
        google_api_key=_get_gemini_api_key(llm_config),
        model=llm_config.get("formatter_model", "gemini-2.5-flash-lite"),
        **llm_config.get("formatter_params", {}),
    )
