from langchain.agents.middleware import wrap_tool_call
from langchain.agents.structured_output import ProviderStrategy
from langchain.messages import ToolMessage
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Dict, Literal, Type
import os
import logging


# Output schema constants (exported for tests)
class SimpleResponse(BaseModel):
    message: str = Field(description="Short natural language response")


class TableResponse(BaseModel):
    columns: list[str] = Field(description="Column headers for the table")
    rows: list[list[str]] = Field(description="Table rows as lists of string values")


SIMPLE_RESPONSE_SCHEMA = {"type": "simple", "description": "Short natural language response"}
TABLE_RESPONSE_SCHEMA = {"type": "table", "description": "Tabular data response"}

SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "simple": SimpleResponse,
    "table": TableResponse,
}


class IntentRouter(BaseModel):
    """Select the output schema required for the user's request."""
    selected_schema: Literal["simple", "table"] = Field(
        description=(
            "Choose 'table' for rankings, lists, comparisons, or any response that should be displayed"
            " as rows/columns. Choose 'simple' for direct explanations or short answers."
        )
    )

_logger = logging.getLogger(__name__)


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages.

    This middleware wraps tool invocation and converts exceptions into a
    structured ToolMessage so the agent receives a friendly error instead of
    an exception bubbling up.
    """
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )
    


@wrap_tool_call
def classify_and_set_schema(request, handler):
    """Classify user intent and set an output schema on the tool_call metadata.

    Uses a small Groq model to decide whether the user's query expects a
    tabular (table) response or a simple natural-language response. The
    decision is recorded at `request.tool_call['metadata']['output_schema']`.
    """
    try:
        messages = getattr(request, "messages", None)
        if not messages:
            return handler(request)

        last = messages[-1]
        role = last.get("role") if isinstance(last, dict) else getattr(last, "role", None)
        if role != "human":
            return handler(request)

        groq_api_key = os.getenv("GROQ_API_KEY")
        selected_key = "simple"
        if groq_api_key:
            try:
                router_llm = init_chat_model("allam-2-7b", model_provider="groq", temperature=0)
                structured_router = router_llm.with_structured_output(IntentRouter)
                routing_decision = structured_router.invoke(messages)
                selected_key = routing_decision.selected_schema
                print(f"Intent classification result: {selected_key}")
            except Exception as e:
                _logger.warning("Groq classification failed: %s", e)

        tool_call = getattr(request, "tool_call", None)
        if isinstance(tool_call, dict):
            meta = tool_call.setdefault("metadata", {})
            meta["output_schema"] = TABLE_RESPONSE_SCHEMA if selected_key == "table" else SIMPLE_RESPONSE_SCHEMA

        target_schema = SCHEMA_REGISTRY[selected_key]
        request.response_format = ProviderStrategy(schema=target_schema)
    except Exception as e:
        _logger.exception("Failed to classify intent: %s", e)

    return handler(request)


