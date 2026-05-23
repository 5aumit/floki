from langchain.agents.middleware import wrap_tool_call
from langchain.agents.structured_output import ProviderStrategy
from langchain.messages import ToolMessage
import logging


# JSON Schema for BlockResponse (strict, provider-agnostic).
BLOCK_RESPONSE_SCHEMA = {
    "title": "BlockResponse",
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "text"},
                            "markdown": {"type": "string"},
                        },
                        "required": ["type", "markdown"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "table"},
                            "markdown": {"type": "string"},
                        },
                        "required": ["type", "markdown"],
                        "additionalProperties": False,
                    },
                ]
            },
        }
    },
    "required": ["blocks"],
    "additionalProperties": False,
}

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
def set_block_response_schema(request, handler):
    """Force the agent to use the BlockResponse JSON schema for structured output."""
    try:
        request.response_format = ProviderStrategy(schema=BLOCK_RESPONSE_SCHEMA)
    except Exception as e:
        _logger.exception("Failed to set block response schema: %s", e)
    return handler(request)
