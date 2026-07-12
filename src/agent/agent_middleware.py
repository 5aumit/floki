from typing import List, Literal, Union

from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from pydantic import BaseModel, Field

from agent.tool_status import emit_tool_status, tool_name_from_request


class TextBlock(BaseModel):
    type: Literal["text"]
    markdown: str


class TableBlock(BaseModel):
    type: Literal["table"]
    markdown: str


class BlockResponse(BaseModel):
    blocks: List[Union[TextBlock, TableBlock]] = Field(min_length=1)


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    emit_tool_status(tool_name_from_request(request))
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"],
        )
