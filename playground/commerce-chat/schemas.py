"""
Pydantic schemas for multi-agent execution plans.

Defines structured data models for the Plan-and-Execute pattern:
- Intent classification
- Tool execution steps
- Complete execution plans
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Intent(str, Enum):
    """Classified user intent."""

    GREETING = "greeting"
    PRODUCT_SEARCH = "product_search"
    PRODUCT_DETAILS = "product_details"
    CART_VIEW = "cart_view"
    CART_ADD = "cart_add"
    CART_REMOVE = "cart_remove"
    CART_UPDATE = "cart_update"
    CHECKOUT = "checkout"
    ORDER_STATUS = "order_status"
    HELP = "help"
    OTHER = "other"


class ToolStep(BaseModel):
    """A single tool execution step."""

    step_id: str = Field(description="Unique identifier for this step (e.g., 'search', 'add_1')")

    tool_name: str = Field(description="Exact name of the MCP tool to call")

    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Known/static parameters for the tool call"
    )

    instruction: str = Field(
        description="Natural language instruction describing what this step does. "
        "Include hints for dynamic values (e.g., 'use cheapest from search', "
        "'use first product', 'add all products from results')"
    )

    depends_on: list[str] | str | None = Field(
        default=None,
        description="Step ID(s) that must complete before this step. "
        "Can be a single string, list of strings, or None. "
        "None or empty = no dependencies (can run first/parallel)",
    )

    @field_validator("depends_on", mode="before")
    @classmethod
    def normalize_depends_on(cls, v: Any) -> list[str] | None:
        """Normalize depends_on to always be a list or None."""
        if v is None:
            return None
        if isinstance(v, str):
            return [v] if v else None
        if isinstance(v, list):
            return [str(item) for item in v if item] or None
        return None


class ExecutionPlan(BaseModel):
    """Structured execution plan from Orchestrator to Executor."""

    # Intent classification
    intent: Intent = Field(description="Classified intent of the user request")

    # Direct response option (skip executor)
    respond_directly: bool = Field(
        default=False,
        description="True if Orchestrator should respond directly without Executor "
        "(for greetings, simple questions, no-tool-needed queries)",
    )

    direct_response: str | None = Field(
        default=None, description="Response content when respond_directly=True"
    )

    # Execution steps (when respond_directly=False)
    steps: list[ToolStep] = Field(
        default_factory=list, description="Ordered list of tool execution steps"
    )

    # Response formatting
    response_instructions: str = Field(
        default="Present the results clearly and concisely to the user.",
        description="Instructions for how Executor should format the final response",
    )

    # User context (from memory)
    user_context: str | None = Field(
        default=None,
        description="Relevant user context from memory (pet info, preferences) "
        "that Executor should use for personalization",
    )

    # Execution hints
    require_all_steps: bool = Field(
        default=True, description="If True, all steps must succeed. If False, partial results OK."
    )
