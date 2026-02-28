"""
Pydantic schemas for structured outputs in the Research & Report pipeline.

Used by agents with enable_json_mode and output_schema.
"""

from pydantic import BaseModel, Field


class ReportSummary(BaseModel):
    """Structured report summary from the writer/reporter agent."""

    title: str = Field(description="Short title of the report")
    sections: list[str] = Field(
        default_factory=list,
        description="List of section summaries or bullet points",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the report (0-1)",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="Key findings or takeaways",
    )


class RouteDecision(BaseModel):
    """Optional structured route decision (if needed for router)."""

    route: str = Field(description="Agent name to route to: research, summarize, or qa")
    reason: str = Field(default="", description="Brief reason for the route")
