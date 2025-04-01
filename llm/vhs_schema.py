from pydantic import BaseModel, Field

class VHSInterpretation(BaseModel):
    """Structured output for VHS interpretation."""
    normal_range: str = Field(description="The normal VHS range for this animal type")
    interpretation: str = Field(description="Clinical interpretation (normal, enlarged, borderline)")
    severity: str = Field(description="If abnormal, severity of cardiomegaly (mild, moderate, severe)")
    possible_conditions: list[str] = Field(description="List of possible conditions associated with this VHS")
    recommendations: list[str] = Field(description="Recommended follow-up actions")
    detailed_explanation: str = Field(description="Detailed explanation of findings in clinical terms")

