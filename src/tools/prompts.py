import os
from typing import List
from pydantic import BaseModel, Field

GEN_PROMPT = """\
You are a UML modeling assistant.
Generate a UML Class Diagram in PlantUML for the system below.

Follow these conventions derived from the ground-truth diagrams:
- Use only PlantUML class-diagram syntax (no Markdown or styling directives).
- Declare classes with `class Name {{` and list attributes as bare names (no types/visibility), one per line.
- Do not invent methods unless explicitly required.
- Add a blank line between class blocks for readability.
- Put multiplicities in quotes next to each class on the relation line (e.g., `A "1" -- "0..*" B`).
- Use `--` for plain associations, `*--` for compositions/whole-part with lifecycle dependency, and `<|--` for inheritance.
- If a relationship needs its own data or represents many-to-many, introduce an explicit class to hold those attributes.
- Never create 2 connections between the same 2 classes; use an association class if needed.

Output rules:
- Output ONLY valid PlantUML code.
- Must include @startuml and @enduml.
- No Markdown.

System description:
{requirements}
"""

EVAL_PROMPT = """\
You are a strict UML reviewer.
Evaluate the following UML Class Diagram (PlantUML) against the requirements.

Requirements:
{requirements}

Candidate PlantUML:
{plantuml}

Also consider these automatic sanity-check issues (if any):
{sanity_issues}

Return a JSON object that matches exactly this schema:
{format_instructions}

Rules:
- Be critical and specific.
- List concrete problems and actionable recommendations.
- If something is missing from requirements, call it out.
"""

class EvalResult(BaseModel):
    syntax_ok: bool = Field(..., description="Is the PlantUML syntactically plausible?")
    syntax_issues: List[str] = Field(default_factory=list)

    semantic_ok: bool = Field(..., description="Does it represent the requirements correctly?")
    semantic_issues: List[str] = Field(default_factory=list)

    pragmatic_ok: bool = Field(..., description="Is it clear/readable/well-structured?")
    pragmatic_issues: List[str] = Field(default_factory=list)

    score_0_100: int = Field(..., ge=0, le=100, description="Overall quality score.")
    recommendations: List[str] = Field(default_factory=list, description="Concrete fixes to improve the diagram.")
