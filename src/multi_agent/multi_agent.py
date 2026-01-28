import os
import sys
import re
import csv
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Add the root folder (let you run the code directly from src/multi_agent)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the tools
from src.tools.utils import (
    invoke_chain,
    extract_plantuml,
    basic_sanity_checks,
    parse_exercise,
    render_success,
    PROJECT_ROOT,
    EXERCISES_DIR
)
from src.tools.prompts import GEN_PROMPT, EVAL_PROMPT, EvalResult
from src.tools.metrics import compute_metrics

# Folder to save the results
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs", "multi_agent")
os.makedirs(OUTPUTS_ROOT, exist_ok=True)

# Multi-agent behavior tuning (defaults chosen to be conservative).
_FIX_TEMPERATURE = float(os.getenv("MULTI_AGENT_FIX_TEMPERATURE", "0.0"))
_SKIP_FIX_CRITIC_SCORE_GE = int(os.getenv("MULTI_AGENT_SKIP_FIX_CRITIC_SCORE_GE", "90"))
_MIN_JUDGE_IMPROVEMENT = float(os.getenv("MULTI_AGENT_MIN_JUDGE_IMPROVEMENT", "1.0"))

# Prompts for each role
CRITIC_PROMPT = """\
You are a strict UML reviewer.
Check the candidate PlantUML against the requirements and the sanity issues.

Requirements:
{requirements}

Candidate PlantUML:
{plantuml}

Sanity-check issues to consider:
{sanity_issues}

Return a JSON object matching exactly this schema:
{format_instructions}

Be conservative, if you think the solution is right, do not suggest changes."""

FIX_PROMPT = """\
You are a UML fixer.
Given the requirements, a flawed PlantUML diagram, and the critique issues,
produce a corrected PlantUML.

Make sure the solution is following these conventions derived from the ground-truth diagrams:
- Use only PlantUML class-diagram syntax (no Markdown or styling directives).
- Declare classes with `class Name {{` and list attributes as bare names (no types/visibility), one per line.
- Do not invent methods unless explicitly required.
- Add a blank line between class blocks for readability.
- Put multiplicities in quotes next to each class on the relation line (e.g., `A "1" -- "0..*" B`).
- Use `--` for plain associations, `*--` for compositions/whole-part with lifecycle dependency, and `<|--` for inheritance.
- If a relationship needs its own data or represents many-to-many, introduce an explicit class to hold those attributes.
- Never create 2 connections between the same 2 classes; use an association class if needed.

Rules:
- Output ONLY valid PlantUML code.
- Must include @startuml and @enduml.
- No Markdown fences.
- Make the smallest set of changes needed to address the critique.
- Preserve existing correct content, names, and structure as much as possible.
- Do NOT add new classes/relations unless explicitly required by the requirements.
- Do NOT remove existing classes/relations unless they are clearly wrong per the requirements.
- If a critique suggestion conflicts with the stated requirements, ignore that suggestion and keep the original content.
- If the critique is low-confidence or ambiguous, prefer leaving the original unchanged.
- If no changes are needed, output the original PlantUML unchanged.

Requirements:
{requirements}

Original PlantUML:
{plantuml}

Critique issues:
{critique}
"""

SELECT_PROMPT = """\
You are the final UML selector.
You are given two candidate PlantUML class diagrams for the same requirements: the initial generation and a fixed version.

Choose which diagram is better with respect to the requirements and conventions. Consider:
- Correctness against the stated requirements (highest priority).
- Rendering viability (if one fails to render, choose the one that renders).
- Alignment with the conventions: attributes as bare names, quoted multiplicities, appropriate connectors, association classes for many-to-many with data, it could never have two relations between the same two classes!
- If both are equivalent, prefer the fixed diagram only if it improves clarity; otherwise keep the generated one.

Return a JSON object with:
- choice: "generated" or "fixed"
- rationale: 1-3 sentences explaining why this choice is better.

Requirements:
{requirements}

Generated PlantUML:
{plantuml_generated}

Fixed PlantUML:
{plantuml_fixed}
"""

class Critique(BaseModel):
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    score_0_100: int = Field(..., ge=0, le=100)

@dataclass
class RunResult:
    plantuml_generated: str
    plantuml_fixed: str
    critique: Critique
    evaluation_generated: EvalResult
    evaluation_fixed: EvalResult
    fix_attempted: bool
    raw_generation: str
    raw_critique: str
    raw_fix: str
    raw_evaluation_generated: str
    raw_evaluation_fixed: str

# Steps of the pipeline
def make_model(role: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)

# Generate the UML
def generate(requirements: str) -> Tuple[str, str]:
    # Create the UML generator
    generator = make_model("generator", temperature=0.2)
    gen_chain = ChatPromptTemplate.from_template(GEN_PROMPT) | generator | StrOutputParser()
    raw_gen = invoke_chain(gen_chain, {"requirements": requirements})
    return extract_plantuml(raw_gen), raw_gen

# Apply the critique to the generate UML diagram
def critique(requirements: str, plantuml: str) -> Tuple[Critique, str]:
    critic = make_model("critic", temperature=0.0)
    sanity = basic_sanity_checks(plantuml)
    sanity_text = "None" if not sanity else "\n".join(f"- {x}" for x in sanity)
    parser = PydanticOutputParser(pydantic_object=Critique)
    prompt = ChatPromptTemplate.from_template(CRITIC_PROMPT).partial(
        format_instructions=parser.get_format_instructions()
    )
    chain = prompt | critic | StrOutputParser()
    raw = invoke_chain(chain, {"requirements": requirements, "plantuml": plantuml, "sanity_issues": sanity_text})
    return parser.parse(raw), raw

# Fix the diagram according to the critique
def fix(requirements: str, plantuml: str, critique: Critique) -> Tuple[str, str]:
    fixer = make_model("fixer", temperature=_FIX_TEMPERATURE)
    prompt = ChatPromptTemplate.from_template(FIX_PROMPT)
    chain = prompt | fixer | StrOutputParser()
    critique_text = "\n".join([f"- {i}" for i in critique.issues] + [f"Recommendation: {r}" for r in critique.recommendations]) or "None"
    raw = invoke_chain(chain, {"requirements": requirements, "plantuml": plantuml, "critique": critique_text})
    return extract_plantuml(raw), raw

# LLM-as-a-judge
def evaluate(requirements: str, plantuml: str) -> Tuple[EvalResult, str]:
    judge = make_model("judge", temperature=0.0)
    sanity = basic_sanity_checks(plantuml)
    sanity_text = "None" if not sanity else "\n".join(f"- {x}" for x in sanity)
    parser = PydanticOutputParser(pydantic_object=EvalResult)
    eval_prompt = ChatPromptTemplate.from_template(EVAL_PROMPT).partial(
        format_instructions=parser.get_format_instructions()
    )
    eval_chain = eval_prompt | judge | StrOutputParser()
    raw_eval = invoke_chain(
        eval_chain,
        {"requirements": requirements, "plantuml": plantuml, "sanity_issues": sanity_text},
    )
    return parser.parse(raw_eval), raw_eval

# Let a selector make a choice over the firstly generated diagram and the fixed one
def select_best(
    requirements: str,
    plantuml_generated: str,
    plantuml_fixed: str,
    render_ok_generated: bool,
    render_ok_fixed: bool,
) -> Tuple[str, str]:
    selector = make_model("selector", temperature=0.0)
    prompt = ChatPromptTemplate.from_template(SELECT_PROMPT)
    chain = prompt | selector | StrOutputParser()

    # If only one renders, pick it immediately.
    if render_ok_generated and not render_ok_fixed:
        return "generated", "Generated renders; fixed does not."
    if render_ok_fixed and not render_ok_generated:
        return "fixed", "Fixed renders; generated does not."

    raw = invoke_chain(
        chain,
        {
            "requirements": requirements,
            "plantuml_generated": plantuml_generated,
            "plantuml_fixed": plantuml_fixed,
        },
    )
    choice_match = re.search(r'"?(generated|fixed)"?', raw, flags=re.IGNORECASE)
    choice = choice_match.group(1).lower() if choice_match else "generated"
    return choice, raw

# Based on the critique result, tells if a fix is needed or not
def _should_attempt_fix(critique_result: Critique, plantuml: str) -> bool:
    sanity = basic_sanity_checks(plantuml)
    if sanity:
        return True
    return critique_result.score_0_100 < _SKIP_FIX_CRITIC_SCORE_GE

# Function to run the pipeline
def run_exercise(exercise_id: str, exercise_path: str) -> Dict[str, float]:
    requirements_text, ground_truth = parse_exercise(exercise_path)
    puml_gen, raw_gen = generate(requirements_text)
    critique_result, raw_critique = critique(requirements_text, puml_gen)

    # Attempt fixing the diagram if needed
    fix_attempted = _should_attempt_fix(critique_result, puml_gen)
    if fix_attempted:
        puml_fixed, raw_fix = fix(requirements_text, puml_gen, critique_result)
    else:
        puml_fixed, raw_fix = puml_gen, ""

    # Evaluation of the diagram
    eval_generated, raw_eval_gen = evaluate(requirements_text, puml_gen)
    if puml_fixed.strip() == puml_gen.strip():
        eval_fixed, raw_eval_fix = eval_generated, raw_eval_gen
    else:
        eval_fixed, raw_eval_fix = evaluate(requirements_text, puml_fixed)

    # Generation of the outputs
    out_dir = os.path.join(OUTPUTS_ROOT, exercise_id)
    os.makedirs(out_dir, exist_ok=True)

    gen_puml_path = os.path.join(out_dir, "diagram_generated.puml")
    fixed_puml_path = os.path.join(out_dir, "diagram_fixed.puml")
    selected_puml_path = os.path.join(out_dir, "diagram_selected.puml")
    gt_puml_path = os.path.join(out_dir, "diagram_ground_truth.puml")
    critique_path = os.path.join(out_dir, "critique.json")
    eval_gen_path = os.path.join(out_dir, "eval_generated.json")
    eval_fix_path = os.path.join(out_dir, "eval_fixed.json")
    eval_sel_path = os.path.join(out_dir, "eval_selected.json")
    selection_path = os.path.join(out_dir, "selection.json")

    with open(gen_puml_path, "w", encoding="utf-8") as f:
        f.write(puml_gen)
    with open(fixed_puml_path, "w", encoding="utf-8") as f:
        f.write(puml_fixed)
    with open(gt_puml_path, "w", encoding="utf-8") as f:
        f.write(ground_truth)
    with open(critique_path, "w", encoding="utf-8") as f:
        f.write(critique_result.model_dump_json(indent=2))
    with open(eval_gen_path, "w", encoding="utf-8") as f:
        f.write(eval_generated.model_dump_json(indent=2))
    with open(eval_fix_path, "w", encoding="utf-8") as f:
        f.write(eval_fixed.model_dump_json(indent=2))

    gen_render_ok = render_success(gen_puml_path, fmt="png")
    fixed_render_ok = render_success(fixed_puml_path, fmt="png")
    gt_render_ok = render_success(gt_puml_path, fmt="png")

    selected_source, raw_selection = select_best(
        requirements_text, puml_gen, puml_fixed, gen_render_ok, fixed_render_ok
    )
    puml_selected = puml_fixed if selected_source == "fixed" else puml_gen
    eval_selected = eval_fixed if selected_source == "fixed" else eval_generated

    with open(selected_puml_path, "w", encoding="utf-8") as f:
        f.write(puml_selected)
    with open(eval_sel_path, "w", encoding="utf-8") as f:
        f.write(eval_selected.model_dump_json(indent=2))
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump({"choice": selected_source, "raw": raw_selection}, f, indent=2)

    selected_render_ok = render_success(selected_puml_path, fmt="png")

    rq2 = compute_metrics(puml_selected, ground_truth)
    rq2["render_success_generated"] = 1.0 if gen_render_ok else 0.0
    rq2["render_success_fixed"] = 1.0 if fixed_render_ok else 0.0
    rq2["render_success_selected"] = 1.0 if selected_render_ok else 0.0
    rq2["render_success_ground_truth"] = 1.0 if gt_render_ok else 0.0
    rq2["exercise_id"] = exercise_id
    rq2["selected_source"] = selected_source
    rq2["judge_score_0_100"] = float(eval_selected.score_0_100)
    rq2["judge_score_generated_0_100"] = float(eval_generated.score_0_100)
    rq2["judge_score_fixed_0_100"] = float(eval_fixed.score_0_100)
    rq2["judge_score_selected_0_100"] = float(eval_selected.score_0_100)
    rq2["critic_score_0_100"] = float(critique_result.score_0_100)
    rq2["fix_attempted"] = 1.0 if fix_attempted else 0.0
    rq2["min_judge_improvement"] = float(_MIN_JUDGE_IMPROVEMENT)

    metrics_path = os.path.join(out_dir, "rq2_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(rq2, f, indent=2)

    print(out_dir)
    return rq2

def main():
    exercise_files = [f"exercise_{i+1:02d}.txt" for i in range(0, 5)]
    rows: List[Dict[str, float]] = []

    # Apply the pipeline for each exercise
    for fname in exercise_files:
        print(f"Trying {fname}")
        path = os.path.join(EXERCISES_DIR, fname)
        exercise_id = fname.replace(".txt", "")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing exercise file: {path}")
        rows.append(run_exercise(exercise_id, path))

    csv_path = os.path.join(OUTPUTS_ROOT, "rq2_results.csv")
    fieldnames = [
        "exercise_id",
        "judge_score_0_100",
        "judge_score_generated_0_100",
        "judge_score_fixed_0_100",
        "judge_score_selected_0_100",
        "critic_score_0_100",
        "render_success_generated",
        "render_success_fixed",
        "render_success_selected",
        "render_success_ground_truth",
        "selected_source",
        "class_precision",
        "class_recall",
        "class_f1",
        "relation_precision",
        "relation_recall",
        "relation_f1",
        "multiplicity_accuracy",
        "n_classes_gen",
        "n_classes_gt",
        "n_relations_gen",
        "n_relations_gt",
        "n_multiplicities_gt",
    ]
    
    # Write CSV summary
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print("\nSaved CSV:", csv_path)

if __name__ == "__main__":
    main()
