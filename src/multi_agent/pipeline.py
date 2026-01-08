import os
import re
import csv
import json
import subprocess
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Shared paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../src/multi_agent
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")
EXERCISES_DIR = os.path.join(PROJECT_ROOT, "data", "exercises")
# OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs", "multi_agent")
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs", "multi_agent_2nd")
os.makedirs(OUTPUTS_ROOT, exist_ok=True)

load_dotenv(dotenv_path=DOTENV_PATH)

# Simple per-process throttle to avoid 429 rate limits on free/low-RPM accounts.
_OPENAI_RPM = int(os.getenv("OPENAI_RPM", "3"))
_RPM_BUFFER_SEC = float(os.getenv("OPENAI_RPM_BUFFER_SEC", "2.0"))
_MIN_INTERVAL = (60.0 / max(_OPENAI_RPM, 1)) + max(_RPM_BUFFER_SEC, 0.0)
_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "1"))
_LOCK = threading.Lock()
_last_call_ts = 0.0  # timestamp of last completed API call (this process)

# Multi-agent behavior tuning (defaults chosen to be conservative).
_FIX_TEMPERATURE = float(os.getenv("MULTI_AGENT_FIX_TEMPERATURE", "0.0"))
_SKIP_FIX_CRITIC_SCORE_GE = int(os.getenv("MULTI_AGENT_SKIP_FIX_CRITIC_SCORE_GE", "90"))
_MIN_JUDGE_IMPROVEMENT = float(os.getenv("MULTI_AGENT_MIN_JUDGE_IMPROVEMENT", "1.0"))


def _wait_for_slot():
    wait_for = (_last_call_ts + _MIN_INTERVAL) - time.time()
    if wait_for > 0:
        time.sleep(wait_for)


def _is_rate_limit_error(exc: Exception) -> bool:
    name = exc.__class__.__name__
    if name == "RateLimitError":
        return True
    msg = str(exc).lower()
    return ("429" in msg and "rate limit" in msg) or ("rate_limit_exceeded" in msg)


def _invoke_chain(chain, inputs: Dict[str, str]) -> str:
    global _last_call_ts

    for attempt in range(_MAX_RETRIES + 1):
        try:
            with _LOCK:
                _wait_for_slot()
                result = chain.invoke(inputs)
                _last_call_ts = time.time()
                return result
        except Exception as e:
            if not _is_rate_limit_error(e) or attempt >= _MAX_RETRIES:
                raise

            msg = str(e)
            retry_after = None
            m = re.search(r"try again in\\s+(\\d+)s", msg, flags=re.IGNORECASE)
            if m:
                try:
                    retry_after = float(m.group(1))
                except Exception:
                    retry_after = None
            time.sleep(retry_after if retry_after is not None else _MIN_INTERVAL)


# ---------------- Utilities ----------------
def extract_plantuml(text: str) -> str:
    m = re.search(r"@startuml[\s\S]*?@enduml", text)
    if not m:
        raise ValueError("No @startuml...@enduml block found.")
    return m.group(0).strip()


def basic_sanity_checks(puml: str) -> List[str]:
    """Catch common formatting issues before asking the critic."""
    issues = []
    if "@startuml" not in puml or "@enduml" not in puml:
        issues.append("Missing @startuml/@enduml.")
    if "```" in puml:
        issues.append("Contains Markdown fences.")
    if len(puml) < 30:
        issues.append("Too short; likely invalid.")
    return issues


def parse_exercise(path: str) -> Tuple[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if "# GROUND_TRUTH_PLANTUML" not in text:
        raise ValueError(f"Exercise file missing '# GROUND_TRUTH_PLANTUML': {path}")

    sw_description = text.split("# GROUND_TRUTH_PLANTUML")[0]
    sw_description = sw_description.replace("# SOFTWARE_DESCRIPTION", "").strip()

    diagram_groundtruth = text.split("# GROUND_TRUTH_PLANTUML", 1)[1].strip()
    return sw_description, diagram_groundtruth


def render_plantuml(puml_path: str, fmt: str = "png") -> Path:
    p = Path(puml_path)
    if not p.exists():
        raise FileNotFoundError(f"PlantUML file not found: {p}")

    cmd = ["plantuml", f"-t{fmt}", str(p)]
    subprocess.run(cmd, check=True)
    return p.with_suffix(f".{fmt}")


def render_success(puml_path: str, fmt: str = "png") -> bool:
    try:
        render_plantuml(puml_path, fmt=fmt)
        return True
    except Exception:
        return False


# ---------------- RQ2 Metrics ----------------
REL_TYPE_MAP = {
    "<|--": "inheritance",
    "--|>": "inheritance",
    "*--": "composition",
    "--*": "composition",
    "o--": "aggregation",
    "--o": "aggregation",
    "..>": "dependency",
    "<..": "dependency",
    "-->": "association",
    "<--": "association",
    "..": "association",
    "--": "association",
}

Relation = Tuple[str, str, str]  # (A, B, type)
Multiplicity = Tuple[str, str, str, str, str]  # (A, B, type, multA, multB)


def _normalize_name(name: str) -> str:
    return name.strip().strip('"').strip()


def _ordered_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a.lower() <= b.lower() else (b, a)


def extract_classes(puml: str) -> Set[str]:
    classes = set()
    for m in re.finditer(r"^\s*class\s+([A-Za-z_]\w*)", puml, flags=re.MULTILINE):
        classes.add(m.group(1))
    for m in re.finditer(r"^\s*(?:abstract\s+class|interface)\s+([A-Za-z_]\w*)", puml, flags=re.MULTILINE):
        classes.add(m.group(1))
    return classes


def detect_relation_type(line: str) -> str:
    for token, rtype in REL_TYPE_MAP.items():
        if token in line:
            return rtype
    return "association"


def extract_relations_and_multiplicities(puml: str) -> Tuple[Set[Relation], Set[Multiplicity]]:
    relations: Set[Relation] = set()
    multiplicities: Set[Multiplicity] = set()

    for raw in puml.splitlines():
        line = raw.strip()
        if not line or line.startswith("'") or line.startswith("//"):
            continue
        if line.startswith("@") or line.lower().startswith("skinparam") or line.lower().startswith("hide"):
            continue
        if not any(tok in line for tok in REL_TYPE_MAP.keys()):
            continue

        m = re.search(
            r'^([A-Za-z_]\w*)\s*(?:"([^"]+)")?\s*([.<|*o-]{2,4}|--|\.{2}>|<\.{2})\s*(?:"([^"]+)")?\s*([A-Za-z_]\w*)',
            line,
        )
        if not m:
            continue

        a = _normalize_name(m.group(1))
        mult_a = m.group(2)
        connector = m.group(3)
        mult_b = m.group(4)
        b = _normalize_name(m.group(5))

        rtype = detect_relation_type(connector)
        x, y = _ordered_pair(a, b)
        relations.add((x, y, rtype))

        if mult_a is not None or mult_b is not None:
            multiplicities.add((x, y, rtype, mult_a or "", mult_b or ""))

    return relations, multiplicities


def prf1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_rq2_metrics(generated_puml: str, ground_truth_puml: str) -> Dict[str, float]:
    gen_classes = extract_classes(generated_puml)
    gt_classes = extract_classes(ground_truth_puml)

    gen_rel, gen_mult = extract_relations_and_multiplicities(generated_puml)
    gt_rel, gt_mult = extract_relations_and_multiplicities(ground_truth_puml)

    tp_c = len(gen_classes & gt_classes)
    fp_c = len(gen_classes - gt_classes)
    fn_c = len(gt_classes - gen_classes)
    class_scores = prf1(tp_c, fp_c, fn_c)

    tp_r = len(gen_rel & gt_rel)
    fp_r = len(gen_rel - gt_rel)
    fn_r = len(gt_rel - gen_rel)
    rel_scores = prf1(tp_r, fp_r, fn_r)

    mult_acc = (len(gen_mult & gt_mult) / len(gt_mult)) if len(gt_mult) > 0 else 0.0

    return {
        "class_precision": class_scores["precision"],
        "class_recall": class_scores["recall"],
        "class_f1": class_scores["f1"],
        "relation_precision": rel_scores["precision"],
        "relation_recall": rel_scores["recall"],
        "relation_f1": rel_scores["f1"],
        "multiplicity_accuracy": mult_acc,
        "n_classes_gen": float(len(gen_classes)),
        "n_classes_gt": float(len(gt_classes)),
        "n_relations_gen": float(len(gen_rel)),
        "n_relations_gt": float(len(gt_rel)),
        "n_multiplicities_gt": float(len(gt_mult)),
    }


# ---------------- Prompts & Roles ----------------
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


class Critique(BaseModel):
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    score_0_100: int = Field(..., ge=0, le=100)


class EvalResult(BaseModel):
    syntax_ok: bool = Field(..., description="Is the PlantUML syntactically plausible?")
    syntax_issues: List[str] = Field(default_factory=list)
    semantic_ok: bool = Field(..., description="Does it represent the requirements correctly?")
    semantic_issues: List[str] = Field(default_factory=list)
    pragmatic_ok: bool = Field(..., description="Is it clear/readable/well-structured?")
    pragmatic_issues: List[str] = Field(default_factory=list)
    score_0_100: int = Field(..., ge=0, le=100, description="Overall quality score.")
    recommendations: List[str] = Field(default_factory=list, description="Concrete fixes to improve the diagram.")


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


# ---------------- Pipeline Steps ----------------
def make_model(role: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=temperature)


def generate(requirements: str) -> Tuple[str, str]:
    generator = make_model("generator", temperature=0.2)
    gen_chain = ChatPromptTemplate.from_template(GEN_PROMPT) | generator | StrOutputParser()
    raw_gen = _invoke_chain(gen_chain, {"requirements": requirements})
    return extract_plantuml(raw_gen), raw_gen


def critique(requirements: str, plantuml: str) -> Tuple[Critique, str]:
    critic = make_model("critic", temperature=0.0)
    sanity = basic_sanity_checks(plantuml)
    sanity_text = "None" if not sanity else "\n".join(f"- {x}" for x in sanity)
    parser = PydanticOutputParser(pydantic_object=Critique)
    prompt = ChatPromptTemplate.from_template(CRITIC_PROMPT).partial(
        format_instructions=parser.get_format_instructions()
    )
    chain = prompt | critic | StrOutputParser()
    raw = _invoke_chain(chain, {"requirements": requirements, "plantuml": plantuml, "sanity_issues": sanity_text})
    return parser.parse(raw), raw


def fix(requirements: str, plantuml: str, critique: Critique) -> Tuple[str, str]:
    fixer = make_model("fixer", temperature=_FIX_TEMPERATURE)
    prompt = ChatPromptTemplate.from_template(FIX_PROMPT)
    chain = prompt | fixer | StrOutputParser()
    critique_text = "\n".join([f"- {i}" for i in critique.issues] + [f"Recommendation: {r}" for r in critique.recommendations]) or "None"
    raw = _invoke_chain(chain, {"requirements": requirements, "plantuml": plantuml, "critique": critique_text})
    return extract_plantuml(raw), raw


def evaluate(requirements: str, plantuml: str) -> Tuple[EvalResult, str]:
    judge = make_model("judge", temperature=0.0)
    sanity = basic_sanity_checks(plantuml)
    sanity_text = "None" if not sanity else "\n".join(f"- {x}" for x in sanity)
    parser = PydanticOutputParser(pydantic_object=EvalResult)
    eval_prompt = ChatPromptTemplate.from_template(EVAL_PROMPT).partial(
        format_instructions=parser.get_format_instructions()
    )
    eval_chain = eval_prompt | judge | StrOutputParser()
    raw_eval = _invoke_chain(
        eval_chain,
        {"requirements": requirements, "plantuml": plantuml, "sanity_issues": sanity_text},
    )
    return parser.parse(raw_eval), raw_eval


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

    raw = _invoke_chain(
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


def _should_attempt_fix(critique_result: Critique, plantuml: str) -> bool:
    sanity = basic_sanity_checks(plantuml)
    if sanity:
        return True
    return critique_result.score_0_100 < _SKIP_FIX_CRITIC_SCORE_GE

# ---------------- Runner ----------------
def run_exercise(exercise_id: str, exercise_path: str) -> Dict[str, float]:
    requirements_text, ground_truth = parse_exercise(exercise_path)
    puml_gen, raw_gen = generate(requirements_text)
    critique_result, raw_critique = critique(requirements_text, puml_gen)

    fix_attempted = _should_attempt_fix(critique_result, puml_gen)
    if fix_attempted:
        puml_fixed, raw_fix = fix(requirements_text, puml_gen, critique_result)
    else:
        puml_fixed, raw_fix = puml_gen, ""

    # eval_generated, raw_eval_gen = evaluate(requirements_text, puml_gen)
    # if puml_fixed.strip() == puml_gen.strip():
    #     eval_fixed, raw_eval_fix = eval_generated, raw_eval_gen
    # else:
    #     eval_fixed, raw_eval_fix = evaluate(requirements_text, puml_fixed)


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
    # with open(eval_gen_path, "w", encoding="utf-8") as f:
    #     f.write(eval_generated.model_dump_json(indent=2))
    # with open(eval_fix_path, "w", encoding="utf-8") as f:
    #     f.write(eval_fixed.model_dump_json(indent=2))

    gen_render_ok = render_success(gen_puml_path, fmt="png")
    fixed_render_ok = render_success(fixed_puml_path, fmt="png")
    gt_render_ok = render_success(gt_puml_path, fmt="png")

    selected_source, raw_selection = select_best(
        requirements_text, puml_gen, puml_fixed, gen_render_ok, fixed_render_ok
    )
    puml_selected = puml_fixed if selected_source == "fixed" else puml_gen
    # eval_selected = eval_fixed if selected_source == "fixed" else eval_generated

    with open(selected_puml_path, "w", encoding="utf-8") as f:
        f.write(puml_selected)
    # with open(eval_sel_path, "w", encoding="utf-8") as f:
    #     f.write(eval_selected.model_dump_json(indent=2))
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump({"choice": selected_source, "raw": raw_selection}, f, indent=2)

    selected_render_ok = render_success(selected_puml_path, fmt="png")

    rq2 = compute_rq2_metrics(puml_selected, ground_truth)
    rq2["render_success_generated"] = 1.0 if gen_render_ok else 0.0
    rq2["render_success_fixed"] = 1.0 if fixed_render_ok else 0.0
    rq2["render_success_selected"] = 1.0 if selected_render_ok else 0.0
    rq2["render_success_ground_truth"] = 1.0 if gt_render_ok else 0.0
    rq2["exercise_id"] = exercise_id
    rq2["selected_source"] = selected_source
    # rq2["judge_score_0_100"] = float(eval_selected.score_0_100)
    # rq2["judge_score_generated_0_100"] = float(eval_generated.score_0_100)
    # rq2["judge_score_fixed_0_100"] = float(eval_fixed.score_0_100)
    # rq2["judge_score_selected_0_100"] = float(eval_selected.score_0_100)
    rq2["critic_score_0_100"] = float(critique_result.score_0_100)
    rq2["fix_attempted"] = 1.0 if fix_attempted else 0.0
    rq2["min_judge_improvement"] = float(_MIN_JUDGE_IMPROVEMENT)

    metrics_path = os.path.join(out_dir, "rq2_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(rq2, f, indent=2)

    print(out_dir)
    return rq2


def main():
    exercise_files = [f"exercise_{i:02d}.txt" for i in range(1, 6)]
    rows: List[Dict[str, float]] = []

    for fname in exercise_files:
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

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print("\nSaved CSV:", csv_path)


if __name__ == "__main__":
    main()
