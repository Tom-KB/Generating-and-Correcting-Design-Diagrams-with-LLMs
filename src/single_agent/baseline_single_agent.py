import os
import sys
import csv
import json
from dataclasses import dataclass
from typing import Dict, List

# Add the root folder (let you run the code directly from src/single_agent)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

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
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs", "single_agent")
os.makedirs(OUTPUTS_ROOT, exist_ok=True)

@dataclass
class RunResult:
    plantuml: str
    evaluation: EvalResult
    raw_generation: str
    raw_evaluation: str

def generate_and_evaluate(requirements: str) -> RunResult:
    # The UML generator
    generator = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
    )

    # LLM-as-a-judge
    judge = ChatOpenAI(
        model=os.getenv("OPENAI_JUDGE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        temperature=0.0,
    )

    # Generation of the UML class diagram
    gen_chain = ChatPromptTemplate.from_template(GEN_PROMPT) | generator | StrOutputParser()
    raw_gen = invoke_chain(gen_chain, {"requirements": requirements})
    puml = extract_plantuml(raw_gen) # Extraction of the plantuml

    # Verification of the sanity of the output
    sanity = basic_sanity_checks(puml)
    sanity_text = "None" if not sanity else "\n".join(f"- {x}" for x in sanity)

    # Evaluation of the result
    parser = PydanticOutputParser(pydantic_object=EvalResult)
    eval_prompt = ChatPromptTemplate.from_template(EVAL_PROMPT).partial(
        format_instructions=parser.get_format_instructions()
    )
    eval_chain = eval_prompt | judge | StrOutputParser()

    raw_eval = invoke_chain(
        eval_chain,
        {"requirements": requirements, "plantuml": puml, "sanity_issues": sanity_text},
    )

    evaluation = parser.parse(raw_eval)

    return RunResult(
        plantuml=puml,
        evaluation=evaluation,
        raw_generation=raw_gen,
        raw_evaluation=raw_eval,
    )

def run_exercise(exercise_id: str, exercise_path: str) -> Dict[str, float]:
    """Runs one exercise end-to-end and returns a flat dict for CSV."""
    requirements_text, ground_truth = parse_exercise(exercise_path)
    result = generate_and_evaluate(requirements_text)

    out_dir = os.path.join(OUTPUTS_ROOT, exercise_id)
    os.makedirs(out_dir, exist_ok=True)

    generated_puml_path = os.path.join(out_dir, "diagram_generated.puml")
    gt_puml_path = os.path.join(out_dir, "diagram_ground_truth.puml")
    eval_path = os.path.join(out_dir, "eval.json")

    with open(generated_puml_path, "w", encoding="utf-8") as f:
        f.write(result.plantuml)

    with open(gt_puml_path, "w", encoding="utf-8") as f:
        f.write(ground_truth)

    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(result.evaluation.model_dump_json(indent=2))

    gen_render_ok = render_success(generated_puml_path, fmt="png")
    gt_render_ok = render_success(gt_puml_path, fmt="png")

    rq1 = compute_metrics(result.plantuml, ground_truth)
    rq1["render_success_generated"] = 1.0 if gen_render_ok else 0.0
    rq1["render_success_ground_truth"] = 1.0 if gt_render_ok else 0.0

    rq1["exercise_id"] = exercise_id
    rq1["judge_score_0_100"] = float(result.evaluation.score_0_100)

    metrics_path = os.path.join(out_dir, "rq1_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(rq1, f, indent=2)

    print(f"[{exercise_id}] Saved outputs to: {out_dir}")
    return rq1

if __name__ == "__main__":
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

    # Write CSV summary
    csv_path = os.path.join(OUTPUTS_ROOT, "rq1_results.csv")
    fieldnames = [
        "exercise_id",
        "judge_score_0_100",
        "render_success_generated",
        "render_success_ground_truth",
        "class_precision", "class_recall", "class_f1",
        "relation_precision", "relation_recall", "relation_f1",
        "multiplicity_accuracy",
        "n_classes_gen", "n_classes_gt",
        "n_relations_gen", "n_relations_gt",
        "n_multiplicities_gt",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print("\nSaved CSV:", csv_path)
