import os, re, csv, json, time, threading
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict
from pathlib import Path
import subprocess
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
#from langchain.output_parsers import PydanticOutputParser  # <-- CORRETO AQUI
import subprocess
from pathlib import Path

load_dotenv()

# Get only the part that matters from the LLM output

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/single_agent
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")
EXERCISES_DIR = os.path.join(PROJECT_ROOT, "data", "exercises")
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs", "single_agent")
os.makedirs(OUTPUTS_ROOT, exist_ok=True)

load_dotenv(dotenv_path=DOTENV_PATH)

# Simple per-process throttle to avoid 429 rate limits on free/low-RPM accounts.
_OPENAI_RPM = int(os.getenv("OPENAI_RPM", "3"))
_RPM_BUFFER_SEC = float(os.getenv("OPENAI_RPM_BUFFER_SEC", "2.0"))
_MIN_INTERVAL = (60.0 / max(_OPENAI_RPM, 1)) + max(_RPM_BUFFER_SEC, 0.0)
_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
_LOCK = threading.Lock()
_last_call_ts = 0.0  # timestamp of last completed API call (this process)


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

def extract_plantuml(text: str) -> str:
    m = re.search(r"@startuml[\s\S]*?@enduml", text)
    if not m:
        raise ValueError("No @startuml...@enduml block found.")
    return m.group(0).strip()

# Identify common errors
def basic_sanity_checks(puml: str) -> List[str]:
    issues = []
    if "@startuml" not in puml or "@enduml" not in puml:
        issues.append("Missing @startuml/@enduml.")
    if "```" in puml:
        issues.append("Contains Markdown fences.")
    if len(puml) < 30:
        issues.append("Too short; likely invalid.")
    return issues


def parse_exercise(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if "# GROUND_TRUTH_PLANTUML" not in text:
        raise ValueError(f"Exercise file missing '# GROUND_TRUTH_PLANTUML': {path}")

    sw_description = text.split("# GROUND_TRUTH_PLANTUML")[0]
    sw_description = sw_description.replace("# SOFTWARE_DESCRIPTION", "").strip()

    diagram_groundtruth = text.split("# GROUND_TRUTH_PLANTUML", 1)[1].strip()
    return sw_description, diagram_groundtruth

def render_plantuml(puml_path: str, fmt: str = "png") -> Path:
    """
    Renders a .puml file into an image using local PlantUML CLI.
    fmt: 'png' or 'svg' (svg is great for papers).
    """
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


# ---------------- RQ1 Metrics (Gold Standard) ----------------
# We keep this parser intentionally simple + robust for class diagrams.
# It works best if your ground truth uses "class X" lines and relationship lines like:
# A "1" -- "0..*" B : label
# A <|-- B
# A *-- B
#
# If you later adopt more PlantUML features, we can upgrade the parser.

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
    # remove quotes and trim
    return name.strip().strip('"').strip()

def _ordered_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a.lower() <= b.lower() else (b, a)

def extract_classes(puml: str) -> Set[str]:
    classes = set()

    # class Foo { ... }
    for m in re.finditer(r'^\s*class\s+([A-Za-z_]\w*)', puml, flags=re.MULTILINE):
        classes.add(m.group(1))

    # also catch "abstract class", "interface" if present
    for m in re.finditer(r'^\s*(?:abstract\s+class|interface)\s+([A-Za-z_]\w*)', puml, flags=re.MULTILINE):
        classes.add(m.group(1))

    return classes

def detect_relation_type(line: str) -> str:
    # find the first connector token in line
    for token, rtype in REL_TYPE_MAP.items():
        if token in line:
            return rtype
    return "association"

def extract_relations_and_multiplicities(puml: str) -> Tuple[Set[Relation], Set[Multiplicity]]:
    relations: Set[Relation] = set()
    multiplicities: Set[Multiplicity] = set()

    # Relationship line heuristic:
    # Start with an identifier, contain connector tokens
    for raw in puml.splitlines():
        line = raw.strip()
        if not line or line.startswith("'") or line.startswith("//"):
            continue
        # ignore directives
        if line.startswith("@") or line.lower().startswith("skinparam") or line.lower().startswith("hide"):
            continue

        # must contain a relationship token
        if not any(tok in line for tok in REL_TYPE_MAP.keys()):
            continue

        # A "1" -- "0..*" B : label
        # Capture: left class, left mult (optional), connector (any), right mult (optional), right class
        m = re.search(
            r'^([A-Za-z_]\w*)\s*(?:"([^"]+)")?\s*([.<|*o-]{2,4}|--|\.{2}>|<\.{2})\s*(?:"([^"]+)")?\s*([A-Za-z_]\w*)',
            line
        )
        if not m:
            continue

        a = _normalize_name(m.group(1))
        mult_a = m.group(2)  # may be None
        connector = m.group(3)
        mult_b = m.group(4)  # may be None
        b = _normalize_name(m.group(5))

        rtype = detect_relation_type(connector)

        # normalize undirected for association/aggregation/composition; keep directed types but still normalize pair
        x, y = _ordered_pair(a, b)
        relations.add((x, y, rtype))

        # multiplicity: only count if at least one side has explicit multiplicity
        if mult_a is not None or mult_b is not None:
            multiplicities.add((x, y, rtype, mult_a or "", mult_b or ""))

    return relations, multiplicities

def prf1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def compute_rq1_metrics(generated_puml: str, ground_truth_puml: str) -> Dict[str, float]:
    gen_classes = extract_classes(generated_puml)
    gt_classes = extract_classes(ground_truth_puml)

    gen_rel, gen_mult = extract_relations_and_multiplicities(generated_puml)
    gt_rel, gt_mult = extract_relations_and_multiplicities(ground_truth_puml)

    # Classes PRF1
    tp_c = len(gen_classes & gt_classes)
    fp_c = len(gen_classes - gt_classes)
    fn_c = len(gt_classes - gen_classes)
    class_scores = prf1(tp_c, fp_c, fn_c)

    # Relations PRF1
    tp_r = len(gen_rel & gt_rel)
    fp_r = len(gen_rel - gt_rel)
    fn_r = len(gt_rel - gen_rel)
    rel_scores = prf1(tp_r, fp_r, fn_r)

    # Multiplicity accuracy: evaluate only GT multiplicities (expected)
    # exact match on (pair, type, multA, multB) after normalization
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

GEN_PROMPT = """\
You are a UML modeling assistant.
Generate a UML Class Diagram in PlantUML for the system below.

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

@dataclass
class RunResult:
    plantuml: str
    evaluation: EvalResult
    raw_generation: str
    raw_evaluation: str

def generate_and_evaluate(requirements: str) -> RunResult:
    generator = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
    )

    # (Recomendado) usar outro modelo/config para judge
    judge = ChatOpenAI(
        model=os.getenv("OPENAI_JUDGE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        temperature=0.0,
    )

    gen_chain = ChatPromptTemplate.from_template(GEN_PROMPT) | generator | StrOutputParser()
    raw_gen = _invoke_chain(gen_chain, {"requirements": requirements})
    puml = extract_plantuml(raw_gen)

    sanity = basic_sanity_checks(puml)
    sanity_text = "None" if not sanity else "\n".join(f"- {x}" for x in sanity)

    parser = PydanticOutputParser(pydantic_object=EvalResult)
    eval_prompt = ChatPromptTemplate.from_template(EVAL_PROMPT).partial(
        format_instructions=parser.get_format_instructions()
    )
    eval_chain = eval_prompt | judge | StrOutputParser()

    raw_eval = _invoke_chain(
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

    rq1 = compute_rq1_metrics(result.plantuml, ground_truth)
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
    # Expecting: exercise_01.txt ... exercise_05.txt
    exercise_files = [f"exercise_{i:02d}.txt" for i in range(1, 6)]
    rows: List[Dict[str, float]] = []

    for fname in exercise_files:
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
