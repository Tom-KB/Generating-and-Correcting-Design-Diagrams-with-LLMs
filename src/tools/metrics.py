from typing import Dict, Set, Tuple
import re

Relation = Tuple[str, str, str]  # (A, B, type)
Multiplicity = Tuple[str, str, str, str, str]  # (A, B, type, multA, multB)

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

def compute_metrics(generated_puml: str, ground_truth_puml: str) -> Dict[str, float]:
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
