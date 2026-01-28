import os
import re
import time
import threading
import subprocess
from pathlib import Path
from typing import List, Dict

import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")
EXERCISES_DIR = os.path.join(PROJECT_ROOT, "data", "exercises")
load_dotenv(dotenv_path=DOTENV_PATH)

# Simple per-process throttle to avoid 429 rate limits on free/low-RPM accounts
_OPENAI_RPM = int(os.getenv("OPENAI_RPM", "3"))
_RPM_BUFFER_SEC = float(os.getenv("OPENAI_RPM_BUFFER_SEC", "2.0"))
_MIN_INTERVAL = (60.0 / max(_OPENAI_RPM, 1)) + max(_RPM_BUFFER_SEC, 0.0)
_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "1"))

_LOCK = threading.Lock()
_last_call_ts = 0.0  # timestamp of last completed API call (this process)

# Basic utilities functions
def extract_plantuml(text: str) -> str:
    m = re.search(r"@startuml[\s\S]*?@enduml", text)
    if not m:
        raise ValueError("No @startuml...@enduml block found.")
    return m.group(0).strip()

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
 
# Gemini rate limit handling
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

def invoke_chain(chain, inputs: Dict[str, str]) -> str:
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
