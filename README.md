# LLM Agentic UML Diagram Generator & Corrector

This project evaluates the performance of Large Language Models in generating **PlantUML Class Diagrams** from textual requirements, it compares a **Single-Agent Baseline** against a **Multi-Agent**.

## Features

* **Single Agent:** Direct generation of PlantUML code from text.
* **Multi-Agent Pipeline:** Add self-correction steps:
* **Generator:** Creates initial diagram.
* **Critic:** Reviews syntax and requirements.
* **Fixer:** Corrects the diagram based on the Critique feedback.
* **Selector:** Chooses the best version (Initial vs Fixed).

* **Automated Metrics:** Calculates Precision, Recall, and F1-Score for classes, relations, and multiplicities compared to a Ground Truth.

## Usage

Configure your `.env` file with the desired model (default: `gpt-4o-mini`) and your API key.

### Run Single Agent Baseline
```bash
python src/single_agent/baseline_single_agent.py
```

### Run Multi-Agent Pipeline
```bash
python src/multi_agent/multi_agent.py
```

Results will be saved in the `outputs/` folder for the selected architecture.