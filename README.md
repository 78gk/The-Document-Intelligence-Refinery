# The Document Intelligence Refinery

A production-grade, 5-stage agentic pipeline for enterprise document intelligence.

## 🚀 Quick Start (< 10 min)

### 1. Prerequisites
- Python 3.10+
- (Optional) OpenRouter API Key for Vision-Augmented extraction.

### 2. Installation
```bash
pip install -e .
```

### 3. Run the Refinery
```bash
python main.py --doc "data/data/CBE ANNUAL REPORT 2023-24.pdf" --query "What is the net profit for 2023?"
```

## 🏗️ Pipeline Stages
1. **Triage Agent**: Classifies document structure and origin.
2. **Extraction Router**: Selects Strategy A (Fast), B (Layout), or C (Vision) with escalation guards.
3. **Semantic Chunker**: Converts content into RAG-optimized LDUs.
4. **PageIndex Builder**: Creates a hierarchical navigation tree.
5. **Query Interface**: LangGraph agent for structured, semantic, and navigational queries.

## 📂 Project Structure
- `src/agents/`: Agent implementations for each stage.
- `src/strategies/`: Extraction strategy implementations.
- `src/models/`: Pydantic data models.
- `rubric/extraction/rules.yaml`: Externalized configuration.
- `.refinery/`: Processed artifacts, profiles, indices, and ledger.

## 🧪 Testing
```bash
pytest
```
