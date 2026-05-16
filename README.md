# AI Multimodal Content Moderator

Production-grade content moderation system using LangGraph, RAG, and Claude 3.5 Sonnet.

## Architecture

- **Input**: Documents, images, text via HuggingFace tasks
- **RAG**: Semantic policy search (pgvector + embeddings)
- **Decision**: Multi-agent LangGraph orchestration
- **Evaluation**: Formal harness with 100-example labeled dataset

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/init_db.py
python scripts/embed_policies.py
```

## Tests

```bash
pytest tests/ -v
```

All tests passing. 3 test files, 13 test cases.

## Components

- `src/hf_tasks/` — Document QA, Image processing, Text classification
- `src/rag/` — Policy library + semantic search
- `src/agents/` — Multi-agent graph (Judge, Retrieval, Audit, Enforce)
- `src/evaluation/` — Evaluation harness + metrics
- `data/eval_dataset.jsonl` — 100 labeled moderation examples

## Evaluation Dataset

100 realistic examples across 6 categories:
- Violence (15)
- Harassment (15)
- Privacy/Doxing (10)
- Misinformation (10)
- Spam (10)
- General (40)

## Current Status

✅ Phase 1-5 complete  
✅ All core code implemented  
✅ Tests passing  
⏳ Phase 6: CI/CD + GitHub Actions  

## Next Steps

- Add GitHub Actions CI/CD pipeline
- Deploy Docker container
- Run full evaluation when API credits available

## License

MIT
