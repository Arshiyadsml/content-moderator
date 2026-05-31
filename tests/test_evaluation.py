import pytest
import json
from pathlib import Path
from src.config import DATA_DIR
from src.evaluation.harness import EvaluationHarness
from src.database import ModeratorDB
from src.rag.policy_library import PolicyLibrary
from src.agents.graph import ModeratorGraph

def test_eval_dataset_loads():
    eval_file = DATA_DIR / "eval_dataset.jsonl"
    assert eval_file.exists()
    with open(eval_file) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 110
    assert lines[0]["id"] == "eval_001"

def test_eval_dataset_valid_format():
    eval_file = DATA_DIR / "eval_dataset.jsonl"
    with open(eval_file) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                assert "id" in item
                assert "content" in item
                assert "ground_truth_label" in item
                assert item["ground_truth_label"] in ["approve", "flag", "quarantine", "reject", "escalate"]

def test_harness_initializes():
    db = ModeratorDB()
    db.connect()
    lib = PolicyLibrary(db)
    graph = ModeratorGraph(db, lib)
    harness = EvaluationHarness(graph, db)
    assert harness is not None
    db.close()
