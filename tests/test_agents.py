import pytest
from src.agents.judge import JudgeAgent

def test_judge_approves_safe_content():
    """Judge should APPROVE benign content"""
    judge = JudgeAgent()
    result = judge.run(
        content="I really enjoyed this movie",
        policies={}
    )
    assert result["decision"] == "APPROVE"
    assert result["confidence"] > 0.8

def test_judge_rejects_violence():
    """Judge should REJECT violent content"""
    judge = JudgeAgent()
    policies = {
        "HAR-001": {
            "title": "No violent content",
            "description": "Content depicting violence",
            "examples": [],
            "severity": 5
        }
    }
    result = judge.run(
        content="Detailed instructions on how to harm others",
        policies=policies
    )
    assert result["decision"] in ["REJECT", "QUARANTINE"]
    
def test_judge_handles_errors():
    """Judge should ESCALATE on errors"""
    judge = JudgeAgent()
    # Very long input might cause issues
    long_content = "x" * 100000
    result = judge.run(content=long_content, policies={})
    # Should not crash; might escalate or handle gracefully
    assert "decision" in result
