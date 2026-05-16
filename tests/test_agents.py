import pytest
from src.agents.judge import JudgeAgent
from unittest.mock import patch, MagicMock


@patch('src.agents.judge.Anthropic')
def test_judge_outputs_valid_decision(mock_anthropic):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"decision": "APPROVE", "confidence": 0.9, "reasoning": "Safe content", "matching_policy_ids": [], "risk_score": 0.1}')]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client
    
    judge = JudgeAgent()
    result = judge.run("This is great!", {})
    assert "decision" in result
    assert result["decision"] in ["APPROVE", "FLAG", "REJECT", "ESCALATE", "QUARANTINE"]

# def test_judge_approves_safe_content():
#    """Judge should APPROVE benign content"""
#   judge = JudgeAgent()
#   result = judge.run(
#       content="I really enjoyed this movie",
#       policies={}
#   )
#   assert result["decision"] == "APPROVE"
#   assert result["confidence"] > 0.8

# def test_judge_rejects_violence():
#  """Judge should REJECT violent content"""
#   judge = JudgeAgent()
#   policies = {
#       "HAR-001": {
#           "title": "No violent content",
#           "description": "Content depicting violence",
#           "examples": [],
#           "severity": 5
#       }
#   }
#   result = judge.run(
#       content="Detailed instructions on how to harm others",
#       policies=policies
#   )
#   assert result["decision"] in ["REJECT", "QUARANTINE"]
    
def test_judge_handles_errors():
    """Judge should ESCALATE on errors"""
    judge = JudgeAgent()
    # Very long input might cause issues
    long_content = "x" * 100000
    result = judge.run(content=long_content, policies={})
    # Should not crash; might escalate or handle gracefully
    assert "decision" in result

def test_judge_parses_json():
    judge = JudgeAgent()
    # Test JSON parsing logic
    response_text = '{"decision": "APPROVE", "confidence": 0.95, "reasoning": "test", "matching_policy_ids": [], "risk_score": 0.1}'
    result = judge._parse_decision(response_text)
    assert result["decision"] == "APPROVE"
    assert result["confidence"] == 0.95
