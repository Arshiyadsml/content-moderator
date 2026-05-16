from anthropic import Anthropic
from src.models import JudgeDecision, DecisionCategory
import json
import logging

logger = logging.getLogger(__name__)

class JudgeAgent:
    """LLM-based decision maker using Claude"""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.client = Anthropic()
        self.model = model
    
    def run(self, content: str, policies: dict) -> dict:
        """
        Ask Claude to make a moderation decision based on:
        1. The flagged content
        2. Relevant policies from RAG
        """
        
        # Format policies for Claude
        policy_context = self._format_policies(policies)
        
        # Build prompt
        prompt = self._build_prompt(content, policy_context)
        
        # Call Claude
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.2,  # Low temp for consistency
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Parse response
            decision_text = response.content[0].text
            decision = self._parse_decision(decision_text)
            
            logger.info(f"Judge decision: {decision['decision']} (confidence: {decision['confidence']})")
            return decision
            
        except Exception as e:
            logger.error(f"Judge failed: {e}")
            return {
                "decision": "ESCALATE",
                "confidence": 0.0,
                "reasoning": f"Error during judgment: {e}"
            }
    
    def _format_policies(self, policies: dict) -> str:
        """Format policy dict for Claude's context"""
        output = "## Applicable Policies\n\n"
        for policy_id, policy_data in policies.items():
            output += f"**{policy_id}: {policy_data['title']}**\n"
            output += f"Description: {policy_data['description']}\n"
            output += f"Severity: {policy_data['severity']}/5\n"
            output += f"Examples: {', '.join(policy_data.get('examples', []))}\n\n"
        return output
    
    def _build_prompt(self, content: str, policy_context: str) -> str:
        """Construct the moderation prompt for Claude"""
        return f"""You are a content moderator. Evaluate the following content against the provided policies.

{policy_context}

## Content to Evaluate
{content}

## Your Task
1. Determine if this content violates any policies
2. Provide a decision: APPROVE, FLAG, QUARANTINE, REJECT, or ESCALATE
3. Explain your reasoning (chain of thought)
4. Rate confidence (0.0-1.0)

Respond ONLY with valid JSON matching this format:
{{
    "decision": "APPROVE" | "FLAG" | "QUARANTINE" | "REJECT" | "ESCALATE",
    "confidence": 0.95,
    "reasoning": "Step-by-step explanation of your decision",
    "matching_policy_ids": ["POL-001", "POL-002"],
    "risk_score": 0.8
}}"""
    
    def _parse_decision(self, response_text: str) -> dict:
        """Extract JSON from Claude's response"""
        try:
            # Try to find JSON block
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                decision_dict = json.loads(json_match.group())
            else:
                decision_dict = json.loads(response_text)
            
            # Validate
            return {
                "decision": decision_dict.get("decision", "ESCALATE"),
                "confidence": float(decision_dict.get("confidence", 0.5)),
                "reasoning": decision_dict.get("reasoning", ""),
                "matching_policy_ids": decision_dict.get("matching_policy_ids", []),
                "risk_score": float(decision_dict.get("risk_score", 0.5))
            }
        except Exception as e:
            logger.error(f"Failed to parse decision: {e}")
            return {
                "decision": "ESCALATE",
                "confidence": 0.0,
                "reasoning": f"Parsing error: {str(e)}"
            }
