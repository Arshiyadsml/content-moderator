import logging

logger = logging.getLogger(__name__)

class EnforcementAgent:
    """Take action based on decision"""
    
    def run(self, judge_decision: dict) -> dict:
        """Execute enforcement action"""
        
        decision = judge_decision.get("decision")
        confidence = judge_decision.get("confidence")
        
        # Decision → Action mapping
        action_map = {
            "APPROVE": {"action": "allow", "visible": True},
            "FLAG": {"action": "flag_user", "visible": True, "add_warning": True},
            "QUARANTINE": {"action": "remove", "visible": False, "notify_mod": True},
            "REJECT": {"action": "remove", "visible": False, "ban_user": False},
            "ESCALATE": {"action": "escalate", "visible": True, "human_review": True},
        }
        
        action = action_map.get(decision, action_map["ESCALATE"])
        
        logger.info(f"Enforcement action: {action['action']} (decision: {decision})")
        
        return {
            "action": action["action"],
            "decision": decision,
            "confidence": confidence,
            **action
        }
