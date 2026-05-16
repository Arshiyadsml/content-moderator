from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AuditAgent:
    """Log all decisions for compliance & debugging"""
    
    def __init__(self, db):
        self.db = db
    
    def run(self, state: dict) -> dict:
        """Record decision to database"""
        
        try:
            self.db.log_decision(
                request_id=state["request_id"],
                user_id=state["user_id"],
                decision=state["judge_decision"].get("decision"),
                confidence=state["judge_decision"].get("confidence"),
                reasoning=state["judge_decision"].get("reasoning"),
                policies=state["judge_decision"].get("matching_policy_ids", []),
                trace={
                    "retrieval": state["retrieval_result"],
                    "judge_decision": state["judge_decision"]
                }
            )
            
            logger.info(f"Audit logged for {state['request_id']}")
            
            return {
                "logged": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
            return {"logged": False, "error": str(e)}
