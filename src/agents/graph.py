from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from src.agents.retrieval import RetrievalAgent
from src.agents.judge import JudgeAgent
from src.agents.audit import AuditAgent
from src.agents.enforce import EnforcementAgent
import logging

logger = logging.getLogger(__name__)

class ModeratorState(TypedDict):
    """Shared state across agents"""
    request_id: str
    content: str
    content_type: str
    user_id: str
    
    # Results from each agent
    extraction: dict  # HF task results
    retrieval_result: dict  # Policy matches
    judge_decision: dict  # Decision + reasoning
    audit_log: dict  # Tracing
    enforcement_action: dict  # Final action

class ModeratorGraph:
    """LangGraph orchestration"""
    
    def __init__(self, db, policy_lib):
        self.db = db
        self.policy_lib = policy_lib
        
        # Initialize agents
        self.retrieval = RetrievalAgent(db, policy_lib)
        self.judge = JudgeAgent()
        self.audit = AuditAgent(db)
        self.enforce = EnforcementAgent()
        
        # Build graph
        self.workflow = StateGraph(ModeratorState)
        self._build_graph()
    
    def _build_graph(self):
        """Define agent nodes and edges"""
        
        # Node 1: Retrieval
        self.workflow.add_node(
            "retrieval",
            lambda state: self._retrieval_node(state)
        )
        
        # Node 2: Judge
        self.workflow.add_node(
            "judge",
            lambda state: self._judge_node(state)
        )
        
        # Node 3: Audit
        self.workflow.add_node(
            "audit",
            lambda state: self._audit_node(state)
        )
        
        # Node 4: Enforce
        self.workflow.add_node(
            "enforce",
            lambda state: self._enforce_node(state)
        )
        
        # Graph edges
        self.workflow.set_entry_point("retrieval")
        self.workflow.add_edge("retrieval", "judge")
        self.workflow.add_edge("judge", "audit")
        self.workflow.add_edge("audit", "enforce")
        self.workflow.add_edge("enforce", END)
        
        self.graph = self.workflow.compile()
        logger.info("✓ LangGraph compiled")
    
    def _retrieval_node(self, state: ModeratorState) -> ModeratorState:
        """Run retrieval agent"""
        logger.info(f"[Retrieval] Processing {state['request_id']}")
        result = self.retrieval.run(state["content"])
        state["retrieval_result"] = result
        return state
    
    def _judge_node(self, state: ModeratorState) -> ModeratorState:
        """Run judge agent"""
        logger.info(f"[Judge] Making decision for {state['request_id']}")
        result = self.judge.run(
            content=state["content"],
            policies=state["retrieval_result"]["policy_texts"]
        )
        state["judge_decision"] = result
        return state
    
    def _audit_node(self, state: ModeratorState) -> ModeratorState:
        """Run audit agent"""
        logger.info(f"[Audit] Logging {state['request_id']}")
        result = self.audit.run(state)
        state["audit_log"] = result
        return state
    
    def _enforce_node(self, state: ModeratorState) -> ModeratorState:
        """Run enforcement agent"""
        logger.info(f"[Enforce] Taking action for {state['request_id']}")
        result = self.enforce.run(state["judge_decision"])
        state["enforcement_action"] = result
        return state
    
    def run(self, request: dict) -> dict:
        """Execute full moderation workflow"""
        initial_state = ModeratorState(
            request_id=request.get("request_id"),
            content=request.get("content"),
            content_type=request.get("content_type", "text"),
            user_id=request.get("user_id"),
            extraction={},
            retrieval_result={},
            judge_decision={},
            audit_log={},
            enforcement_action={}
        )
        
        final_state = self.graph.invoke(initial_state)
        return final_state
