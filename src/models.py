from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

# === Input/Output Models ===

class DecisionCategory(str, Enum):
    """Moderation decision categories"""
    APPROVE = "approve"
    FLAG = "flag"
    QUARANTINE = "quarantine"
    REJECT = "reject"
    ESCALATE = "escalate"

class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"
    MULTIMODAL = "multimodal"

# Input
class ModerationRequest(BaseModel):
    """User-facing API request"""
    content: str
    content_type: ContentType = ContentType.TEXT
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = Field(default_factory=lambda: "req_" + datetime.now().isoformat())

# Extracted data
class ExtractedContent(BaseModel):
    """Normalized content after HF processing"""
    text: str
    images: List[str] = []
    metadata: Dict[str, Any]
    extraction_method: str

# Agent states
class RetrievalResult(BaseModel):
    """Retrieval agent output"""
    matching_policies: List[str]
    policy_texts: Dict[str, str]
    retrieval_confidence: float
    search_query: str

class JudgeDecision(BaseModel):
    """LLM Judge output"""
    decision: DecisionCategory
    confidence: float
    reasoning: str
    matching_policy_ids: List[str]
    risk_score: float

class AuditLog(BaseModel):
    """Audit trail entry"""
    timestamp: datetime
    request_id: str
    user_id: Optional[str]
    decision: DecisionCategory
    confidence: float
    reasoning: str
    policies_matched: List[str]
    agent_trace: Dict[str, Any]

class ModerationResponse(BaseModel):
    """Final API response"""
    request_id: str
    decision: DecisionCategory
    confidence: float
    reasoning: str
    audit_id: str
    timestamp: datetime

# Evaluation
class EvalExample(BaseModel):
    """Ground-truth evaluation example"""
    id: str
    content: str
    content_type: ContentType
    ground_truth_label: DecisionCategory
    ground_truth_reasoning: str
    policy_category: str
    difficulty: str = "medium"

class EvalResult(BaseModel):
    """Per-example evaluation result"""
    example_id: str
    predicted_decision: DecisionCategory
    predicted_confidence: float
    ground_truth: DecisionCategory
    correct: bool
    match_confidence: float

class MetricsReport(BaseModel):
    """Aggregate evaluation metrics"""
    timestamp: datetime
    total_examples: int
    accuracy: float
    precision: Dict[DecisionCategory, float]
    recall: Dict[DecisionCategory, float]
    f1: Dict[DecisionCategory, float]
    macro_f1: float
    weighted_f1: float
