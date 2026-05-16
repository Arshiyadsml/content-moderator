import json
import logging
from pathlib import Path
from anthropic import Anthropic
import numpy as np
from src.config import DATA_DIR

logger = logging.getLogger(__name__)

class PolicyLibrary:
    """
    Builds & manages policy embeddings for RAG.
    Policies are semantic rules about what content is allowed.
    """
    
    def __init__(self, db):
        self.db = db
        self.client = Anthropic()
        self.policies = self.load_seed_policies()
    
    def load_seed_policies(self) -> dict:
        """Load initial policy definitions from JSON"""
        policy_file = DATA_DIR / "policies.json"
        if not policy_file.exists():
            return self._create_default_policies()
        
        with open(policy_file) as f:
            return json.load(f)
    
    def _create_default_policies(self) -> dict:
        """Seed policies (customize for your domain)"""
        return {
            "HAR-001": {
                "title": "No violent content",
                "description": "Content depicting, glorifying, or inciting violence",
                "examples": [
                    "Graphic descriptions of physical harm",
                    "Calls to harm specific groups",
                    "Detailed harm instructions"
                ],
                "severity": 5,
                "category": "violence"
            },
            "HAR-002": {
                "title": "No harassment or hate speech",
                "description": "Content targeting individuals/groups with slurs or abuse",
                "examples": [
                    "Ethnic slurs or discriminatory language",
                    "Targeted abuse campaigns",
                    "Dehumanizing comparisons"
                ],
                "severity": 5,
                "category": "harassment"
            },
            "SPM-001": {
                "title": "No spam or manipulation",
                "description": "Repetitive, misleading, or manipulative content",
                "examples": [
                    "Engagement-baiting tactics",
                    "Duplicated content across accounts",
                    "Misleading URLs or phishing attempts"
                ],
                "severity": 2,
                "category": "spam"
            },
            "PRI-001": {
                "title": "No sharing private information",
                "description": "Posting PII without consent (doxing)",
                "examples": [
                    "Home addresses of individuals",
                    "Credit card numbers",
                    "Non-public medical records"
                ],
                "severity": 4,
                "category": "privacy"
            },
            "MIS-001": {
                "title": "No health misinformation",
                "description": "False medical claims that could cause harm",
                "examples": [
                    "Unproven cancer cures",
                    "Vaccine misinformation",
                    "Harmful supplement claims"
                ],
                "severity": 3,
                "category": "misinformation"
            }
        }
    
    def embed_policies(self):
        """
        Generate embeddings for all policies using Claude.
        This makes them searchable via semantic similarity.
        """
        for policy_id, policy_data in self.policies.items():
            # Combine title + description for embedding
            text = f"{policy_data['title']}. {policy_data['description']}"
            
            try:
                # Use Claude to embed (via API)
                embedding = self._embed_text(text)
                
                # Store in database
                self.db.insert_policy(
                    policy_id=policy_id,
                    title=policy_data["title"],
                    description=policy_data["description"],
                    embedding=embedding,
                    category=policy_data["category"],
                    severity=policy_data["severity"],
                    examples=policy_data["examples"]
                )
                logger.info(f"Embedded policy {policy_id}")
            except Exception as e:
                logger.error(f"Failed to embed {policy_id}: {e}")
    
    def _embed_text(self, text: str) -> np.ndarray:
        """
        Claude doesn't have native embeddings, but we can:
        1. Use OpenAI API for embeddings (if budget allows)
        2. Use open-source sentence-transformers
        3. Use simple TF-IDF (fast, not ML-based)
        
        For now, we'll use sentence-transformers (free, offline).
        """
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode(text)
        return embedding
    
    def search_policies(self, query: str, top_k: int = 5) -> list:
        """Find relevant policies via semantic search"""
        query_embedding = self._embed_text(query)
        results = self.db.search_policies(query_embedding, limit=top_k)
        return results
