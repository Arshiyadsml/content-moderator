from src.rag.policy_library import PolicyLibrary
import logging

logger = logging.getLogger(__name__)

class RAGRetriever:
    """Semantic search over policy library"""
    
    def __init__(self, db, policy_library: PolicyLibrary):
        self.db = db
        self.policy_lib = policy_library
    
    def retrieve(self, query: str, top_k: int = 5) -> dict:
        """
        Find relevant policies for a content moderation query.
        
        Example query:
          "User posted a video showing violence against minorities"
        
        Should retrieve:
          - HAR-001 (violent content)
          - HAR-002 (hate speech)
        """
        matching_policies = self.policy_lib.search_policies(query, top_k)
        
        policy_dict = {}
        for policy_id, title, description, category, severity, similarity in matching_policies:
            policy_dict[policy_id] = {
                "title": title,
                "description": description,
                "category": category,
                "severity": severity,
                "relevance_score": similarity
            }
        
        logger.info(f"Retrieved {len(policy_dict)} policies for query: {query[:50]}...")
        
        return {
            "matching_policies": list(policy_dict.keys()),
            "policy_texts": policy_dict,
            "search_query": query
        }
