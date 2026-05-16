from src.rag.retriever import RAGRetriever
import logging

logger = logging.getLogger(__name__)

class RetrievalAgent:
    """Find relevant policies via RAG"""
    
    def __init__(self, db, policy_lib):
        self.retriever = RAGRetriever(db, policy_lib)
    
    def run(self, content: str) -> dict:
        """Retrieve policies most relevant to content"""
        result = self.retriever.retrieve(content, top_k=5)
        return result
