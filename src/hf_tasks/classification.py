from transformers import pipeline, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class ClassificationTask:
    """Text & token classification for flagging"""
    
    def __init__(self):
        # Toxicity detection
        self.toxicity = pipeline(
            "text-classification",
            model="michellejieli/NSFW_text_classifier"
        )
        
        # Token classification (NER-like, for PII/sensitive terms)
        self.token_classifier = pipeline(
            "token-classification",
            model="bert-base-cased"
        )
    
    def toxicity_score(self, text: str) -> dict:
        """Score text for toxicity"""
        try:
            result = self.toxicity(text[:512])  # Truncate to avoid memory issues
            return {
                "label": result[0]["label"],
                "score": result[0]["score"]
            }
        except Exception as e:
            logger.error(f"Toxicity scoring failed: {e}")
            return {"label": "unknown", "score": 0.0}
    
    def extract_sensitive_terms(self, text: str) -> list:
        """Identify sensitive entities (names, emails, etc.)"""
        try:
            tokens = self.token_classifier(text[:512])
            return [t for t in tokens if t["entity"] in ["B-PER", "B-ORG", "B-EMAIL"]]
        except Exception as e:
            logger.error(f"Token classification failed: {e}")
            return []

classification = ClassificationTask()
