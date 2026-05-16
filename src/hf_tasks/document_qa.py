from langchain.document_loaders import PyPDFLoader, UnstructuredDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class DocumentQATask:
    """
    Extract claims from PDFs/docs using HF Document QA task.
    (In production, use LLaVA or Claude's document API, but this is the HF way)
    """
    
    def __init__(self):
        # HF pipeline for document Q&A
        self.qa_pipeline = pipeline(
            "document-question-answering",
            model="impira/layoutlm-document-qa"  # Or any HF QA model
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
    
    def load_document(self, file_path: str) -> list:
        """Load PDF/docx and chunk"""
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = UnstructuredDocumentLoader(file_path)
            
            docs = loader.load()
            chunks = self.text_splitter.split_documents(docs)
            logger.info(f"Loaded {len(chunks)} chunks from {file_path}")
            return chunks
        except Exception as e:
            logger.error(f"Document load failed: {e}")
            return []
    
    def extract_claims(self, document_text: str, policy_keywords: list) -> dict:
        """
        Ask targeted questions to extract policy-relevant claims.
        E.g., "Does this document make health claims?" "Does it promote violence?"
        """
        questions = [
            "What are the main claims made?",
            "Who is the intended audience?",
            "Are there any warnings or disclaimers?",
        ]
        
        results = {}
        for question in questions:
            try:
                answer = self.qa_pipeline(question, document_text)
                results[question] = answer
            except Exception as e:
                logger.warning(f"QA failed on '{question}': {e}")
        
        return results

# Singleton instance
document_qa = DocumentQATask()
