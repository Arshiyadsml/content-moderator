from transformers import pipeline, CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class ImageTextTask:
    """Multi-task image-text processing using HF models"""
    
    def __init__(self):
        # Image-to-Text (caption generation)
        self.image_to_text = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base"
        )
        
        # VQA (visual question answering)
        self.vqa_pipeline = pipeline(
            "visual-question-answering",
            model="dandelin/vilt-b32-finetuned-vqa"
        )
        
        # CLIP for semantic similarity (image-text matching)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def load_image(self, image_source: str) -> Image.Image:
        """Load image from URL or local path"""
        if image_source.startswith("http"):
            response = requests.get(image_source)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_source)
        return image
    
    def image_to_caption(self, image: Image.Image) -> str:
        """Generate text caption from image"""
        try:
            caption = self.image_to_text(image)
            return caption[0]["generated_text"]
        except Exception as e:
            logger.error(f"Image-to-text failed: {e}")
            return ""
    
    def visual_qa(self, image: Image.Image, question: str) -> str:
        """Ask questions about image content"""
        try:
            answer = self.vqa_pipeline(image, question)
            return answer[0]["answer"]
        except Exception as e:
            logger.error(f"VQA failed: {e}")
            return ""
    
    def policy_qa_batch(self, image: Image.Image, policy_questions: list) -> dict:
        """Ask multiple policy-relevant questions"""
        results = {}
        for q in policy_questions:
            results[q] = self.visual_qa(image, q)
        return results

# Singleton
image_text_task = ImageTextTask()
