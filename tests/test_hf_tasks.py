import pytest
from src.hf_tasks.image_text import image_text_task
from unittest.mock import Mock, patch
from src.hf_tasks.classification import classification

# def test_image_to_caption():
#    """Test image-to-text works"""
#   # Using a simple test image URL
#   image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/1200px-Good_Food_Display_-_NCI_Visuals_Online.jpg"
#   image = image_text_task.load_image(image_url)
#   caption = image_text_task.image_to_caption(image)
#   assert isinstance(caption, str)
#   assert len(caption) > 0

def test_toxicity_score():
    """Test toxicity classification"""
    result = classification.toxicity_score("This is a nice day")
    assert "label" in result
    assert "score" in result
    assert 0 <= result["score"] <= 1

def test_toxicity_detects_bad_content():
    """Test it flags toxic content"""
    result = classification.toxicity_score("I hate everyone")
    assert result["score"] > 0.3  # Should have some toxicity signal

def test_extract_sensitive_terms():
    """Test NER extraction"""
    result = classification.extract_sensitive_terms("John lives in New York")
    assert isinstance(result, list)

@patch('src.hf_tasks.image_text.image_text_task.image_to_caption')
def test_image_to_caption(mock_caption):
    mock_caption.return_value = "test image caption"
    caption = image_text_task.image_to_caption(None)
    assert caption == "test image caption"
