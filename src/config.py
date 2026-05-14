import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set in .env")

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://localhost/moderator"
)

# LLM Settings
MODEL_NAME = "claude-3-5-sonnet-20241022"
MAX_TOKENS = 1024
TEMPERATURE = 0.2

# Moderation
DECISION_TIMEOUT_SECONDS = 5
BATCH_SIZE = 32

# Evaluation
EVAL_SAMPLE_SIZE = 100
EVAL_METRICS = ["precision", "recall", "f1"]

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = ENVIRONMENT == "development"
