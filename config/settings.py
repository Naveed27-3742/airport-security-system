from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"
SAMPLES_DIR = BASE_DIR / "data" / "samples"

MODEL_DIR = BASE_DIR / "models"

MODEL_NAME = os.getenv("MODEL_NAME", "yolo11n.pt")

MODEL_PATH = MODEL_DIR / MODEL_NAME

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

DATABASE_PATH = os.getenv("DATABASE_PATH",str(BASE_DIR / "db" / "events.db"))

MAX_FRAMES = int(os.getenv("MAX_FRAMES", "0"))



