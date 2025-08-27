from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

HF_DATASET = "Dingdong-Inc/FreshRetailNet-50K"
SAMPLE_ROWS = 200_000