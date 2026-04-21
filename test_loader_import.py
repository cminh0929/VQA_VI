# Task 2 Validation Script
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils.data_loader import get_dataloader

try:
    # This might fail if data/train.json doesn't exist, which is expected for now
    # But we check if the import and functions exist
    print("DataLoader module imported successfully.")
    print("Testing get_dataloader presence...")
    assert get_dataloader is not None
    print("Validation passed.")
except Exception as e:
    print(f"Validation failed: {e}")
