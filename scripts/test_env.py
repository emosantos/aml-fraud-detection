# scripts/check_env.py
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import DB_CONFIG
import os
from dotenv import load_dotenv

# Force reload .env
load_dotenv(override=True)


print("\nDirect from .env file:")
print(f"DB_PASSWORD from os.getenv: '{os.getenv('DB_PASSWORD')}'")

print("\nFrom DB_CONFIG:")
print(f"Password: '{DB_CONFIG['password']}'")
print(f"Password length: {len(DB_CONFIG['password'])} characters")

print("\nExpected password: 'aml_password_123'")
print(f"Expected length: {len('aml_password_123')} characters")

print("\nDo they match?", DB_CONFIG['password'] == 'aml_password_123')

# Check for hidden characters
print("\nPassword bytes:", DB_CONFIG['password'].encode())
print("Expected bytes:", 'aml_password_123'.encode())