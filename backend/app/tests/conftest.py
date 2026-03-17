import os
import sys
import pytest
from fastapi.testclient import TestClient

# Deterministic/offline test defaults
os.environ["LLM_PROVIDER"] = "stub"
os.environ.pop("OPENROUTER_API_KEY", None)
os.environ["VECTOR_STORE"] = "memory"

# Ensure package import works in container
sys.path.insert(0, "/app")

from app.main import app

@pytest.fixture(scope="session")
def client():
    return TestClient(app)