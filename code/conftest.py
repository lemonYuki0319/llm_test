import json
import pytest
from config.settings import DEFAULT_MODEL
from utils.llm_client import LLMClient


@pytest.fixture(scope="session")
def llm_client():
    return LLMClient(model=DEFAULT_MODEL)


@pytest.fixture(scope="session")
def golden_set():
    with open("testdata/golden_set.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def security_payloads():
    with open("testdata/security_payloads.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def robustness_inputs():
    with open("testdata/robustness_inputs.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def cot_questions():
    with open("testdata/cot_questions.json", "r", encoding="utf-8") as f:
        return json.load(f)
