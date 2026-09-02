"""Mock heavy deps so unit tests run without full install."""
import sys
import types
import unittest.mock as mock

# Mock litellm
if "litellm" not in sys.modules:
    sys.modules["litellm"] = mock.MagicMock()

# Mock backoff already installed, but ensure present
# Mock docker already installed
# Mock OpenAI provider specific
for mod in ["openai", "anthropic", "google.genai"]:
    if mod not in sys.modules:
        sys.modules[mod] = mock.MagicMock()

# Mock dotenv
if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = mock.MagicMock()

# Mock tqdm
if "tqdm" not in sys.modules:
    sys.modules["tqdm"] = mock.MagicMock()

# Mock datasets
if "datasets" not in sys.modules:
    sys.modules["datasets"] = mock.MagicMock()

# Mock git for utils.git_utils — already installed, keep real
# Mock pandas — mock if not installed
try:
    import pandas  # noqa: F401
except Exception:
    sys.modules["pandas"] = mock.MagicMock()

# Mock matplotlib
try:
    import matplotlib  # noqa: F401
except Exception:
    sys.modules["matplotlib"] = mock.MagicMock()
    sys.modules["matplotlib.pyplot"] = mock.MagicMock()

# Mock hydra
if "hydra" not in sys.modules:
    sys.modules["hydra"] = mock.MagicMock()
