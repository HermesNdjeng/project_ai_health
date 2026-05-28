import os

# Prevent ChatOpenAI from raising at import time when no real key is set.
os.environ.setdefault("OPENAI_API_KEY", "test-key-placeholder")
