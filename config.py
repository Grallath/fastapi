# File: config.py
import os

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

OPENAI_API_KEY_SET = bool(os.getenv("OPENAI_API_KEY"))
