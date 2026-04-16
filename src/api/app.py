"""Compatibility wrapper for the new server layer.

Prefer:
    uvicorn server.app:app --host 127.0.0.1 --port 8001 --reload
"""

from server.app import app
