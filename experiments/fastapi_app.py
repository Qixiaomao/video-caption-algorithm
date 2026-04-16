#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compatibility entrypoint for the legacy FastAPI prototype.

Prefer `uvicorn server.app:app --host 127.0.0.1 --port 8001 --reload` for the product v2 layout.
This module stays as a thin wrapper so existing commands can roll back
without losing their import target.
"""

from server.app import app


