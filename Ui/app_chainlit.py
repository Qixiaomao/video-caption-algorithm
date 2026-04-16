#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compatibility Chainlit entry.

Prefer:
    chainlit run frontend/chainlit_app.py -w --host 127.0.0.1 --port 8000

This wrapper keeps the old command usable:
    chainlit run Ui/app_chainlit.py -w
"""

from frontend.chainlit_app import *  # noqa: F401,F403
