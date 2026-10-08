# ml_backend/ls_client.py
"""
Label Studio client singleton.

This module provides the Label Studio client instance.
Separated from ml_backend.py to avoid circular imports.
"""

import os
import logging
from label_studio_sdk.client import LabelStudio

logger = logging.getLogger("LSClient")

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_PERSONAL_TOKEN")

# Create singleton client instance
_ls_client = None


def get_ls_client() -> LabelStudio:
    """Get Label Studio client instance (singleton)."""
    global _ls_client
    
    if _ls_client is None:
        _ls_client = LabelStudio(
            base_url=LABEL_STUDIO_URL,
            api_key=LABEL_STUDIO_TOKEN
        )
        logger.info(f"Connected to Label Studio at: {LABEL_STUDIO_URL}")
    
    return _ls_client


# For backwards compatibility, also expose as `ls`
def _get_ls():
    return get_ls_client()


ls = property(lambda self: get_ls_client())
