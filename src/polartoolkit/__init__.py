"""Helpful tools for polar researchers"""

import logging

from ._version import version as __version__

__all__ = ["__version__"]


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

logger.addHandler(logging.NullHandler())
