"""Compatibility import for the manifest adapter moved in Architecture v2.

New code should import :mod:`crowdtensor.adapters.manifests`.  Keeping this
small forwarding module avoids breaking older local manifests while removing
the former implementation from the package root.
"""

from .adapters.manifests import *  # noqa: F401,F403
