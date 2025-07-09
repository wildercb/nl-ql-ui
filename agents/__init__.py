"""
Alias package to forward `agents` imports to `backend.agents`.
"""

import importlib, sys
module = importlib.import_module('backend.agents')
sys.modules[__name__] = module 