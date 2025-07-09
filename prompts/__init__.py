"""
Alias package to forward `prompts` imports to `backend.prompts`.
"""

import importlib, sys
module = importlib.import_module('backend.prompts')
sys.modules[__name__] = module 