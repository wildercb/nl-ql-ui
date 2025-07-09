"""
Alias package to forward `models` imports to `backend.models`.
"""
import importlib, sys
module = importlib.import_module('backend.models')
# Forward attributes
globals().update(module.__dict__)
# Register submodules
for name, submodule in list(sys.modules.items()):
    if name.startswith('backend.models.'):
        alias = name.replace('backend.models', 'models')
        sys.modules[alias] = submodule
# Ensure the alias resolves to backend.models
sys.modules[__name__] = module 