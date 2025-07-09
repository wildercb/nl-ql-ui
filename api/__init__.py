"""Alias package to forward `api` imports to `backend.api`."""
import importlib, sys

module = importlib.import_module("backend.api")
# Export backend.api symbols
globals().update(module.__dict__)

# Register submodules under api.* for any already-imported backend.api.* modules
for name, submodule in list(sys.modules.items()):
    if name.startswith("backend.api."):
        alias = name.replace("backend.api", "api")
        sys.modules[alias] = submodule

# Ensure `import api` returns the backend.api module
sys.modules[__name__] = module 