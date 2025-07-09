"""Alias package to forward `mcp_server` imports to `backend.mcp_server`."""
import importlib, sys
module = importlib.import_module('backend.mcp_server')
globals().update(module.__dict__)
for name, submodule in list(sys.modules.items()):
    if name.startswith('backend.mcp_server.'):
        alias = name.replace('backend.mcp_server', 'mcp_server')
        sys.modules[alias] = submodule
sys.modules[__name__] = module 