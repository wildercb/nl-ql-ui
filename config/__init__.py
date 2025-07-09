"""Alias package to forward `config` imports to `backend.config`."""
import importlib, sys
module = importlib.import_module('backend.config')
# Forward top-level attributes
globals().update(module.__dict__)

# Register backend.config.* submodules under the `config` namespace so that
# `import config.xxx` works even if the submodule was imported elsewhere first.
for name, submodule in list(sys.modules.items()):
    if name.startswith('backend.config.'):
        alias = name.replace('backend.config', 'config')
        sys.modules[alias] = submodule

# Ensure the alias package itself resolves to the backend.config module
sys.modules[__name__] = module 