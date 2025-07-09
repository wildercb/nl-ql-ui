"""
Alias package to forward `services` imports to `backend.services`.
"""

import importlib, sys
# Map backend.services.llm_tracking_service early
sys.modules['services.llm_tracking_service'] = importlib.import_module('backend.services.llm_tracking_service')

module = importlib.import_module('backend.services')
# Forward attributes
globals().update(module.__dict__)

# Register submodules
for name, submodule in list(sys.modules.items()):
    if name.startswith('backend.services.'):
        alias = name.replace('backend.services', 'services')
        sys.modules[alias] = submodule
sys.modules[__name__] = module 
importlib.import_module('backend.services.database_service') 
# Re-register in case new submodules were imported above
for name, submodule in list(sys.modules.items()):
    if name.startswith('backend.services.'):
        alias = name.replace('backend.services', 'services')
        sys.modules[alias] = submodule 