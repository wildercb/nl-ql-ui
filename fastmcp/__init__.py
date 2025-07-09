"""
Stub implementation of FastMCP for development/testing.
This replaces the external fastmcp dependency so the application can run.
"""

class FastMCP:
    def __init__(self, *args, **kwargs):
        pass
    
    async def run(self, *args, **kwargs):
        return {"status": "stub running"}

__all__ = ["FastMCP"] 