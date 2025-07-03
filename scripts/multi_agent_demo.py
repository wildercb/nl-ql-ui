#!/usr/bin/env python3
"""Quick demo for the multi-agent pipeline.

Run with:
    python scripts/multi_agent_demo.py

Make sure Ollama (or your chosen provider) is running and the required models
are pulled. Adjust model names at the bottom if needed.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH when run directly
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))

from backend.services.agent_orchestration_service import AgentOrchestrationService


async def main() -> None:
    query = (
        os.environ.get("DEMO_QUERY")
        or "Which authors published blog posts about GraphQL in the last two weeks?"
    )

    service = AgentOrchestrationService()
    result = await service.process_query(
        query,
        pre_model=os.environ.get("PRE_MODEL", "llama2"),
        translator_model=os.environ.get("TRANSLATOR_MODEL", "llama2"),
        review_model=os.environ.get("REVIEW_MODEL", "llama2"),
    )

    print("\n=== Multi-Agent Result ===\n")
    print(result.to_json())


if __name__ == "__main__":
    asyncio.run(main()) 