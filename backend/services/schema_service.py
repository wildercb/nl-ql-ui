"""GraphQL Schema Service

Fetches and caches the GraphQL SDL from the Neo4j GraphQL endpoint (or any
GraphQL server).  Agents can include this SDL in their prompts to guarantee
queries match the current schema.

The service performs a standard GraphQL introspection query, converts the
result into SDL using `graphql-core` if available, and keeps an in-memory
cache with a configurable TTL so we don't hammer the endpoint.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, Tuple

import httpx  # type: ignore
from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore
from bson import ObjectId  # type: ignore

try:
    # Optional – prettier SDL conversion if graphql-core is installed
    from graphql import get_introspection_query, build_client_schema, print_schema  # type: ignore
    _HAS_GRAPHQL_CORE = True
except ModuleNotFoundError:  # pragma: no cover – optional dependency
    _HAS_GRAPHQL_CORE = False

from config.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_SCHEMAS: Dict[str, Tuple[float, str]] = {}
_DEFAULT_TTL = 60 * 30  # 30 minutes

_INTROSPECTION_QUERY = (
    get_introspection_query(descriptions=True) if _HAS_GRAPHQL_CORE else
    """query IntrospectionQuery { __schema { types { name kind description fields {\n      name description args { name description type { kind ofType { kind name } } }\n      type { kind name ofType { kind name } }\n    } } } }"""
)

# ---------------------------------------------------------------------------
# Mongo Fallback
# ---------------------------------------------------------------------------

_BASIC_TYPE_MAP = {
    str: 'String',
    int: 'Int',
    float: 'Float',
    bool: 'Boolean'
}

async def _derive_sdl_from_mongo(uri: str, db_name: str, sample_size: int = 1) -> str:
    """Derive a very simple GraphQL SDL from Mongo collections.

    This inspects one sample document per collection and maps field types.
    Complex / nested structures default to JSON scalar for brevity.
    """
    client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
    db = client[db_name]
    collection_names = await db.list_collection_names()

    type_defs = ["scalar JSON"]
    query_fields: list[str] = []

    for coll_name in collection_names:
        coll = db[coll_name]
        doc = await coll.find_one()
        if not doc:
            continue
        fields = []
        for k, v in doc.items():
            if k == "_id":
                continue
            gql_type = _BASIC_TYPE_MAP.get(type(v))
            if gql_type is None:
                # Simple heuristics for lists
                if isinstance(v, list):
                    elem_type = _BASIC_TYPE_MAP.get(type(v[0]), 'JSON') if v else 'JSON'
                    gql_type = f'[{elem_type}]'
                elif isinstance(v, dict):
                    gql_type = 'JSON'
                elif isinstance(v, ObjectId):
                    gql_type = 'ID'
                else:
                    gql_type = 'JSON'
            fields.append(f"  {k}: {gql_type}")
        type_defs.append(f"type {coll_name} {{\n" + "\n".join(fields) + "\n}")
        query_fields.append(f"  {coll_name}(limit: Int): [{coll_name}]")

    query_type = "type Query {\n" + "\n".join(query_fields) + "\n}"
    type_defs.append(query_type)

    return "\n\n".join(type_defs)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_schema_sdl(ttl: int = _DEFAULT_TTL) -> str:
    """Return the current GraphQL schema SDL as a string (cached)."""
    settings = get_settings()
    url = settings.neo4j.graphql_endpoint.rstrip('/')

    # Return cached copy if fresh
    now = time.time()
    if url in _SCHEMAS and now - _SCHEMAS[url][0] < ttl:
        return _SCHEMAS[url][1]

    logger.info("🌐 Fetching GraphQL schema via introspection @ %s", url)
    
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, json={"query": _INTROSPECTION_QUERY})
            r.raise_for_status()
            data = r.json().get("data")
            if not data:
                raise RuntimeError("No introspection data returned from endpoint")

        if _HAS_GRAPHQL_CORE:
            # Convert to SDL
            schema = build_client_schema(data)
            sdl: str = print_schema(schema)
        else:
            # Fallback – store the JSON as a string (LLMs can still read it)
            sdl = json.dumps(data, indent=2)

        _SCHEMAS[url] = (now, sdl)
        logger.info("✅ GraphQL SDL cached (%d chars)", len(sdl))
        return sdl
    except Exception as e:
        logger.error("❌ Error fetching GraphQL schema: %s", e)
        s = get_settings()
        return await _derive_sdl_from_mongo(s.data_database.url, s.data_database.database)

# Synchronous helper (rarely used)
def get_schema_sdl_sync(ttl: int = _DEFAULT_TTL) -> str:
    return asyncio.get_event_loop().run_until_complete(get_schema_sdl(ttl)) 