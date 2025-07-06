"""GraphQL endpoint backed by the content Mongo instance.

The schema is generated dynamically from sample documents in each collection so
it automatically reflects the structure seeded by `seed_content.js`.

Supports:
  • top-level query field per collection in the form:
      collectionName(limit: Int = 20, filter: JSON): [CollectionType]
  • simple JSON scalar filter which is passed straight to Mongo `find`.
  • optional `limit` (default 20, max 100).

More complex mutations/subscriptions are out of scope for now but can be added
using the same pattern.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter
from fastapi import Request
from starlette.responses import JSONResponse

from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore
from bson import ObjectId  # type: ignore
from ariadne import make_executable_schema, gql, QueryType, ScalarType
from ariadne.asgi import GraphQL

from config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["GraphQL"], prefix="/api")

# ---------------------------------------------------------------------------
# Dynamic SDL generation
# ---------------------------------------------------------------------------

_BASIC_TYPE_MAP = {
    str: "String",
    int: "Int",
    float: "Float",
    bool: "Boolean",
}

JSON_SCALAR = ScalarType("JSON")

@JSON_SCALAR.serializer
def _serialize_json(value):
    return value

@JSON_SCALAR.value_parser
def _parse_json(value):
    return value

async def _derive_schema_and_resolvers() -> Tuple[str, Dict[str, Any]]:
    """Return SDL and resolver map built from Mongo collections."""
    settings = get_settings()
    client = AsyncIOMotorClient(settings.data_database.url, serverSelectionTimeoutMS=5000)  # type: ignore[arg-type]
    db = client[settings.data_database.database]

    collection_names = await db.list_collection_names()

    typedefs: List[str] = ["scalar JSON"]
    query_fields: List[str] = []
    query_resolver = QueryType()

    async def _build_type_and_resolver(coll_name: str):
        sample = await db[coll_name].find_one()
        if not sample:
            return
        fields = []
        for k, v in sample.items():
            if k == "_id":
                continue
            gql_type = _BASIC_TYPE_MAP.get(type(v))
            if gql_type is None:
                if isinstance(v, list):
                    elem_type = _BASIC_TYPE_MAP.get(type(v[0]), "JSON") if v else "JSON"
                    gql_type = f"[{elem_type}]"
                elif isinstance(v, dict):
                    gql_type = "JSON"
                elif isinstance(v, ObjectId):
                    gql_type = "ID"
                else:
                    gql_type = "JSON"
            fields.append(f"  {k}: {gql_type}")
        typedefs.append(f"type {coll_name} {{\n" + "\n".join(fields) + "\n}")
        query_fields.append(f"  {coll_name}(limit: Int = 20, filter: JSON): [{coll_name}]")

        # Attach resolver dynamically
        async def _resolver(_, info, limit: int = 20, filter: Dict[str, Any] | None = None):
            limit = max(1, min(limit, 100))
            filter = filter or {}
            cursor = db[coll_name].find(filter).limit(limit)
            result: List[Dict[str, Any]] = []
            async for doc in cursor:
                doc.pop("_id", None)
                result.append(doc)
            return result

        query_resolver.field(coll_name)(_resolver)

    # Build types and resolvers for each collection
    await asyncio.gather(*[_build_type_and_resolver(c) for c in collection_names])

    typedefs.append("type Query {\n" + "\n".join(query_fields) + "\n}")
    sdl = "\n\n".join(typedefs)
    schema = make_executable_schema(gql(sdl), [query_resolver, JSON_SCALAR])
    return schema, sdl

# ---------------------------------------------------------------------------
# Mount ASGI GraphQL app at /api/graphql
# ---------------------------------------------------------------------------

_schema_ready: asyncio.Event | None = None
_gql_app: GraphQL | None = None
_sdl_cache: str = ""

async def _init_schema():
    global _schema_ready, _gql_app, _sdl_cache
    if _schema_ready is not None:
        await _schema_ready.wait()
        return

    _schema_ready = asyncio.Event()
    try:
        schema, sdl = await _derive_schema_and_resolvers()
        _gql_app = GraphQL(schema, debug=get_settings().api.debug)
        _sdl_cache = sdl
        logger.info("✅ Mongo GraphQL schema initialised (%d chars)", len(sdl))
    finally:
        _schema_ready.set()

@router.on_event("startup")
async def _startup():
    asyncio.create_task(_init_schema())

@router.api_route("/graphql", methods=["GET", "POST", "OPTIONS"])
async def graphql_http(request: Request):
    await _init_schema()
    if _gql_app is None:  # pragma: no cover
        return JSONResponse({"error": "Schema not ready"}, status_code=503)
    return await _gql_app.handle_request(request.scope, request.receive, request.send)

@router.get("/graphql/schema")
async def get_sdl():
    await _init_schema()
    return JSONResponse({"sdl": _sdl_cache}) 