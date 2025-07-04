"""Service to execute GraphQL-like read-only queries against the content Mongo database.

This is intentionally *minimal* – it only supports the following GraphQL pattern for now:

query { collectionName(limit: 10) { fieldA fieldB ... } }

It maps directly to: db.collectionName.find({}, { fieldA: 1, fieldB: 1, _id: 0 }).limit(10)

If no limit is supplied, a default of 20 is used (capped at 100).

Future extensions can swap this naïve translator with a full GraphQL engine or aggregation-pipeline builder.
"""

from __future__ import annotations

import re
import logging
from typing import Any, List, Dict

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pydantic import BaseModel, Field, ValidationError

from config.settings import get_settings

logger = logging.getLogger(__name__)


class ParsedQuery(BaseModel):
    collection: str
    projection: List[str] = Field(default_factory=list)
    limit: int = 20


class DataQueryService:
    """Connects to the secondary Mongo instance and runs read-only queries."""

    _client: AsyncIOMotorClient | None = None
    _db: AsyncIOMotorDatabase | None = None

    def __init__(self):
        settings = get_settings()
        self._uri = settings.data_database.url
        self._db_name = settings.data_database.database
        self._min_pool = settings.data_database.min_connections
        self._max_pool = settings.data_database.max_connections

    async def _get_db(self) -> AsyncIOMotorDatabase:
        if self._db is None:
            logger.info(f"🔌 Connecting to content MongoDB @ {self._uri}")
            self._client = AsyncIOMotorClient(self._uri, minPoolSize=self._min_pool, maxPoolSize=self._max_pool)
            self._db = self._client[self._db_name]
        return self._db

    # ----------------------------- Parsing ---------------------------------

    # Match queries that look like:
    #   query { collectionName(limit: 10) { fieldA fieldB } }
    # The pattern is intentionally *simple* – it does **not** support nested
    # selections or aliases.  It extracts the collection name, an optional
    # numeric limit, and the list of space-separated field names.
    _PATTERN = re.compile(
        r"query\s*"                 # literal 'query' keyword
        r"\{\s*"                  # opening brace
        r"(?P<col>\w+)"           # collection name
        r"\s*"                    # optional whitespace
        r"(?:"                     # --- optional '(limit: N)' ---
            r"\(\s*limit\s*:\s*(?P<limit>\d+)\s*\)"  # capture limit
        r")?"                      # --- end optional ---
        r"\s*"                    # optional whitespace
        r"\{\s*"                 # opening brace for field list
        r"(?P<fields>[^}]+?)"      # everything up to the next '}' (non-greedy)
        r"\s*\}\s*"             # closing brace for field list
        r"\}\s*"                 # closing brace for entire query
        , re.S
    )

    @classmethod
    def _parse_graphql(cls, graphql: str) -> ParsedQuery:
        match = cls._PATTERN.search(graphql)
        if not match:
            raise ValueError("Unsupported query pattern")
        collection = match.group("col")
        limit = int(match.group("limit") or 20)
        limit = max(1, min(limit, 100))
        raw_fields = match.group("fields")
        projection = [f.strip() for f in raw_fields.split() if f.strip()]
        if not projection:
            raise ValueError("No fields requested")
        return ParsedQuery(collection=collection, projection=projection, limit=limit)

    # ------------------------------ Public ---------------------------------

    async def run_query(self, graphql_query: str) -> List[Dict[str, Any]]:
        """Execute GraphQL string and return list of documents (dicts)."""
        try:
            parsed = self._parse_graphql(graphql_query)
        except (ValidationError, ValueError) as exc:
            logger.error(f"Failed to parse GraphQL: {exc}")
            raise

        db = await self._get_db()
        coll = db[parsed.collection]
        projection = {field: 1 for field in parsed.projection}
        cursor = coll.find({}, projection).limit(parsed.limit)
        docs: List[Dict[str, Any]] = []
        async for doc in cursor:
            # Remove _id for cleanliness
            doc.pop("_id", None)
            docs.append(doc)
        return docs


# Singleton helper
_data_query_service: DataQueryService | None = None


def get_data_query_service() -> DataQueryService:
    global _data_query_service
    if _data_query_service is None:
        _data_query_service = DataQueryService()
    return _data_query_service 