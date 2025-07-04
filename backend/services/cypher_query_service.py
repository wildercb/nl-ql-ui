"""Service that converts limited GraphQL queries into Cypher and executes them against Neo4j."""

from __future__ import annotations

import logging
import re
from typing import List, Dict, Any

from neo4j import AsyncGraphDatabase, Neo4jDriver, Result
from pydantic import BaseModel, Field
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ParsedQuery(BaseModel):
    label: str
    fields: List[str]
    limit: int = 20


class CypherQueryService:
    """Very naive GraphQL→Cypher translator and executor."""

    _driver: Neo4jDriver | None = None

    _PATTERN = re.compile(
        r"query\s*"    # keyword
        r"{\s*"        # open brace
        r"(?P<label>\w+)"  # node label
        r"(?:\(\s*limit\s*:\s*(?P<limit>\d+)\s*\))?"  # optional limit argument
        r"\s*{\s*"   # open fields brace
        r"(?P<fields>[^}]+?)"  # field list
        r"\s*}\s*}"  # close braces
        , re.S
    )

    def __init__(self):
        settings = get_settings()
        self._uri = settings.neo4j.uri
        self._user = settings.neo4j.user
        self._password = settings.neo4j.password

    def _ensure_driver(self):
        if self._driver is None:
            logger.info(f"🔌 Connecting to Neo4j @ {self._uri}")
            self._driver = AsyncGraphDatabase.driver(self._uri, auth=(self._user, self._password))

    @classmethod
    def _parse_graphql(cls, graphql: str) -> ParsedQuery:
        m = cls._PATTERN.search(graphql)
        if not m:
            raise ValueError("Unsupported GraphQL pattern for Cypher translation")
        label = m.group("label")
        limit = int(m.group("limit") or 20)
        limit = max(1, min(limit, 100))
        fields = [f.strip() for f in m.group("fields").split() if f.strip()]
        return ParsedQuery(label=label, fields=fields, limit=limit)

    async def run_query(self, graphql_query: str) -> List[Dict[str, Any]]:
        parsed = self._parse_graphql(graphql_query)
        self._ensure_driver()

        # Build cypher
        proj = ", ".join([f"n.{f} as {f}" for f in parsed.fields])
        cypher = (
            f"MATCH (n:{parsed.label}) "
            f"RETURN {proj} "
            f"LIMIT {parsed.limit}"
        )
        logger.debug(f"Cypher query built: {cypher}")

        async with self._driver.session() as session:
            result: Result = await session.run(cypher)
            records = await result.data()
            return records

_cypher_service: CypherQueryService | None = None


def get_cypher_service() -> CypherQueryService:
    global _cypher_service
    if _cypher_service is None:
        _cypher_service = CypherQueryService()
    return _cypher_service 