"""
Neo4j Schema Service

This service provides functionality to fetch and generate GraphQL schema
from Neo4j database for use in translation and review agents.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from neo4j import AsyncGraphDatabase
from config.settings import get_settings

logger = logging.getLogger(__name__)


class Neo4jSchemaService:
    """Service for fetching and generating GraphQL schema from Neo4j."""
    
    def __init__(self):
        self.settings = get_settings()
        self.driver = None
        self._schema_cache = None
        self._cache_timestamp = 0
        self._cache_ttl = 300  # 5 minutes cache
    
    async def get_driver(self):
        """Get Neo4j driver instance."""
        if self.driver is None:
            try:
                self.driver = AsyncGraphDatabase.driver(
                    self.settings.neo4j.uri,
                    auth=(self.settings.neo4j.username, self.settings.neo4j.password)
                )
                logger.info("✅ Neo4j driver initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Neo4j driver: {e}")
                return None
        return self.driver
    
    async def close(self):
        """Close Neo4j driver connection."""
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("🔌 Neo4j driver closed")
    
    async def get_graphql_schema(self, force_refresh: bool = False) -> Optional[str]:
        """
        Get GraphQL schema from Neo4j database.
        
        Args:
            force_refresh: Force refresh the cached schema
            
        Returns:
            GraphQL schema as string or None if unavailable
        """
        import time
        
        # Check cache first
        current_time = time.time()
        if not force_refresh and self._schema_cache and (current_time - self._cache_timestamp) < self._cache_ttl:
            logger.info("📋 Using cached Neo4j schema")
            return self._schema_cache
        
        try:
            driver = await self.get_driver()
            if not driver:
                logger.warning("⚠️ Neo4j driver not available, using fallback schema")
                return self._get_fallback_schema()
            
            # Fetch schema from Neo4j
            schema = await self._fetch_neo4j_schema(driver)
            if schema:
                self._schema_cache = schema
                self._cache_timestamp = current_time
                logger.info("✅ Neo4j schema fetched and cached")
                return schema
            else:
                logger.warning("⚠️ Could not fetch Neo4j schema, using fallback")
                return self._get_fallback_schema()
                
        except Exception as e:
            logger.error(f"❌ Error fetching Neo4j schema: {e}")
            return self._get_fallback_schema()
    
    async def _fetch_neo4j_schema(self, driver) -> Optional[str]:
        """Fetch schema from Neo4j using Cypher queries."""
        try:
            async with driver.session() as session:
                # Get all node labels
                node_labels = await self._get_node_labels(session)
                
                # Get all relationship types
                relationship_types = await self._get_relationship_types(session)
                
                # Get property schemas for each label
                property_schemas = await self._get_property_schemas(session, node_labels)
                
                # Generate GraphQL schema
                schema = self._generate_graphql_schema(node_labels, relationship_types, property_schemas)
                
                return schema
                
        except Exception as e:
            logger.error(f"❌ Error in Neo4j schema fetch: {e}")
            return None
    
    async def _get_node_labels(self, session) -> List[str]:
        """Get all node labels from Neo4j."""
        try:
            result = await session.run("CALL db.labels() YIELD label RETURN label")
            labels = []
            async for record in result:
                labels.append(record["label"])
            return labels
        except Exception as e:
            logger.error(f"❌ Error getting node labels: {e}")
            return []
    
    async def _get_relationship_types(self, session) -> List[str]:
        """Get all relationship types from Neo4j."""
        try:
            result = await session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
            types = []
            async for record in result:
                types.append(record["relationshipType"])
            return types
        except Exception as e:
            logger.error(f"❌ Error getting relationship types: {e}")
            return []
    
    async def _get_property_schemas(self, session, labels: List[str]) -> Dict[str, Dict[str, str]]:
        """Get property schemas for each node label."""
        schemas = {}
        
        for label in labels:
            try:
                # Get sample nodes for this label
                query = f"""
                MATCH (n:{label})
                RETURN n
                LIMIT 10
                """
                result = await session.run(query)
                
                properties = {}
                async for record in result:
                    node = record["n"]
                    for key, value in node.items():
                        if key not in properties:
                            properties[key] = self._infer_graphql_type(value)
                
                schemas[label] = properties
                
            except Exception as e:
                logger.error(f"❌ Error getting properties for label {label}: {e}")
                schemas[label] = {}
        
        return schemas
    
    def _infer_graphql_type(self, value) -> str:
        """Infer GraphQL type from Python value."""
        if value is None:
            return "String"
        elif isinstance(value, bool):
            return "Boolean"
        elif isinstance(value, int):
            return "Int"
        elif isinstance(value, float):
            return "Float"
        elif isinstance(value, str):
            return "String"
        elif isinstance(value, list):
            if value:
                element_type = self._infer_graphql_type(value[0])
                return f"[{element_type}]"
            else:
                return "[String]"
        elif isinstance(value, dict):
            return "JSON"
        else:
            return "String"
    
    def _generate_graphql_schema(self, labels: List[str], relationships: List[str], properties: Dict[str, Dict[str, str]]) -> str:
        """Generate GraphQL schema from Neo4j metadata."""
        schema_parts = []
        
        # Add scalar definitions
        schema_parts.append("scalar JSON")
        schema_parts.append("scalar DateTime")
        schema_parts.append("")
        
        # Generate types for each label
        for label in labels:
            type_name = self._to_pascal_case(label)
            props = properties.get(label, {})
            
            if not props:
                # Default properties if none found
                props = {
                    "id": "ID",
                    "name": "String",
                    "created_at": "DateTime"
                }
            
            fields = []
            for prop_name, prop_type in props.items():
                graphql_name = self._to_camel_case(prop_name)
                fields.append(f"  {graphql_name}: {prop_type}")
            
            # Add relationships
            for rel_type in relationships:
                rel_name = self._to_camel_case(rel_type)
                fields.append(f"  {rel_name}: [{type_name}]")
            
            type_def = f"type {type_name} {{\n" + "\n".join(fields) + "\n}}"
            schema_parts.append(type_def)
            schema_parts.append("")
        
        # Generate Query type
        query_fields = []
        for label in labels:
            type_name = self._to_pascal_case(label)
            query_name = self._to_camel_case(label)
            query_fields.append(f"  {query_name}(limit: Int = 20, filter: JSON): [{type_name}]")
            query_fields.append(f"  {query_name}ById(id: ID!): {type_name}")
        
        query_type = "type Query {\n" + "\n".join(query_fields) + "\n}"
        schema_parts.append(query_type)
        
        return "\n".join(schema_parts)
    
    def _to_pascal_case(self, text: str) -> str:
        """Convert text to PascalCase."""
        return "".join(word.capitalize() for word in text.split("_"))
    
    def _to_camel_case(self, text: str) -> str:
        """Convert text to camelCase."""
        words = text.split("_")
        return words[0] + "".join(word.capitalize() for word in words[1:])
    
    def _get_fallback_schema(self) -> str:
        """Get fallback schema when Neo4j is not available."""
        return """
scalar JSON
scalar DateTime

type Query {
  thermalScans(limit: Int = 20, filter: JSON): [ThermalScan]
  thermalScansById(id: ID!): ThermalScan
  maintenanceLogs(limit: Int = 20, filter: JSON): [MaintenanceLog]
  maintenanceLogsById(id: ID!): MaintenanceLog
  pointClouds(limit: Int = 20, filter: JSON): [PointCloud]
  pointCloudsById(id: ID!): PointCloud
  machineAudioAlerts(limit: Int = 20, filter: JSON): [MachineAudioAlert]
  machineAudioAlertsById(id: ID!): MachineAudioAlert
}

type ThermalScan {
  id: ID
  scanId: String
  maxTemperature: Float
  temperatureReadings: [Float]
  hotspotCoordinates: [Int]
  dateTaken: DateTime
  locationDetails: String
  equipmentId: String
  processType: String
}

type MaintenanceLog {
  id: ID
  logId: String
  notes: String
  operatorId: String
  timestamp: DateTime
  equipmentId: String
  maintenanceType: String
  status: String
}

type PointCloud {
  id: ID
  cloudId: String
  deviationMap: JSON
  spatialRegion: JSON
  quality: JSON
  timestamp: DateTime
}

type MachineAudioAlert {
  id: ID
  alertId: String
  timestamp: DateTime
  frequencyAnalysis: JSON
  pattern: String
  machineType: String
  severity: String
}
"""
    
    async def test_connection(self) -> bool:
        """Test Neo4j connection."""
        try:
            driver = await self.get_driver()
            if not driver:
                return False
            
            async with driver.session() as session:
                result = await session.run("RETURN 1 as test")
                record = await result.single()
                return record["test"] == 1
                
        except Exception as e:
            logger.error(f"❌ Neo4j connection test failed: {e}")
            return False


# Global instance
_neo4j_schema_service: Optional[Neo4jSchemaService] = None


def get_neo4j_schema_service() -> Neo4jSchemaService:
    """Get global Neo4j schema service instance."""
    global _neo4j_schema_service
    if _neo4j_schema_service is None:
        _neo4j_schema_service = Neo4jSchemaService()
    return _neo4j_schema_service


async def get_neo4j_graphql_schema(force_refresh: bool = False) -> Optional[str]:
    """Convenience function to get Neo4j GraphQL schema."""
    service = get_neo4j_schema_service()
    return await service.get_graphql_schema(force_refresh) 