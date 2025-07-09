"""
Content Seed Service - Handles seeding of demo content data.

This service manages the seeding of demo content into MongoDB
for testing and demonstration purposes.
"""

import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime

# Use direct backend import to avoid aliasing issues during module initialization
from backend.services.database_service import get_database_service

logger = logging.getLogger(__name__)


class ContentSeedService:
    """Service for seeding demo content data."""
    
    def __init__(self):
        self.db_service = None
        logger.info("ContentSeedService initialized")
    
    async def _get_db_service(self):
        """Get database service instance."""
        if not self.db_service:
            self.db_service = await get_database_service()
        return self.db_service
    
    async def seed_once(self, force: bool = False) -> int:
        """
        Seed demo data once.
        
        Args:
            force: If True, force re-seeding even if data exists
            
        Returns:
            Number of documents inserted
        """
        try:
            db_service = await self._get_db_service()
            db = db_service.database
            
            inserted_count = 0
            
            # Check if content database exists
            content_db = db_service.client["mppw_content"]
            
            # Seed maintenance logs
            if force or await content_db.maintenanceLogs.count_documents({}) == 0:
                maintenance_logs = [
                    {
                        "logID": 1,
                        "equipmentID": "printer_XYZ",
                        "notes": "Thermal anomaly detected near extruder – operator adjusted cooling settings.",
                        "operatorID": "op_42",
                        "timestamp": datetime(2024, 6, 12, 14, 23, 0),
                        "imageUrl": "https://picsum.photos/seed/anomaly1/400/250"
                    },
                    {
                        "logID": 2,
                        "equipmentID": "printer_XYZ",
                        "notes": "Routine check – no issues.",
                        "operatorID": "op_17",
                        "timestamp": datetime(2024, 6, 20, 9, 11, 0)
                    },
                    {
                        "logID": 3,
                        "equipmentID": "printer_XYZ",
                        "notes": "Uneven powder spreading captured on video; operator planning blade replacement.",
                        "operatorID": "op_55",
                        "timestamp": datetime(2024, 6, 25, 10, 45, 0),
                        "imageUrl": "https://picsum.photos/seed/anomaly3/400/250",
                        "videoUrl": "https://samplelib.com/lib/preview/mp4/sample-5s.mp4"
                    }
                ]
                await content_db.maintenanceLogs.insert_many(maintenance_logs)
                inserted_count += len(maintenance_logs)
                logger.info(f"Inserted {len(maintenance_logs)} maintenance logs")
            
            # Seed thermal scans
            if force or await content_db.thermalScans.count_documents({}) == 0:
                thermal_scans = [
                    {
                        "scanID": "ts_1001",
                        "jobID": "job_789",
                        "maxTemperature": 67.3,
                        "hotspotCoordinates": [123, 456],
                        "context": {"processType": "additive_manufacturing"},
                        "thermalImage": "https://picsum.photos/seed/thermal2/600/300"
                    },
                    {
                        "scanID": "ts_1002", 
                        "jobID": "job_790",
                        "maxTemperature": 55.1,
                        "hotspotCoordinates": [],
                        "context": {"processType": "additive_manufacturing"}
                    }
                ]
                await content_db.thermalScans.insert_many(thermal_scans)
                inserted_count += len(thermal_scans)
                logger.info(f"Inserted {len(thermal_scans)} thermal scans")
            
            # Seed other collections as needed
            collections_data = {
                "pointClouds": [
                    {
                        "cloudID": "pc_5001",
                        "spatialRegion": {"bounds": [10, 20, 30, 40, 50, 60]},
                        "quality": {"layerDeviation": 0.12},
                        "deviationMap": "https://example.com/models/pc_5001.glb"
                    }
                ],
                "machineAudioAlerts": [
                    {
                        "alertID": "ma_1",
                        "machineID": "cnc_22",
                        "pattern": "bearing_wear",
                        "timestamp": datetime.utcnow(),
                        "frequencyAnalysis": [50, 120, 500],
                        "audioClip": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
                    }
                ],
                "crossModalAnalysis": [
                    {
                        "jobID": "job_123",
                        "thermalDefects": ["warping"],
                        "vibrationSpectra": [3.2, 5.5, 1.1],
                        "reportPdf": "https://example.com/reports/job_123.pdf"
                    }
                ]
            }
            
            for collection_name, data in collections_data.items():
                collection = content_db[collection_name]
                if force or await collection.count_documents({}) == 0:
                    await collection.insert_many(data)
                    inserted_count += len(data)
                    logger.info(f"Inserted {len(data)} documents into {collection_name}")
            
            logger.info(f"✅ Content seeding complete! Inserted {inserted_count} total documents")
            return inserted_count
            
        except Exception as e:
            logger.error(f"❌ Content seeding failed: {e}")
            raise
    
    async def clear_content(self) -> bool:
        """Clear all demo content data."""
        try:
            db_service = await self._get_db_service()
            content_db = db_service.client["mppw_content"]
            
            collections = [
                "maintenanceLogs", "thermalScans", "pointClouds",
                "machineAudioAlerts", "crossModalAnalysis", "materialCorrelation",
                "energyThermalSync", "surfaceQuality", "logImageCrosswalk",
                "porosityModelResults"
            ]
            
            for collection_name in collections:
                result = await content_db[collection_name].delete_many({})
                logger.info(f"Cleared {result.deleted_count} documents from {collection_name}")
            
            logger.info("✅ All demo content cleared")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clear content: {e}")
            return False


# Global service instance
_content_seed_service = None


async def get_seed_service() -> ContentSeedService:
    """Get the global content seed service instance."""
    global _content_seed_service
    if _content_seed_service is None:
        _content_seed_service = ContentSeedService()
    return _content_seed_service 