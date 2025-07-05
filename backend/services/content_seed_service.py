from __future__ import annotations

"""Utility to seed the `mppw_content` Mongo database with demo documents used by the UI examples.

This duplicates the data from `backend/database/seed_content.js` in pure Python so it can be
invoked on-demand from an API route without recreating the Docker volume.
"""

import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Dict, Any, List

from config.settings import get_settings

logger = logging.getLogger(__name__)

# ----- demo documents -------------------------------------------------------

_DEMO_DOCS: Dict[str, List[Dict[str, Any]]] = {
    "maintenanceLogs": [
        {
            "logID": 1,
            "equipmentID": "printer_XYZ",
            "notes": "Thermal anomaly detected near extruder – operator adjusted cooling settings.",
            "operatorID": "op_42",
            "timestamp": "2024-06-12T14:23:00Z",
            "imageUrl": "https://picsum.photos/seed/anomaly1/400/250",
        },
        {
            "logID": 2,
            "equipmentID": "printer_XYZ",
            "notes": "Routine check – no issues.",
            "operatorID": "op_17",
            "timestamp": "2024-06-20T09:11:00Z",
        },
        {
            "logID": 3,
            "equipmentID": "printer_XYZ",
            "notes": "Uneven powder spreading captured on video; operator planning blade replacement.",
            "operatorID": "op_55",
            "timestamp": "2024-06-25T10:45:00Z",
            "imageUrl": "https://picsum.photos/seed/anomaly3/400/250",
            "videoUrl": "https://samplelib.com/lib/preview/mp4/sample-5s.mp4",
        },
    ],
    "thermalScans": [
        {
            "scanID": "ts_1001",
            "jobID": "job_789",
            "maxTemperature": 67.3,
            "hotspotCoordinates": [123, 456],
            "context": {"processType": "additive_manufacturing"},
            "thermalImage": "https://picsum.photos/seed/thermal2/600/300",
        },
        {
            "scanID": "ts_1002",
            "jobID": "job_790",
            "maxTemperature": 55.1,
            "hotspotCoordinates": [],
            "context": {"processType": "additive_manufacturing"},
        },
    ],
    "pointClouds": [
        {
            "cloudID": "pc_5001",
            "spatialRegion": {"bounds": [10, 20, 30, 40, 50, 60]},
            "quality": {"layerDeviation": 0.12},
            "deviationMap": "https://example.com/models/pc_5001.glb",
        }
    ],
    "machineAudioAlerts": [
        {
            "alertID": "ma_1",
            "machineID": "cnc_22",
            "pattern": "bearing_wear",
            "timestamp": "2025-07-05T00:00:00Z",
            "frequencyAnalysis": [50, 120, 500],
            "audioClip": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
        }
    ],
    "crossModalAnalysis": [
        {
            "jobID": "job_123",
            "thermalDefects": ["warping"],
            "vibrationSpectra": [3.2, 5.5, 1.1],
            "reportPdf": "https://example.com/reports/job_123.pdf",
        }
    ],
    "materialCorrelation": [
        {
            "correlationScore": 0.83,
            "frequencyPeaks": [4.4, 9.1, 15.2],
            "stressStrainCurve": "https://example.com/plots/ss_job_99.png",
        }
    ],
    "energyThermalSync": [
        {
            "jobID": "dmls_777",
            "layer": 45,
            "tempDelta": 18,
            "powerVar": 12,
            "thermalMatrix": "https://picsum.photos/seed/thermal3/500/300",
            "powerTimeSeries": "https://example.com/data/dmls_777_pwr.csv",
        }
    ],
    "surfaceQuality": [
        {
            "frameID": "frm_88",
            "toolpath": "spiral",
            "Ra": 1.2,
            "Rz": 6.8,
            "frameImage": "https://picsum.photos/seed/frame88/500/300",
        }
    ],
    "logImageCrosswalk": [
        {
            "logExcerpt": "Powder spreading issues observed at layer 33 – recoater streaks.",
            "imageEmbedding": "https://picsum.photos/seed/recoater1/400/250",
            "timestamp": "2024-05-15T12:00:00Z",
        }
    ],
    "porosityModelResults": [
        {
            "porosityPercentage": 2.1,
            "confidenceInterval": [1.8, 2.4],
            "featureContributions": [
                {"sensor": "ctScan", "weight": 0.6},
                {"sensor": "photodiode", "weight": 0.4},
            ],
            "reportUrl": "https://example.com/reports/porosity_2_1.pdf",
        }
    ],
}

# ----- seed helper ----------------------------------------------------------

class ContentSeedService:
    """Lazily connects to the content DB and inserts demo docs if collections are empty."""

    _client: AsyncIOMotorClient | None = None
    _db: AsyncIOMotorDatabase | None = None

    async def _get_db(self) -> AsyncIOMotorDatabase:
        if self._db is None:
            settings = get_settings()
            self._client = AsyncIOMotorClient(settings.data_database.url)
            self._db = self._client[settings.data_database.database]
        return self._db

    async def seed_once(self, force: bool = False) -> dict[str, int]:
        """Insert demo docs.

        If *force* is False (default) we only insert into an empty collection.  If *force* is True
        we first delete all existing documents, then insert fresh copies so media additions are
        visible without manual cleanup.
        Returns a mapping of collection → inserted_count.
        """
        db = await self._get_db()
        inserted: dict[str, int] = {}
        for coll_name, docs in _DEMO_DOCS.items():
            coll = db[coll_name]
            count = await coll.count_documents({})
            if force and count:
                await coll.delete_many({})
                count = 0

            if count == 0:
                result = await coll.insert_many(docs)
                inserted[coll_name] = len(result.inserted_ids)
                logger.info("🌱 Seeded %s with %d docs", coll_name, len(result.inserted_ids))
            else:
                logger.debug("%s already has %d docs – skipping", coll_name, count)
        return inserted


_seed_service: ContentSeedService | None = None

def get_seed_service() -> ContentSeedService:
    global _seed_service
    if _seed_service is None:
        _seed_service = ContentSeedService()
    return _seed_service 