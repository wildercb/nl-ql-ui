"""
In-Context Learning (ICL) Examples Configuration

This module provides the specific manufacturing/3D printing ICL examples
for natural language to GraphQL translation. Built with scalability in mind
for future vector database integration with similarity search and reranking.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import hashlib
import json


@dataclass
class ICLExample:
    """A single in-context learning example with metadata for future vector search."""
    id: str
    natural: str
    graphql: str
    category: str = "manufacturing"
    tags: List[str] = field(default_factory=list)
    complexity_score: float = 1.0
    
    def __post_init__(self):
        # Auto-generate ID from content hash if not provided
        if not self.id:
            content = f"{self.natural}|{self.graphql}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:12]


class ICLExampleManager:
    """
    Manager for in-context learning examples.
    
    Built for future scalability with vector database integration for:
    - Similarity search
    - Semantic reranking 
    - Dynamic example selection
    """
    
    def __init__(self):
        self.examples: List[ICLExample] = []
        self._load_manufacturing_examples()
    
    def _load_manufacturing_examples(self):
        """Load the specific manufacturing/3D printing ICL examples."""
        
        # Manufacturing & Maintenance
        self.examples.extend([
            ICLExample(
                id="maint_thermal_001",
                natural="Show operator notes from June 2024 mentioning thermal anomalies in printer XYZ",
                graphql='''query {
  maintenanceLogs(
    filter: {
      timestamp: { between: ["2024-06-01", "2024-06-30"] }
      notes: { contains: "thermal anomaly" }
      equipmentID: "printer_XYZ"
    }
  ) {
    logID
    notes
    operatorID
  }
}''',
                tags=["maintenance", "thermal", "logs", "equipment"],
                complexity_score=0.7
            ),
            
            ICLExample(
                id="thermal_imaging_001",
                natural="Find all thermal scans from additive manufacturing jobs with hotspots exceeding 60°C",
                graphql='''query {
  thermalScans(
    filter: { 
      maxTemperature: { gt: 60 }
      context: { processType: "additive_manufacturing" }
    }
  ) {
    scanID
    jobID
    hotspotCoordinates
  }
}''',
                tags=["thermal", "additive_manufacturing", "temperature", "quality"],
                complexity_score=0.6
            ),
            
            ICLExample(
                id="point_cloud_001", 
                natural="Identify point clouds in sector X-Y-Z where layer thickness deviates >0.1mm from CAD specs",
                graphql='''query {
  pointClouds(
    filter: {
      spatialRegion: { bounds: [10,20,30,40,50,60] }
      quality: { layerDeviation: { gt: 0.1 } }
    }
  ) {
    cloudID
    deviationMap
  }
}''',
                tags=["point_cloud", "metrology", "deviation", "quality_control"],
                complexity_score=0.8
            ),
            
            ICLExample(
                id="audio_analysis_001",
                natural="Alert when CNC machine audio shows bearing wear patterns in last 24 hours", 
                graphql='''subscription {
  machineAudioAlerts(
    trigger: {
      pattern: "bearing_wear"
      timeframe: "24h"
      machineType: "CNC"
    }
  ) {
    alertID
    timestamp
    frequencyAnalysis
  }
}''',
                tags=["audio", "CNC", "bearing_wear", "subscription", "real_time"],
                complexity_score=0.9
            ),
            
            ICLExample(
                id="cross_modal_001",
                natural="Correlate thermal images showing warping with extruder motor vibration data from same jobs",
                graphql='''query {
  crossModalAnalysis(
    filters: [
      { modality: "thermal", condition: "defectType = 'warping'" }
      { modality: "vibration", condition: "motorType = 'extruder'" }
    ]
    correlation: "timestamp"
  ) {
    jobID
    thermalDefects
    vibrationSpectra
  }
}''',
                tags=["cross_modal", "thermal", "vibration", "correlation", "multi_sensor"],
                complexity_score=1.0
            ),
            
            ICLExample(
                id="material_acoustic_001",
                natural="Correlate tensile strength test results from printed titanium alloys with in-situ microphone recordings of printing sounds",
                graphql='''query {
  materialCorrelation(
    filters: [
      { modality: "mechanical", testType: "tensile_strength" }
      { modality: "audio", feature: "acoustic_emission_spectrum" }
    ]
    joinKey: "batch_id"
  ) {
    correlationScore
    frequencyPeaks
    stressStrainCurve
  }
}''',
                tags=["materials", "acoustic_emission", "mechanical_testing", "titanium", "correlation"],
                complexity_score=1.0
            ),
            
            ICLExample(
                id="ir_power_001",
                natural="Identify relationships between layer 45's infrared thermal patterns and laser power fluctuations in DMLS jobs",
                graphql='''subscription {
  energyThermalSync(
    layer: 45 
    process: "DMLS"
    thresholds: {
      tempDelta: 15°C
      powerVar: ±10%
    }
  ) {
    jobID
    thermalMatrix
    powerTimeSeries
  }
}''',
                tags=["infrared", "laser_power", "DMLS", "subscription", "layer_analysis"],
                complexity_score=0.9
            ),
            
            ICLExample(
                id="video_roughness_001",
                natural="Align CNC milling video frames of AM parts with post-machining roughness measurements from 3D profilometry",
                graphql='''query {
  surfaceQuality(
    filters: {
      video: { operation: "finish_milling" }
      metrology: { instrument: "3D_profilometer" }
    }
    temporalWindow: "5s"
  ) {
    frameID
    toolpath
    Ra
    Rz
  }
}''',
                tags=["video", "surface_roughness", "CNC_milling", "profilometry", "temporal_alignment"],
                complexity_score=1.0
            ),
            
            ICLExample(
                id="operator_powder_001",
                natural="Find entries where operator notes mention 'powder spreading issues' and correlate with coater blade camera images from those timestamps",
                graphql='''query {
  logImageCrosswalk(
    textFilters: { 
      keywords: ["powder spreading"]
      dateRange: "2024-05-01 to 2024-05-31" 
    }
    imageFilters: {
      source: "recoater_cam"
      defectTags: ["streaks", "uneven"]
    }
  ) {
    logExcerpt
    imageEmbedding
    timestamp
  }
}''',
                tags=["operator_logs", "powder_spreading", "camera", "text_image_correlation"],
                complexity_score=0.8
            ),
            
            ICLExample(
                id="porosity_fusion_001",
                natural="Predict internal porosity levels by combining X-ray CT slice predictions with in-process photodiode meltpool signals",
                graphql='''mutation {
  porosityModel(
    inputs: {
      ctScan: "base64EncodedSlice"
      photodiode: {
        frequency: 100kHz
        amplitudeRange: [0.2V, 1.8V]
      }
    }
  ) {
    porosityPercentage
    confidenceInterval
    featureContributions {
      sensor
      weight
    }
  }
}''',
                tags=["porosity", "CT_scan", "photodiode", "mutation", "ML_prediction", "multi_sensor"],
                complexity_score=1.0
            )
        ])
    
    def get_all_examples(self) -> List[ICLExample]:
        """Get all ICL examples."""
        return self.examples.copy()
    
    def get_examples_by_tags(self, tags: List[str], limit: int = 3) -> List[ICLExample]:
        """
        Get examples filtered by tags.
        
        Future: This will be replaced with vector similarity search.
        """
        filtered = []
        for example in self.examples:
            if any(tag in example.tags for tag in tags):
                filtered.append(example)
        
        # Sort by complexity score (descending) and return top N
        filtered.sort(key=lambda x: x.complexity_score, reverse=True)
        return filtered[:limit]
    
    def get_random_examples(self, count: int = 3) -> List[ICLExample]:
        """Get random examples for diversity."""
        import random
        return random.sample(self.examples, min(count, len(self.examples)))
    
    def get_examples_for_query(self, query: str, limit: int = 3) -> List[ICLExample]:
        """
        Get best examples for a given query.
        
        Current: Simple keyword matching
        Future: Vector similarity search with semantic embeddings
        """
        query_lower = query.lower()
        scored_examples = []
        
        for example in self.examples:
            score = 0
            
            # Simple keyword scoring (to be replaced with vector similarity)
            for tag in example.tags:
                if tag.replace("_", " ") in query_lower:
                    score += 1
            
            # Check for keywords in natural language
            natural_words = example.natural.lower().split()
            query_words = query_lower.split()
            overlap = len(set(natural_words) & set(query_words))
            score += overlap * 0.1
            
            if score > 0:
                scored_examples.append((example, score))
        
        # Sort by score and return top examples
        scored_examples.sort(key=lambda x: x[1], reverse=True)
        return [ex for ex, score in scored_examples[:limit]]
    
    # Future vector database methods (placeholder)
    def prepare_for_vector_search(self) -> List[Dict[str, Any]]:
        """
        Prepare examples for vector database indexing.
        
        Returns format ready for embedding generation and vector storage.
        """
        return [
            {
                "id": ex.id,
                "text": f"{ex.natural} -> {ex.graphql}",
                "natural": ex.natural,
                "graphql": ex.graphql,
                "tags": ex.tags,
                "complexity_score": ex.complexity_score,
                "metadata": {
                    "category": ex.category,
                    "graphql_type": "query" if "query" in ex.graphql else "subscription" if "subscription" in ex.graphql else "mutation"
                }
            }
            for ex in self.examples
        ]
    
    def add_example(self, example: ICLExample):
        """Add a new example (for future dynamic updates)."""
        self.examples.append(example)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the example collection."""
        all_tags = set()
        for ex in self.examples:
            all_tags.update(ex.tags)
        
        return {
            "total_examples": len(self.examples),
            "unique_tags": len(all_tags),
            "avg_complexity": sum(ex.complexity_score for ex in self.examples) / len(self.examples),
            "tags": sorted(list(all_tags))
        }


# Global manager instance
_icl_manager: Optional[ICLExampleManager] = None


def get_icl_manager() -> ICLExampleManager:
    """Get the global ICL example manager."""
    global _icl_manager
    if _icl_manager is None:
        _icl_manager = ICLExampleManager()
    return _icl_manager


# Convenience functions for backward compatibility
def get_icl_examples(category: str = "manufacturing", limit: int = 3) -> List[ICLExample]:
    """Get ICL examples (simplified interface)."""
    manager = get_icl_manager()
    return manager.get_random_examples(limit)


def get_smart_examples(query: str, category: str = "manufacturing", limit: int = 3) -> List[ICLExample]:
    """Get smart examples based on query analysis."""
    manager = get_icl_manager()
    return manager.get_examples_for_query(query, limit) 