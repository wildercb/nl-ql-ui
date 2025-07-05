"""Initial In-Context Learning (ICL) Examples for Natural Language to GraphQL Translation."""

from typing import List, Dict

# List of initial ICL examples for seeding
# Each example is a dictionary with 'natural' (natural language query) and 'graphql' (corresponding GraphQL query)
INITIAL_ICL_EXAMPLES: List[Dict[str, str]] = [
    {
        "natural": "Show operator notes from June 2024 mentioning thermal anomalies in printer XYZ",
        "graphql": "query {\n  maintenanceLogs(\n    filter: {\n      timestamp: { between: [\"2024-06-01\", \"2024-06-30\"] }\n      notes: { contains: \"thermal anomaly\" }\n      equipmentID: \"printer_XYZ\"\n    }\n  ) {\n    logID\n    notes\n    operatorID\n  }\n}"
    },
    {
        "natural": "Find all thermal scans from additive manufacturing jobs with hotspots exceeding 60°C",
        "graphql": "query {\n  thermalScans(\n    filter: { \n      maxTemperature: { gt: 60 }\n      context: { processType: \"additive_manufacturing\" }\n    }\n  ) {\n    scanID\n    jobID\n    hotspotCoordinates\n  }\n}"
    },
    {
        "natural": "Identify point clouds in sector X-Y-Z where layer thickness deviates >0.1mm from CAD specs",
        "graphql": "query {\n  pointClouds(\n    filter: {\n      spatialRegion: { bounds: [10,20,30,40,50,60] }\n      quality: { layerDeviation: { gt: 0.1 } }\n    }\n  ) {\n    cloudID\n    deviationMap\n  }\n}"
    },
    {
        "natural": "Alert when CNC machine audio shows bearing wear patterns in last 24 hours",
        "graphql": "subscription {\n  machineAudioAlerts(\n    trigger: {\n      pattern: \"bearing_wear\"\n      timeframe: \"24h\"\n      machineType: \"CNC\"\n    }\n  ) {\n    alertID\n    timestamp\n    frequencyAnalysis\n  }\n}"
    },
    {
        "natural": "Correlate thermal images showing warping with extruder motor vibration data from same jobs",
        "graphql": "query {\n  crossModalAnalysis(\n    filters: [\n      { modality: \"thermal\", condition: \"defectType = 'warping'\" },\n      { modality: \"vibration\", condition: \"motorType = 'extruder'\" }\n    ],\n    correlation: \"timestamp\"\n  ) {\n    jobID\n    thermalDefects\n    vibrationSpectra\n  }\n}"
    },
    {
        "natural": "Correlate tensile strength test results from printed titanium alloys with in-situ microphone recordings of printing sounds",
        "graphql": "query {\n  materialCorrelation(\n    filters: [\n      { modality: \"mechanical\", testType: \"tensile_strength\" },\n      { modality: \"audio\", feature: \"acoustic_emission_spectrum\" }\n    ],\n    joinKey: \"batch_id\"\n  ) {\n    correlationScore\n    frequencyPeaks\n    stressStrainCurve\n  }\n}"
    },
    {
        "natural": "Identify relationships between layer 45's infrared thermal patterns and laser power fluctuations in DMLS jobs",
        "graphql": "subscription {\n  energyThermalSync(\n    layer: 45, \n    process: \"DMLS\",\n    thresholds: {\n      tempDelta: 15°C,\n      powerVar: ±10%\n    }\n  ) {\n    jobID\n    thermalMatrix\n    powerTimeSeries\n  }\n}"
    },
    {
        "natural": "Align CNC milling video frames of AM parts with post-machining roughness measurements from 3D profilometry",
        "graphql": "query {\n  surfaceQuality(\n    filters: {\n      video: { operation: \"finish_milling\" },\n      metrology: { instrument: \"3D_profilometer\" }\n    },\n    temporalWindow: \"5s\"\n  ) {\n    frameID\n    toolpath\n    Ra\n    Rz\n  }\n}"
    },
    {
        "natural": "Find entries where operator notes mention 'powder spreading issues' and correlate with coater blade camera images from those timestamps",
        "graphql": "query {\n  logImageCrosswalk(\n    textFilters: { \n      keywords: [\"powder spreading\"],\n      dateRange: \"2024-05-01 to 2024-05-31\" \n    },\n    imageFilters: {\n      source: \"recoater_cam\",\n      defectTags: [\"streaks\", \"uneven\"]\n    }\n  ) {\n    logExcerpt\n    imageEmbedding\n    timestamp\n  }\n}"
    },
    {
        "natural": "Predict internal porosity levels by combining X-ray CT slice predictions with in-process photodiode meltpool signals",
        "graphql": "mutation {\n  porosityModel(\n    inputs: {\n      ctScan: \"base64EncodedSlice\",\n      photodiode: {\n        frequency: 100kHz,\n        amplitudeRange: [0.2V, 1.8V]\n      }\n    }\n  ) {\n    porosityPercentage\n    confidenceInterval\n    featureContributions {\n      sensor\n      weight\n    }\n  }\n}"
    }
]

def get_initial_icl_examples() -> List[str]:
    """Format initial ICL examples as strings for inclusion in prompts."""
    formatted_examples = []
    for example in INITIAL_ICL_EXAMPLES:
        formatted_examples.append(f"Natural: {example['natural']}\nGraphQL: {example['graphql']}")
    return formatted_examples 