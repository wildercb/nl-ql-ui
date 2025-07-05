// 🚀 Content Database Seed Script for MPPW MCP
// This script populates the secondary Mongo instance (mppw_content)
// with sample documents so the demo GraphQL queries used in the ICL
// examples return interesting results – including some media files that
// DataResults.vue can render.

print("🌱 Seeding content database with demo documents …");

// Switch to the content DB
const content = db.getSiblingDB("mppw_content");

// Helper – insert documents only if the collection is empty (allows re-runs)
function safeInsertMany(collection, docs) {
  if (content[collection].countDocuments() === 0) {
    content[collection].insertMany(docs);
    print(`  ✔️  Inserted ${docs.length} docs into ${collection}`);
  } else {
    print(`  ↩️  ${collection} already populated – skipping`);
  }
}

// 1️⃣ maintenanceLogs
safeInsertMany("maintenanceLogs", [
  {
    logID: 1,
    equipmentID: "printer_XYZ",
    notes: "Thermal anomaly detected near extruder – operator adjusted cooling settings.",
    operatorID: "op_42",
    timestamp: ISODate("2024-06-12T14:23:00Z"),
    imageUrl: "https://picsum.photos/seed/anomaly1/400/250"
  },
  {
    logID: 2,
    equipmentID: "printer_XYZ",
    notes: "Routine check – no issues.",
    operatorID: "op_17",
    timestamp: ISODate("2024-06-20T09:11:00Z")
  }
]);

// 2️⃣ thermalScans
safeInsertMany("thermalScans", [
  {
    scanID: "ts_1001",
    jobID: "job_789",
    maxTemperature: 67.3,
    hotspotCoordinates: [123, 456],
    context: { processType: "additive_manufacturing" },
    thermalImage: "https://picsum.photos/seed/thermal2/600/300"
  },
  {
    scanID: "ts_1002",
    jobID: "job_790",
    maxTemperature: 55.1,
    hotspotCoordinates: [],
    context: { processType: "additive_manufacturing" }
  }
]);

// 3️⃣ pointClouds
safeInsertMany("pointClouds", [
  {
    cloudID: "pc_5001",
    spatialRegion: { bounds: [10, 20, 30, 40, 50, 60] },
    quality: { layerDeviation: 0.12 },
    deviationMap: "https://example.com/models/pc_5001.glb"
  }
]);

// 4️⃣ machineAudioAlerts
safeInsertMany("machineAudioAlerts", [
  {
    alertID: "ma_1",
    machineID: "cnc_22",
    pattern: "bearing_wear",
    timestamp: ISODate(),
    frequencyAnalysis: [50, 120, 500],
    audioClip: "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
  }
]);

// 5️⃣ crossModalAnalysis
safeInsertMany("crossModalAnalysis", [
  {
    jobID: "job_123",
    thermalDefects: ["warping"],
    vibrationSpectra: [3.2, 5.5, 1.1],
    reportPdf: "https://example.com/reports/job_123.pdf"
  }
]);

// 6️⃣ materialCorrelation
safeInsertMany("materialCorrelation", [
  {
    correlationScore: 0.83,
    frequencyPeaks: [4.4, 9.1, 15.2],
    stressStrainCurve: "https://example.com/plots/ss_job_99.png"
  }
]);

// 7️⃣ energyThermalSync
safeInsertMany("energyThermalSync", [
  {
    jobID: "dmls_777",
    layer: 45,
    tempDelta: 18,
    powerVar: 12,
    thermalMatrix: "https://picsum.photos/seed/thermal3/500/300",
    powerTimeSeries: "https://example.com/data/dmls_777_pwr.csv"
  }
]);

// 8️⃣ surfaceQuality
safeInsertMany("surfaceQuality", [
  {
    frameID: "frm_88",
    toolpath: "spiral",
    Ra: 1.2,
    Rz: 6.8,
    frameImage: "https://picsum.photos/seed/frame88/500/300"
  }
]);

// 9️⃣ logImageCrosswalk
safeInsertMany("logImageCrosswalk", [
  {
    logExcerpt: "Powder spreading issues observed at layer 33 – recoater streaks.",
    imageEmbedding: "https://picsum.photos/seed/recoater1/400/250",
    timestamp: ISODate("2024-05-15T12:00:00Z")
  }
]);

// 10️⃣ porosityModel (results from a mutation)
//     We treat this like a historical record of past predictions.
safeInsertMany("porosityModelResults", [
  {
    porosityPercentage: 2.1,
    confidenceInterval: [1.8, 2.4],
    featureContributions: [
      { sensor: "ctScan", weight: 0.6 },
      { sensor: "photodiode", weight: 0.4 }
    ],
    reportUrl: "https://example.com/reports/porosity_2_1.pdf"
  }
]);

print("✅ Content database seeding complete!"); 