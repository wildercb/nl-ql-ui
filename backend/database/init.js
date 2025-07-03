// MongoDB Initialization Script for MPPW MCP
// This script sets up collections, indexes, and initial data

print("🚀 Initializing MPPW MCP MongoDB Database...");

// Switch to the application database
db = db.getSiblingDB('mppw_mcp');

// Create collections with validation schemas
print("📋 Creating collections with validation...");

// Users collection
db.createCollection("users", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["username", "email", "hashed_password"],
      properties: {
        _id: { bsonType: "objectId" },
        uuid: { bsonType: "string" },
        username: { bsonType: "string" },
        email: { bsonType: "string" },
        hashed_password: { bsonType: "string" },
        is_active: { bsonType: "bool" },
        is_verified: { bsonType: "bool" },
        created_at: { bsonType: "date" },
        updated_at: { bsonType: "date" }
      }
    }
  }
});

// Queries collection
db.createCollection("queries", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["natural_query", "graphql_query", "model_used", "confidence"],
      properties: {
        _id: { bsonType: "objectId" },
        uuid: { bsonType: "string" },
        natural_query: { bsonType: "string" },
        graphql_query: { bsonType: "string" },
        schema_context: { bsonType: "string" },
        model_used: { bsonType: "string" },
        confidence: { bsonType: "number" },
        created_at: { bsonType: "date" },
        updated_at: { bsonType: "date" }
      }
    }
  }
});

// Query results collection
db.createCollection("query_results");

// Query sessions collection
db.createCollection("query_sessions");

// User sessions collection
db.createCollection("user_sessions");

// User API keys collection
db.createCollection("user_api_keys");

// Query feedback collection
db.createCollection("query_feedback");

// User preferences collection
db.createCollection("user_preferences");

// LLM interactions collection (new for tracking)
db.createCollection("llm_interactions", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["session_id", "model", "prompt", "response", "timestamp"],
      properties: {
        _id: { bsonType: "objectId" },
        session_id: { bsonType: "string" },
        model: { bsonType: "string" },
        prompt: { bsonType: "string" },
        response: { bsonType: "string" },
        timestamp: { bsonType: "date" },
        processing_time: { bsonType: "number" },
        tokens_used: { bsonType: "number" }
      }
    }
  }
});

print("🔍 Creating indexes for optimal performance...");

// Users indexes
db.users.createIndex({ "username": 1 }, { unique: true });
db.users.createIndex({ "email": 1 }, { unique: true });
db.users.createIndex({ "uuid": 1 }, { unique: true });
db.users.createIndex({ "created_at": 1 });

// Queries indexes
db.queries.createIndex({ "uuid": 1 }, { unique: true });
db.queries.createIndex({ "user_id": 1 });
db.queries.createIndex({ "model_used": 1 });
db.queries.createIndex({ "confidence": -1 });
db.queries.createIndex({ "created_at": -1 });
db.queries.createIndex({ "natural_query": "text", "graphql_query": "text" });

// Query results indexes
db.query_results.createIndex({ "query_id": 1 });
db.query_results.createIndex({ "is_successful": 1 });
db.query_results.createIndex({ "created_at": -1 });

// Sessions indexes
db.query_sessions.createIndex({ "user_id": 1 });
db.query_sessions.createIndex({ "is_active": 1 });
db.query_sessions.createIndex({ "created_at": -1 });

db.user_sessions.createIndex({ "user_id": 1 });
db.user_sessions.createIndex({ "session_token": 1 }, { unique: true });
db.user_sessions.createIndex({ "expires_at": 1 });

// API keys indexes
db.user_api_keys.createIndex({ "user_id": 1 });
db.user_api_keys.createIndex({ "key_hash": 1 }, { unique: true });
db.user_api_keys.createIndex({ "is_active": 1 });

// LLM interactions indexes (for tracking)
db.llm_interactions.createIndex({ "session_id": 1 });
db.llm_interactions.createIndex({ "model": 1 });
db.llm_interactions.createIndex({ "timestamp": -1 });
db.llm_interactions.createIndex({ "user_id": 1 });

print("📊 Creating compound indexes for complex queries...");

// Compound indexes for better query performance
db.queries.createIndex({ "user_id": 1, "created_at": -1 });
db.queries.createIndex({ "model_used": 1, "confidence": -1 });
db.llm_interactions.createIndex({ "session_id": 1, "timestamp": -1 });
db.llm_interactions.createIndex({ "model": 1, "timestamp": -1 });

print("🎯 Setting up TTL indexes for automatic cleanup...");

// TTL indexes for automatic cleanup
db.user_sessions.createIndex({ "expires_at": 1 }, { expireAfterSeconds: 0 });
db.llm_interactions.createIndex({ "timestamp": 1 }, { expireAfterSeconds: 7776000 }); // 90 days

print("✅ MongoDB initialization complete!");
print("📈 Database ready for MPPW MCP application with enhanced LLM tracking");

// Display collection status
print("\n📊 Collection Summary:");
print("Collections created: " + db.getCollectionNames().length);
db.getCollectionNames().forEach(function(collection) {
    print("  - " + collection + ": " + db[collection].getIndexes().length + " indexes");
}); 