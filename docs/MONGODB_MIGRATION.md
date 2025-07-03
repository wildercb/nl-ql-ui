# MongoDB Migration & Enhanced LLM Tracking

This document describes the complete migration from PostgreSQL to MongoDB and the implementation of comprehensive LLM interaction tracking in the MPPW MCP application.

## 🔄 Migration Overview

### What Changed
- **Database**: PostgreSQL → MongoDB with Beanie ODM
- **Architecture**: Document-based storage for flexible data modeling
- **LLM Tracking**: Comprehensive interaction logging and analytics
- **Performance**: Optimized for real-time analytics and monitoring

### Why MongoDB?
1. **Flexible Schema**: JSON documents ideal for LLM interaction data
2. **Scalability**: Better horizontal scaling for analytics workloads
3. **Rich Queries**: Native support for complex aggregations
4. **Real-time Analytics**: Efficient time-series data handling

## 📊 Enhanced LLM Tracking System

### Features Added
- **Complete Interaction Logging**: Every prompt and response captured
- **Performance Metrics**: Processing time, token usage, confidence scores
- **Session Management**: Grouped interactions for analysis
- **Real-time Analytics**: Live dashboards and monitoring
- **Data Export**: Full session and interaction export capabilities

### Tracking Coverage
- ✅ **Ollama Service**: All model interactions
- ✅ **Translation Service**: Natural language to GraphQL
- ✅ **Validation Service**: Query validation and suggestions
- ✅ **FastMCP Tools**: All 17+ FastMCP tool interactions
- ✅ **API Endpoints**: Complete request/response logging

## 🏗️ New Architecture

### Database Collections

```
MongoDB Collections:
├── users                    # User accounts and profiles
├── user_sessions           # Authentication sessions
├── user_api_keys          # API key management
├── user_preferences       # User settings and preferences
├── queries                # Translation queries and results
├── query_results          # Query execution results
├── query_sessions         # Query session management
├── query_feedback         # User feedback on translations
└── llm_interactions       # 🆕 Comprehensive LLM tracking
```

### Key MongoDB Features
- **Automatic Indexing**: Optimized queries for all collections
- **TTL Indexes**: Automatic cleanup of old sessions and interactions
- **Validation Schemas**: Document structure validation
- **Aggregation Pipelines**: Advanced analytics and reporting

## 🔧 Technical Implementation

### 1. Database Models (Beanie ODM)

```python
# Example: LLM Interaction Model
class LLMInteraction(BaseDocument):
    session_id: str
    model: str
    provider: str
    prompt: str
    response: str
    processing_time: float
    tokens_used: Optional[int]
    confidence_score: Optional[float]
    timestamp: datetime
    # ... additional metadata
```

### 2. LLM Tracking Service

```python
# Context manager for automatic tracking
async with tracking_service.track_interaction(
    session_id="session-123",
    model="llama2",
    provider="ollama",
    interaction_type="translation"
) as tracker:
    # Set input data
    tracker.set_prompt(prompt)
    tracker.set_parameters(temperature=0.7)
    
    # Make model call
    response = await model_call()
    
    # Track output
    tracker.set_response(response)
    tracker.set_performance_metrics(processing_time=1.5)
```

### 3. Analytics API Endpoints

```
GET /analytics/overview              # High-level analytics dashboard
GET /analytics/sessions/{id}         # Detailed session analysis
GET /analytics/interactions/{id}     # Full interaction details
GET /analytics/interactions          # Paginated interaction list
GET /analytics/models/stats          # Model usage statistics
GET /analytics/database/stats        # Database health and metrics
POST /analytics/cleanup              # Data retention management
GET /analytics/export/session/{id}   # Complete session export
```

## 🚀 Quick Start Guide

### 1. Start the Application

```bash
# Using the enhanced monitoring script
./scripts/run-with-monitoring.sh
```

This will:
- ✅ Start MongoDB container
- ✅ Initialize collections and indexes  
- ✅ Start FastMCP server with tracking
- ✅ Start REST API with analytics endpoints
- ✅ Start frontend with debug utilities
- ✅ Set up comprehensive logging

### 2. Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:3000 | Main UI with debug tools |
| **REST API** | http://localhost:8000 | Translation API + Analytics |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **Analytics** | http://localhost:8000/analytics/overview | LLM tracking dashboard |
| **MongoDB** | mongodb://localhost:27017 | Direct database access |

### 3. View LLM Interactions

```bash
# Get recent interactions
curl http://localhost:8000/analytics/interactions?limit=10

# Get session analytics
curl http://localhost:8000/analytics/sessions/{session_id}

# Export full session data
curl http://localhost:8000/analytics/export/session/{session_id}
```

## 📈 Monitoring & Analytics

### Real-time Dashboards

1. **Analytics Overview** (`/analytics/overview`)
   - Total interactions and success rates
   - Top performing models
   - Recent activity timeline
   - Interaction type breakdown

2. **Session Analytics** (`/analytics/sessions/{id}`)
   - Complete session timeline
   - Performance metrics per interaction
   - Model usage patterns
   - Quality score trends

3. **Model Statistics** (`/analytics/models/stats`)
   - Usage statistics by model
   - Performance comparisons
   - Success rate analysis
   - Token consumption metrics

### Database Monitoring

```bash
# View database statistics
curl http://localhost:8000/analytics/database/stats

# Monitor collection sizes
docker exec -it mongo mongo mppw_mcp --eval "
db.runCommand('dbStats')
"

# View recent interactions
docker exec -it mongo mongo mppw_mcp --eval "
db.llm_interactions.find().sort({timestamp: -1}).limit(5).pretty()
"
```

## 🔍 Debugging & Troubleshooting

### Enhanced Debug Features

1. **Frontend Debug Tools** (`/src/utils/debug.ts`)
   - Real-time API call monitoring
   - User interaction tracking
   - Performance metrics collection
   - Local storage management

2. **Backend Structured Logging**
   - JSON formatted logs
   - Request/response tracking
   - Error stack traces
   - Performance timing

3. **Database Query Logging**
   - MongoDB operation logging
   - Index usage analysis
   - Query performance metrics

### Common Issues & Solutions

#### MongoDB Connection Issues
```bash
# Check MongoDB container
docker-compose logs mongo

# Verify database initialization
docker exec -it mongo mongo mppw_mcp --eval "show collections"
```

#### LLM Tracking Not Working
```bash
# Check tracking service status
curl http://localhost:8000/analytics/overview

# Verify environment variables
echo $LLM_TRACKING_ENABLED
```

#### Performance Issues
```bash
# Check database indexes
docker exec -it mongo mongo mppw_mcp --eval "
db.llm_interactions.getIndexes()
"

# Monitor collection stats
curl http://localhost:8000/analytics/database/stats
```

## 🔄 Data Migration (if needed)

### From Existing PostgreSQL Data

If you have existing PostgreSQL data to migrate:

```python
# Example migration script
async def migrate_postgresql_to_mongodb():
    # 1. Export PostgreSQL data
    # 2. Transform to MongoDB document format
    # 3. Import with proper validation
    # 4. Verify data integrity
    pass
```

### Backup and Restore

```bash
# MongoDB backup
docker exec mongo mongodump --db mppw_mcp --out /backup

# MongoDB restore
docker exec mongo mongorestore --db mppw_mcp /backup/mppw_mcp
```

## 🎯 Next Steps

### Immediate Benefits
- ✅ Real-time LLM interaction monitoring
- ✅ Comprehensive session analytics
- ✅ Enhanced debugging capabilities
- ✅ Flexible data modeling
- ✅ Improved scalability

### Future Enhancements
- 🔮 Machine learning on interaction patterns
- 🔮 Predictive model performance optimization
- 🔮 Advanced visualization dashboards
- 🔮 Real-time alerting and monitoring
- 🔮 A/B testing for model selection

## 📝 Configuration Reference

### MongoDB Settings
```env
MONGODB_URL=mongodb://mongo:27017
MONGODB_DATABASE=mppw_mcp
MONGODB_MIN_CONNECTIONS=10
MONGODB_MAX_CONNECTIONS=100
```

### LLM Tracking Settings
```env
LLM_TRACKING_ENABLED=true
LLM_TRACKING_STORE_PROMPTS=true
LLM_TRACKING_STORE_RESPONSES=true
LLM_TRACKING_RETENTION_DAYS=90
LLM_TRACKING_MAX_PROMPT_LENGTH=10000
LLM_TRACKING_MAX_RESPONSE_LENGTH=10000
```

---

**Migration Complete!** 🎉

Your MPPW MCP application now runs on MongoDB with comprehensive LLM tracking and real-time analytics. All previous functionality is preserved while adding powerful new monitoring and debugging capabilities. 