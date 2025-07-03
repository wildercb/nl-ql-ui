# Complete Testing & Monitoring Guide

This guide provides comprehensive instructions for running and testing the entire MPPW MCP application with full visibility into all inputs, outputs, and data flows.

## 🚀 Quick Start

### 1. Start the Complete Application Stack

```bash
# Run the complete application with monitoring
./scripts/run-with-monitoring.sh
```

This script will:
- Start all Docker services (Backend API, FastMCP Server, Frontend, Ollama, PostgreSQL, Redis)
- Configure maximum debug logging
- Create individual monitoring scripts for each service
- Set up comprehensive testing utilities
- Verify all services are healthy

### 2. Access Points

Once running, you can access:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Ollama**: http://localhost:11434

## 📊 Monitoring & Logging

### Real-time Log Monitoring

1. **Monitor All Services Simultaneously**:
   ```bash
   ./logs/monitor_all.sh
   ```

2. **Monitor Individual Services**:
   ```bash
   ./logs/monitor_backend.sh     # Backend API logs
   ./logs/monitor_fastmcp.sh     # FastMCP server logs
   ./logs/monitor_frontend.sh    # Frontend logs
   ./logs/monitor_ollama.sh      # Ollama model server logs
   ```

### Log Levels & Content

The application logs at multiple levels with structured data:

#### Backend API Logs
- **INFO**: General application flow
- **DEBUG**: Detailed request/response data
- **ERROR**: Error conditions with stack traces

Example log entry:
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "DEBUG",
  "logger": "api.routes.translation",
  "message": "Translation request received",
  "extra": {
    "natural_query": "Get all users with emails",
    "model": "llama2",
    "schema_context": "User { id name email }"
  }
}
```

#### FastMCP Server Logs
- **Context tracking**: Every tool execution with input/output
- **Performance metrics**: Execution times and resource usage
- **Resource access**: All data retrieval operations

#### Ollama Model Logs
Enhanced logging shows:
- **🤖 Model Requests**: Full prompts and parameters
- **✅ Model Responses**: Generated text and performance metrics
- **⚡ Performance**: Token/second, duration, memory usage

Example:
```
🤖 Ollama Model Request: model=llama2, prompt_length=245, temperature=0.7
✅ Ollama Model Response: response_length=156, tokens_per_second=45.2, duration=3.4s
```

#### Frontend Debug Logs
The frontend automatically logs:
- **API Calls**: All requests/responses with timing
- **User Interactions**: Clicks, form submissions, navigation
- **Performance**: Page load times, render performance
- **Errors**: JavaScript errors with stack traces

## 🧪 Testing the Complete Flow

### 1. Automated API Testing

```bash
./logs/test_complete_flow.sh
```

This tests:
- Health endpoints
- Translation API
- Validation services
- Model management
- FastMCP server connectivity

### 2. Manual Frontend Testing

1. **Open the frontend**: http://localhost:3000
2. **Open browser console** (F12) to see detailed logs
3. **Test the translation flow**:
   - Enter a natural language query
   - Select a model
   - Observe the complete data flow in console logs

### 3. FastMCP Tool Testing

Use an MCP client to test FastMCP tools:

```bash
# Example: Test translation tool
mcp-client call translate_query \
  --natural_query "Get all users with their names and emails" \
  --schema_context "User { id name email }" \
  --model "llama2"
```

## 🔍 Debugging Specific Components

### Database Operations

To see all SQL queries and database operations:

```bash
# Enable database query logging
export DATABASE_ECHO=true
docker-compose restart backend
```

### Redis Cache Operations

Monitor Redis operations:

```bash
# Connect to Redis and monitor commands
docker-compose exec redis redis-cli MONITOR
```

### GraphQL Schema Operations

Test schema introspection and validation:

```bash
# Test GraphQL validation
curl -X POST http://localhost:8000/api/v1/validation/validate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "{ users { id name email } }",
    "schema": "type User { id: ID! name: String! email: String! }"
  }'
```

## 📈 Performance Monitoring

### Application Performance

1. **Frontend Performance**:
   - Open browser dev tools > Performance tab
   - Use the debug utility: `window.debugManager.exportLogs()`

2. **Backend Performance**:
   - Monitor response times in logs
   - Check database query performance
   - Monitor Ollama model response times

3. **Model Performance**:
   - Token generation speed
   - Memory usage
   - Model loading times

### Resource Usage

Monitor Docker container resources:

```bash
# Real-time resource monitoring
docker stats

# Specific container logs with timestamps
docker-compose logs -f --timestamps backend
```

## 🛠️ Advanced Debugging

### Environment Variables for Maximum Debug

Use these environment variables for enhanced debugging:

```bash
# Maximum logging
export LOG_LEVEL=DEBUG
export DATABASE_ECHO=true
export OLLAMA_DEBUG=1
export OLLAMA_VERBOSE=1

# Frontend debugging
export VITE_DEBUG=true
export VITE_LOG_LEVEL=debug

# API debugging
export API_DEBUG=true
export GRAPHQL_INTROSPECTION=true
```

### Custom Debug Configuration

Create a custom debug configuration:

```bash
# Copy debug config
cp config/debug.env .env.local

# Modify as needed
vim .env.local

# Restart with new config
docker-compose --env-file .env.local up --build
```

### Network Traffic Monitoring

Monitor HTTP traffic between services:

```bash
# Monitor all container network traffic
docker run -it --rm --net container:mppw-mcp-backend-1 nicolaka/netshoot tcpdump -i eth0

# Monitor specific ports
docker run -it --rm --net container:mppw-mcp-backend-1 nicolaka/netshoot netstat -tulpn
```

## 🔧 Troubleshooting

### Common Issues

1. **Services not starting**:
   ```bash
   # Check Docker resources
   docker system df
   docker system prune -f
   
   # Restart with clean state
   docker-compose down -v
   docker-compose up --build
   ```

2. **Model not responding**:
   ```bash
   # Check Ollama status
   curl http://localhost:11434/api/tags
   
   # Pull a model manually
   docker-compose exec ollama ollama pull llama2
   ```

3. **Database connection issues**:
   ```bash
   # Check PostgreSQL logs
   docker-compose logs postgres
   
   # Test connection
   docker-compose exec postgres psql -U postgres -d mppw_mcp
   ```

### Log Analysis

1. **Search logs for errors**:
   ```bash
   docker-compose logs | grep -i error
   ```

2. **Export logs for analysis**:
   ```bash
   docker-compose logs > full-application-logs.txt
   ```

3. **Filter by service and time**:
   ```bash
   docker-compose logs --since="1h" backend | jq '.message'
   ```

## 📊 Debug Dashboard

### Browser Console Commands

In the frontend console, you can:

```javascript
// View all debug logs
debugManager.getAllLogs()

// Export logs to file
debugManager.exportLogs()

// Clear all logs
debugManager.clearLogs()

// Monitor API calls
debugManager.getAPILogs().filter(log => log.url.includes('translate'))

// Performance analysis
debugManager.getPerformanceLogs()
```

### Health Check Endpoints

Monitor service health:

```bash
# Backend health
curl http://localhost:8000/health

# Individual service health
curl http://localhost:8000/health/database
curl http://localhost:8000/health/ollama
curl http://localhost:8000/health/redis
```

## 🎯 Testing Scenarios

### 1. Complete Translation Flow

1. Start monitoring: `./logs/monitor_all.sh`
2. Open frontend: http://localhost:3000
3. Enter query: "Show me all products with their prices"
4. Watch logs for:
   - Frontend user interaction
   - API request to backend
   - FastMCP tool execution
   - Ollama model processing
   - Database queries (if applicable)
   - Response generation and delivery

### 2. Error Handling Testing

1. Test with invalid query
2. Test with unavailable model
3. Test with network interruption
4. Observe error propagation through logs

### 3. Performance Testing

1. Submit multiple queries simultaneously
2. Monitor response times
3. Check memory usage
4. Analyze token generation speed

## 📋 Monitoring Checklist

Before testing, ensure:

- [ ] All services are running and healthy
- [ ] Debug logging is enabled
- [ ] Monitoring scripts are accessible
- [ ] Browser console is open (for frontend testing)
- [ ] Ollama has at least one model available
- [ ] Database is accessible and initialized

During testing, monitor:

- [ ] Request/response flow through all services
- [ ] Model input prompts and output responses  
- [ ] Database queries and results
- [ ] Frontend user interactions
- [ ] Error handling and recovery
- [ ] Performance metrics

## 🔄 Continuous Monitoring

For ongoing development, set up:

1. **File watching**: Changes auto-restart services
2. **Log aggregation**: Centralized log collection
3. **Performance baseline**: Track performance over time
4. **Error alerting**: Automatic error notifications

```bash
# Watch for file changes
docker-compose up --build --watch

# Continuous health monitoring
watch -n 5 'curl -s http://localhost:8000/health | jq .'
```

This comprehensive setup gives you complete visibility into every aspect of your application's operation, from user interactions in the frontend to model processing in Ollama. 