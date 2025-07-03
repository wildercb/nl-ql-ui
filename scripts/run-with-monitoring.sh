#!/bin/bash

# MPPW MCP Application - Complete Monitoring Setup
# This script starts the entire application stack with comprehensive logging

set -e

echo "🚀 Starting MPPW MCP Application with Full Monitoring..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create logs directory
mkdir -p logs

# Set development environment with debug logging
export ENV=development
export DEBUG=true
export LOG_LEVEL=DEBUG
export LOG_FORMAT=json
export DATABASE_ECHO=true
export PROMETHEUS_METRICS=true

# Ollama configuration for detailed logging
export OLLAMA_DEBUG=1
export OLLAMA_VERBOSE=1

# API configuration for maximum visibility
export API_DEBUG=true
export GRAPHQL_INTROSPECTION=true
export GRAPHQL_PLAYGROUND=true

print_status "Environment configured for maximum debugging"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Clean up any existing containers
print_status "Cleaning up existing containers..."
docker-compose down --remove-orphans

# Pull latest images
print_status "Pulling latest images..."
docker-compose pull

# Start the application stack
print_status "Starting application stack..."
docker-compose up --build -d

# Wait for services to be ready
print_status "Waiting for services to initialize..."
sleep 10

# Check service health
check_service() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Checking $service..."
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            print_status "$service is ready! ✅"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    print_error "$service failed to start after $max_attempts attempts"
    return 1
}

# Check all services
check_service "Backend API" "http://localhost:8000/health"
check_service "Frontend" "http://localhost:3000"
check_service "Ollama" "http://localhost:11434/api/tags"

# Pull a model if none exists
print_status "Ensuring Ollama has a model available..."
docker-compose exec ollama ollama pull llama2 || print_warning "Failed to pull llama2 model"

# Display service URLs
echo ""
echo "🌟 Application is running! Access points:"
echo "  Frontend:     http://localhost:3000"
echo "  Backend API:  http://localhost:8000"
echo "  API Docs:     http://localhost:8000/docs"
echo "  Ollama:       http://localhost:11434"
echo ""

# Start log monitoring
print_status "Starting log monitoring in separate terminals..."

# Function to create monitoring script
create_monitor_script() {
    local service=$1
    local title=$2
    cat > "logs/monitor_${service}.sh" << EOF
#!/bin/bash
echo "📊 Monitoring ${title} logs..."
echo "Press Ctrl+C to stop"
docker-compose logs -f --tail=50 ${service}
EOF
    chmod +x "logs/monitor_${service}.sh"
}

# Create monitoring scripts for each service
create_monitor_script "backend" "Backend API"
create_monitor_script "fastmcp" "FastMCP Server"
create_monitor_script "frontend" "Frontend"
create_monitor_script "ollama" "Ollama"

echo "📝 Log monitoring scripts created:"
echo "  Backend logs:   ./logs/monitor_backend.sh"
echo "  FastMCP logs:   ./logs/monitor_fastmcp.sh"
echo "  Frontend logs:  ./logs/monitor_frontend.sh"
echo "  Ollama logs:    ./logs/monitor_ollama.sh"
echo ""

# Create combined monitoring script
cat > "logs/monitor_all.sh" << 'EOF'
#!/bin/bash
echo "📊 Starting combined log monitoring..."
echo "This will show logs from all services. Press Ctrl+C to stop."
echo ""

# Start monitoring in background with prefixes
(docker-compose logs -f --tail=20 backend 2>&1 | sed 's/^/[BACKEND] /') &
(docker-compose logs -f --tail=20 fastmcp 2>&1 | sed 's/^/[FASTMCP] /') &
(docker-compose logs -f --tail=20 frontend 2>&1 | sed 's/^/[FRONTEND] /') &
(docker-compose logs -f --tail=20 ollama 2>&1 | sed 's/^/[OLLAMA] /') &

# Wait for all background jobs
wait
EOF
chmod +x "logs/monitor_all.sh"

echo "🔍 Combined monitoring: ./logs/monitor_all.sh"
echo ""

# Create testing script
cat > "logs/test_complete_flow.sh" << 'EOF'
#!/bin/bash

echo "🧪 Testing Complete Application Flow..."

# Test 1: Health check
echo "1️⃣ Testing health endpoints..."
curl -s http://localhost:8000/health | jq '.' || echo "❌ Backend health check failed"

# Test 2: MCP server info
echo -e "\n2️⃣ Testing FastMCP server..."
# Note: This would require MCP client connection

# Test 3: Translation flow
echo -e "\n3️⃣ Testing translation API..."
curl -X POST http://localhost:8000/api/v1/translation/translate \
  -H "Content-Type: application/json" \
  -d '{
    "natural_query": "Get all users with their names and emails",
    "schema_context": "User { id name email }",
    "model": "llama2"
  }' | jq '.' || echo "❌ Translation test failed"

# Test 4: Validation
echo -e "\n4️⃣ Testing validation..."
curl -X POST http://localhost:8000/api/v1/validation/validate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "{ users { id name email } }"
  }' | jq '.' || echo "❌ Validation test failed"

# Test 5: Model info
echo -e "\n5️⃣ Testing model management..."
curl -s http://localhost:8000/api/v1/models/list | jq '.' || echo "❌ Model list failed"

echo -e "\n✅ Test flow completed!"
EOF
chmod +x "logs/test_complete_flow.sh"

echo "🧪 Testing script: ./logs/test_complete_flow.sh"
echo ""

print_status "Setup complete! You can now:"
echo "  1. Open http://localhost:3000 in your browser"
echo "  2. Run ./logs/monitor_all.sh to see all logs"
echo "  3. Run ./logs/test_complete_flow.sh to test the API"
echo "  4. Check individual service logs with monitor_<service>.sh scripts"
echo ""

print_warning "To stop the application: docker-compose down" 