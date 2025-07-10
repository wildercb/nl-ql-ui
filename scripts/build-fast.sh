#!/bin/bash

# Fast Docker build script using BuildKit
# This script optimizes Docker builds for speed without removing functionality

set -e

echo "🚀 Starting fast Docker build with BuildKit..."

# Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Build arguments for optimization
BUILD_ARGS="--build-arg BUILDKIT_INLINE_CACHE=1"

# Function to build with cache
build_with_cache() {
    local service=$1
    echo "📦 Building $service with cache optimization..."
    
    # Pull latest images for cache
    docker pull mppw-mcp-$service:latest 2>/dev/null || true
    
    # Build with cache
    docker-compose build --parallel --build-arg BUILDKIT_INLINE_CACHE=1 $service
    
    echo "✅ $service build completed"
}

# Function to build all services
build_all() {
    echo "🔨 Building all services in parallel..."
    
    # Pull latest images for cache
    docker pull mppw-mcp-backend:latest 2>/dev/null || true
    docker pull mppw-mcp-frontend:latest 2>/dev/null || true
    
    # Build all services in parallel
    docker-compose build --parallel --build-arg BUILDKIT_INLINE_CACHE=1
    
    echo "✅ All builds completed"
}

# Function to show build stats
show_stats() {
    echo ""
    echo "📊 Build Statistics:"
    echo "==================="
    docker images | grep mppw-mcp
    echo ""
    echo "💾 Disk usage:"
    docker system df
}

# Main execution
case "${1:-all}" in
    "backend")
        build_with_cache "backend"
        ;;
    "frontend")
        build_with_cache "frontend"
        ;;
    "all"|*)
        build_all
        ;;
esac

show_stats

echo "🎉 Fast build completed successfully!"
echo ""
echo "To start the services:"
echo "  docker-compose up -d"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f" 