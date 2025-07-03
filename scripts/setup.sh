#!/bin/bash

# MPPW MCP Project Setup Script
echo "🚀 Setting up MPPW MCP - Natural Language to GraphQL Translation Service"
echo "========================================================================"

# Check if required commands exist
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ $1 is not installed. Please install $1 and try again."
        exit 1
    fi
}

echo "📋 Checking prerequisites..."
check_command "python3"
check_command "node"
check_command "npm"
check_command "docker"

# Get Python and Node versions
PYTHON_VERSION=$(python3 --version)
NODE_VERSION=$(node --version)
NPM_VERSION=$(npm --version)
DOCKER_VERSION=$(docker --version)

echo "✅ Prerequisites check passed:"
echo "   $PYTHON_VERSION"
echo "   Node.js $NODE_VERSION"
echo "   npm $NPM_VERSION"
echo "   $DOCKER_VERSION"
echo ""

# Setup backend
echo "🐍 Setting up Python backend..."
cd backend

echo "   Creating virtual environment..."
python3 -m venv venv

echo "   Activating virtual environment..."
source venv/bin/activate

echo "   Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "   ✅ Backend setup complete!"
cd ..

# Setup frontend
echo "🌐 Setting up Vue.js frontend..."
cd frontend

echo "   Installing Node.js dependencies..."
npm install

echo "   ✅ Frontend setup complete!"
cd ..

# Create environment files
echo "⚙️  Creating environment configuration..."

# Backend .env
if [ ! -f "backend/.env" ]; then
    echo "   Creating backend/.env from .env.example..."
    cp backend/.env.example backend/.env 2>/dev/null || echo "   Warning: backend/.env.example not found, you'll need to create backend/.env manually"
fi

# Generate a random secret key for the backend
if [ -f "backend/.env" ]; then
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/your-secret-key-here-change-in-production/$SECRET_KEY/" backend/.env
    else
        # Linux
        sed -i "s/your-secret-key-here-change-in-production/$SECRET_KEY/" backend/.env
    fi
    echo "   ✅ Generated secure secret key for backend"
fi

# Docker setup
echo "🐳 Setting up Docker environment..."
echo "   Building Docker images..."
docker-compose build

echo ""
echo "🎉 Setup complete! Here's what you can do next:"
echo ""
echo "🚀 Quick Start:"
echo "   1. Start Ollama service:"
echo "      ollama serve"
echo ""
echo "   2. Pull a model (in another terminal):"
echo "      ollama pull llama2"
echo ""
echo "   3. Start the full application with Docker:"
echo "      docker-compose up -d"
echo ""
echo "   4. Or start services individually:"
echo ""
echo "      Backend API:"
echo "      cd backend && source venv/bin/activate && python -m uvicorn api.main:app --reload"
echo ""
echo "      MCP Server:"
echo "      cd backend && source venv/bin/activate && python run_mcp_server.py"
echo ""
echo "      Frontend:"
echo "      cd frontend && npm run dev"
echo ""
echo "📱 Access Points:"
echo "   Frontend:        http://localhost:3000"
echo "   Backend API:     http://localhost:8000"
echo "   API Docs:        http://localhost:8000/docs"
echo "   Health Check:    http://localhost:8000/health"
echo ""
echo "📚 Next Steps:"
echo "   - Configure your database connection in backend/.env"
echo "   - Set up your preferred Ollama models"
echo "   - Customize the GraphQL schema context"
echo "   - Review the documentation at http://localhost:3000/docs"
echo ""
echo "🔧 Troubleshooting:"
echo "   - Check logs: docker-compose logs -f [service-name]"
echo "   - Reset database: docker-compose down -v && docker-compose up -d"
echo "   - Update dependencies: git pull && ./scripts/setup.sh"
echo ""
echo "Happy coding! 🎯" 