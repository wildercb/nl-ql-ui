# MPPW MCP - Natural Language to GraphQL Translation Service

**Full Name**: MPPW (Multi-Provider Protocol Workbench) Model Context Protocol

A powerful, scalable FastMCP server that translates natural language queries into GraphQL using AI models through Ollama, with comprehensive validation, schema introspection, interactive assistance capabilities, and **real-time LLM interaction tracking** using MongoDB.

## 🚀 New FastMCP Architecture

This project has been completely rewritten using **FastMCP 2.10.1**, providing:

- **Modern MCP Implementation**: Using the latest FastMCP framework with decorator-based APIs
- **Organized Structure**: Clear separation of tools, resources, and prompts for easy scaling
- **Rich Context Management**: Comprehensive logging and progress reporting
- **Interactive Prompts**: Step-by-step guidance for users at all skill levels
- **Resource Access**: Query history, templates, and analytics via MCP resources
- **MongoDB Integration**: Document-based storage with flexible schema design
- **🆕 LLM Tracking**: Comprehensive monitoring of all model interactions
- **Real-time Analytics**: Live dashboards and performance monitoring

## 🗄️ Enhanced Database & Monitoring

### MongoDB Document Storage
- **Flexible Schema**: JSON documents ideal for LLM interaction data
- **Scalable Analytics**: Native aggregation pipelines for real-time insights
- **Automatic Indexing**: Optimized queries for all collections
- **TTL Cleanup**: Automatic cleanup of old sessions and interactions

### 📊 Comprehensive LLM Tracking
- **Every Interaction Logged**: Complete prompt and response capture
- **Performance Metrics**: Processing time, token usage, confidence scores
- **Session Management**: Grouped interactions for analysis
- **Model Analytics**: Usage patterns and performance comparisons
- **Data Export**: Full session and interaction export capabilities

### 🔍 Real-time Monitoring
- **Analytics Dashboard**: `/analytics/overview` - High-level metrics and trends
- **Session Details**: `/analytics/sessions/{id}` - Complete interaction timeline
- **Model Statistics**: `/analytics/models/stats` - Performance by model
- **Database Health**: `/analytics/database/stats` - Collection metrics and indexes

### Access Points for Monitoring
| Endpoint | Purpose |
|----------|---------|
| `GET /analytics/overview` | System-wide analytics dashboard |
| `GET /analytics/interactions` | Paginated interaction history |
| `GET /analytics/sessions/{id}` | Detailed session analysis |
| `GET /analytics/export/session/{id}` | Complete session export |

## 📁 Project Structure

```
backend/
├── fastmcp/                    # FastMCP Server Implementation
│   ├── main.py                # Main server with core tools
│   ├── tools/                 # MCP Tools (organized by function)
│   │   ├── translation.py     # Natural language to GraphQL translation
│   │   ├── validation.py      # GraphQL query validation & analysis
│   │   ├── models.py          # AI model management (Ollama)
│   │   └── schema.py          # Schema introspection & analysis
│   ├── resources/             # MCP Resources (data access)
│   │   ├── queries.py         # Query history & templates
│   │   └── history.py         # User sessions & analytics
│   └── prompts/               # MCP Prompts (interactive guidance)
│       ├── translation.py     # Translation assistance & learning
│       └── analysis.py        # Schema exploration & performance
├── services/                  # Core Business Logic
│   ├── translation_service.py # AI-powered translation engine
│   ├── validation_service.py  # GraphQL validation & optimization
│   └── ollama_service.py      # Ollama model integration
├── api/                       # FastAPI REST Interface
├── models/                    # Database Models (SQLAlchemy)
├── config/                    # Configuration Management
└── cli.py                     # FastMCP Server Entry Point

frontend/                      # Vue.js 3 Application
docs/                         # Comprehensive Documentation
```

## 🛠 FastMCP Tools

### Translation Tools
- **`translate_query`**: Convert natural language to GraphQL with confidence scoring
- **`batch_translate`**: Process multiple queries efficiently with concurrency control
- **`translate_with_context`**: Enhanced translation using schema and examples
- **`improve_translation`**: Refine translations based on user feedback

### Validation Tools
- **`validate_graphql`**: Comprehensive query validation with best practices analysis
- **`validate_with_schema`**: Schema-aware validation with field and type checking
- **`get_validation_suggestions`**: Detailed improvement recommendations
- **`analyze_query_complexity`**: Performance analysis with optimization hints

### Model Management Tools
- **`list_available_models`**: View all Ollama models
- **`get_model_info`**: Detailed model information and capabilities
- **`pull_model`**: Download new AI models
- **`delete_model`**: Remove unused models

### Schema Tools
- **`introspect_schema`**: Discover GraphQL schema from endpoints
- **`analyze_schema`**: Comprehensive schema analysis with insights
- **`generate_query_examples`**: Create example queries from schema

## 📚 MCP Resources

### Query Resources
- **`query://saved/{query_id}`**: Access saved translations
- **`query://recent`**: Recent translation history
- **`query://popular`**: Popular query patterns
- **`query://templates/{category}`**: Domain-specific templates
- **`query://statistics`**: Usage analytics and metrics

### History Resources
- **`history://user/{user_id}`**: User translation history
- **`history://session/{session_id}`**: Session-specific data

### Configuration Resources
- **`config://server`**: Server configuration and status

## 🎓 Interactive Prompts

### Translation Assistance
- **`translation_assistant`**: Skill-level appropriate guidance (beginner/intermediate/advanced)
- **`query_optimization`**: Performance optimization guidance
- **`domain_translation`**: Domain-specific patterns (e-commerce, social, analytics)

### Analysis Guidance
- **`schema_exploration`**: Step-by-step schema understanding
- **`performance_analysis`**: Performance bottleneck identification and optimization

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Ollama (for local AI models)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mppw-mcp
   ```

2. **Run the setup script**
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. **Start with comprehensive monitoring**
   ```bash
   # Use the enhanced monitoring script for complete visibility
   chmod +x scripts/run-with-monitoring.sh
   ./scripts/run-with-monitoring.sh
   ```

   This will start:
   - ✅ MongoDB database with collections and indexes
   - ✅ FastMCP server with LLM tracking enabled
   - ✅ REST API with analytics endpoints
   - ✅ Frontend with debug utilities
   - ✅ Comprehensive logging and monitoring

4. **Alternative: Standard Docker Compose**
   ```bash
   docker-compose up -d
   ```

### Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:3000 | Main UI with debug tools |
| **REST API** | http://localhost:8000 | Translation API + Analytics |
| **API Documentation** | http://localhost:8000/docs | Interactive API docs |
| **LLM Analytics** | http://localhost:8000/analytics/overview | Real-time tracking dashboard |
| **MongoDB** | mongodb://localhost:27017 | Direct database access |

### Running the FastMCP Server

**Direct Python execution:**
```bash
cd backend
python cli.py
```

**With specific configuration:**
```bash
OLLAMA_BASE_URL=http://localhost:11434 python cli.py
```

### Quick Monitoring Commands

```bash
# View recent LLM interactions
curl http://localhost:8000/analytics/interactions?limit=10

# Get system overview
curl http://localhost:8000/analytics/overview

# Export a session's complete data
curl http://localhost:8000/analytics/export/session/{session_id}

# Check database health
curl http://localhost:8000/analytics/database/stats
```

## 🔧 Usage Examples

### Basic Translation
```python
# Using the MCP client
result = await mcp_client.call_tool("translate_query", {
    "natural_query": "Find users with gmail addresses",
    "schema_context": "type User { id: ID! name: String! email: String! }"
})

print(result["graphql_query"])
# Output: { users(where: { email: { contains: "gmail" } }) { id name email } }
```

### Batch Processing
```python
result = await mcp_client.call_tool("batch_translate", {
    "queries": [
        "Find active users",
        "Get products under $50",
        "Show recent orders"
    ],
    "concurrent_limit": 3
})

print(f"Translated {result['successful']}/{result['total_queries']} queries")
```

### Schema Introspection
```python
schema_data = await mcp_client.call_tool("introspect_schema", {
    "endpoint": "https://api.example.com/graphql",
    "headers": {"Authorization": "Bearer token"}
})

print(f"Found {len(schema_data['schema']['types'])} types")
```

### Interactive Learning
```python
# Get beginner-friendly guidance
guidance = await mcp_client.get_prompt("translation_assistant", {
    "difficulty": "beginner",
    "query_type": "ecommerce"
})
```

## 🔍 Advanced Features

### Custom Model Providers
The system supports multiple AI providers through a plugin architecture:

```python
# Custom provider implementation
class CustomProvider(BaseModelProvider):
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        # Your custom model integration
        pass

# Register with the factory
ModelProviderFactory.register("custom", CustomProvider)
```

### Domain-Specific Validation
```python
# E-commerce specific validation
class EcommerceValidator(BaseValidator):
    def validate_product_query(self, query: str) -> ValidationResult:
        # Custom domain validation
        pass
```

### Performance Monitoring
```python
# Built-in performance tracking
performance_data = await mcp_client.get_resource("query://statistics")
print(f"Average confidence: {performance_data['average_confidence']}")
print(f"Success rate: {performance_data['success_rate']}%")
```

## 🌐 API Endpoints

The FastAPI REST interface provides HTTP access to core functionality:

- **POST** `/api/v1/translation/translate` - Translate natural language to GraphQL
- **POST** `/api/v1/translation/batch` - Batch translation
- **POST** `/api/v1/validation/validate` - Validate GraphQL queries
- **GET** `/api/v1/models/` - List available AI models
- **GET** `/health/` - Health check

## 📖 Documentation

- **[Usage Guide](docs/USAGE_GUIDE.md)** - Comprehensive usage instructions
- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and patterns
- **[Examples Guide](docs/EXAMPLES.md)** - Practical implementation examples
- **[Configuration Guide](docs/CONFIGURATION.md)** - Environment setup
- **[Validation Guide](docs/VALIDATION_GUIDE.md)** - Custom validation rules

## 🛡 Security & Performance

### Security Features
- Rate limiting and request validation
- Secure configuration management
- SQL injection prevention
- Input sanitization

### Performance Optimizations
- Connection pooling for databases
- Redis caching for frequent queries
- Async processing for I/O operations
- Query complexity analysis

## 🧪 Testing

```bash
# Run all tests
pytest

# Test specific components
pytest tests/test_translation.py
pytest tests/test_validation.py

# Test FastMCP tools
pytest tests/test_fastmcp_tools.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions  
- **Documentation**: `/docs` directory

---

**Built with FastMCP 2.10.1** - Modern, scalable, and developer-friendly MCP implementation. 