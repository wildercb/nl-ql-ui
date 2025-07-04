# Data Pipeline – Query → Data → UI

This document explains **exactly** how the natural-language→GraphQL→MongoDB data flow works in the MPPW-MCP stack after the July 2025 multi-agent upgrade.

---

## 1. High-level flow

```
User NL query → Translator (phi3) → GraphQL ➜ DataQueryService ➜ Mongo-content DB ➜ JSON results → Vue UI
```

1. **HomeView.vue** – user enters NL query and presses *Multi-Agent*.
2. **Agent orchestration** produces:
   * rewritten query (Rewriter agent)
   * GraphQL query (Translator agent)
   * review verdict (Reviewer agent)
3. When the `review` SSE event fires **and** `placeholder.graphqlQuery` is set, the front-end calls
   ```http
   POST /api/data/query { "graphql_query": "…" }
   ```
4. **DataQueryService** (new) parses the GraphQL string, converts to a *very simple* Mongo `find()` + projection + limit, and runs it against the **mongo_data** container.
5. Results are streamed back to the front-end and rendered by **DataResults.vue**.

---

## 2. Docker services

```yaml
docker-compose.yml
services:
  mongo:          # original metadata DB (beanie models)
  mongo_data:     # NEW – houses actual content you want to retrieve
```

Ports:
* `mongo`     – localhost:27017
* `mongo_data`– localhost:27018 (internal 27017 in container)

Environment vars passed to **backend** service:
```
DATA_MONGODB_URL=mongodb://mongo_data:27017
DATA_MONGODB_DATABASE=mppw_content
```

---

## 3. Configuration (backend/config/settings.py)

```python
class DataDatabaseSettings(BaseSettings):
    url: str = "mongodb://mongo_data:27018"
    database: str = "mppw_content"
    …

class Settings(BaseSettings):
    data_database: DataDatabaseSettings = Field(default_factory=DataDatabaseSettings)
```

Change values at runtime via env vars `DATA_MONGODB_URL`, `DATA_MONGODB_DATABASE`.

---

## 4. Backend components

### services/data_query_service.py
* Lazily opens a Motor connection to the *content* DB.
* Regex-parses the limited GraphQL pattern:  
  `query { <collection>(limit: <n>) { fieldA fieldB } }`
* Maps → `db.<collection>.find({}, {fieldA:1, fieldB:1}).limit(<n>)`
* Returns docs with `_id` stripped.

### api/routes/data_query.py
`POST /api/data/query` – calls service and wraps results.

### Inclusion
Added to `api/main.py` via `app.include_router(data_query.router)`.

### Cypher / Neo4j upgrade (July 2025)

We now forward the generated GraphQL to Neo4j via a *very naive* GraphQL→Cypher translator in `services/cypher_query_service.py`.

* Pattern supported: `query { NodeLabel(limit: N) { field1 field2 } }`
* Cypher produced: `MATCH (n:NodeLabel) RETURN n.field1 AS field1, … LIMIT N`.
* Connection details controlled by `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` env-vars (see `Neo4jSettings`).

The front-end displays the generated GraphQL in **GraphQLQueryBox**; the user can inspect/edit later iterations and press **Send** to execute.  Toggle `autoSendGraphQL` in `HomeView.vue` to run automatically.

---

## 5. Front-end components

### components/ChatStream.vue
Streams agent conversation.

### components/DataResults.vue *(NEW)*
* Accepts `results` + `loading` props.
* Displays a tidy scrollable card list.
* Auto-detects `.png/.jpg/.gif/...` → `<img>` and `.mp4/.webm/...` → `<video controls>`.

### HomeView.vue
* Maintains `dataResults` + `isDataLoading` ref.
* After SSE `complete`, calls `fetchDataResults()` which hits `/api/data/query`.
* Renders `<DataResults>` under chat stream.

---

## 6. Updating the schema / mapping

Because the translator currently emits **raw GraphQL**, only the collection & field names must match your Mongo documents.

* **Add a new collection** → simply insert docs into `mongo_data`.
* **Rename a field** → update both the data and ensure translator agents know the new name.
* **Want advanced filtering/joins?**
  * Replace the regex translator in `DataQueryService._parse_graphql()` with a proper GraphQL → aggregation mapper.
  * Or inject your own `mapping.yaml` and read it before building the Mongo query.

---

## 7. Extension hooks for future agents

1. **Better parser** – swap `_parse_graphql` implementation.
2. **Aggregation pipelines** – after parsing, call `coll.aggregate([...])` instead of `find`.
3. **Access control** – validate `_user` inside the route (currently no auth required).
4. **Streaming** – instead of returning JSON in one go, yield SSE events for large result sets.

---

## 8. Running locally

```bash
# Rebuild containers
docker compose up --build

# Wait until backend healthcheck passes then visit
http://localhost:3000  # Vue front-end
```

To inspect content DB:
```
docker exec -it mppw-mcp-mongo_data-1 mongosh mppw_content
```

---

## 9. Testing quick-start

```js
// inside the mongo_data shell
use mppw_content

// sample collection and doc
insert into getThermalScans
{
  id: "scan1",
  timestamp: ISODate(),
  location: "Printer12",
  temperature_reading: 72,
  image_url: "https://picsum.photos/200"
}
```
Run in UI: "Get all thermal scans (limit 10)"  → Agents → GraphQL → DataResults card shows doc & image preview.

---

## 10. File overview

```
backend/
  services/data_query_service.py
  api/routes/data_query.py
frontend/
  components/DataResults.vue
  views/HomeView.vue  <-- integration
docker-compose.yml    <-- mongo_data service
```

---

Happy querying! 🎉 