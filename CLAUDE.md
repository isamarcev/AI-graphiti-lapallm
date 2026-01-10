# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hackathon project demonstrating an AI agent with long-term memory using:
- **Lapa LLM** - Ukrainian language model based on Gemma 12B
- **Graphiti** - Temporal knowledge graph for storing agent memory
- **LangGraph** - Framework for building stateful agents
- **Neo4j** - Graph database backend

The agent maintains conversation history in a knowledge graph, enabling it to remember facts across sessions and provide personalized responses in Ukrainian.

## Development Setup

### Prerequisites
- Python 3.12+
- Docker and Docker Compose (for Neo4j)
- 16GB+ RAM recommended (for 12B model)

### Initial Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Or use pyproject.toml
uv sync  # if using uv
# or
pip install -e .

# Copy environment template
cp .env.example .env

# Start Neo4j
docker-compose up -d

# Verify Neo4j is running
docker-compose ps
```

### Running the LLM Server

**Default: Hosted Lapathon API (Recommended)**

The project is configured to use the hosted Lapathon API by default:

```bash
# Configuration in .env
VLLM_BASE_URL=http://146.59.127.106:4000
VLLM_API_KEY=sk-your-team-key
VLLM_MODEL_NAME=lapa
USE_HOSTED_EMBEDDINGS=true
```

No local vLLM server needed! Just ensure your `.env` has the correct API key from your team administrator.

**Alternative: Local vLLM (for development)**

If you want to run locally instead:

```bash
# Option 1: vLLM local (requires GPU)
vllm serve lapa-llm/lapa-v0.1.2-instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096

# Option 2: vLLM via Docker
docker run --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model lapa-llm/lapa-v0.1.2-instruct

# Then update .env:
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL_NAME=lapa-llm/lapa-v0.1.2-instruct
USE_HOSTED_EMBEDDINGS=false
```

### Running Tests/Demos
```bash
# Launch Jupyter for demo notebook
jupyter notebook demo_flow.ipynb

# Or Jupyter Lab
jupyter lab demo_flow.ipynb
```

### Verification Commands
```bash
# Check Neo4j (via browser at http://localhost:7474)
# Login: neo4j / password123
docker-compose ps

# Test hosted API connection (optional)
python -c "
import asyncio
from openai import AsyncOpenAI

async def test():
    client = AsyncOpenAI(
        api_key='your-api-key',
        base_url='http://146.59.127.106:4000'
    )
    response = await client.chat.completions.create(
        model='lapa',
        messages=[{'role': 'user', 'content': 'Привіт!'}]
    )
    print('✅ Hosted API works!')

asyncio.run(test())
"
```

## Architecture

### Agent Flow
The LangGraph agent follows a 3-node pipeline:
1. **retrieve_memory** (agent/nodes.py:22) - Search Graphiti for relevant context
2. **generate_response** (agent/nodes.py:90) - Generate response with LLM using retrieved context
3. **save_to_memory** (agent/nodes.py:163) - Save conversation episode to graph

Graph definition in agent/graph.py defines the edge connections.

### Key Components

#### LLM Client (clients/llm_client.py)
- Wraps OpenAI-compatible API (vLLM or OpenAI)
- Supports structured outputs via JSON schema (vLLM) or native (OpenAI)
- Both async and sync interfaces available
- LangSmith tracing integrated via `@traceable` decorator

**Important**: For structured outputs with vLLM, requires vLLM 0.8.5+ with JSON mode support. Fallback to OpenAI API available via `USE_OPENAI_FALLBACK=true` in .env.

#### Graphiti Client (clients/graphiti_client.py)
- Supports both hosted Qwen embeddings and local sentence-transformers
- Uses `OpenAIGenericClient` from graphiti-core for vLLM compatibility
- Async context manager pattern: `async with GraphitiClient() as client:`
- BGE cross-encoder for reranking search results

**Embedder Selection:**
- `USE_HOSTED_EMBEDDINGS=true`: Uses hosted text-embedding-qwen API
- `USE_HOSTED_EMBEDDINGS=false`: Uses local sentence-transformers (paraphrase-multilingual-mpnet-base-v2)

**Critical**: Graphiti requires `build_indices_and_constraints()` on first run to create Neo4j indexes.

#### State Management (agent/state.py)
`AgentState` TypedDict maintains:
- `messages`: LangChain message history (accumulated via `add_messages`)
- `retrieved_context`: Search results from Graphiti
- `user_id`, `session_id`: For memory isolation
- `needs_memory_update`: Flag for save node
- `timestamp`: Current interaction time

### Configuration (config/settings.py)
Uses pydantic-settings to load from .env file:
- LLM: vLLM base URL, model name, temperature, max_tokens
- Neo4j: connection URI, credentials
- Graphiti: episode length, search limits, relevance threshold
- Embeddings: multilingual model for Ukrainian
- Agent: system prompt, conversation history limit

## Code Patterns

### LLM Generation with Structured Output
```python
from clients.llm_client import get_llm_client
from models.schemas import YourSchema

llm = get_llm_client()
result = await llm.generate_async(
    messages=[{"role": "user", "content": "..."}],
    response_format=YourSchema  # Pydantic model
)
```

### Graphiti Memory Operations
```python
from clients.graphiti_client import get_graphiti_client

graphiti = await get_graphiti_client()

# Search
results = await graphiti.search(query="...", limit=10)

# Add episode
await graphiti.add_episode(
    episode_body="User: ... Assistant: ...",
    episode_name=f"user_{timestamp}",
    source_description=f"user:{user_id}",
    reference_time=datetime.now()
)
```

### Agent Invocation
```python
from agent.graph import get_agent_app
from langchain_core.messages import HumanMessage

agent = get_agent_app()
result = await agent.ainvoke({
    "messages": [HumanMessage(content="Привіт!")],
    "user_id": "user_1",
    "session_id": "session_1",
    "timestamp": datetime.now(),
    "retrieved_context": None,
    "current_query": None,
    "needs_memory_update": False,
    "search_results": None,
    "message_count": 0
})
```

## Common Issues

### Hosted API Connection Issues
If you get connection errors with the hosted API:
- Verify your API key in `.env` (should start with `sk-`)
- Check network connectivity to `http://146.59.127.106:4000`
- Confirm with team administrator that your key is active
- Try the verification command above to test connection

### Embedding Dimension Mismatch
If Graphiti fails with vector dimension errors:
```python
# Test embeddings dimension
from openai import AsyncOpenAI
import asyncio

async def check_dim():
    client = AsyncOpenAI(
        api_key='your-key',
        base_url='http://146.59.127.106:4000'
    )
    response = await client.embeddings.create(
        input='test',
        model='text-embedding-qwen'
    )
    print(f"Dimension: {len(response.data[0].embedding)}")

asyncio.run(check_dim())
```
Then update `EMBEDDING_DIMENSION` in `.env` to match and rebuild Neo4j indices.

### vLLM Structured Output Failures
If LLM doesn't return valid JSON:
- Verify vLLM version: `pip show vllm` (need 0.8.5+)
- Check vLLM logs for JSON schema errors
- Use OpenAI fallback for testing: set `USE_OPENAI_FALLBACK=true`

### Graphiti Index Errors
First-time setup requires index creation:
```python
graphiti = await get_graphiti_client()
await graphiti.graphiti.build_indices_and_constraints()
```

### Neo4j Connection Issues
```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs neo4j

# Restart
docker-compose restart neo4j
```

### Memory/Performance
The 12B model requires significant resources:
- 16GB RAM minimum for CPU inference
- 12GB VRAM for GPU inference
- First embedding model load downloads ~400MB
- Consider quantized models for lower memory usage

## LangSmith Tracing (Optional)

To enable tracing for debugging:
```bash
# In .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_PROJECT=graphiti-lapa-demo
```

All LLM calls and agent nodes are decorated with `@traceable` for automatic instrumentation.

## Neo4j Queries

Useful Cypher queries for debugging:

```cypher
// View all nodes and relationships
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100

// Search for entities
MATCH (n:Entity) WHERE n.name CONTAINS 'keyword' RETURN n

// View episodes
MATCH (e:Episode) RETURN e ORDER BY e.created_at DESC LIMIT 10

// Count graph stats
MATCH (n) RETURN count(n) as nodes
MATCH ()-[r]->() RETURN count(r) as relationships
```

## File Structure Reference

- `config/settings.py` - All configuration from .env
- `clients/llm_client.py` - vLLM/OpenAI wrapper with structured outputs
- `clients/graphiti_client.py` - Graphiti setup with custom embedder
- `agent/state.py` - LangGraph state definition
- `agent/nodes.py` - Three processing nodes (retrieve, generate, save)
- `agent/graph.py` - Graph assembly and compilation
- `models/schemas.py` - Pydantic models for structured outputs
- `docker-compose.yml` - Neo4j 5.26 with APOC plugin

## Notes

- **Hosted by default**: Project configured for hosted Lapathon API (no local vLLM needed)
- **Embeddings**: Uses hosted text-embedding-qwen for faster setup
- **System prompt**: In Ukrainian (config/settings.py:86)
- **Context window**: Agent uses last N messages (configurable via MAX_CONVERSATION_HISTORY)
- **Temporal memory**: Episodes saved with timestamp for temporal graph queries
- **Privacy**: Memory isolation by user_id prevents cross-user information leakage
- **Models available**: `lapa` (default), `mamay` (alternative)