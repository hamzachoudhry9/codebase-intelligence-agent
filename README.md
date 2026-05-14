# ◈ Codebase Intelligence Agent

> A fully local, offline AI agent that ingests any software repository, understands it at the function level, and lets engineers query it conversationally. Zero cloud APIs. Zero data exposure.

**Evaluation results on 50 test cases:**

| Metric | Score |
|--------|-------|
| Task completion | **100%** (50/50, 0 errors, 0 timeouts) |
| LLM faithfulness | **0.856** |
| Keyword faithfulness | 0.769 |
| Tool precision | 0.543 |
| Tool recall | 0.770 |
| Avg latency | **30s** (first query) / **22s** (subsequent) |

---

## What it does

Large engineering organisations lose months of productivity when engineers onboard to an unfamiliar codebase. Senior engineers spend 30% of their time answering repeated questions from juniors. This agent solves that.

You type `"my CUDA kernel is giving an out-of-memory error"` and the agent:

1. **Checks its memory** — retrieves any past session where this error was solved
2. **Makes a plan** — decides which tools to use and in what order
3. **Executes the plan** — searches your codebase semantically, runs code, searches the web if needed
4. **Recovers from failures** — replans up to 2 times if a tool step fails
5. **Synthesises an answer** — with exact file names, function names, and working code
6. **Saves to memory** — the next time this error appears, the answer is already known

### Why this is different

| Capability | Most RAG tools | This agent |
|-----------|---------------|------------|
| Code understanding | Document-level | **Function-level** (tree-sitter) |
| Memory | Per-session only | **Persistent across all sessions** |
| LLM provider | Cloud APIs required | **100% local via Ollama** |
| IDE integration | None | **MCP server** (Cursor, VS Code) |
| Data privacy | Sent to cloud | **Never leaves the machine** |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Agent                          │
│                                                             │
│  memory_retrieval → planning → execution → [replan?]        │
│                                    │                        │
│                              ┌─────┴──────┐                 │
│                         tool_outputs      │                 │
│                              └─────┬──────┘                 │
│                                    ▼                        │
│                              synthesis → save_memory        │
└─────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
   FastAPI Server                       ChromaDB
   /query (blocking)                  ┌──────────────┐
   /query/stream (SSE)                │ project_docs │ ← code chunks
   /health                            │ session_mem  │ ← past sessions
   /sessions                          │ stackoverflow│ ← Q&A pairs
         │                            └──────────────┘
         ▼
   MCP Server ──→ Cursor / VS Code / Claude Desktop
         │
         ▼
   Streamlit UI (localhost:8501)
```

### LangGraph node graph

```
start
  │
  ▼
memory_retrieval_node   ← ChromaDB semantic search over past sessions
  │
  ▼
planning_node           ← Ollama llama3.1:8b generates JSON plan
  │
  ▼
execution_node ◄────────┐
  │                     │
  ├── success ──────────┤ (next step)
  │                     │
  └── failure ──► replan_node (max 2 times)
                        │
                        └── give up → synthesis_node
  │
  ▼ (all steps done)
synthesis_node          ← Ollama synthesises tool outputs into answer
  │
  ▼
save_memory_node        ← Embeds and stores session in ChromaDB
  │
  ▼
end
```

### Tools

| Tool | Description | Backed by |
|------|-------------|-----------|
| `search_docs` | Semantic search over ingested codebase | ChromaDB direct query + HuggingFace BGE |
| `web_search` | External docs and Stack Overflow | DuckDuckGo (3-attempt retry) |
| `execute_code` | Sandboxed Python subprocess | `subprocess.run` (10s timeout, import blocking) |
| `retrieve_memory` | Past debugging sessions | ChromaDB `session_memory` collection |

### Code-aware chunking

Files are chunked at the function and class level using [tree-sitter](https://tree-sitter.github.io/tree-sitter/), not at fixed token counts. Each chunk carries structured metadata:

```json
{
  "file": "agent/nodes.py",
  "function": "planning_node",
  "start_line": 63,
  "end_line": 108,
  "type": "function",
  "language": "python",
  "docstring": "Generate a minimal, tool-annotated plan..."
}
```

Supported languages: Python, C, C++, CUDA (`.cu`, `.cuh`), Markdown, RST.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | [python.org](https://python.org) |
| Ollama | Latest | [ollama.com](https://ollama.com) |
| Git | Any | For cloning |
| Docker (optional) | 24+ | For production deploy |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/codebase-intelligence-agent.git
cd codebase-intelligence-agent
```

### 2. Install Ollama and pull the model

**macOS:**
```bash
brew install ollama
ollama serve &
ollama pull llama3.1:8b
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
systemctl start ollama
ollama pull llama3.1:8b
```

**Windows:**
```
# Download the installer from https://ollama.com/download/windows
# After installing, open Ollama from the Start menu, then in PowerShell:
ollama pull llama3.1:8b
```

Verify:
```bash
curl http://localhost:11434/api/tags
# Should return JSON listing llama3.1:8b
```

### 3. Create a virtual environment and install dependencies

```bash
python -m venv venv

# macOS / Linux:
source venv/bin/activate

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# The defaults work for local development — no changes needed
```

`.env.example` contents:
```env
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PERSIST_DIR=./chroma_db
DOCS_DIR=./docs
REPO_ROOT=.
AGENT_API_KEY=dev-key-change-in-production
PORT=8000
AGENT_API_URL=http://localhost:8000
```

### 5. Build the knowledge base

```bash
# Index this repository (source code + docs):
python ingest/build_index.py --repo . --docs docs/

# Optionally add Stack Overflow Q&A (cached after first run):
python ingest/build_index.py --repo . --docs docs/ --scrape-so

# Index a different codebase:
python ingest/build_index.py --repo /path/to/your/repo
```

Expected output:
```
Loading embedding model (BAAI/bge-small-en-v1.5)... Ready in 3.2s
Found 37 files in .
Total chunks: 228
  python: 120 chunks
  markdown: 108 chunks
'project_docs' → 228 chunks

Sample chunks:
  [python  ] agent/nodes.py:planning_node — def planning_node(state: AgentState) ...
  [python  ] agent/tools.py:search_docs — @tool def search_docs(query: str) ...
```

---

## Running the agent

### Development mode (3 terminals)

**Terminal 1 — API server:**
```bash
source venv/bin/activate   # or .\venv\Scripts\Activate.ps1 on Windows
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Wait for:
```
[info] preload_graph_complete
[info] ollama_warmup_complete
[info] all_components_ready
INFO: Application startup complete.
```

**Terminal 2 — Streamlit UI:**
```bash
source venv/bin/activate
streamlit run ui/app.py
# Opens at http://localhost:8501
```

**Terminal 3 (optional) — MCP server:**
```bash
source venv/bin/activate
python mcp_server/server.py
```

### Production mode (Docker)

```bash
docker-compose up --build

# On first run, build the knowledge base inside the container:
docker-compose exec agent python ingest/build_index.py --repo . --docs docs/
```

Services:
- API: http://localhost:8000
- UI: http://localhost:8501
- Ollama: http://localhost:11434

---

## API reference

### POST /query

Blocking endpoint. Returns the complete result.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-change-in-production" \
  -d '{"query": "What does the planning_node function do?"}'
```

Response:
```json
{
  "answer": "**Summary:** The `planning_node` function is defined in `agent/nodes.py` (lines 63–108)...",
  "plan": ["[search_docs] What does the planning_node function do?", "[retrieve_memory] Check past sessions..."],
  "tools_used": ["search_docs", "retrieve_memory"],
  "replan_count": 0,
  "latency_s": 22.3
}
```

### POST /query/stream

Server-Sent Events. Returns events as the agent works.

```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-change-in-production" \
  -d '{"query": "Debug a CUDA out-of-memory error"}' \
  --no-buffer
```

Event types:
```
data: {"type": "plan",        "data": ["[search_docs] ...", "[execute_code] ..."]}
data: {"type": "tool_call",   "data": {"tool": "search_docs", "task": "..."}}
data: {"type": "tool_result", "data": {"tool": "search_docs", "result": "...", "success": true}}
data: {"type": "replan",      "data": {"count": 1}}
data: {"type": "answer",      "data": "Full synthesised answer text..."}
data: {"type": "done",        "data": {"latency_s": 22.3}}
```

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "2.1.0",
  "ollama_warmed_up": true,
  "index": {
    "project_docs_chunks": 228,
    "session_memory_sessions": 12
  }
}
```

### GET /sessions

Returns the 20 most recent memory sessions.

```bash
curl http://localhost:8000/sessions \
  -H "X-API-Key: dev-key-change-in-production"
```

---

## IDE integration via MCP

The MCP server exposes 5 tools to any MCP-compatible IDE.

### Cursor

Create `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "codebase-intelligence": {
      "command": "python",
      "args": ["/absolute/path/to/project/mcp_server/server.py"]
    }
  }
}
```

Restart Cursor. The tools appear in the tool panel.

### VS Code + Continue extension

Add to `~/.continue/config.json`:

```json
{
  "mcpServers": [
    {
      "name": "codebase-intelligence",
      "command": "python",
      "args": ["/absolute/path/to/project/mcp_server/server.py"]
    }
  ]
}
```

### Available MCP tools

| Tool | Description |
|------|-------------|
| `mcp_search_docs` | Semantic search over ingested codebase |
| `mcp_web_search` | DuckDuckGo web search |
| `mcp_execute_code` | Sandboxed Python execution |
| `mcp_retrieve_memory` | Past session retrieval |
| `mcp_full_query` | Full agent pipeline via API |

---

## Running evaluations

```bash
# API server must be running
python eval/evaluator.py --cases eval/test_cases.json --out eval/results.json

# Keyword-only mode (faster, less accurate):
python eval/evaluator.py --no-llm-judge
```

### Results (50 test cases, 3 categories)

```
=== Evaluation Summary ===
task_completion_rate : 1.000   (50/50)
avg_llm_faithfulness : 0.856
avg_kw_faithfulness  : 0.769
avg_tool_precision   : 0.543
avg_tool_recall      : 0.770
avg_latency_s        : 30.02
n_errors             : 0
n_timeouts           : 0
n_low_faithfulness   : 0

By category:
  documentation_lookup            n=24  faithfulness=0.85  precision=0.729
  code_generation_and_verification n=20  faithfulness=0.85  precision=0.217
  debugging                        n=6   faithfulness=0.90  precision=0.889
```

---

## Adding your own codebase

```bash
# Point the indexer at any local repository:
python ingest/build_index.py \
  --repo /path/to/your/company/repo \
  --docs /path/to/your/company/docs

# Add error log examples (improves debugging answers):
mkdir -p data/error_logs
# Drop .txt files with error tracebacks + fixes into data/error_logs/
python ingest/build_index.py --repo . --no-wipe  # append without wiping
```

---

## Project structure

```
.
├── agent/
│   ├── graph.py          # LangGraph StateGraph definition
│   ├── nodes.py          # 6 nodes: memory, planning, execution, replan, synthesis, save
│   ├── state.py          # AgentState TypedDict
│   └── tools.py          # 4 tools: search_docs, web_search, execute_code, retrieve_memory
├── api/
│   └── main.py           # FastAPI server (blocking + streaming endpoints)
├── ingest/
│   ├── build_index.py    # Knowledge base builder (code + docs + Stack Overflow)
│   └── code_chunker.py   # Tree-sitter chunking (Python, C++, CUDA, Markdown)
├── memory/
│   └── session_store.py  # ChromaDB session memory with BGE embeddings
├── mcp_server/
│   └── server.py         # FastMCP server exposing 5 tools to IDEs
├── ui/
│   └── app.py            # Streamlit interface with SSE streaming
├── eval/
│   ├── evaluator.py      # LLM-as-judge + keyword evaluation harness
│   └── test_cases.json   # 50 test cases across 3 categories
├── docs/                 # Documentation ingested into the knowledge base
├── data/
│   └── error_logs/       # Error log examples (add your own .txt files here)
├── chroma_settings.py    # ChromaDB client factory (telemetry disabled)
├── Dockerfile            # API server container
├── Dockerfile.ui         # Streamlit container
├── docker-compose.yml    # Orchestrates Ollama + agent + UI
├── requirements.txt      # Python dependencies
└── .env.example          # Environment variable template
```

---

## Technical decisions

**Why Ollama instead of OpenAI/Anthropic?**
Enterprise codebases cannot send proprietary source code to external APIs. Ollama runs `llama3.1:8b` entirely on-device. Zero tokens leave the machine.

**Why direct ChromaDB queries instead of LlamaIndex?**
The index is built using raw ChromaDB inserts with custom metadata. LlamaIndex's `ChromaVectorStore` wrapper expects its own internal node format (`_node_content`, `_node_type`) and returns 0 results when reading externally-inserted embeddings. Direct queries against the collection are faster, simpler, and always work.

**Why tree-sitter instead of fixed-token chunking?**
Fixed-token chunking splits functions in half. A search for `planning_node` can return the bottom half of `execution_node` followed by the top half of `planning_node` — neither is usable. Tree-sitter parses the AST and chunks at function/class boundaries, so every chunk is a complete, semantically meaningful unit.

**Why LLM-as-judge for evaluation?**
Keyword matching counts `"memory exhausted"` as 0 against a test case expecting `"out of memory"`. The local Ollama model judges semantic correctness, handling synonyms, paraphrasing, and partial answers — giving a metric that can actually be defended.

---

## License

MIT

---

## Author

Built as a demonstration of production-grade agentic AI systems for enterprise developer productivity use cases.
