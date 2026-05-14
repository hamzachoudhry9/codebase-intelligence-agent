import json, os, sys, time, traceback
from typing import AsyncIterator

os.environ["ANONYMIZED_TELEMETRY"] = "False"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import structlog, uvicorn
from chroma_settings import get_chroma_client
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

load_dotenv()
log = structlog.get_logger()

_API_KEY = os.getenv("AGENT_API_KEY", "dev-key-change-in-production")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str = Security(_api_key_header)):
    if _API_KEY == "dev-key-change-in-production":
        return
    if key != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")

app = FastAPI(
    title="Codebase Intelligence Agent",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent_graph = None

def get_graph():
    global _agent_graph
    if _agent_graph is None:
        from agent.graph import agent_graph
        _agent_graph = agent_graph
    return _agent_graph

@app.on_event("startup")
async def preload_components():
    import httpx
    log.info("preload_graph_started")
    get_graph()
    log.info("preload_graph_complete")
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            log.info("ollama_warmup_started", model="llama3.1:8b")
            await client.post(
                f"{base}/api/generate",
                json={"model": "llama3.1:8b", "prompt": "ready", "stream": False},
            )
        log.info("ollama_warmup_complete")
    except Exception as e:
        log.warning("ollama_warmup_failed", error=str(e))
    log.info("all_components_ready")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    plan: list
    tools_used: list
    replan_count: int
    latency_s: float

def _make_initial_state(query: str) -> dict:
    return {
        "query": query,
        "past_context": "",
        "plan": [],
        "current_step_index": 0,
        "messages": [],
        "tool_outputs": [],
        "final_answer": "",
        "replan_count": 0,
        "done": False,
    }

@app.post("/query", response_model=QueryResponse)
async def query_agent(req: QueryRequest, _: None = Depends(verify_api_key)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    t0 = time.time()
    log.info("query_received", query=req.query[:120])
    try:
        result = get_graph().invoke(_make_initial_state(req.query))
    except Exception as exc:
        log.error("agent_failed", error=str(exc), traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}")
    tools_used = list({o["tool"] for o in result.get("tool_outputs", []) if o["tool"] != "none"})
    latency = round(time.time() - t0, 2)
    log.info("query_complete", latency_s=latency, tools_used=tools_used,
             replan_count=result.get("replan_count", 0))
    return QueryResponse(
        answer=result.get("final_answer", ""),
        plan=result.get("plan", []),
        tools_used=tools_used,
        replan_count=result.get("replan_count", 0),
        latency_s=latency,
    )

@app.post("/query/stream")
async def query_stream(req: QueryRequest, _: None = Depends(verify_api_key)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    t0 = time.time()

    async def generate() -> AsyncIterator[str]:
        def emit(type_: str, data) -> str:
            return f"data: {json.dumps({'type': type_, 'data': data})}\n\n"
        try:
            for event in get_graph().stream(_make_initial_state(req.query)):
                for node_name, output in event.items():
                    if node_name == "planning" and "plan" in output:
                        yield emit("plan", output["plan"])
                    elif node_name == "execution" and "tool_outputs" in output:
                        outs = output.get("tool_outputs", [])
                        if outs:
                            last = outs[-1]
                            yield emit("tool_call", {"tool": last["tool"], "task": last.get("task", "")})
                            yield emit("tool_result", {
                                "tool": last["tool"],
                                "result": last.get("result", "")[:300],
                                "success": last.get("success", True),
                            })
                    elif node_name == "replan":
                        yield emit("replan", {"count": output.get("replan_count", 1)})
                    elif node_name == "synthesis" and "final_answer" in output:
                        yield emit("answer", output["final_answer"])
            yield emit("done", {"latency_s": round(time.time() - t0, 2)})
        except Exception as exc:
            yield emit("error", {"message": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/health")
async def health():
    status = {"status": "ok", "version": "2.1.0", "index": {}}
    try:
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        client = get_chroma_client(path=chroma_dir)
        docs = client.get_collection("project_docs")
        mem  = client.get_or_create_collection("session_memory")
        status["index"] = {
            "project_docs_chunks": docs.count(),
            "session_memory_sessions": mem.count(),
        }
    except Exception as e:
        status["index"] = {"error": str(e)}
    return status

@app.get("/sessions")
async def list_sessions(_: None = Depends(verify_api_key)):
    try:
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        client = get_chroma_client(path=chroma_dir)
        collection = client.get_or_create_collection("session_memory")
        count = collection.count()
        if count == 0:
            return {"sessions": [], "total": 0}
        results = collection.peek(limit=min(20, count))
        return {
            "sessions": [
                {"summary": doc, "metadata": meta}
                for doc, meta in zip(results["documents"], results["metadatas"])
            ],
            "total": count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", "8000")),
                reload=os.getenv("ENV", "dev") == "dev",
                log_level="warning")
