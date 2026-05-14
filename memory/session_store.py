import json, os, uuid
from datetime import datetime
import chromadb
from chroma_settings import get_chroma_client
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

_EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_embed_model = None
_store_instance = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbedding(model_name=_EMBED_MODEL_NAME)
    return _embed_model

def get_session_store():
    global _store_instance
    if _store_instance is None:
        _store_instance = SessionMemoryStore()
    return _store_instance

class SessionMemoryStore:
    COLLECTION_NAME = "session_memory"

    def __init__(self):
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.client = get_chroma_client(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(self.COLLECTION_NAME)
        self.embed_model = _get_embed_model()

    def save_session(self, query, plan, result, tools_used):
        session_id = str(uuid.uuid4())
        summary = f"Query: {query}\nPlan: {'; '.join(plan)}\nResult: {result[:500]}"
        embedding = self.embed_model.get_text_embedding(summary)
        self.collection.add(
            ids=[session_id],
            embeddings=[embedding],
            documents=[summary],
            metadatas=[{
                "query": query,
                "tools_used": json.dumps(tools_used),
                "timestamp": datetime.utcnow().isoformat(),
                "plan_steps": len(plan),
            }],
        )
        return session_id

    def retrieve_relevant_sessions(self, query, top_k=3):
        count = self.collection.count()
        if count == 0:
            return []
        embedding = self.embed_model.get_text_embedding(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, count),
        )
        return [
            {"summary": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    def list_recent_sessions(self, limit=20):
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.peek(limit=min(limit, count))
        return [
            {"summary": doc, "metadata": meta}
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
