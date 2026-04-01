"""Swarm Long-Term Memory via ChromaDB."""
import uuid
from datetime import datetime
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from liquid_swarm.state import SwarmState

def get_vector_store():
    # Use Chroma to persist long-term global memory
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        collection_name="supernova_global_memory",
        embedding_function=embeddings,
        persist_directory="supernova_memory.db"
    )
    return vector_store

async def bootstrap_node(state: SwarmState) -> dict:
    """Retrieves relevant past context from long-term memory."""
    if not state.get("tasks"):
        return {"global_context": "No tasks provided."}
        
    # Approximate main goal by combining queries or taking the first
    query_hint = state["tasks"][0].query
    
    try:
        vs = get_vector_store()
        results = await vs.asimilarity_search(query_hint, k=3)
        global_context = "\n".join([r.page_content for r in results]) if results else "No prior knowledge available."
    except Exception as e:
        # ChromaDB might fail if it's completely empty on the very first run in some configurations,
        # or due to network issues with OpenAI embeddings
        global_context = f"No prior knowledge available (Memory init missed: {e})."
        
    return {"global_context": global_context}

async def archivar_node(state: SwarmState) -> dict:
    """Saves successful findings to the persistent vector database."""
    final_results = state.get("final_results", [])
    if not final_results:
        return {}
        
    try:
        vs = get_vector_store()
        
        docs = []
        metadatas = []
        ids = []
        
        for r in final_results:
            if r.status == "success" and r.data.get("result"):
                docs.append(str(r.data["result"]))
                metadatas.append({
                    "task_id": r.task_id, 
                    "timestamp": datetime.now().isoformat()
                })
                ids.append(str(uuid.uuid4()))
                
        if docs:
            await vs.aadd_texts(texts=docs, metadatas=metadatas, ids=ids)
            
    except Exception as e:
        print(f"[Archivar] Failed to save memory: {e}")
        
    return {}
