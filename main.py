# main.py

from datetime import datetime
from typing import Optional, Dict, List, Any
from uuid import uuid4
from math import inf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import faiss
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)
from langchain_core.documents import Document


app = FastAPI(title="Generative-Agent API (Refactored)")

@app.get("/")
async def health_check():
    return {"status": "ok"}

# ——— shared LLM & embeddings ——————————————————

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7) # Consider making model configurable
emb = OpenAIEmbeddings()

def _new_agent_instance(
    name: str,
    age: int,
    traits: str,
    status: str,
    summary_refresh_seconds: int,
    reflection_threshold: int,
    verbose: bool
) -> GenerativeAgent:
    try:
        probe = emb.embed_query("probe")
    except Exception as e:
        # Handle potential issues with OpenAI API key or connectivity early
        raise HTTPException(status_code=500, detail=f"Failed to get embedding dimension: {e}")

    dim = len(probe)

    index = faiss.IndexFlatL2(dim)
    vectorstore = FAISS(
        embedding_function=emb,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )

    retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, k=15, decay_rate=0.01 # Default k
    )

    actual_reflect = reflection_threshold if reflection_threshold > 0 else inf
    actual_refresh = summary_refresh_seconds if summary_refresh_seconds > 0 else inf

    memory = GenerativeAgentMemory(
        llm=llm,
        memory_retriever=retriever,
        reflection_threshold=actual_reflect,
        # verbose=verbose, # verbose in GenerativeAgentMemory is for logging reflections
    )

    agent = GenerativeAgent(
        name=name,
        age=age,
        traits=traits,
        status=status,
        memory=memory,
        llm=llm,
        summary_refresh_seconds=actual_refresh,
        verbose=verbose, # verbose in GenerativeAgent is for general logging
    )
    return agent

agents: Dict[str, GenerativeAgent] = {}

# ——— Pydantic models ———————————————————————————————

class CreateAgentReq(BaseModel):
    name: str
    age: int
    traits: str
    status: str
    agent_id: Optional[str] = None
    summary_refresh_seconds: int = Field(default=0, ge=0) # 0 → never
    reflection_threshold: int = Field(default=0, ge=0)   # 0 → never
    verbose: bool = False

class GenerateResponseReq(BaseModel):
    prompt: str
    k: Optional[int] = Field(default=None, gt=0) # If None or 0, uses agent's default k

class AddMemoryReq(BaseModel):
    text_to_memorize: str

class FetchMemoriesReq(BaseModel):
    observation: str
    k: Optional[int] = Field(default=None, gt=0) # If None or 0, uses agent's default k

# ——— Endpoints ———————————————————————————————————————

@app.post("/agents", status_code=201)
def create_agent(req: CreateAgentReq):
    aid = req.agent_id or str(uuid4())
    if aid in agents:
        raise HTTPException(status_code=400, detail=f"Agent with agent_id '{aid}' already exists.")
    
    try:
        agents[aid] = _new_agent_instance(
            name=req.name,
            age=req.age,
            traits=req.traits,
            status=req.status,
            summary_refresh_seconds=req.summary_refresh_seconds,
            reflection_threshold=req.reflection_threshold,
            verbose=req.verbose,
        )
    except Exception as e:
        # Catch errors during agent initialization (e.g., OpenAI API issues)
        raise HTTPException(status_code=500, detail=f"Failed to initialize agent: {e}")
        
    return {"agent_id": aid, "name": req.name}

@app.get("/agents")
def list_agents():
    return {"agents": list(agents.keys())}

@app.post("/agents/{agent_id}/generate_response")
def generate_response(agent_id: str, req: GenerateResponseReq):
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    
    text = req.prompt.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Prompt may not be empty.")
    
    agent = agents[agent_id]
    original_k = agent.memory.memory_retriever.k
    
    try:
        if req.k is not None and req.k > 0:
            agent.memory.memory_retriever.k = req.k
        
        # generate_dialogue_response returns: (Boolean indicating if observation should be added to memory, String response)
        prompt_is_important, response_text = agent.generate_dialogue_response(text, datetime.now())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during dialogue generation: {e}")
    finally:
        # Restore original k value for the agent's retriever
        agent.memory.memory_retriever.k = original_k
        
    return {
        "agent_name": agent.name, 
        "response": response_text,
        "prompt_is_important_to_memorize": prompt_is_important
    }

@app.post("/agents/{agent_id}/add_memory")
def add_memory(agent_id: str, req: AddMemoryReq):
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    
    text_to_add = req.text_to_memorize.strip()
    if not text_to_add:
        raise HTTPException(status_code=400, detail="Memory text may not be empty.")
    
    try:
        agents[agent_id].memory.add_memory(text_to_add, now=datetime.now())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding memory: {e}")

    return {"added_memory": text_to_add}

@app.post("/agents/{agent_id}/fetch_memories")
def fetch_memories(agent_id: str, req: FetchMemoriesReq):
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")

    observation = req.observation.strip()
    if not observation:
        raise HTTPException(status_code=400, detail="Observation text may not be empty for fetching memories.")

    agent = agents[agent_id]
    original_k = agent.memory.memory_retriever.k
    
    try:
        if req.k is not None and req.k > 0:
            agent.memory.memory_retriever.k = req.k
        
        # fetch_memories returns List[Document]
        retrieved_docs: List[Document] = agent.memory.fetch_memories(observation, now=datetime.now())
        memories_content: List[str] = [doc.page_content for doc in retrieved_docs]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching memories: {e}")
    finally:
        # Restore original k value
        agent.memory.memory_retriever.k = original_k
        
    return {"memories": memories_content}


@app.get("/agents/{agent_id}/summary")
def get_summary(agent_id: str): # Changed function name for consistency
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    
    try:
        summary_text = agents[agent_id].get_summary(force_refresh=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {e}")
        
    return {"agent_id": agent_id, "summary": summary_text}

@app.delete("/agents/{agent_id}")
def delete_agent(agent_id: str):
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    
    del agents[agent_id]
    return {"deleted_agent_id": agent_id, "status": "success"}

# Optional: Add exception handlers for more graceful error responses
# from fastapi.responses import JSONResponse
# @app.exception_handler(ValueError) # Example
# async def value_error_exception_handler(request: Any, exc: ValueError):
#    return JSONResponse(status_code=400, content={"message": str(exc)})
