from datetime import datetime
from typing import Optional, Dict, List, Any
from uuid import uuid4
from math import inf
import os # For checking environment variable
import traceback # For printing stack traces

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

print("DEBUG: FastAPI application starting up...", flush=True)

@app.get("/")
async def health_check():
    print("DEBUG: Health check '/' endpoint hit.", flush=True)
    return {"status": "ok"}

# ——— shared embeddings (LLM is now per-agent) ——————————————————
print("DEBUG: Attempting to initialize global OpenAIEmbeddings instance 'emb'...", flush=True)
GLOBAL_EMBEDDING_INSTANCE = None
try:
    if not os.getenv("OPENAI_API_KEY"):
        print("CRITICAL_WARNING: OPENAI_API_KEY environment variable is NOT SET at the time of global 'emb' initialization. This will likely cause failure when 'emb' is used.", flush=True)
    
    GLOBAL_EMBEDDING_INSTANCE = OpenAIEmbeddings()
    # Try a dummy embedding to catch API key issues early for embeddings
    GLOBAL_EMBEDDING_INSTANCE.embed_query("startup_test_embedding")
    print("DEBUG: Global 'GLOBAL_EMBEDDING_INSTANCE' initialized and tested successfully.", flush=True)
except Exception as e:
    print(f"CRITICAL_ERROR: Failed to initialize or test global 'GLOBAL_EMBEDDING_INSTANCE' OpenAIEmbeddings: {e}", flush=True)
    print("Full stack trace for global embedding initialization failure:", flush=True)
    traceback.print_exc()
    # Application might still start but will fail if embeddings are needed.
    # Depending on strictness, you might want to raise SystemExit here.

def _new_agent_instance(
    name: str,
    age: int,
    traits: str,
    status: str,
    summary_refresh_seconds: int,
    reflection_threshold: int,
    verbose: bool,
    model_name: Optional[str] = None
) -> GenerativeAgent:
    print(f"DEBUG: _new_agent_instance called for agent '{name}' with model_name: '{model_name}'", flush=True)

    if not os.getenv("OPENAI_API_KEY"):
        print("CRITICAL_ERROR: OPENAI_API_KEY environment variable is NOT SET within _new_agent_instance. OpenAI calls will fail.", flush=True)
        # This is a serious issue, the calls below will likely raise exceptions.
    
    effective_model_name = model_name if model_name and model_name.strip() else "gpt-4o-mini"
    print(f"DEBUG: Effective model name for agent '{name}': {effective_model_name}", flush=True)

    agent_llm = None
    try:
        print(f"DEBUG: Attempting to initialize ChatOpenAI for agent '{name}' with model: {effective_model_name}", flush=True)
        agent_llm = ChatOpenAI(model_name=effective_model_name, temperature=0.7)
        # It's good practice to test the LLM instance if possible, e.g., a small invoke, but that can be slow/costly.
        # For now, we assume instantiation implies it's ready or will fail on first actual use by Langchain.
        print(f"DEBUG: ChatOpenAI for agent '{name}' (model {effective_model_name}) initialized.", flush=True)
    except Exception as e:
        print(f"ERROR_STACKTRACE: Failed to initialize LLM for agent '{name}' with model '{effective_model_name}': {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to initialize LLM with model '{effective_model_name}': {e}")

    if GLOBAL_EMBEDDING_INSTANCE is None:
        print("CRITICAL_ERROR: Global 'GLOBAL_EMBEDDING_INSTANCE' is None. Cannot proceed with agent creation that requires embeddings.", flush=True)
        raise HTTPException(status_code=500, detail="Core embedding system failed to initialize. Agent creation aborted.")

    # Embedding dimension check (already performed at startup for GLOBAL_EMBEDDING_INSTANCE)
    # We rely on the startup check. If we reach here, GLOBAL_EMBEDDING_INSTANCE should be working.
    # For safety, you could re-check or get dim directly if needed, but let's assume startup check is enough.
    try:
        # This call is just to get the dimension, assumes GLOBAL_EMBEDDING_INSTANCE is valid.
        probe_for_dim = GLOBAL_EMBEDDING_INSTANCE.embed_query("get_dim_probe")
        dim = len(probe_for_dim)
        print(f"DEBUG: Embedding dimension confirmed as {dim} for agent '{name}'.", flush=True)
    except Exception as e:
        print(f"ERROR_STACKTRACE: Failed to confirm embedding dimension for agent '{name}': {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to confirm embedding dimension: {e}")


    print(f"DEBUG: Setting up FAISS index for agent '{name}'...", flush=True)
    try:
        index = faiss.IndexFlatL2(dim)
        vectorstore = FAISS(
            embedding_function=GLOBAL_EMBEDDING_INSTANCE, # Use the global, tested instance
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )
        retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=vectorstore, k=15, decay_rate=0.01
        )
        print(f"DEBUG: FAISS setup complete for agent '{name}'.", flush=True)
    except Exception as e:
        print(f"ERROR_STACKTRACE: Failed during FAISS setup for agent '{name}': {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during FAISS setup: {e}")

     print(f"DEBUG: Setting up GenerativeAgentMemory for agent '{name}'...", flush=True)
    try:
        # If reflection_threshold from input is 0 or less, pass None to GenerativeAgentMemory.
        # The GenerativeAgentMemory itself handles None by setting a very high internal threshold (effectively infinity).
        actual_reflect_for_memory = reflection_threshold if reflection_threshold > 0 else None
        
        memory = GenerativeAgentMemory(
            llm=agent_llm,
            memory_retriever=retriever,
            reflection_threshold=actual_reflect_for_memory, # Pass None if 0 or less
        )
        print(f"DEBUG: GenerativeAgentMemory setup complete for agent '{name}'.", flush=True)
    except Exception as e:
        print(f"ERROR_STACKTRACE: Failed during GenerativeAgentMemory setup for agent '{name}': {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during GenerativeAgentMemory setup: {e}")

    print(f"DEBUG: Initializing GenerativeAgent '{name}'...", flush=True)
    try:
        # For GenerativeAgent, if summary_refresh_seconds from input is 0 or less, pass None.
        # The GenerativeAgent class might also interpret None as "never refresh" or use a default.
        actual_refresh_for_agent = summary_refresh_seconds if summary_refresh_seconds > 0 else None

        agent = GenerativeAgent(
            name=name,
            age=age,
            traits=traits,
            status=status,
            memory=memory,
            llm=agent_llm,
            summary_refresh_seconds=actual_refresh_for_agent, # Pass None if 0 or less
            verbose=verbose,
        )
        print(f"DEBUG: GenerativeAgent '{name}' initialized successfully.", flush=True)
        return agent
    except Exception as e:
        print(f"ERROR_STACKTRACE: Failed during GenerativeAgent initialization for agent '{name}': {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during GenerativeAgent initialization: {e}")
agents: Dict[str, GenerativeAgent] = {}

# ——— Pydantic models ———————————————————————————————

class CreateAgentReq(BaseModel):
    name: str
    age: int
    traits: str
    status: str
    agent_id: Optional[str] = None
    summary_refresh_seconds: int = Field(default=0, ge=0)
    reflection_threshold: int = Field(default=0, ge=0)
    verbose: bool = False
    model_name: Optional[str] = None

class GenerateResponseReq(BaseModel):
    prompt: str
    k: Optional[int] = Field(default=None, gt=0) 

class AddMemoryReq(BaseModel):
    text_to_memorize: str

class FetchMemoriesReq(BaseModel):
    observation: str
    k: Optional[int] = Field(default=None, gt=0)

# ——— Endpoints ———————————————————————————————————————

@app.post("/agents", status_code=201)
def create_agent(req: CreateAgentReq):
    print(f"DEBUG: /agents POST request received: {req.model_dump_json(exclude_none=True)}", flush=True) # Log request
    aid = req.agent_id or str(uuid4())
    if aid in agents:
        print(f"WARN: Agent with agent_id '{aid}' already exists.", flush=True)
        raise HTTPException(status_code=400, detail=f"Agent with agent_id '{aid}' already exists.")
    
    if GLOBAL_EMBEDDING_INSTANCE is None:
        print("CRITICAL_ERROR: Global embedding instance not available. Aborting agent creation.", flush=True)
        raise HTTPException(status_code=500, detail="Server's core embedding system is not operational.")

    try:
        print(f"DEBUG: Calling _new_agent_instance for agent_id '{aid}' with name '{req.name}'", flush=True)
        agents[aid] = _new_agent_instance(
            name=req.name,
            age=req.age,
            traits=req.traits,
            status=req.status,
            summary_refresh_seconds=req.summary_refresh_seconds,
            reflection_threshold=req.reflection_threshold,
            verbose=req.verbose,
            model_name=req.model_name
        )
        print(f"DEBUG: Agent '{aid}' (name: '{req.name}') created and stored.", flush=True)
    except HTTPException as e: # This will catch HTTPExceptions raised from _new_agent_instance
        print(f"DEBUG: HTTPException caught in create_agent for agent_id '{aid}': {e.detail}", flush=True)
        raise e 
    except Exception as e: # Catch any other unexpected errors during the call to _new_agent_instance or storage
        print(f"ERROR_STACKTRACE: Unexpected error in create_agent endpoint for agent_id '{aid}': {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected server error during agent creation: {e}")
        
    model_actually_used = "unknown"
    try:
        current_agent_llm = agents[aid].llm
        if hasattr(current_agent_llm, 'model_name'):
            model_actually_used = current_agent_llm.model_name
        elif hasattr(current_agent_llm, 'model'):
             model_actually_used = current_agent_llm.model
    except KeyError: # Should not happen if agent creation was successful
        print(f"ERROR: Agent {aid} not found in agents dict immediately after creation attempt.", flush=True)
        # This case implies something went very wrong, but _new_agent_instance should have raised.
        # Fallback to a generic error if this state is reached.
        raise HTTPException(status_code=500, detail="Internal inconsistency after agent creation.")


    print(f"DEBUG: Agent '{aid}' (name: '{req.name}') creation successful. Model used: {model_actually_used}. Responding to client.", flush=True)
    return {"agent_id": aid, "name": req.name, "model_used": model_actually_used}

@app.get("/agents")
def list_agents():
    print("DEBUG: /agents GET request received (list_agents)", flush=True)
    if GLOBAL_EMBEDDING_INSTANCE is None and len(agents) > 0 : # If agents exist but embedding failed, it's an issue
         print("WARN: Listing agents, but global embedding instance is not available.", flush=True)

    agent_details = []
    for agent_id, agent_instance in agents.items():
        model_used = "unknown"
        # Ensure agent_instance and its llm attribute exist before accessing model_name
        if agent_instance and hasattr(agent_instance, 'llm') and agent_instance.llm:
            if hasattr(agent_instance.llm, 'model_name'):
                model_used = agent_instance.llm.model_name
            elif hasattr(agent_instance.llm, 'model'):
                model_used = agent_instance.llm.model
        else:
            print(f"WARN: Agent {agent_id} found in dict, but has no LLM instance or is malformed.", flush=True)

        agent_details.append({"agent_id": agent_id, "name": agent_instance.name if agent_instance else "Unknown Name", "model": model_used})
    print(f"DEBUG: Returning {len(agent_details)} agents.", flush=True)
    return {"agents": agent_details}


# ... (rest of your endpoints: generate_response, add_memory, fetch_memories, get_summary, delete_agent) ...
# You can add similar "DEBUG" print statements at the beginning of each endpoint if needed.

@app.post("/agents/{agent_id}/generate_response")
def generate_response(agent_id: str, req: GenerateResponseReq):
    print(f"DEBUG: /generate_response for agent {agent_id} with prompt: '{req.prompt[:50]}...'", flush=True)
    if agent_id not in agents:
        print(f"ERROR: Agent '{agent_id}' not found for generate_response.", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    # ... (rest of the logic)
    agent = agents[agent_id]
    original_k = agent.memory.memory_retriever.k
    
    try:
        if req.k is not None and req.k > 0:
            agent.memory.memory_retriever.k = req.k
        
        prompt_is_important, response_text = agent.generate_dialogue_response(req.prompt.strip(), datetime.now())
    
    except Exception as e:
        print(f"ERROR_STACKTRACE: Error during dialogue generation for agent {agent_id}: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during dialogue generation: {e}")
    finally:
        agent.memory.memory_retriever.k = original_k # Ensure k is reset
        
    return {
        "agent_name": agent.name, 
        "response": response_text,
        "prompt_is_important_to_memorize": prompt_is_important
    }

@app.post("/agents/{agent_id}/add_memory")
def add_memory(agent_id: str, req: AddMemoryReq):
    print(f"DEBUG: /add_memory for agent {agent_id} with text: '{req.text_to_memorize[:50]}...'", flush=True)
    if agent_id not in agents:
        print(f"ERROR: Agent '{agent_id}' not found for add_memory.", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    # ... (rest of the logic)
    text_to_add = req.text_to_memorize.strip()
    if not text_to_add:
        raise HTTPException(status_code=400, detail="Memory text may not be empty.")
    
    try:
        agents[agent_id].memory.add_memory(text_to_add, now=datetime.now())
    except Exception as e:
        print(f"ERROR_STACKTRACE: Error adding memory for agent {agent_id}: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error adding memory: {e}")

    return {"added_memory": text_to_add}

@app.post("/agents/{agent_id}/fetch_memories")
def fetch_memories(agent_id: str, req: FetchMemoriesReq):
    print(f"DEBUG: /fetch_memories for agent {agent_id} with observation: '{req.observation[:50]}...'", flush=True)
    if agent_id not in agents:
        print(f"ERROR: Agent '{agent_id}' not found for fetch_memories.", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    # ... (rest of the logic)
    observation = req.observation.strip()
    if not observation:
        raise HTTPException(status_code=400, detail="Observation text may not be empty for fetching memories.")

    agent = agents[agent_id]
    original_k = agent.memory.memory_retriever.k
    
    try:
        if req.k is not None and req.k > 0:
            agent.memory.memory_retriever.k = req.k
        
        retrieved_docs: List[Document] = agent.memory.fetch_memories(observation, now=datetime.now())
        memories_content: List[str] = [doc.page_content for doc in retrieved_docs]

    except Exception as e:
        print(f"ERROR_STACKTRACE: Error fetching memories for agent {agent_id}: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching memories: {e}")
    finally:
        agent.memory.memory_retriever.k = original_k # Ensure k is reset
        
    return {"memories": memories_content}


@app.get("/agents/{agent_id}/summary")
def get_summary(agent_id: str): 
    print(f"DEBUG: /summary for agent {agent_id}", flush=True)
    if agent_id not in agents:
        print(f"ERROR: Agent '{agent_id}' not found for get_summary.", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    # ... (rest of the logic)
    try:
        summary_text = agents[agent_id].get_summary(force_refresh=True)
    except Exception as e:
        print(f"ERROR_STACKTRACE: Error generating summary for agent {agent_id}: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating summary: {e}")
        
    return {"agent_id": agent_id, "summary": summary_text}

@app.delete("/agents/{agent_id}")
def delete_agent(agent_id: str):
    print(f"DEBUG: /delete_agent for agent {agent_id}", flush=True)
    if agent_id not in agents:
        print(f"ERROR: Agent '{agent_id}' not found for delete_agent.", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    
    del agents[agent_id]
    print(f"DEBUG: Agent '{agent_id}' deleted successfully.", flush=True)
    return {"deleted_agent_id": agent_id, "status": "success"}

print("DEBUG: FastAPI application finished loading.", flush=True)
