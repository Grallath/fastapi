from datetime import datetime
from typing import Optional, Dict, List, Any
from uuid import uuid4
# from math import inf # No longer needed for these parameters
import os 
import traceback 

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

# Default model names if not provided in requests
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small" 

@app.get("/")
async def health_check():
    print("DEBUG: Health check '/' endpoint hit.", flush=True)
    return {"status": "ok"}

if not os.getenv("OPENAI_API_KEY"):
    print("CRITICAL_WARNING: OPENAI_API_KEY environment variable is NOT SET. OpenAI calls will likely fail.", flush=True)
else:
    print("DEBUG: OPENAI_API_KEY environment variable is detected.", flush=True)


def _new_agent_instance(
    name: str,
    age: int,
    traits: str,
    status: str,
    summary_refresh_seconds: int, # This is an int from the request
    reflection_threshold: int,    # This is an int from the request
    verbose: bool,
    llm_model_name: Optional[str] = None,
    embedding_model_name: Optional[str] = None
) -> GenerativeAgent:
    print(f"DEBUG: _new_agent_instance called for agent '{name}'", flush=True)
    print(f"DEBUG:   LLM Model Request: '{llm_model_name}', Embedding Model Request: '{embedding_model_name}'", flush=True)
    print(f"DEBUG:   Input summary_refresh_seconds: {summary_refresh_seconds}, Input reflection_threshold: {reflection_threshold}", flush=True)

    effective_llm_model = llm_model_name if llm_model_name and llm_model_name.strip() else DEFAULT_CHAT_MODEL
    effective_embedding_model = embedding_model_name if embedding_model_name and embedding_model_name.strip() else DEFAULT_EMBEDDING_MODEL
    
    print(f"DEBUG:   Effective LLM Model: '{effective_llm_model}'", flush=True)
    print(f"DEBUG:   Effective Embedding Model: '{effective_embedding_model}'", flush=True)

    agent_llm = None
    try:
        print(f"DEBUG: Attempting to initialize ChatOpenAI for agent '{name}' with model: {effective_llm_model}", flush=True)
        agent_llm = ChatOpenAI(model_name=effective_llm_model, temperature=0.7)
        print(f"DEBUG: ChatOpenAI for agent '{name}' (model {effective_llm_model}) initialized.", flush=True)
    except Exception as e:
        print(f"ERROR_STACKTRACE: Failed to initialize LLM for agent '{name}' with model '{effective_llm_model}': {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to initialize LLM with model '{effective_llm_model}': {e}")

    agent_embeddings = None
    dim = 0 
    try:
        print(f"DEBUG: Attempting to initialize OpenAIEmbeddings for agent '{name}' with model: {effective_embedding_model}", flush=True)
        agent_embeddings = OpenAIEmbeddings(model=effective_embedding_model)
        probe_for_dim = agent_embeddings.embed_query("get_dim_probe_for_agent")
        dim = len(probe_for_dim)
        print(f"DEBUG: OpenAIEmbeddings for agent '{name}' (model {effective_embedding_model}, dim {dim}) initialized and tested.", flush=True)
    except Exception as e:
        print(f"ERROR_STACKTRACE: Failed to initialize or test OpenAIEmbeddings for agent '{name}' with model '{effective_embedding_model}': {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to initialize/test embeddings with model '{effective_embedding_model}': {e}")

    print(f"DEBUG: Setting up FAISS index for agent '{name}' (dim: {dim})...", flush=True)
    try:
        index = faiss.IndexFlatL2(dim)
        vectorstore = FAISS(
            embedding_function=agent_embeddings,
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
        # For GenerativeAgentMemory, reflection_threshold CAN be None.
        # If input reflection_threshold is 0 (or less), pass None.
        actual_reflect_for_memory = reflection_threshold if reflection_threshold > 0 else None
        print(f"DEBUG: actual_reflect_for_memory (for GenerativeAgentMemory) will be: {actual_reflect_for_memory}", flush=True)
        
        memory = GenerativeAgentMemory(
            llm=agent_llm,
            memory_retriever=retriever,
            reflection_threshold=actual_reflect_for_memory, 
        )
        print(f"DEBUG: GenerativeAgentMemory setup complete for agent '{name}'.", flush=True)
    except Exception as e:
        print(f"ERROR_STACKTRACE: Failed during GenerativeAgentMemory setup for agent '{name}': {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during GenerativeAgentMemory setup: {e}")

    print(f"DEBUG: Initializing GenerativeAgent '{name}'...", flush=True)
    try:
        # For GenerativeAgent, summary_refresh_seconds MUST be an int.
        # If input summary_refresh_seconds is 0, passing 0 achieves "never refresh automatically"
        # due to internal logic `self.summary_refresh_seconds > 0`.
        actual_refresh_for_agent = summary_refresh_seconds # Pass the int (e.g., 0) directly.
                                                            # Pydantic on CreateAgentReq already ensures it's >= 0.
        
        print(f"DEBUG: actual_refresh_for_agent (for GenerativeAgent) will be: {actual_refresh_for_agent}", flush=True)

        agent = GenerativeAgent(
            name=name,
            age=age,
            traits=traits,
            status=status,
            memory=memory,
            llm=agent_llm,
            summary_refresh_seconds=actual_refresh_for_agent, 
            verbose=verbose,
        )
        print(f"DEBUG: GenerativeAgent '{name}' initialized successfully.", flush=True)
        return agent
    except Exception as e:
        print(f"ERROR_STACKTRACE: Failed during GenerativeAgent initialization for agent '{name}': {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during GenerativeAgent initialization: {e}")

agents: Dict[str, GenerativeAgent] = {}

class CreateAgentReq(BaseModel):
    name: str
    age: int
    traits: str
    status: str
    agent_id: Optional[str] = None
    summary_refresh_seconds: int = Field(default=0, ge=0) # Ensures it's an int >= 0
    reflection_threshold: int = Field(default=0, ge=0)   # Ensures it's an int >= 0
    verbose: bool = False
    model_name: Optional[str] = None
    embedding_model_name: Optional[str] = None

class GenerateResponseReq(BaseModel):
    prompt: str
    k: Optional[int] = Field(default=None, gt=0) 

class AddMemoryReq(BaseModel):
    text_to_memorize: str

class FetchMemoriesReq(BaseModel):
    observation: str
    k: Optional[int] = Field(default=None, gt=0)

@app.post("/agents", status_code=201)
def create_agent(req: CreateAgentReq):
    print(f"DEBUG: /agents POST request received: {req.model_dump_json(exclude_none=True)}", flush=True) 
    aid = req.agent_id or str(uuid4())
    if aid in agents:
        print(f"WARN: Agent with agent_id '{aid}' already exists.", flush=True)
        raise HTTPException(status_code=400, detail=f"Agent with agent_id '{aid}' already exists.")
    
    current_agent_instance = None
    try:
        print(f"DEBUG: Calling _new_agent_instance for agent_id '{aid}' with name '{req.name}'", flush=True)
        current_agent_instance = _new_agent_instance( 
            name=req.name,
            age=req.age,
            traits=req.traits,
            status=req.status,
            summary_refresh_seconds=req.summary_refresh_seconds, # Pass as int
            reflection_threshold=req.reflection_threshold,       # Pass as int
            verbose=req.verbose,
            llm_model_name=req.model_name, 
            embedding_model_name=req.embedding_model_name 
        )
        agents[aid] = current_agent_instance 
        print(f"DEBUG: Agent '{aid}' (name: '{req.name}') created and stored.", flush=True)
    except HTTPException as e: 
        print(f"DEBUG: HTTPException caught in create_agent for agent_id '{aid}': {e.detail}", flush=True)
        raise e 
    except Exception as e: 
        print(f"ERROR_STACKTRACE: Unexpected error in create_agent endpoint for agent_id '{aid}': {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected server error during agent creation: {e}")
        
    llm_model_used = "unknown"
    embedding_model_used = "unknown"

    if current_agent_instance:
        if hasattr(current_agent_instance, 'llm') and current_agent_instance.llm and hasattr(current_agent_instance.llm, 'model_name'):
            llm_model_used = current_agent_instance.llm.model_name
        
        if (hasattr(current_agent_instance, 'memory') and current_agent_instance.memory and
            hasattr(current_agent_instance.memory, 'memory_retriever') and current_agent_instance.memory.memory_retriever and
            hasattr(current_agent_instance.memory.memory_retriever, 'vectorstore') and current_agent_instance.memory.memory_retriever.vectorstore and
            hasattr(current_agent_instance.memory.memory_retriever.vectorstore, 'embedding_function') and 
            current_agent_instance.memory.memory_retriever.vectorstore.embedding_function and
            hasattr(current_agent_instance.memory.memory_retriever.vectorstore.embedding_function, 'model')):
            embedding_model_used = current_agent_instance.memory.memory_retriever.vectorstore.embedding_function.model
    else:
        print(f"ERROR: current_agent_instance is None after creation attempt for agent_id '{aid}'. This indicates a logic flaw.", flush=True)
        if aid in agents and agents[aid] is not None: # Check if it's in dict and not None
            retrieved_agent = agents[aid]
            if hasattr(retrieved_agent, 'llm') and retrieved_agent.llm and hasattr(retrieved_agent.llm, 'model_name'):
                 llm_model_used = retrieved_agent.llm.model_name
            if (hasattr(retrieved_agent, 'memory') and hasattr(retrieved_agent.memory, 'memory_retriever') and 
                hasattr(retrieved_agent.memory.memory_retriever, 'vectorstore') and hasattr(retrieved_agent.memory.memory_retriever.vectorstore, 'embedding_function')
                and hasattr(retrieved_agent.memory.memory_retriever.vectorstore.embedding_function, 'model')):
                 embedding_model_used = retrieved_agent.memory.memory_retriever.vectorstore.embedding_function.model

    print(f"DEBUG: Agent '{aid}' creation processing complete. LLM: {llm_model_used}, Embedding: {embedding_model_used}. Responding.", flush=True)
    return {
        "agent_id": aid, 
        "name": req.name, 
        "llm_model_used": llm_model_used,
        "embedding_model_used": embedding_model_used
    }

@app.get("/agents")
def list_agents():
    print("DEBUG: /agents GET request received (list_agents)", flush=True)
    agent_details = []
    for agent_id, agent_instance_from_dict in agents.items():
        llm_model = "unknown"
        emb_model = "unknown"
        agent_name_from_instance = "Unknown Name" 

        if agent_instance_from_dict: 
            agent_name_from_instance = agent_instance_from_dict.name
            if hasattr(agent_instance_from_dict, 'llm') and agent_instance_from_dict.llm and hasattr(agent_instance_from_dict.llm, 'model_name'):
                llm_model = agent_instance_from_dict.llm.model_name
            
            if (hasattr(agent_instance_from_dict, 'memory') and agent_instance_from_dict.memory and
                hasattr(agent_instance_from_dict.memory, 'memory_retriever') and agent_instance_from_dict.memory.memory_retriever and
                hasattr(agent_instance_from_dict.memory.memory_retriever, 'vectorstore') and agent_instance_from_dict.memory.memory_retriever.vectorstore and
                hasattr(agent_instance_from_dict.memory.memory_retriever.vectorstore, 'embedding_function') and 
                agent_instance_from_dict.memory.memory_retriever.vectorstore.embedding_function and
                hasattr(agent_instance_from_dict.memory.memory_retriever.vectorstore.embedding_function, 'model')):
                emb_model = agent_instance_from_dict.memory.memory_retriever.vectorstore.embedding_function.model
        else:
            print(f"WARN: Agent {agent_id} found in agents dictionary, but its instance is None.", flush=True)

        agent_details.append({
            "agent_id": agent_id, 
            "name": agent_name_from_instance, 
            "llm_model": llm_model,
            "embedding_model": emb_model
        })
    print(f"DEBUG: Returning {len(agent_details)} agents.", flush=True)
    return {"agents": agent_details}


@app.post("/agents/{agent_id}/generate_response")
def generate_response(agent_id: str, req: GenerateResponseReq):
    print(f"DEBUG: /generate_response for agent {agent_id} with prompt: '{req.prompt[:50]}...'", flush=True)
    if agent_id not in agents or agents[agent_id] is None:
        print(f"ERROR: Agent '{agent_id}' not found or is None for generate_response.", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")
    
    agent = agents[agent_id]
    original_k = -1 
    if hasattr(agent, 'memory') and hasattr(agent.memory, 'memory_retriever'):
         original_k = agent.memory.memory_retriever.k
    else:
        print(f"WARN: Agent {agent_id} missing memory or retriever for k value backup in generate_response.", flush=True)

    try:
        if req.k is not None and req.k > 0:
            if hasattr(agent, 'memory') and hasattr(agent.memory, 'memory_retriever'):
                agent.memory.memory_retriever.k = req.k
            else:
                print(f"WARN: Cannot set k for agent {agent_id}; missing memory or retriever.", flush=True)

        prompt_is_important, response_text = agent.generate_dialogue_response(req.prompt.strip(), datetime.now())
    
    except Exception as e:
        print(f"ERROR_STACKTRACE: Error during dialogue generation for agent {agent_id}: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during dialogue generation: {e}")
    finally:
        if original_k != -1 and hasattr(agent, 'memory') and hasattr(agent.memory, 'memory_retriever'): 
            agent.memory.memory_retriever.k = original_k 
        
    return {
        "agent_name": agent.name, 
        "response": response_text,
        "prompt_is_important_to_memorize": prompt_is_important
    }

@app.post("/agents/{agent_id}/add_memory")
def add_memory(agent_id: str, req: AddMemoryReq):
    print(f"DEBUG: /add_memory for agent {agent_id} with text: '{req.text_to_memorize[:50]}...'", flush=True)
    if agent_id not in agents or agents[agent_id] is None:
        print(f"ERROR: Agent '{agent_id}' not found or is None for add_memory.", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")
    
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
    if agent_id not in agents or agents[agent_id] is None:
        print(f"ERROR: Agent '{agent_id}' not found or is None for fetch_memories.", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")

    observation = req.observation.strip()
    if not observation:
        raise HTTPException(status_code=400, detail="Observation text may not be empty for fetching memories.")

    agent = agents[agent_id]
    original_k = -1
    if hasattr(agent, 'memory') and hasattr(agent.memory, 'memory_retriever'):
        original_k = agent.memory.memory_retriever.k
    else:
        print(f"WARN: Agent {agent_id} missing memory or retriever for k value backup in fetch_memories.", flush=True)
    
    try:
        if req.k is not None and req.k > 0:
            if hasattr(agent, 'memory') and hasattr(agent.memory, 'memory_retriever'):
                agent.memory.memory_retriever.k = req.k
            else:
                print(f"WARN: Cannot set k for agent {agent_id}; missing memory or retriever.", flush=True)

        retrieved_docs: List[Document] = agent.memory.fetch_memories(observation, now=datetime.now())
        memories_content: List[str] = [doc.page_content for doc in retrieved_docs]

    except Exception as e:
        print(f"ERROR_STACKTRACE: Error fetching memories for agent {agent_id}: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching memories: {e}")
    finally:
        if original_k != -1 and hasattr(agent, 'memory') and hasattr(agent.memory, 'memory_retriever'): 
             agent.memory.memory_retriever.k = original_k
        
    return {"memories": memories_content}


@app.get("/agents/{agent_id}/summary")
def get_summary(agent_id: str): 
    print(f"DEBUG: /summary for agent {agent_id}", flush=True)
    if agent_id not in agents or agents[agent_id] is None:
        print(f"ERROR: Agent '{agent_id}' not found or is None for get_summary.", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")
    
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
