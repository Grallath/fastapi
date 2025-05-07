# File: main.py
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from uuid import uuid4
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

# ANSI Color Codes
class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'

    # Custom colors for scores/importance
    IMPORTANCE_HIGH = OKGREEN
    IMPORTANCE_MEDIUM = WARNING # Yellow
    IMPORTANCE_LOW = FAIL # Red
    
    METADATA_KEY = OKCYAN
    METADATA_VALUE = OKBLUE
    CONTENT_COLOR = ENDC # Default color for main content
    SEPARATOR = DIM # Dim color for separators

app = FastAPI(title="Generative-Agent API (Refactored)")

print(f"{BColors.OKGREEN}DEBUG: FastAPI application starting up...{BColors.ENDC}", flush=True)

# Default model names if not provided in requests
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small" 

@app.get("/")
async def health_check():
    print(f"{BColors.OKGREEN}DEBUG: Health check '/' endpoint hit.{BColors.ENDC}", flush=True)
    return {"status": "ok"}

if not os.getenv("OPENAI_API_KEY"):
    print(f"{BColors.FAIL}CRITICAL_WARNING: OPENAI_API_KEY environment variable is NOT SET. OpenAI calls will likely fail.{BColors.ENDC}", flush=True)
else:
    print(f"{BColors.OKGREEN}DEBUG: OPENAI_API_KEY environment variable is detected.{BColors.ENDC}", flush=True)


def _new_agent_instance(
    name: str,
    age: int,
    traits: str,
    status: str,
    summary_refresh_seconds: int,
    reflection_threshold: int,
    verbose: bool,
    llm_model_name: Optional[str] = None,
    embedding_model_name: Optional[str] = None
) -> GenerativeAgent:
    print(f"{BColors.OKBLUE}DEBUG: _new_agent_instance called for agent '{name}'{BColors.ENDC}", flush=True)
    print(f"{BColors.DIM}  LLM Model Request: '{llm_model_name}', Embedding Model Request: '{embedding_model_name}'{BColors.ENDC}", flush=True)
    print(f"{BColors.DIM}  Input summary_refresh_seconds: {summary_refresh_seconds}, Input reflection_threshold: {reflection_threshold}{BColors.ENDC}", flush=True)

    effective_llm_model = llm_model_name if llm_model_name and llm_model_name.strip() else DEFAULT_CHAT_MODEL
    effective_embedding_model = embedding_model_name if embedding_model_name and embedding_model_name.strip() else DEFAULT_EMBEDDING_MODEL
    
    print(f"{BColors.DIM}  Effective LLM Model: '{effective_llm_model}'{BColors.ENDC}", flush=True)
    print(f"{BColors.DIM}  Effective Embedding Model: '{effective_embedding_model}'{BColors.ENDC}", flush=True)

    agent_llm = None
    try:
        print(f"{BColors.DIM}DEBUG: Attempting to initialize ChatOpenAI for agent '{name}' with model: {effective_llm_model}{BColors.ENDC}", flush=True)
        agent_llm = ChatOpenAI(model_name=effective_llm_model, temperature=0.7)
        print(f"{BColors.OKGREEN}DEBUG: ChatOpenAI for agent '{name}' (model {effective_llm_model}) initialized.{BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed to initialize LLM for agent '{name}' with model '{effective_llm_model}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to initialize LLM with model '{effective_llm_model}': {e}")

    agent_embeddings = None
    dim = 0 
    try:
        print(f"{BColors.DIM}DEBUG: Attempting to initialize OpenAIEmbeddings for agent '{name}' with model: {effective_embedding_model}{BColors.ENDC}", flush=True)
        agent_embeddings = OpenAIEmbeddings(model=effective_embedding_model)
        probe_for_dim = agent_embeddings.embed_query("get_dim_probe_for_agent")
        dim = len(probe_for_dim)
        print(f"{BColors.OKGREEN}DEBUG: OpenAIEmbeddings for agent '{name}' (model {effective_embedding_model}, dim {dim}) initialized and tested.{BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed to initialize or test OpenAIEmbeddings for agent '{name}' with model '{effective_embedding_model}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to initialize/test embeddings with model '{effective_embedding_model}': {e}")

    print(f"{BColors.DIM}DEBUG: Setting up FAISS index for agent '{name}' (dim: {dim})...{BColors.ENDC}", flush=True)
    try:
        index = faiss.IndexFlatL2(dim)
        vectorstore = FAISS(
            embedding_function=agent_embeddings,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )
        retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=vectorstore, k=15, decay_rate=0.01 # k here is default, can be overridden by requests
        )
        print(f"{BColors.OKGREEN}DEBUG: FAISS setup complete for agent '{name}'.{BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed during FAISS setup for agent '{name}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during FAISS setup: {e}")

    print(f"{BColors.DIM}DEBUG: Setting up GenerativeAgentMemory for agent '{name}'...{BColors.ENDC}", flush=True)
    try:
        actual_reflect_for_memory = reflection_threshold if reflection_threshold > 0 else None
        print(f"{BColors.DIM}DEBUG: actual_reflect_for_memory (for GenerativeAgentMemory) will be: {actual_reflect_for_memory}{BColors.ENDC}", flush=True)
        
        memory = GenerativeAgentMemory(
            llm=agent_llm,
            memory_retriever=retriever,
            reflection_threshold=actual_reflect_for_memory, 
        )
        print(f"{BColors.OKGREEN}DEBUG: GenerativeAgentMemory setup complete for agent '{name}'.{BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed during GenerativeAgentMemory setup for agent '{name}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during GenerativeAgentMemory setup: {e}")

    print(f"{BColors.DIM}DEBUG: Initializing GenerativeAgent '{name}'...{BColors.ENDC}", flush=True)
    try:
        actual_refresh_for_agent = summary_refresh_seconds
        print(f"{BColors.DIM}DEBUG: actual_refresh_for_agent (for GenerativeAgent) will be: {actual_refresh_for_agent}{BColors.ENDC}", flush=True)

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
        print(f"{BColors.OKGREEN}DEBUG: GenerativeAgent '{name}' initialized successfully.{BColors.ENDC}", flush=True)
        return agent
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed during GenerativeAgent initialization for agent '{name}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during GenerativeAgent initialization: {e}")

agents: Dict[str, GenerativeAgent] = {}

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
    print(f"{BColors.HEADER}DEBUG: /agents POST request received: {req.model_dump_json(exclude_none=True)}{BColors.ENDC}", flush=True) 
    aid = req.agent_id or str(uuid4())
    if aid in agents:
        print(f"{BColors.WARNING}WARN: Agent with agent_id '{aid}' already exists.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=400, detail=f"Agent with agent_id '{aid}' already exists.")
    
    current_agent_instance = None
    try:
        print(f"{BColors.DIM}DEBUG: Calling _new_agent_instance for agent_id '{aid}' with name '{req.name}'{BColors.ENDC}", flush=True)
        current_agent_instance = _new_agent_instance( 
            name=req.name,
            age=req.age,
            traits=req.traits,
            status=req.status,
            summary_refresh_seconds=req.summary_refresh_seconds,
            reflection_threshold=req.reflection_threshold,
            verbose=req.verbose,
            llm_model_name=req.model_name, 
            embedding_model_name=req.embedding_model_name 
        )
        agents[aid] = current_agent_instance 
        print(f"{BColors.OKGREEN}DEBUG: Agent '{aid}' (name: '{req.name}') created and stored.{BColors.ENDC}", flush=True)
    except HTTPException as e: 
        print(f"{BColors.WARNING}DEBUG: HTTPException caught in create_agent for agent_id '{aid}': {e.detail}{BColors.ENDC}", flush=True)
        raise e 
    except Exception as e: 
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Unexpected error in create_agent endpoint for agent_id '{aid}': {e}{BColors.ENDC}", flush=True)
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
        # This case should ideally not happen if creation was successful
        print(f"{BColors.FAIL}ERROR: current_agent_instance is None after creation attempt for agent_id '{aid}'. This indicates a logic flaw.{BColors.ENDC}", flush=True)
        # Attempt to retrieve from dict anyway as a fallback, though it implies an issue
        if aid in agents and agents[aid] is not None: 
            retrieved_agent = agents[aid]
            if hasattr(retrieved_agent, 'llm') and retrieved_agent.llm and hasattr(retrieved_agent.llm, 'model_name'):
                 llm_model_used = retrieved_agent.llm.model_name
            if (hasattr(retrieved_agent, 'memory') and hasattr(retrieved_agent.memory, 'memory_retriever') and 
                hasattr(retrieved_agent.memory.memory_retriever, 'vectorstore') and hasattr(retrieved_agent.memory.memory_retriever.vectorstore, 'embedding_function')
                and hasattr(retrieved_agent.memory.memory_retriever.vectorstore.embedding_function, 'model')):
                 embedding_model_used = retrieved_agent.memory.memory_retriever.vectorstore.embedding_function.model

    print(f"{BColors.OKGREEN}DEBUG: Agent '{aid}' creation processing complete. LLM: {llm_model_used}, Embedding: {embedding_model_used}. Responding.{BColors.ENDC}", flush=True)
    return {
        "agent_id": aid, 
        "name": req.name, 
        "llm_model_used": llm_model_used,
        "embedding_model_used": embedding_model_used
    }

@app.get("/agents")
def list_agents():
    print(f"{BColors.HEADER}DEBUG: /agents GET request received (list_agents){BColors.ENDC}", flush=True)
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
            print(f"{BColors.WARNING}WARN: Agent {agent_id} found in agents dictionary, but its instance is None.{BColors.ENDC}", flush=True)

        agent_details.append({
            "agent_id": agent_id, 
            "name": agent_name_from_instance, 
            "llm_model": llm_model,
            "embedding_model": emb_model
        })
    print(f"{BColors.DIM}DEBUG: Returning {len(agent_details)} agents.{BColors.ENDC}", flush=True)
    return {"agents": agent_details}


@app.post("/agents/{agent_id}/generate_response")
def generate_response(agent_id: str, req: GenerateResponseReq):
    print(f"{BColors.HEADER}DEBUG: /generate_response for agent {agent_id} with prompt: '{req.prompt[:50]}...' (K={req.k or 'default'}){BColors.ENDC}", flush=True)
    if agent_id not in agents or agents[agent_id] is None:
        print(f"{BColors.FAIL}ERROR: Agent '{agent_id}' not found or is None for generate_response.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")
    
    agent = agents[agent_id]
    original_k = -1 
    retriever = agent.memory.memory_retriever
    if hasattr(retriever, 'k'):
         original_k = retriever.k
    else:
        print(f"{BColors.WARNING}WARN: Agent {agent_id} retriever missing k attribute for k value backup in generate_response.{BColors.ENDC}", flush=True)

    try:
        if req.k is not None and req.k > 0:
            if hasattr(retriever, 'k'):
                retriever.k = req.k
            else:
                print(f"{BColors.WARNING}WARN: Cannot set k for agent {agent_id}; retriever missing k attribute.{BColors.ENDC}", flush=True)

        prompt_is_important, response_text = agent.generate_dialogue_response(req.prompt.strip(), datetime.now())
    
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Error during dialogue generation for agent {agent_id}: {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during dialogue generation: {e}")
    finally:
        if original_k != -1 and hasattr(retriever, 'k'): 
            retriever.k = original_k 
        
    return {
        "agent_name": agent.name, 
        "response": response_text,
        "prompt_is_important_to_memorize": prompt_is_important
    }

@app.post("/agents/{agent_id}/add_memory")
def add_memory(agent_id: str, req: AddMemoryReq):
    print(f"{BColors.HEADER}DEBUG: /add_memory for agent {agent_id} with text: '{req.text_to_memorize[:50]}...'{BColors.ENDC}", flush=True)
    if agent_id not in agents or agents[agent_id] is None:
        print(f"{BColors.FAIL}ERROR: Agent '{agent_id}' not found or is None for add_memory.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")
    
    text_to_add = req.text_to_memorize.strip()
    if not text_to_add:
        print(f"{BColors.WARNING}WARN: Memory text may not be empty.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=400, detail="Memory text may not be empty.")
    
    try:
        agents[agent_id].memory.add_memory(text_to_add, now=datetime.now())
        print(f"{BColors.OKGREEN}DEBUG: Memory added successfully for agent {agent_id}.{BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Error adding memory for agent {agent_id}: {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error adding memory: {e}")

    return {"added_memory": text_to_add}

@app.post("/agents/{agent_id}/fetch_memories")
def fetch_memories(agent_id: str, req: FetchMemoriesReq):
    print(f"{BColors.HEADER}>>> Incoming Fetch Memories Request for Agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER} <<<", flush=True)
    print(f"{BColors.DIM}Observation: '{req.observation[:100]}...' (K={req.k or 'default'}){BColors.ENDC}", flush=True)

    if agent_id not in agents or agents[agent_id] is None:
        print(f"{BColors.FAIL}ERROR: Agent '{agent_id}' not found or is None for fetch_memories.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")

    observation = req.observation.strip()
    if not observation:
        print(f"{BColors.FAIL}ERROR: Observation text may not be empty for fetching memories.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=400, detail="Observation text may not be empty for fetching memories.")

    agent = agents[agent_id]
    original_k = -1
    
    retriever = agent.memory.memory_retriever
    if hasattr(retriever, 'k'):
        original_k = retriever.k
    
    retrieved_docs_for_response: List[str] = []

    try:
        if req.k is not None and req.k > 0:
            if hasattr(retriever, 'k'):
                retriever.k = req.k
            else:
                print(f"{BColors.WARNING}WARN: Cannot set k for agent {agent_id}; retriever does not have 'k' attribute.{BColors.ENDC}", flush=True)

        retrieved_docs: List[Document] = agent.memory.fetch_memories(observation, now=datetime.now())
        retrieved_docs_for_response = [doc.page_content for doc in retrieved_docs]

        # --- Enhanced Logging Block ---
        print(f"\n{BColors.OKBLUE}--- Detailed Fetched Memories (Agent: {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.OKBLUE}) ---{BColors.ENDC}", flush=True)
        if not retrieved_docs:
            print(f"{BColors.WARNING}No memories were fetched for this observation.{BColors.ENDC}", flush=True)
        else:
            print(f"{BColors.DIM}Displaying {len(retrieved_docs)} fetched memories:{BColors.ENDC}", flush=True)
            for i, doc in enumerate(retrieved_docs):
                importance = doc.metadata.get('importance', 0.0)
                created_at_raw = doc.metadata.get('created_at')
                last_accessed_at_raw = doc.metadata.get('last_accessed_at')
                buffer_idx = doc.metadata.get('buffer_idx', 'N/A')

                created_at_str = created_at_raw.strftime("%Y-%m-%d %H:%M:%S") if hasattr(created_at_raw, 'strftime') else str(created_at_raw)
                last_accessed_at_str = last_accessed_at_raw.strftime("%Y-%m-%d %H:%M:%S") if hasattr(last_accessed_at_raw, 'strftime') else str(last_accessed_at_raw)

                importance_color = BColors.IMPORTANCE_LOW
                if importance >= 0.7:
                    importance_color = BColors.IMPORTANCE_HIGH
                elif importance >= 0.4:
                    importance_color = BColors.IMPORTANCE_MEDIUM
                
                print(f"{BColors.SEPARATOR}{'-'*70}{BColors.ENDC}", flush=True)
                print(f"{BColors.BOLD}Memory #{i+1}:{BColors.ENDC}", flush=True)
                print(f"  {BColors.METADATA_KEY}Static Importance:{BColors.ENDC} {importance_color}{importance:.3f}{BColors.ENDC}", flush=True)
                print(f"  {BColors.METADATA_KEY}Content:{BColors.ENDC}\n{BColors.CONTENT_COLOR}    \"{doc.page_content.strip()}\"{BColors.ENDC}", flush=True) # Added quotes around content
                print(f"  {BColors.DIM}{BColors.METADATA_KEY}Details:{BColors.ENDC}")
                print(f"    {BColors.METADATA_KEY}Created At:{BColors.ENDC} {BColors.METADATA_VALUE}{created_at_str}{BColors.ENDC}", flush=True)
                print(f"    {BColors.METADATA_KEY}Last Accessed:{BColors.ENDC} {BColors.METADATA_VALUE}{last_accessed_at_str}{BColors.ENDC}", flush=True)
                print(f"    {BColors.METADATA_KEY}Buffer Idx:{BColors.ENDC} {BColors.METADATA_VALUE}{buffer_idx}{BColors.ENDC}", flush=True)
        print(f"{BColors.SEPARATOR}{'-'*70}{BColors.ENDC}", flush=True)
        print(f"{BColors.OKBLUE}--- End of Detailed Fetched Memories ---{BColors.ENDC}\n", flush=True)
        # --- End of Enhanced Logging Block ---

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Error fetching memories for agent {agent_id}: {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching memories: {e}")
    finally:
        if original_k != -1 and hasattr(retriever, 'k'): 
             retriever.k = original_k
        
    print(f"{BColors.HEADER}<<< Completing Fetch Memories Request for Agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER} >>>{BColors.ENDC}\n", flush=True)
    return {"memories": retrieved_docs_for_response}


@app.get("/agents/{agent_id}/summary")
def get_summary(agent_id: str): 
    print(f"{BColors.HEADER}DEBUG: /summary for agent {agent_id}{BColors.ENDC}", flush=True)
    if agent_id not in agents or agents[agent_id] is None:
        print(f"{BColors.FAIL}ERROR: Agent '{agent_id}' not found or is None for get_summary.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")
    
    summary_text = "Error generating summary."
    try:
        summary_text = agents[agent_id].get_summary(force_refresh=True)
        print(f"{BColors.OKGREEN}DEBUG: Summary generated successfully for agent {agent_id}.{BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Error generating summary for agent {agent_id}: {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating summary: {e}")
        
    return {"agent_id": agent_id, "summary": summary_text}

@app.delete("/agents/{agent_id}")
def delete_agent(agent_id: str):
    print(f"{BColors.HEADER}DEBUG: /delete_agent for agent {agent_id}{BColors.ENDC}", flush=True)
    if agent_id not in agents: # No need to check agents[agent_id] is None if key not present
        print(f"{BColors.FAIL}ERROR: Agent '{agent_id}' not found for delete_agent.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    
    del agents[agent_id]
    print(f"{BColors.OKGREEN}DEBUG: Agent '{agent_id}' deleted successfully.{BColors.ENDC}", flush=True)
    return {"deleted_agent_id": agent_id, "status": "success"}

print(f"{BColors.OKGREEN}DEBUG: FastAPI application finished loading.{BColors.ENDC}", flush=True)
