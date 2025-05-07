# File: main.py (Complete and Modified)
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple

from uuid import uuid4
import os
import traceback
import numpy as np # Ensure numpy is imported

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import faiss
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_experimental.generative_agents import GenerativeAgentMemory # Still needed for memory
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import DistanceStrategy

# --- IMPORT THE CUSTOM AGENT ---
from custom_agent import AutonomousGenerativeAgent, BColors # Import custom class and colors

app = FastAPI(title="Autonomous Generative-Agent API") # Updated title

print(f"{BColors.OKGREEN}DEBUG: FastAPI application starting up... (Using Autonomous Agents){BColors.ENDC}", flush=True)

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


# Health Check endpoint remains the same
@app.get("/")
async def health_check():
    print(f"{BColors.OKGREEN}DEBUG: Health check '/' endpoint hit.{BColors.ENDC}", flush=True)
    return {"status": "ok"}


# API Key check remains the same
if not os.getenv("OPENAI_API_KEY"):
    print(f"{BColors.FAIL}CRITICAL_WARNING: OPENAI_API_KEY environment variable is NOT SET. OpenAI calls likely fail.{BColors.ENDC}", flush=True)
else:
    print(f"{BColors.OKGREEN}DEBUG: OPENAI_API_KEY environment variable is detected.{BColors.ENDC}", flush=True)


def _new_agent_instance( # Keep function signature the same
    name: str,
    age: int,
    traits: str,
    status: str,
    summary_refresh_seconds: int,
    reflection_threshold: int, # Keep as int
    verbose: bool,
    llm_model_name: Optional[str] = None,
    embedding_model_name: Optional[str] = None
) -> AutonomousGenerativeAgent: # Return type is now the custom class
    print(f"{BColors.OKBLUE}DEBUG: _new_agent_instance called for agent '{BColors.BOLD}{name}{BColors.ENDC}{BColors.OKBLUE}' (Using Autonomous Class){BColors.ENDC}", flush=True)

    # --- LLM Initialization ---
    effective_llm_model = llm_model_name if llm_model_name and llm_model_name.strip() else DEFAULT_CHAT_MODEL
    agent_llm = None
    try:
        print(f"{BColors.DIM}DEBUG: Attempting to initialize ChatOpenAI for agent '{name}' with model: {effective_llm_model}{BColors.ENDC}", flush=True)
        agent_llm = ChatOpenAI(model_name=effective_llm_model, temperature=0.7)
        print(f"{BColors.OKGREEN}DEBUG: ChatOpenAI for agent '{name}' (model {effective_llm_model}) initialized.{BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed to initialize LLM for agent '{name}' with model '{effective_llm_model}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to initialize LLM with model '{effective_llm_model}': {e}")

    # --- Embedding Initialization ---
    effective_embedding_model = embedding_model_name if embedding_model_name and embedding_model_name.strip() else DEFAULT_EMBEDDING_MODEL
    agent_embeddings = None
    dim = 0
    try:
        print(f"{BColors.DIM}DEBUG: Attempting to initialize OpenAIEmbeddings for agent '{name}' with model: {effective_embedding_model}{BColors.ENDC}", flush=True)
        agent_embeddings = OpenAIEmbeddings(model=effective_embedding_model)
        # Probe to get embedding dimension
        probe_for_dim = agent_embeddings.embed_query("get_dim_probe_for_agent")
        dim = len(probe_for_dim)
        print(f"{BColors.OKGREEN}DEBUG: OpenAIEmbeddings for agent '{name}' (model {effective_embedding_model}, dim {dim}) initialized and tested.{BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed to initialize or test OpenAIEmbeddings for agent '{name}' with model '{effective_embedding_model}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to initialize/test embeddings with model '{effective_embedding_model}': {e}")

    # --- FAISS Setup ---
    print(f"{BColors.DIM}DEBUG: Setting up FAISS index for agent '{name}' (dim: {dim}). Using Inner Product.{BColors.ENDC}", flush=True)
    try:
        # Using IndexFlatIP for Inner Product. L2 normalization is important.
        index = faiss.IndexFlatIP(dim)
        vectorstore = FAISS(
            embedding_function=agent_embeddings,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
            normalize_L2=True, # Normalize vectors for IP distance
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT # Use IP strategy
        )
        # Consider adjusting decay_rate and k based on experimentation
        retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=vectorstore, k=15, decay_rate=0.01 # Default k=15
        )
        print(f"{BColors.OKGREEN}DEBUG: FAISS setup complete for agent '{name}'. (Index: IP, Normalize: True, Strategy: MAX_INNER_PRODUCT){BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed during FAISS setup for agent '{name}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during FAISS setup: {e}")

    # --- Memory Setup ---
    print(f"{BColors.DIM}DEBUG: Setting up GenerativeAgentMemory for agent '{name}'...{BColors.ENDC}", flush=True)
    try:
        # Reflection threshold needs to be Optional[float] for base GenerativeAgentMemory
        actual_reflect_for_memory = float(reflection_threshold) if reflection_threshold > 0 else None
        print(f"{BColors.DIM}DEBUG: actual_reflect_for_memory (for GenerativeAgentMemory) will be: {actual_reflect_for_memory}{BColors.ENDC}", flush=True)
        memory_instance = GenerativeAgentMemory(
            llm=agent_llm,
            memory_retriever=retriever,
            reflection_threshold=actual_reflect_for_memory, # Use the float/None value
            verbose=verbose,
            # Ensure necessary kwargs are passed if GenerativeAgentMemory's init changes
        )
        print(f"{BColors.OKGREEN}DEBUG: GenerativeAgentMemory setup complete for agent '{name}'.{BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed during GenerativeAgentMemory setup for agent '{name}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during GenerativeAgentMemory setup: {e}")

    # --- INSTANTIATE THE CUSTOM AGENT CLASS ---
    print(f"{BColors.DIM}DEBUG: Initializing AutonomousGenerativeAgent '{name}'...{BColors.ENDC}", flush=True)
    try:
        # summary_refresh_seconds expects int in base GenerativeAgent
        agent = AutonomousGenerativeAgent( # Use the custom class
            name=name,
            age=age,
            traits=traits,
            status=status,
            memory=memory_instance, # Pass the GenerativeAgentMemory instance
            llm=agent_llm,
            summary_refresh_seconds=summary_refresh_seconds, # Pass int directly
            verbose=verbose,
            # Add other necessary kwargs if the __init__ signature requires them
        )
        # Immediately initialize the custom chains after the agent object is created
        agent._initialize_chains()
        print(f"{BColors.OKGREEN}DEBUG: AutonomousGenerativeAgent '{name}' initialized successfully.{BColors.ENDC}", flush=True)
        return agent
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed during AutonomousGenerativeAgent initialization for agent '{name}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during AutonomousGenerativeAgent initialization: {e}")


# --- Ensure agents dictionary holds the correct type ---
agents: Dict[str, AutonomousGenerativeAgent] = {} # Use the custom class type

# --- Pydantic Models ---
class CreateAgentReq(BaseModel):
    name: str
    age: int
    traits: str
    status: str
    agent_id: Optional[str] = None
    summary_refresh_seconds: int = Field(default=3600, ge=0) # Default to 1 hour
    reflection_threshold: int = Field(default=0, ge=0) # Default 0 = disabled (API expects int)
    verbose: bool = False
    model_name: Optional[str] = None
    embedding_model_name: Optional[str] = None

class GenerateResponseReq(BaseModel):
    prompt: str # Use 'prompt' as the observation/input text
    k: Optional[int] = Field(default=None, gt=0) # Optional override for retriever K

# Structure for the new response
class GenerateReactionResponse(BaseModel):
    agent_name: str
    reaction_type: str # SAY, THINK, DO, IGNORE, UNKNOWN
    content: str       # The actual dialogue, thought, or action description
    observation_was_important: bool # Based on initial poignancy score

class AddMemoryReq(BaseModel):
    text_to_memorize: str

class FetchMemoriesReq(BaseModel):
    observation: str
    k: Optional[int] = Field(default=None, gt=0)

class UpdateStatusReq(BaseModel):
    new_status: str


# --- Endpoints ---

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
            reflection_threshold=req.reflection_threshold, # Pass int
            verbose=req.verbose,
            llm_model_name=req.model_name,
            embedding_model_name=req.embedding_model_name
        )
        agents[aid] = current_agent_instance
        print(f"{BColors.OKGREEN}DEBUG: Agent '{BColors.BOLD}{aid}{BColors.ENDC}{BColors.OKGREEN}' (name: '{req.name}') created and stored.{BColors.ENDC}", flush=True)
    except HTTPException as http_exc:
        print(f"{BColors.WARNING}DEBUG: HTTPException caught in create_agent for agent_id '{aid}': {http_exc.detail}{BColors.ENDC}", flush=True)
        raise http_exc
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Unexpected error in create_agent endpoint for agent_id '{aid}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        if aid in agents: del agents[aid]
        raise HTTPException(status_code=500, detail=f"Unexpected server error during agent creation: {e}")

    # --- Get model names used ---
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
        print(f"{BColors.FAIL}ERROR: current_agent_instance is None after creation attempt for agent_id '{aid}'. This indicates a logic flaw.{BColors.ENDC}", flush=True)
        if aid in agents and agents[aid] is not None:
             retrieved_agent = agents[aid]
             if hasattr(retrieved_agent, 'llm') and retrieved_agent.llm and hasattr(retrieved_agent.llm, 'model_name'):
                 llm_model_used = retrieved_agent.llm.model_name
             if (hasattr(retrieved_agent, 'memory') and hasattr(retrieved_agent.memory, 'memory_retriever') and
                 hasattr(retrieved_agent.memory.memory_retriever, 'vectorstore') and hasattr(retrieved_agent.memory.memory_retriever.vectorstore, 'embedding_function')
                 and hasattr(retrieved_agent.memory.memory_retriever.vectorstore.embedding_function, 'model')):
                 embedding_model_used = retrieved_agent.memory.memory_retriever.vectorstore.embedding_function.model

    print(f"{BColors.OKGREEN}DEBUG: Agent '{BColors.BOLD}{aid}{BColors.ENDC}{BColors.OKGREEN}' creation processing complete. LLM: {llm_model_used}, Embedding: {embedding_model_used}. Responding.{BColors.ENDC}", flush=True)
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
        current_status = "Unknown Status"

        if agent_instance_from_dict:
            agent_name_from_instance = agent_instance_from_dict.name
            current_status = agent_instance_from_dict.status
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
            "status": current_status,
            "llm_model": llm_model,
            "embedding_model": emb_model
        })
    print(f"{BColors.DIM}DEBUG: Returning {len(agent_details)} agents.{BColors.ENDC}", flush=True)
    return {"agents": agent_details}


@app.post("/agents/{agent_id}/update_status")
def update_agent_status(agent_id: str, req: UpdateStatusReq):
    print(f"{BColors.HEADER}>>> Incoming Update Status Request for Agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER} <<<", flush=True)
    print(f"{BColors.DIM}New Status: '{req.new_status}'{BColors.ENDC}", flush=True)

    if agent_id not in agents or agents[agent_id] is None:
        print(f"{BColors.FAIL}ERROR: Agent '{BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.FAIL}' not found or is None for update_status.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")

    agent = agents[agent_id]

    try:
        new_status_stripped = req.new_status.strip()
        if not new_status_stripped:
            raise HTTPException(status_code=400, detail="New status cannot be empty.")
        agent.status = new_status_stripped
        print(f"{BColors.OKGREEN}SUCCESS: Agent '{BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.OKGREEN}' status updated to: '{agent.status}'{BColors.ENDC}", flush=True)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Error updating status for agent '{BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.FAIL}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error updating agent status: {e}")

    print(f"{BColors.HEADER}<<< Completing Update Status Request for Agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER} >>>{BColors.ENDC}\n", flush=True)
    return {"agent_id": agent_id, "status": agent.status}


@app.post("/agents/{agent_id}/generate_response", response_model=GenerateReactionResponse)
def generate_response(agent_id: str, req: GenerateResponseReq):
    observation = req.prompt.strip()
    print(f"{BColors.HEADER}DEBUG: /generate_response for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER} with observation: '{observation[:50]}...' (K={req.k or 'default'}){BColors.ENDC}", flush=True)
    if agent_id not in agents or agents[agent_id] is None:
        print(f"{BColors.FAIL}ERROR: Agent '{agent_id}' not found or is None for generate_response.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")

    if not observation:
         print(f"{BColors.WARNING}WARN: Observation cannot be empty.{BColors.ENDC}", flush=True)
         raise HTTPException(status_code=400, detail="Observation cannot be empty.")

    agent = agents[agent_id]
    original_k = -1
    retriever = agent.memory.memory_retriever
    if hasattr(retriever, 'k'):
         original_k = retriever.k
    else:
        print(f"{BColors.WARNING}WARN: Agent {agent_id} retriever missing k attribute for k value backup in generate_response.{BColors.ENDC}", flush=True)

    reaction_string = ""
    observation_was_important = False
    try:
        if req.k is not None and req.k > 0:
            if hasattr(retriever, 'k'):
                retriever.k = req.k
                print(f"{BColors.DIM}DEBUG: Temporarily set retriever k to {req.k} for agent {agent_id}.{BColors.ENDC}", flush=True)
            else:
                print(f"{BColors.WARNING}WARN: Cannot set k for agent {agent_id}; retriever missing k attribute.{BColors.ENDC}", flush=True)

        print(f"{BColors.DIM}Agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.DIM} current status before reaction: '{agent.status}'{BColors.ENDC}", flush=True)

        # --- CALL THE NEW generate_reaction METHOD ---
        observation_was_important, reaction_string = agent.generate_reaction(observation, datetime.now())
        print(f"{BColors.OKGREEN}DEBUG: agent.generate_reaction completed for {agent_id}. Raw output: '{reaction_string}' | Observation important flag: {observation_was_important}{BColors.ENDC}", flush=True)

    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Error during reaction generation for agent {agent_id}: {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during reaction generation: {e}")
    finally:
        if original_k != -1 and hasattr(retriever, 'k'):
             retriever.k = original_k
             print(f"{BColors.DIM}DEBUG: Restored retriever k to {original_k} for agent {agent_id}.{BColors.ENDC}", flush=True)

    # --- PARSE THE REACTION STRING ---
    reaction_type = "UNKNOWN"
    content = ""

    if reaction_string.startswith("SAY:"):
        reaction_type = "SAY"
        content = reaction_string[len("SAY:"):].strip()
    elif reaction_string.startswith("THINK:"):
        reaction_type = "THINK"
        content = reaction_string[len("THINK:"):].strip()
    elif reaction_string.startswith("DO:"):
        reaction_type = "DO"
        content = reaction_string[len("DO:"):].strip()
    elif not reaction_string:
        reaction_type = "IGNORE"
        content = ""
    else:
        print(f"{BColors.WARNING}WARN: Unexpected reaction string format from agent {agent_id}: '{reaction_string}'. Setting type to UNKNOWN.{BColors.ENDC}", flush=True)
        content = reaction_string

    return GenerateReactionResponse(
        agent_name=agent.name,
        reaction_type=reaction_type,
        content=content,
        observation_was_important=observation_was_important
    )


@app.post("/agents/{agent_id}/add_memory")
def add_memory(agent_id: str, req: AddMemoryReq):
    print(f"{BColors.HEADER}DEBUG: /add_memory for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER} with text: '{req.text_to_memorize[:50]}...'{BColors.ENDC}", flush=True)
    if agent_id not in agents or agents[agent_id] is None:
        print(f"{BColors.FAIL}ERROR: Agent '{agent_id}' not found or is None for add_memory.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")

    text_to_add = req.text_to_memorize.strip()
    if not text_to_add:
        print(f"{BColors.WARNING}WARN: Memory text may not be empty.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=400, detail="Memory text may not be empty.")

    agent = agents[agent_id]
    try:
        if not agent.memory:
             print(f"{BColors.FAIL}ERROR: Agent {agent_id} memory object is None.{BColors.ENDC}", flush=True)
             raise HTTPException(status_code=500, detail=f"Agent {agent_id} memory not initialized.")

        # Add memory using the agent's memory object
        agent.memory.add_memory(text_to_add, now=datetime.now())
        print(f"{BColors.OKGREEN}DEBUG: Memory added successfully for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.OKGREEN}.{BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Error adding memory for agent {agent_id}: {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error adding memory: {e}")

    return {"status": "success", "added_memory": text_to_add}


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
    if not agent.memory or not agent.memory.memory_retriever:
         print(f"{BColors.FAIL}ERROR: Agent {agent_id} memory or retriever not initialized.{BColors.ENDC}", flush=True)
         raise HTTPException(status_code=500, detail=f"Agent {agent_id} memory/retriever not initialized.")

    original_k = -1
    retriever = agent.memory.memory_retriever
    if hasattr(retriever, 'k'):
        original_k = retriever.k

    retrieved_docs_for_response_payload: List[Dict[str, Any]] = []

    try:
        requested_k = retriever.k
        if req.k is not None and req.k > 0:
            if hasattr(retriever, 'k'):
                requested_k = req.k
                retriever.k = requested_k
                print(f"{BColors.DIM}DEBUG: Temporarily set retriever k to {req.k} for fetch_memories.{BColors.ENDC}", flush=True)
            else:
                print(f"{BColors.WARNING}WARN: Cannot set k for agent {agent_id}; retriever does not have 'k' attribute.{BColors.ENDC}", flush=True)

        # Use the memory's fetch method
        retrieved_docs_only: List[Document] = agent.memory.fetch_memories(observation, now=datetime.now())

        # Get scores separately if possible
        docs_and_scores: List[Tuple[Document, float]] = []
        if (hasattr(agent.memory.memory_retriever, "vectorstore") and
            agent.memory.memory_retriever.vectorstore is not None and
            hasattr(agent.memory.memory_retriever.vectorstore, "similarity_search_with_relevance_scores")):
            print(f"{BColors.DIM}DEBUG: Fetching scores using similarity_search_with_relevance_scores for agent {agent_id}.{BColors.ENDC}", flush=True)
            try:
                docs_and_scores = agent.memory.memory_retriever.vectorstore.similarity_search_with_relevance_scores(
                    observation,
                    k=requested_k,
                )
            except Exception as sim_exc:
                 print(f"{BColors.WARNING}WARN: similarity_search_with_relevance_scores failed: {sim_exc}. Assigning 0.0 scores.{BColors.ENDC}", flush=True)
                 docs_and_scores = [(doc, 0.0) for doc in retrieved_docs_only]
        else:
            print(f"{BColors.WARNING}WARN: Cannot get relevance scores reliably for agent {agent_id}. Scores will be 0.0.{BColors.ENDC}", flush=True)
            docs_and_scores = [(doc, 0.0) for doc in retrieved_docs_only]

        # Prepare response payload
        for doc, score in docs_and_scores:
            serializable_metadata = {}
            for k, v in doc.metadata.items():
                if isinstance(v, datetime):
                    serializable_metadata[k] = v.isoformat()
                # Add handling for other non-serializable types if necessary
                elif isinstance(v, np.ndarray):
                    serializable_metadata[k] = v.tolist() # Example for numpy arrays
                elif isinstance(v, (np.float32, np.float64)):
                     serializable_metadata[k] = float(v) # Convert numpy floats
                elif isinstance(v, (np.int32, np.int64)):
                     serializable_metadata[k] = int(v) # Convert numpy ints
                else:
                    serializable_metadata[k] = v

            retrieved_docs_for_response_payload.append({
                "content": doc.page_content,
                "metadata": serializable_metadata,
                "relevance_score": score
            })

        # --- Detailed Logging ---
        print(f"\n{BColors.OKBLUE}--- Detailed Fetched Memories (Agent: {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.OKBLUE}) ---{BColors.ENDC}", flush=True)
        if not docs_and_scores:
            print(f"{BColors.WARNING}No memories were fetched for this observation.{BColors.ENDC}", flush=True)
        else:
            print(f"{BColors.DIM}Displaying {len(docs_and_scores)} fetched memories:{BColors.ENDC}", flush=True)
            for i, (doc, score) in enumerate(docs_and_scores):
                importance = doc.metadata.get('importance', 0.0)
                created_at_raw = doc.metadata.get('created_at')
                last_accessed_at_raw = doc.metadata.get('last_accessed_at')
                buffer_idx = doc.metadata.get('buffer_idx', 'N/A')
                created_at_str = created_at_raw.isoformat() if isinstance(created_at_raw, datetime) else str(created_at_raw)
                last_accessed_at_str = last_accessed_at_raw.isoformat() if isinstance(last_accessed_at_raw, datetime) else str(last_accessed_at_raw)
                importance_color = BColors.IMPORTANCE_LOW
                if importance >= 0.7: importance_color = BColors.IMPORTANCE_HIGH
                elif importance >= 0.4: importance_color = BColors.IMPORTANCE_MEDIUM
                relevance_color = BColors.IMPORTANCE_LOW
                if score >= 0.8: relevance_color = BColors.IMPORTANCE_HIGH
                elif score >= 0.7: relevance_color = BColors.IMPORTANCE_MEDIUM
                print(f"{BColors.SEPARATOR}{'-'*70}{BColors.ENDC}", flush=True)
                print(f"{BColors.BOLD}Memory #{i+1}:{BColors.ENDC}", flush=True)
                print(f"  {BColors.METADATA_KEY}Relevance Score:{BColors.ENDC} {relevance_color}{score:.4f}{BColors.ENDC}", flush=True)
                print(f"  {BColors.METADATA_KEY}Static Importance:{BColors.ENDC} {importance_color}{importance:.3f}{BColors.ENDC}", flush=True)
                print(f"  {BColors.METADATA_KEY}Content:{BColors.ENDC}\n{BColors.CONTENT_COLOR}    \"{doc.page_content.strip()}\"{BColors.ENDC}", flush=True)
                print(f"  {BColors.DIM}{BColors.METADATA_KEY}Details:{BColors.ENDC}")
                print(f"    {BColors.METADATA_KEY}Created At:{BColors.ENDC} {BColors.METADATA_VALUE}{created_at_str}{BColors.ENDC}", flush=True)
                print(f"    {BColors.METADATA_KEY}Last Accessed:{BColors.ENDC} {BColors.METADATA_VALUE}{last_accessed_at_str}{BColors.ENDC}", flush=True)
                print(f"    {BColors.METADATA_KEY}Buffer Idx:{BColors.ENDC} {BColors.METADATA_VALUE}{buffer_idx}{BColors.ENDC}", flush=True)
                other_meta = {k: v for k, v in doc.metadata.items() if k not in ['importance', 'created_at', 'last_accessed_at', 'buffer_idx']}
                if other_meta:
                    # Attempt to display other meta, converting non-serializable if known
                    display_meta = {}
                    for mk, mv in other_meta.items():
                         if isinstance(mv, np.ndarray): display_meta[mk] = mv.tolist()
                         elif isinstance(mv, (np.float32, np.float64)): display_meta[mk] = float(mv)
                         elif isinstance(mv, (np.int32, np.int64)): display_meta[mk] = int(mv)
                         else: display_meta[mk] = mv
                    print(f"    {BColors.METADATA_KEY}Other Meta:{BColors.ENDC} {BColors.METADATA_VALUE}{display_meta}{BColors.ENDC}", flush=True)

        if docs_and_scores:
            print(f"{BColors.SEPARATOR}{'-'*70}{BColors.ENDC}", flush=True)
        print(f"{BColors.OKBLUE}--- End of Detailed Fetched Memories ---{BColors.ENDC}\n", flush=True)

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Error fetching memories for agent {agent_id}: {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching memories: {e}")
    finally:
        if original_k != -1 and hasattr(retriever, 'k'):
             retriever.k = original_k
             print(f"{BColors.DIM}DEBUG: Restored retriever k to {original_k} for fetch_memories.{BColors.ENDC}", flush=True)

    print(f"{BColors.HEADER}<<< Completing Fetch Memories Request for Agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER} >>>{BColors.ENDC}\n", flush=True)
    return {"memories": retrieved_docs_for_response_payload }


@app.get("/agents/{agent_id}/summary")
def get_summary(agent_id: str):
    print(f"{BColors.HEADER}DEBUG: /summary GET request for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER}", flush=True)
    if agent_id not in agents or agents[agent_id] is None:
        print(f"{BColors.FAIL}ERROR: Agent '{agent_id}' not found or is None for get_summary.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")

    agent = agents[agent_id]
    summary_text = "Error generating summary."
    try:
        summary_text = agent.get_summary(force_refresh=True)
        print(f"{BColors.OKGREEN}DEBUG: Summary generated successfully for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.OKGREEN}.{BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Error generating summary for agent {agent_id}: {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        summary_text = f"Error generating summary: {e}"
        # Consider raising HTTPException if summary failure is critical
        # raise HTTPException(status_code=500, detail=f"Error generating summary: {e}")

    return {"agent_id": agent_id, "summary": summary_text}


@app.delete("/agents/{agent_id}")
def delete_agent(agent_id: str):
    print(f"{BColors.HEADER}DEBUG: /delete_agent DELETE request for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER}", flush=True)
    if agent_id not in agents:
        print(f"{BColors.FAIL}ERROR: Agent '{agent_id}' not found for delete_agent.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")

    try:
        agent_instance = agents.pop(agent_id)
        del agent_instance
        print(f"{BColors.OKGREEN}DEBUG: Agent '{BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.OKGREEN}' deleted successfully.{BColors.ENDC}", flush=True)
    except KeyError:
        print(f"{BColors.FAIL}ERROR: Agent '{agent_id}' was not in the dictionary during deletion attempt.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found during deletion.")
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Unexpected error deleting agent {agent_id}: {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error deleting agent: {e}")

    return {"deleted_agent_id": agent_id, "status": "success"}


print(f"{BColors.OKGREEN}DEBUG: FastAPI application finished loading. (Using Autonomous Agents){BColors.ENDC}", flush=True)

# Run with: uvicorn main:app --reload
