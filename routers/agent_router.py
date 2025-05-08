# File: routers/agent_router.py
import traceback
from datetime import datetime
from typing import Dict, List, Any, Tuple
from uuid import uuid4
import numpy as np

from fastapi import APIRouter, HTTPException, Depends
from langchain_core.documents import Document

from custom_agent import AutonomousGenerativeAgent # Needs to be accessible
from agent_factory import create_new_agent_instance
from schemas import (
    CreateAgentReq, GenerateResponseReq, GenerateReactionResponse, AddMemoryReq,
    FetchMemoriesReq, UpdateStatusReq, AgentListResponse, AgentDetail,
    AgentCreationResponse, SimpleStatusResponse, AddedMemoryResponse,
    FetchedMemoriesResponse, FetchedMemoriesDocument, AgentSummaryResponse,
    DeletedAgentResponse
)
from utils import BColors

router = APIRouter(
    prefix="/agents",
    tags=["agents"]
)

# This is a global-like instance shared across requests for this router.
# In a more complex app, this might be managed by app.state or a database.
agents_db: Dict[str, AutonomousGenerativeAgent] = {}


# --- Helper Function for Model Name Retrieval ---
def get_agent_model_details(agent: AutonomousGenerativeAgent) -> Tuple[str, str]:
    llm_model_used = "unknown"
    embedding_model_used = "unknown"

    if agent:
        if hasattr(agent, 'llm') and agent.llm and hasattr(agent.llm, 'model_name'):
            llm_model_used = agent.llm.model_name

        if (hasattr(agent, 'memory') and agent.memory and
            hasattr(agent.memory, 'memory_retriever') and agent.memory.memory_retriever and
            hasattr(agent.memory.memory_retriever, 'vectorstore') and agent.memory.memory_retriever.vectorstore and
            hasattr(agent.memory.memory_retriever.vectorstore, 'embedding_function') and
            agent.memory.memory_retriever.vectorstore.embedding_function and
            hasattr(agent.memory.memory_retriever.vectorstore.embedding_function, 'model')):
            embedding_model_used = agent.memory.memory_retriever.vectorstore.embedding_function.model
    return llm_model_used, embedding_model_used


# --- Endpoints ---
@router.post("", status_code=201, response_model=AgentCreationResponse)
def create_agent_endpoint(req: CreateAgentReq):
    print(f"{BColors.HEADER}DEBUG: /agents POST request received: {req.model_dump_json(exclude_none=True)}{BColors.ENDC}", flush=True)
    aid = req.agent_id or str(uuid4())
    if aid in agents_db:
        print(f"{BColors.WARNING}WARN: Agent with agent_id '{aid}' already exists.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=400, detail=f"Agent with agent_id '{aid}' already exists.")

    current_agent_instance = None
    try:
        print(f"{BColors.DIM}DEBUG: Calling create_new_agent_instance for agent_id '{aid}' with name '{req.name}'{BColors.ENDC}", flush=True)
        current_agent_instance = create_new_agent_instance( # Use the factory function
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
        agents_db[aid] = current_agent_instance
        print(f"{BColors.OKGREEN}DEBUG: Agent '{BColors.BOLD}{aid}{BColors.ENDC}{BColors.OKGREEN}' (name: '{req.name}') created and stored.{BColors.ENDC}", flush=True)
    except HTTPException as http_exc:
        print(f"{BColors.WARNING}DEBUG: HTTPException caught in create_agent for agent_id '{aid}': {http_exc.detail}{BColors.ENDC}", flush=True)
        raise http_exc
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Unexpected error in create_agent endpoint for agent_id '{aid}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        if aid in agents_db: del agents_db[aid] # Clean up if partially created
        raise HTTPException(status_code=500, detail=f"Unexpected server error during agent creation: {e}")

    llm_model_used, embedding_model_used = "unknown", "unknown"
    if current_agent_instance:
        llm_model_used, embedding_model_used = get_agent_model_details(current_agent_instance)
    else: # Should ideally not happen if creation was successful
        print(f"{BColors.FAIL}ERROR: current_agent_instance is None after creation attempt for agent_id '{aid}'. This indicates a logic flaw.{BColors.ENDC}", flush=True)
        if aid in agents_db and agents_db[aid] is not None:
             llm_model_used, embedding_model_used = get_agent_model_details(agents_db[aid])


    print(f"{BColors.OKGREEN}DEBUG: Agent '{BColors.BOLD}{aid}{BColors.ENDC}{BColors.OKGREEN}' creation processing complete. LLM: {llm_model_used}, Embedding: {embedding_model_used}. Responding.{BColors.ENDC}", flush=True)
    return AgentCreationResponse(
        agent_id=aid,
        name=req.name,
        llm_model_used=llm_model_used,
        embedding_model_used=embedding_model_used
    )

@router.get("", response_model=AgentListResponse)
def list_agents_endpoint():
    print(f"{BColors.HEADER}DEBUG: /agents GET request received (list_agents){BColors.ENDC}", flush=True)
    agent_details = []
    for agent_id, agent_instance in agents_db.items():
        llm_model, emb_model = "unknown", "unknown"
        agent_name = "Unknown Name"
        current_status = "Unknown Status"

        if agent_instance:
            agent_name = agent_instance.name
            current_status = agent_instance.status
            llm_model, emb_model = get_agent_model_details(agent_instance)
        else:
            print(f"{BColors.WARNING}WARN: Agent {agent_id} found in agents_db, but its instance is None.{BColors.ENDC}", flush=True)

        agent_details.append(AgentDetail(
            agent_id=agent_id, name=agent_name, status=current_status,
            llm_model=llm_model, embedding_model=emb_model
        ))
    print(f"{BColors.DIM}DEBUG: Returning {len(agent_details)} agents.{BColors.ENDC}", flush=True)
    return AgentListResponse(agents=agent_details)

@router.post("/{agent_id}/update_status", response_model=AgentDetail)
def update_agent_status_endpoint(agent_id: str, req: UpdateStatusReq):
    print(f"{BColors.HEADER}>>> Incoming Update Status Request for Agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER} <<<", flush=True)
    print(f"{BColors.DIM}New Status: '{req.new_status}'{BColors.ENDC}", flush=True)

    if agent_id not in agents_db or agents_db[agent_id] is None:
        print(f"{BColors.FAIL}ERROR: Agent '{BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.FAIL}' not found or is None.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")

    agent = agents_db[agent_id]
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

    llm_model, emb_model = get_agent_model_details(agent)
    print(f"{BColors.HEADER}<<< Completing Update Status Request for Agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER} >>>{BColors.ENDC}\n", flush=True)
    return AgentDetail(agent_id=agent_id, name=agent.name, status=agent.status, llm_model=llm_model, embedding_model=emb_model)


@router.post("/{agent_id}/generate_response", response_model=GenerateReactionResponse)
def generate_response_endpoint(agent_id: str, req: GenerateResponseReq):
    observation = req.prompt.strip()
    print(f"{BColors.HEADER}DEBUG: /generate_response for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER} with observation: '{observation[:50]}...' (K={req.k or 'default'}){BColors.ENDC}", flush=True)
    if agent_id not in agents_db or agents_db[agent_id] is None:
        print(f"{BColors.FAIL}ERROR: Agent '{agent_id}' not found or is None.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")
    if not observation:
         print(f"{BColors.WARNING}WARN: Observation cannot be empty.{BColors.ENDC}", flush=True)
         raise HTTPException(status_code=400, detail="Observation cannot be empty.")

    agent = agents_db[agent_id]
    original_k = -1
    retriever = agent.memory.memory_retriever
    if hasattr(retriever, 'k'):
         original_k = retriever.k

    reaction_string = ""
    observation_was_important = False
    try:
        if req.k is not None and req.k > 0 and hasattr(retriever, 'k'):
            retriever.k = req.k
            print(f"{BColors.DIM}DEBUG: Temporarily set retriever k to {req.k} for agent {agent_id}.{BColors.ENDC}", flush=True)
        
        print(f"{BColors.DIM}Agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.DIM} current status before reaction: '{agent.status}'{BColors.ENDC}", flush=True)
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

    reaction_type, content = "UNKNOWN", reaction_string
    if reaction_string.startswith("SAY:"): reaction_type, content = "SAY", reaction_string[len("SAY:"):].strip()
    elif reaction_string.startswith("THINK:"): reaction_type, content = "THINK", reaction_string[len("THINK:"):].strip()
    elif reaction_string.startswith("DO:"): reaction_type, content = "DO", reaction_string[len("DO:"):].strip()
    elif not reaction_string: reaction_type, content = "IGNORE", ""
    else: print(f"{BColors.WARNING}WARN: Unexpected reaction string format from agent {agent_id}: '{reaction_string}'.{BColors.ENDC}", flush=True)

    return GenerateReactionResponse(
        agent_name=agent.name, reaction_type=reaction_type, content=content,
        observation_was_important=observation_was_important
    )

@router.post("/{agent_id}/add_memory", response_model=AddedMemoryResponse)
def add_memory_endpoint(agent_id: str, req: AddMemoryReq):
    print(f"{BColors.HEADER}DEBUG: /add_memory for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER} with text: '{req.text_to_memorize[:50]}...'{BColors.ENDC}", flush=True)
    if agent_id not in agents_db or agents_db[agent_id] is None:
        print(f"{BColors.FAIL}ERROR: Agent '{agent_id}' not found or is None.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")
    text_to_add = req.text_to_memorize.strip()
    if not text_to_add:
        print(f"{BColors.WARNING}WARN: Memory text may not be empty.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=400, detail="Memory text may not be empty.")

    agent = agents_db[agent_id]
    try:
        if not agent.memory:
             raise HTTPException(status_code=500, detail=f"Agent {agent_id} memory not initialized.")
        agent.memory.add_memory(text_to_add, now=datetime.now())
        print(f"{BColors.OKGREEN}DEBUG: Memory added successfully for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.OKGREEN}.{BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Error adding memory for agent {agent_id}: {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error adding memory: {e}")
    return AddedMemoryResponse(status="success", added_memory=text_to_add)

@router.post("/{agent_id}/fetch_memories", response_model=FetchedMemoriesResponse)
def fetch_memories_endpoint(agent_id: str, req: FetchMemoriesReq):
    print(f"{BColors.HEADER}>>> Incoming Fetch Memories Request for Agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER} <<<", flush=True)
    print(f"{BColors.DIM}Observation: '{req.observation[:100]}...' (K={req.k or 'default'}){BColors.ENDC}", flush=True)

    if agent_id not in agents_db or agents_db[agent_id] is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")
    observation = req.observation.strip()
    if not observation:
        raise HTTPException(status_code=400, detail="Observation text may not be empty for fetching memories.")

    agent = agents_db[agent_id]
    if not agent.memory or not agent.memory.memory_retriever:
         raise HTTPException(status_code=500, detail=f"Agent {agent_id} memory/retriever not initialized.")

    original_k, retriever = -1, agent.memory.memory_retriever
    if hasattr(retriever, 'k'): original_k = retriever.k
    
    response_payload: List[FetchedMemoriesDocument] = []
    try:
        requested_k = retriever.k
        if req.k is not None and req.k > 0 and hasattr(retriever, 'k'):
            requested_k = req.k
            retriever.k = requested_k
            print(f"{BColors.DIM}DEBUG: Temporarily set retriever k to {req.k}.{BColors.ENDC}", flush=True)

        retrieved_docs_only: List[Document] = agent.memory.fetch_memories(observation, now=datetime.now())
        docs_and_scores: List[Tuple[Document, float]] = []

        if (hasattr(retriever, "vectorstore") and retriever.vectorstore and
            hasattr(retriever.vectorstore, "similarity_search_with_relevance_scores")):
            try:
                docs_and_scores = retriever.vectorstore.similarity_search_with_relevance_scores(observation, k=requested_k)
            except Exception as sim_exc:
                 print(f"{BColors.WARNING}WARN: similarity_search_with_relevance_scores failed: {sim_exc}. Scores 0.0.{BColors.ENDC}", flush=True)
                 docs_and_scores = [(doc, 0.0) for doc in retrieved_docs_only]
        else:
            print(f"{BColors.WARNING}WARN: Cannot get relevance scores. Scores 0.0.{BColors.ENDC}", flush=True)
            docs_and_scores = [(doc, 0.0) for doc in retrieved_docs_only]

        for doc, score in docs_and_scores:
            serializable_metadata = {}
            for k, v in doc.metadata.items():
                if isinstance(v, datetime): serializable_metadata[k] = v.isoformat()
                elif isinstance(v, np.ndarray): serializable_metadata[k] = v.tolist()
                elif isinstance(v, (np.float32, np.float64)): serializable_metadata[k] = float(v)
                elif isinstance(v, (np.int32, np.int64)): serializable_metadata[k] = int(v)
                else: serializable_metadata[k] = v
            response_payload.append(FetchedMemoriesDocument(
                content=doc.page_content, metadata=serializable_metadata, relevance_score=score
            ))
        # Detailed Logging (omitted for brevity here, but keep if useful for your debugging)
        print(f"{BColors.OKBLUE}--- Detailed Fetched Memories (Agent: {agent_id}) Logged to console ---{BColors.ENDC}")

    except HTTPException as e: raise e
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Error fetching memories for agent {agent_id}: {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching memories: {e}")
    finally:
        if original_k != -1 and hasattr(retriever, 'k'):
             retriever.k = original_k
             print(f"{BColors.DIM}DEBUG: Restored retriever k to {original_k}.{BColors.ENDC}", flush=True)
    print(f"{BColors.HEADER}<<< Completing Fetch Memories Request for Agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER} >>>{BColors.ENDC}\n", flush=True)
    return FetchedMemoriesResponse(memories=response_payload)

@router.get("/{agent_id}/summary", response_model=AgentSummaryResponse)
def get_summary_endpoint(agent_id: str):
    print(f"{BColors.HEADER}DEBUG: /summary GET for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER}", flush=True)
    if agent_id not in agents_db or agents_db[agent_id] is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")
    agent = agents_db[agent_id]
    summary_text = "Error generating summary."
    try:
        summary_text = agent.get_summary(force_refresh=True) # Force refresh for this direct endpoint
        print(f"{BColors.OKGREEN}DEBUG: Summary generated for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.OKGREEN}.{BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Error generating summary for {agent_id}: {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        # Potentially raise HTTPException here if summary is critical
        summary_text = f"Error generating summary: {e}"
    return AgentSummaryResponse(agent_id=agent_id, summary=summary_text)

@router.delete("/{agent_id}", response_model=DeletedAgentResponse)
def delete_agent_endpoint(agent_id: str):
    print(f"{BColors.HEADER}DEBUG: /delete_agent DELETE for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER}", flush=True)
    if agent_id not in agents_db:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    try:
        agent_instance = agents_db.pop(agent_id)
        del agent_instance # Explicitly delete
        print(f"{BColors.OKGREEN}DEBUG: Agent '{BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.OKGREEN}' deleted.{BColors.ENDC}", flush=True)
    except KeyError: # Should be caught by 'not in agents_db' but good for robustness
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found during deletion.")
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Unexpected error deleting agent {agent_id}: {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error deleting agent: {e}")
    return DeletedAgentResponse(deleted_agent_id=agent_id, status="success")
