# File: routers/agent_router.py
import traceback
from datetime import datetime
from typing import Dict, List, Any, Tuple
from uuid import uuid4
import numpy as np

from fastapi import APIRouter, HTTPException
from langchain_core.documents import Document # Required for fetch_memories type hint

from custom_agent import AutonomousGenerativeAgent
from agent_factory import create_new_agent_instance
from schemas import (
    CreateAgentReq, GenerateResponseReq, GenerateReactionResponse, AddMemoryReq,
    FetchMemoriesReq, UpdateStatusReq, AgentListResponse, AgentDetail,
    AgentCreationResponse, AddedMemoryResponse, FetchedMemoriesResponse,
    FetchedMemoriesDocument, AgentSummaryResponse, DeletedAgentResponse
)
from utils import BColors

router = APIRouter(
    prefix="/agents",
    tags=["agents"]
)

agents_db: Dict[str, AutonomousGenerativeAgent] = {}


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


@router.post("", status_code=201, response_model=AgentCreationResponse)
def create_agent_endpoint(req: CreateAgentReq):
    print(f"{BColors.HEADER}DEBUG: /agents POST request received: {req.model_dump_json(exclude_none=True)}{BColors.ENDC}", flush=True)
    aid = req.agent_id or str(uuid4())
    if aid in agents_db:
        raise HTTPException(status_code=400, detail=f"Agent with agent_id '{aid}' already exists.")
    try:
        current_agent_instance = create_new_agent_instance(
            name=req.name, age=req.age, traits=req.traits, status=req.status,
            summary_refresh_seconds=req.summary_refresh_seconds,
            reflection_threshold=req.reflection_threshold, verbose=req.verbose,
            llm_model_name=req.model_name, embedding_model_name=req.embedding_model_name
        )
        agents_db[aid] = current_agent_instance
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        if aid in agents_db: del agents_db[aid]
        raise HTTPException(status_code=500, detail=f"Unexpected server error during agent creation: {e}")

    llm_model_used, embedding_model_used = get_agent_model_details(current_agent_instance)
    return AgentCreationResponse(
        agent_id=aid, name=req.name,
        llm_model_used=llm_model_used, embedding_model_used=embedding_model_used
    )

@router.get("", response_model=AgentListResponse)
def list_agents_endpoint():
    agent_details_list = []
    for agent_id, agent_instance in agents_db.items():
        name, status, llm_model, emb_model = "Unknown", "Unknown", "unknown", "unknown"
        if agent_instance:
            name = agent_instance.name
            status = agent_instance.status
            llm_model, emb_model = get_agent_model_details(agent_instance)
        agent_details_list.append(AgentDetail(
            agent_id=agent_id, name=name, status=status,
            llm_model=llm_model, embedding_model=emb_model
        ))
    return AgentListResponse(agents=agent_details_list)

@router.post("/{agent_id}/update_status", response_model=AgentDetail)
def update_agent_status_endpoint(agent_id: str, req: UpdateStatusReq):
    if agent_id not in agents_db or agents_db[agent_id] is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")
    agent = agents_db[agent_id]
    new_status_stripped = req.new_status.strip()
    if not new_status_stripped:
        raise HTTPException(status_code=400, detail="New status cannot be empty.")
    agent.status = new_status_stripped
    llm_model, emb_model = get_agent_model_details(agent)
    return AgentDetail(
        agent_id=agent_id, name=agent.name, status=agent.status,
        llm_model=llm_model, embedding_model=emb_model
    )

@router.post("/{agent_id}/generate_response", response_model=GenerateReactionResponse)
def generate_response_endpoint(agent_id: str, req: GenerateResponseReq):
    observation = req.prompt.strip()
    print(f"{BColors.HEADER}DEBUG: /generate_response for agent {BColors.BOLD}{agent_id}{BColors.ENDC} with '{observation[:50]}...' (K={req.k or 'default'}){BColors.ENDC}", flush=True)
    if agent_id not in agents_db or agents_db[agent_id] is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")
    if not observation:
         raise HTTPException(status_code=400, detail="Observation cannot be empty.")

    agent: AutonomousGenerativeAgent = agents_db[agent_id]
    original_k = -1
    retriever = agent.memory.memory_retriever
    if hasattr(retriever, 'k'):
         original_k = retriever.k

    api_reaction_type, api_content, observation_was_important_flag = "UNKNOWN", "", False
    try:
        if req.k is not None and req.k > 0 and hasattr(retriever, 'k'):
            retriever.k = req.k
        
        api_reaction_type, api_content, observation_was_important_flag = agent.get_interpreted_reaction(observation, datetime.now())
        print(f"{BColors.OKGREEN}DEBUG: agent.get_interpreted_reaction completed. API Type: '{api_reaction_type}', Important: {observation_was_important_flag}{BColors.ENDC}", flush=True)

    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Error during reaction generation for agent {agent_id}: {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during reaction generation: {e}")
    finally:
        if original_k != -1 and hasattr(retriever, 'k'):
             retriever.k = original_k

    return GenerateReactionResponse(
        agent_name=agent.name, reaction_type=api_reaction_type,
        content=api_content, observation_was_important=observation_was_important_flag
    )

@router.post("/{agent_id}/add_memory", response_model=AddedMemoryResponse)
def add_memory_endpoint(agent_id: str, req: AddMemoryReq):
    if agent_id not in agents_db or agents_db[agent_id] is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")
    text_to_add = req.text_to_memorize.strip()
    if not text_to_add:
        raise HTTPException(status_code=400, detail="Memory text may not be empty.")
    agent = agents_db[agent_id]
    try:
        if not agent.memory:
             raise HTTPException(status_code=500, detail=f"Agent {agent_id} memory not initialized.")
        agent.memory.add_memory(text_to_add, now=datetime.now())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding memory: {e}")
    return AddedMemoryResponse(status="success", added_memory=text_to_add)

@router.post("/{agent_id}/fetch_memories", response_model=FetchedMemoriesResponse)
def fetch_memories_endpoint(agent_id: str, req: FetchMemoriesReq):
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
    
    response_payload_docs: List[FetchedMemoriesDocument] = []
    try:
        requested_k = retriever.k # Default to retriever's k
        if req.k is not None and req.k > 0 and hasattr(retriever, 'k'):
            requested_k = req.k
            retriever.k = requested_k

        docs_and_scores: List[Tuple[Document, float]] = []
        if (hasattr(retriever, "vectorstore") and retriever.vectorstore and
            hasattr(retriever.vectorstore, "similarity_search_with_relevance_scores")):
            docs_and_scores = retriever.vectorstore.similarity_search_with_relevance_scores(observation, k=requested_k)
        else:
            fetched_docs_only: List[Document] = agent.memory.fetch_memories(observation, now=datetime.now())
            docs_and_scores = [(doc, 0.0) for doc in fetched_docs_only]

        for doc, score in docs_and_scores:
            serializable_metadata = {
                k: v.isoformat() if isinstance(v, datetime) else
                   v.tolist() if isinstance(v, np.ndarray) else
                   float(v) if isinstance(v, (np.float32, np.float64)) else
                   int(v) if isinstance(v, (np.int32, np.int64)) else v
                for k, v in doc.metadata.items()
            }
            response_payload_docs.append(FetchedMemoriesDocument(
                content=doc.page_content, metadata=serializable_metadata, relevance_score=score
            ))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching memories: {e}")
    finally:
        if original_k != -1 and hasattr(retriever, 'k'):
             retriever.k = original_k
    return FetchedMemoriesResponse(memories=response_payload_docs)

@router.get("/{agent_id}/summary", response_model=AgentSummaryResponse)
def get_summary_endpoint(agent_id: str):
    if agent_id not in agents_db or agents_db[agent_id] is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")
    agent = agents_db[agent_id]
    summary_text = "Error generating summary."
    try:
        summary_text = agent.get_summary(force_refresh=True)
    except Exception as e:
        summary_text = f"Error generating summary: {e}"
    return AgentSummaryResponse(agent_id=agent_id, summary=summary_text)

@router.delete("/{agent_id}", response_model=DeletedAgentResponse)
def delete_agent_endpoint(agent_id: str):
    if agent_id not in agents_db:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    try:
        agents_db.pop(agent_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error deleting agent: {e}")
    return DeletedAgentResponse(deleted_agent_id=agent_id, status="success")
