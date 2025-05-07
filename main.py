# File: main.py
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
from langchain_experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain.chains.llm import LLMChain # Needed for the chain in memory and agent

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

    IMPORTANCE_HIGH = OKGREEN
    IMPORTANCE_MEDIUM = WARNING
    IMPORTANCE_LOW = FAIL

    METADATA_KEY = OKCYAN
    METADATA_VALUE = OKBLUE
    CONTENT_COLOR = ENDC
    SEPARATOR = DIM

app = FastAPI(title="Generative-Agent API (Refactored)")

print(f"{BColors.OKGREEN}DEBUG: FastAPI application starting up...{BColors.ENDC}", flush=True)

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# --- Custom Generative Agent Memory ---
class CustomGenerativeAgentMemory(GenerativeAgentMemory):
    def __init__(
        self,
        *,
        llm: BaseLanguageModel,
        memory_retriever: TimeWeightedVectorStoreRetriever,
        verbose: bool = False,
        reflection_threshold: Optional[float] = None,
        **kwargs: Any,
    ):
        print(f"{BColors.HEADER}{BColors.BOLD}DEBUG_INIT: CustomGenerativeAgentMemory IS BEING INITIALIZED.{BColors.ENDC}", flush=True)
        super().__init__(
            llm=llm,
            memory_retriever=memory_retriever,
            verbose=verbose,
            reflection_threshold=reflection_threshold,
            **kwargs
        )
        self.verbose = verbose

    def _get_summary_of_relevant_context(
        self,
        agent_name_alpha: str,
        agent_name_beta: str,
        observation: str,
        now: Optional[datetime] = None,
    ) -> str:
        print(f"{BColors.OKBLUE}{BColors.BOLD}DEBUG_TRACE: CustomGenerativeAgentMemory._get_summary_of_relevant_context IS BEING CALLED for {agent_name_alpha} and {agent_name_beta}{BColors.ENDC}", flush=True)
        print(f"{BColors.OKBLUE}DEBUG_TRACE: CustomMemory - Observation for _get_summary_of_relevant_context: {observation[:200]}...{BColors.ENDC}", flush=True)

        relationship_query = f"{agent_name_alpha}'s relationship with {agent_name_beta}"
        relevant_memories = self.fetch_memories(relationship_query, now=now)
        relevant_memories_string = self.aggregate_memories(relevant_memories, prefix=False)
        if not relevant_memories_string.strip():
            relevant_memories_string = "No specific memories found regarding this entity."

        prompt_template = PromptTemplate.from_template(
            "You are an AI assistant analyzing the relationship between two entities based on a detailed observation and existing memories.\n\n"
            "Current Detailed Observation of the Scene (focus on descriptions of entities involved):\n"
            "\"\"\"\n{observation_text}\n\"\"\"\n\n"
            "Agent Alpha: {agent_name_alpha}\n"
            "Agent Beta (the other primary entity observed in the scene): {agent_name_beta}\n\n"
            "Context from {agent_name_alpha}'s memory regarding any prior relationship, knowledge, or thoughts about {agent_name_beta}:\n"
            "\"\"\"\n{relevant_memories}\n\"\"\"\n"
            "(If these memories are empty or non-specific about {agent_name_beta}, it implies {agent_name_alpha} likely has no direct prior memory of {agent_name_beta} as described in the observation.)\n\n"
            "Considering the \"Current Detailed Observation\" first and foremost, and then referencing {agent_name_alpha}'s memories, "
            "what is the most likely current relationship between {agent_name_alpha} and {agent_name_beta} in the context of this specific scene? \n"
            "Specifically:\n"
            "1. Are they the same entity? (Compare their descriptions in the observation if both are detailed there, or {agent_name_alpha}'s known self-description vs. {agent_name_beta}'s observed description). List key differences if they are not the same.\n"
            "2. If different entities, are they known to each other from the past, or are they strangers meeting now?\n\n"
            "Provide a concise summary of this relationship analysis:\n"
            "Relationship Summary:"
        )
        formatted_prompt_string = prompt_template.format(
            agent_name_alpha=agent_name_alpha,
            agent_name_beta=agent_name_beta,
            observation_text=observation,
            relevant_memories=relevant_memories_string
        )
        print(f"{BColors.OKBLUE}{BColors.BOLD}DEBUG_TRACE: CustomMemory - FULLY FORMATTED PROMPT for _get_summary_of_relevant_context chain:\n{formatted_prompt_string}{BColors.ENDC}", flush=True)
        
        result = self.chain.run(prompt=formatted_prompt_string, callbacks=self.callbacks)
        print(f"{BColors.OKBLUE}{BColors.BOLD}DEBUG_TRACE: CustomMemory - RAW RESULT from _get_summary_of_relevant_context chain:\n{result}\n{BColors.ENDC}", flush=True)
        return result
# --- End of Custom Generative Agent Memory ---


# --- Custom Generative Agent ---
class CustomGenerativeAgent(GenerativeAgent):
    def __init__(self, *, memory: CustomGenerativeAgentMemory, name: str, **kwargs: Any): # Added name for debug print
        print(f"{BColors.HEADER}{BColors.BOLD}DEBUG_INIT: CustomGenerativeAgent IS BEING INITIALIZED for agent: {name}{BColors.ENDC}", flush=True)
        super().__init__(memory=memory, name=name, **kwargs)

    def _get_dialogue_observation_relevance(
        self, observation: str, other_agent_name: str, now: Optional[datetime] = None
    ) -> str:
        print(f"{BColors.FAIL}{BColors.BOLD}DEBUG_TRACE: CustomGenerativeAgent._get_dialogue_observation_relevance IS BEING CALLED for {self.name} and {other_agent_name}{BColors.ENDC}", flush=True)
        print(f"{BColors.FAIL}DEBUG_TRACE: CustomAgent - Observation for dialogue relevance: {observation[:200]}...{BColors.ENDC}", flush=True)

        agent_alpha_self_description = self.status # Using current agent status

        prompt_template = PromptTemplate.from_template(
            "Task: Determine the relationship between Agent Alpha and an Observed Entity based *only* on the provided \"Full Current Observation\" and Alpha's known self-description. Then, check if any existing memories contradict this initial assessment.\n\n"
            "Agent Alpha: {agent_name_alpha}\n"
            "Agent Alpha's Known Self-Description (how Alpha currently presents himself to the world or his current state):\n"
            "\"\"\"\n{agent_alpha_self_description}\n\"\"\"\n\n"
            "Observed Entity (referred to as Agent Beta in this analysis): {other_agent_name}\n\n"
            "Full Current Observation of the Scene (This is the primary source of truth for visual descriptions and immediate interactions):\n"
            "\"\"\"\n{observation}\n\"\"\"\n\n"
            "Instructions for Analysis:\n"
            "Step 1: Visual Comparison based SOLELY on the \"Full Current Observation\" and Alpha's Self-Description.\n"
            "   - Describe Agent Alpha's appearance as per his \"Known Self-Description\".\n"
            "   - Describe the \"Observed Entity\" ({other_agent_name})'s appearance and actions *as detailed in the* \"Full Current Observation\".\n"
            "   - Critical Question: Based *only* on these visual descriptions, are Agent Alpha and the Observed Entity the same person? Answer Yes or No.\n"
            "   - If No, list at least three key visual/descriptive differences between Agent Alpha (from self-description) and the Observed Entity (from observation).\n"
            "   - If they are different entities, does the \"Full Current Observation\" provide any clues about whether they know each other or if this is a first encounter? (e.g., signs of recognition, specific reactions).\n\n"
            "Step 2: Consideration of Agent Alpha's Memories about {other_agent_name}.\n"
            "   Memories of {agent_name_alpha} specifically about {other_agent_name}:\n"
            "   \"\"\"\n{relevant_memories_string}\n\"\"\"\n"
            "   - Do these memories (if any) confirm or contradict the assessment from Step 1? Be specific. If memories are general or not about {other_agent_name}, state that.\n\n"
            "Step 3: Final Concise Relationship Summary for Dialogue Context.\n"
            "   Based *primarily* on the visual evidence in the \"Full Current Observation\" (Step 1), and then secondarily on memories (Step 2), provide a very concise summary of the current relationship. Examples: 'They are strangers; Alpha is observing a new arrival.', 'They are the same person seen from a different perspective.', 'They are old acquaintances meeting again.', 'The observation does not describe Alpha, only a new entity.'\n\n"
            "Concise Relationship and Context Summary:"
        )
        
        agent_S_memories = self.memory.fetch_memories(
            f"{self.name}'s relationship with {other_agent_name}", now=now
        )
        relevant_memories_string = "\n".join(
            [memory.page_content for memory in agent_S_memories]
        )
        if not relevant_memories_string.strip():
            relevant_memories_string = "No specific memories found regarding this entity."

        formatted_prompt_string = prompt_template.format(
            agent_name_alpha=self.name,
            agent_alpha_self_description=agent_alpha_self_description,
            other_agent_name=other_agent_name,
            observation=observation,
            relevant_memories_string=relevant_memories_string
        )

        print(f"{BColors.WARNING}{BColors.BOLD}DEBUG_TRACE: CustomAgent - FULLY FORMATTED PROMPT for _get_dialogue_observation_relevance chain:\n{formatted_prompt_string}{BColors.ENDC}", flush=True)

        result = self.chain.run(prompt=formatted_prompt_string, callbacks=self.callbacks)
        
        print(f"{BColors.OKGREEN}{BColors.BOLD}DEBUG_TRACE: CustomAgent - RAW RESULT from _get_dialogue_observation_relevance chain:\n{result}\n{BColors.ENDC}", flush=True)
        return result.strip()
# --- End of Custom Generative Agent ---


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
    print(f"{BColors.OKBLUE}DEBUG: _new_agent_instance called for agent '{BColors.BOLD}{name}{BColors.ENDC}{BColors.OKBLUE}'{BColors.ENDC}", flush=True)
    # ... (rest of the LLM and Embedding setup remains the same) ...
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

    print(f"{BColors.DIM}DEBUG: Setting up FAISS index for agent '{name}' (dim: {dim}). Using Inner Product.{BColors.ENDC}", flush=True)
    try:
        index = faiss.IndexFlatIP(dim)
        vectorstore = FAISS(
            embedding_function=agent_embeddings,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
            normalize_L2=True,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
        )
        retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=vectorstore, k=15, decay_rate=0.01
        )
        print(f"{BColors.OKGREEN}DEBUG: FAISS setup complete for agent '{name}'. (Index: IP, Normalize: True, Strategy: MAX_INNER_PRODUCT){BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed during FAISS setup for agent '{name}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during FAISS setup: {e}")

    print(f"{BColors.DIM}DEBUG: Setting up CustomGenerativeAgentMemory for agent '{name}'...{BColors.ENDC}", flush=True)
    try:
        actual_reflect_for_memory = reflection_threshold if reflection_threshold > 0 else None
        print(f"{BColors.DIM}DEBUG: actual_reflect_for_memory (for CustomGenerativeAgentMemory) will be: {actual_reflect_for_memory}{BColors.ENDC}", flush=True)

        memory_instance = CustomGenerativeAgentMemory(
            llm=agent_llm,
            memory_retriever=retriever,
            reflection_threshold=actual_reflect_for_memory,
            verbose=verbose,
        )
        print(f"{BColors.OKGREEN}DEBUG: CustomGenerativeAgentMemory setup complete for agent '{name}'.{BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed during CustomGenerativeAgentMemory setup for agent '{name}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during CustomGenerativeAgentMemory setup: {e}")

    print(f"{BColors.DIM}DEBUG: Initializing CustomGenerativeAgent '{name}'...{BColors.ENDC}", flush=True)
    try:
        actual_refresh_for_agent = summary_refresh_seconds
        print(f"{BColors.DIM}DEBUG: actual_refresh_for_agent (for CustomGenerativeAgent) will be: {actual_refresh_for_agent}{BColors.ENDC}", flush=True)

        # Pass all necessary kwargs from GenerativeAgent's __init__ signature
        # Check GenerativeAgent source for full list if more are used/needed
        agent = CustomGenerativeAgent(
            name=name,
            age=age,
            traits=traits,
            status=status,
            memory=memory_instance,
            llm=agent_llm,
            summary_refresh_seconds=actual_refresh_for_agent,
            verbose=verbose,
            # Default values for other GenerativeAgent __init__ params if not passed:
            # dialogue_llm=None, # Will default to llm
            # max_tokens_for_summary=500, # Default in GenerativeAgent
            # dialogue_prompt=None, # Will use default
            # reflection_llm=None, # Will default to llm
            # Chain_healing is specific to some chains, not a direct GenerativeAgent param.
            # add_all_details_to_creations = False # default
            # aggregate_reflection_related_memories = True #default
            # max_reflection_thoughts = 50 # default
            # relevant_memories_type = None # default
            # k_relevant_memories_for_summary = 200 # default
        )
        print(f"{BColors.OKGREEN}DEBUG: CustomGenerativeAgent '{name}' initialized successfully.{BColors.ENDC}", flush=True)
        return agent
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed during CustomGenerativeAgent initialization for agent '{name}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during CustomGenerativeAgent initialization: {e}")


agents: Dict[str, GenerativeAgent] = {}

# Pydantic Models
# ... (No changes to Pydantic models: CreateAgentReq, GenerateResponseReq, etc.)
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

class UpdateStatusReq(BaseModel):
    new_status: str


# Endpoints
# ... (No changes to endpoint implementations: /agents, /generate_response, etc.)
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
        print(f"{BColors.OKGREEN}DEBUG: Agent '{BColors.BOLD}{aid}{BColors.ENDC}{BColors.OKGREEN}' (name: '{req.name}') created and stored.{BColors.ENDC}", flush=True)
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
        agent.status = req.new_status.strip()
        print(f"{BColors.OKGREEN}SUCCESS: Agent '{BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.OKGREEN}' status updated to: '{agent.status}'{BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Error updating status for agent '{BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.FAIL}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error updating agent status: {e}")

    print(f"{BColors.HEADER}<<< Completing Update Status Request for Agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER} >>>{BColors.ENDC}\n", flush=True)
    return {"agent_id": agent_id, "status": agent.status}


@app.post("/agents/{agent_id}/generate_response")
def generate_response(agent_id: str, req: GenerateResponseReq):
    print(f"{BColors.HEADER}DEBUG: /generate_response for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER} with prompt: '{req.prompt[:50]}...' (K={req.k or 'default'}){BColors.ENDC}", flush=True)
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

        print(f"{BColors.DIM}Agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.DIM} current status before generation: '{agent.status}'{BColors.ENDC}", flush=True)
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
    print(f"{BColors.HEADER}DEBUG: /add_memory for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER} with text: '{req.text_to_memorize[:50]}...'{BColors.ENDC}", flush=True)
    if agent_id not in agents or agents[agent_id] is None:
        print(f"{BColors.FAIL}ERROR: Agent '{agent_id}' not found or is None for add_memory.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")

    text_to_add = req.text_to_memorize.strip()
    if not text_to_add:
        print(f"{BColors.WARNING}WARN: Memory text may not be empty.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=400, detail="Memory text may not be empty.")

    try:
        agents[agent_id].memory.add_memory(text_to_add, now=datetime.now())
        print(f"{BColors.OKGREEN}DEBUG: Memory added successfully for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.OKGREEN}.{BColors.ENDC}", flush=True)
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

    retrieved_docs_for_response_payload: List[Dict[str, Any]] = []

    try:
        if req.k is not None and req.k > 0:
            if hasattr(retriever, 'k'):
                retriever.k = req.k
            else:
                print(f"{BColors.WARNING}WARN: Cannot set k for agent {agent_id}; retriever does not have 'k' attribute.{BColors.ENDC}", flush=True)

        docs_and_scores: List[Tuple[Document, float]]
        if hasattr(agent.memory.memory_retriever, "vectorstore") and \
           hasattr(agent.memory.memory_retriever.vectorstore, "similarity_search_with_relevance_scores"):
            print(f"{BColors.DIM}DEBUG: Fetching memories using similarity_search_with_relevance_scores for agent {agent_id}.{BColors.ENDC}", flush=True)
            docs_and_scores = agent.memory.memory_retriever.vectorstore.similarity_search_with_relevance_scores(
                observation,
                k=retriever.k if hasattr(retriever, 'k') else 15,
            )
        else:
            print(f"{BColors.WARNING}WARN: Using agent.memory.fetch_memories() for agent {agent_id}, scores might not be raw relevance scores.{BColors.ENDC}", flush=True)
            retrieved_docs_only: List[Document] = agent.memory.fetch_memories(observation, now=datetime.now())
            docs_and_scores = [(doc, doc.metadata.get('relevance_score', 0.0)) for doc in retrieved_docs_only]

        for doc, score in docs_and_scores:
            retrieved_docs_for_response_payload.append({
                "content": doc.page_content,
                "relevance_score": score
            })

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

                created_at_str = created_at_raw.strftime("%Y-%m-%d %H:%M:%S") if hasattr(created_at_raw, 'strftime') else str(created_at_raw)
                last_accessed_at_str = last_accessed_at_raw.strftime("%Y-%m-%d %H:%M:%S") if hasattr(last_accessed_at_raw, 'strftime') else str(last_accessed_at_raw)

                importance_color = BColors.IMPORTANCE_LOW
                if importance >= 0.7: importance_color = BColors.IMPORTANCE_HIGH
                elif importance >= 0.4: importance_color = BColors.IMPORTANCE_MEDIUM

                relevance_color = BColors.IMPORTANCE_LOW
                if score >= 0.7: relevance_color = BColors.IMPORTANCE_HIGH
                elif score >= 0.4: relevance_color = BColors.IMPORTANCE_MEDIUM

                print(f"{BColors.SEPARATOR}{'-'*70}{BColors.ENDC}", flush=True)
                print(f"{BColors.BOLD}Memory #{i+1}:{BColors.ENDC}", flush=True)
                print(f"  {BColors.METADATA_KEY}Relevance Score:{BColors.ENDC} {relevance_color}{score:.4f}{BColors.ENDC}", flush=True)
                print(f"  {BColors.METADATA_KEY}Static Importance:{BColors.ENDC} {importance_color}{importance:.3f}{BColors.ENDC}", flush=True)
                print(f"  {BColors.METADATA_KEY}Content:{BColors.ENDC}\n{BColors.CONTENT_COLOR}    \"{doc.page_content.strip()}\"{BColors.ENDC}", flush=True)
                print(f"  {BColors.DIM}{BColors.METADATA_KEY}Details:{BColors.ENDC}")
                print(f"    {BColors.METADATA_KEY}Created At:{BColors.ENDC} {BColors.METADATA_VALUE}{created_at_str}{BColors.ENDC}", flush=True)
                print(f"    {BColors.METADATA_KEY}Last Accessed:{BColors.ENDC} {BColors.METADATA_VALUE}{last_accessed_at_str}{BColors.ENDC}", flush=True)
                print(f"    {BColors.METADATA_KEY}Buffer Idx:{BColors.ENDC} {BColors.METADATA_VALUE}{buffer_idx}{BColors.ENDC}", flush=True)
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

    print(f"{BColors.HEADER}<<< Completing Fetch Memories Request for Agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER} >>>{BColors.ENDC}\n", flush=True)
    return {"memories": retrieved_docs_for_response_payload }


@app.get("/agents/{agent_id}/summary")
def get_summary(agent_id: str):
    print(f"{BColors.HEADER}DEBUG: /summary for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER}", flush=True)
    if agent_id not in agents or agents[agent_id] is None:
        print(f"{BColors.FAIL}ERROR: Agent '{agent_id}' not found or is None for get_summary.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found or invalid.")

    summary_text = "Error generating summary."
    try:
        summary_text = agents[agent_id].get_summary(force_refresh=True)
        print(f"{BColors.OKGREEN}DEBUG: Summary generated successfully for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.OKGREEN}.{BColors.ENDC}", flush=True)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Error generating summary for agent {agent_id}: {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating summary: {e}")

    return {"agent_id": agent_id, "summary": summary_text}

@app.delete("/agents/{agent_id}")
def delete_agent(agent_id: str):
    print(f"{BColors.HEADER}DEBUG: /delete_agent for agent {BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.HEADER}", flush=True)
    if agent_id not in agents:
        print(f"{BColors.FAIL}ERROR: Agent '{agent_id}' not found for delete_agent.{BColors.ENDC}", flush=True)
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")

    del agents[agent_id]
    print(f"{BColors.OKGREEN}DEBUG: Agent '{BColors.BOLD}{agent_id}{BColors.ENDC}{BColors.OKGREEN}' deleted successfully.{BColors.ENDC}", flush=True)
    return {"deleted_agent_id": agent_id, "status": "success"}

print(f"{BColors.OKGREEN}DEBUG: FastAPI application finished loading.{BColors.ENDC}", flush=True)
