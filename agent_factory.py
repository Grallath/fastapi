# File: agent_factory.py
import traceback
from typing import Optional

import faiss
from fastapi import HTTPException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_experimental.generative_agents import GenerativeAgentMemory
from langchain_community.vectorstores.utils import DistanceStrategy

from custom_agent import AutonomousGenerativeAgent
from utils import BColors
from config import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL


def create_new_agent_instance(
    name: str,
    age: int,
    traits: str,
    status: str,
    summary_refresh_seconds: int,
    reflection_threshold: int,
    verbose: bool,
    llm_model_name: Optional[str] = None,
    embedding_model_name: Optional[str] = None
) -> AutonomousGenerativeAgent:
    print(f"{BColors.OKBLUE}DEBUG: create_new_agent_instance called for agent '{BColors.BOLD}{name}{BColors.ENDC}{BColors.OKBLUE}'{BColors.ENDC}", flush=True)

    effective_llm_model = llm_model_name if llm_model_name and llm_model_name.strip() else DEFAULT_CHAT_MODEL
    try:
        agent_llm = ChatOpenAI(model_name=effective_llm_model, temperature=0.7)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed to initialize LLM for agent '{name}' with model '{effective_llm_model}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to initialize LLM with model '{effective_llm_model}': {e}")

    effective_embedding_model = embedding_model_name if embedding_model_name and embedding_model_name.strip() else DEFAULT_EMBEDDING_MODEL
    dim = 0
    try:
        agent_embeddings = OpenAIEmbeddings(model=effective_embedding_model)
        probe_for_dim = agent_embeddings.embed_query("get_dim_probe_for_agent")
        dim = len(probe_for_dim)
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed to initialize or test OpenAIEmbeddings for agent '{name}' with model '{effective_embedding_model}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to initialize/test embeddings with model '{effective_embedding_model}': {e}")

    print(f"{BColors.DIM}DEBUG: Setting up FAISS index for agent '{name}' (dim: {dim}). Using Inner Product (for Cosine).{BColors.ENDC}", flush=True)
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
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed during FAISS setup for agent '{name}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during FAISS setup: {e}")

    print(f"{BColors.DIM}DEBUG: Setting up GenerativeAgentMemory for agent '{name}'...{BColors.ENDC}", flush=True)
    try:
        actual_reflect_for_memory = float(reflection_threshold) if reflection_threshold > 0 else None
        memory_instance = GenerativeAgentMemory(
            llm=agent_llm,
            memory_retriever=retriever,
            reflection_threshold=actual_reflect_for_memory,
            verbose=verbose,
        )
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed during GenerativeAgentMemory setup for agent '{name}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during GenerativeAgentMemory setup: {e}")

    print(f"{BColors.DIM}DEBUG: Initializing AutonomousGenerativeAgent '{name}'...{BColors.ENDC}", flush=True)
    try:
        agent = AutonomousGenerativeAgent(
            name=name,
            age=age,
            traits=traits,
            status=status,
            memory=memory_instance,
            llm=agent_llm,
            summary_refresh_seconds=summary_refresh_seconds,
            verbose=verbose,
        )
        print(f"{BColors.OKGREEN}DEBUG: AutonomousGenerativeAgent '{name}' initialized successfully.{BColors.ENDC}", flush=True)
        return agent
    except Exception as e:
        print(f"{BColors.FAIL}ERROR_STACKTRACE: Failed during AutonomousGenerativeAgent initialization for agent '{name}': {e}{BColors.ENDC}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed during AutonomousGenerativeAgent initialization: {e}")
