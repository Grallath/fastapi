# main.py

from datetime import datetime
from typing import Optional, Dict
from uuid import uuid4
from math import inf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import faiss
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)

app = FastAPI(title="Generative-Agent API (Dynamic)")

@app.get("/")
async def health_check():
    return {"status": "ok"}

# ——— shared LLM & embeddings ——————————————————

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
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
    # Probe embedding size
    probe = emb.embed_query("probe")
    dim = len(probe)

    # Build an empty FAISS index
    index = faiss.IndexFlatL2(dim)
    vectorstore = FAISS(
        embedding_function=emb,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )

    # Time‐weighted retriever
    retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, k=15, decay_rate=0.01
    )

    # reflection_threshold: 0 → never; else as given
    actual_reflect = reflection_threshold if reflection_threshold > 0 else inf

    # summary_refresh_seconds: 0 → never; else as given
    actual_refresh = summary_refresh_seconds if summary_refresh_seconds > 0 else inf

    memory = GenerativeAgentMemory(
        llm=llm,
        memory_retriever=retriever,
        reflection_threshold=actual_reflect,
    )

    agent = GenerativeAgent(
        name=name,
        age=age,
        traits=traits,
        status=status,
        memory=memory,
        llm=llm,
        summary_refresh_seconds=actual_refresh,
        verbose=verbose,
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
    summary_refresh_seconds: int = 0
    reflection_threshold: int = 0   # 0 → never; 1 → after each obs; etc.
    verbose: bool = False

class TalkReq(BaseModel):
    prompt: str
    k: Optional[int] = None            # override top-k memories
    write_memory: Optional[bool] = True  # whether to add prompt to memory

class ObserveReq(BaseModel):
    observation: str

# ——— Endpoints ———————————————————————————————————————

@app.post("/agents")
def create_agent(req: CreateAgentReq):
    aid = req.agent_id or str(uuid4())
    if aid in agents:
        raise HTTPException(400, f"agent_id '{aid}' already exists")
    agents[aid] = _new_agent_instance(
        name=req.name,
        age=req.age,
        traits=req.traits,
        status=req.status,
        summary_refresh_seconds=req.summary_refresh_seconds,
        reflection_threshold=req.reflection_threshold,
        verbose=req.verbose,
    )
    return {"agent_id": aid}

@app.get("/agents")
def list_agents():
    return {"agents": list(agents.keys())}

@app.post("/agents/{agent_id}/talk")
def talk(agent_id: str, req: TalkReq):
    if agent_id not in agents:
        raise HTTPException(404, f"No such agent '{agent_id}'")
    text = req.prompt.strip()
    if not text:
        raise HTTPException(400, "Prompt may not be empty")
    agent = agents[agent_id]

    # override top-k if provided
    if req.k is not None:
        agent.memory.memory_retriever.k = req.k

    should_write, response = agent.generate_dialogue_response(text, datetime.now())

    # only write into memory if desired
    if req.write_memory and should_write:
        agent.memory.add_memory(text)

    return {"agent": agent.name, "response": response}

@app.post("/agents/{agent_id}/observe")
def observe(agent_id: str, req: ObserveReq):
    if agent_id not in agents:
        raise HTTPException(404, f"No such agent '{agent_id}'")
    obs = req.observation.strip()
    if not obs:
        raise HTTPException(400, "Observation may not be empty")
    agents[agent_id].memory.add_memory(obs)
    return {"stored": obs}

@app.get("/agents/{agent_id}/summary")
def summary(agent_id: str):
    if agent_id not in agents:
        raise HTTPException(404, f"No such agent '{agent_id}'")
    text = agents[agent_id].get_summary(force_refresh=True)
    return {"agent_id": agent_id, "summary": text}

@app.delete("/agents/{agent_id}")
def delete_agent(agent_id: str):
    if agent_id not in agents:
        raise HTTPException(404, f"No such agent '{agent_id}'")
    del agents[agent_id]
    return {"deleted": agent_id}
