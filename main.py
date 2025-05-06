# main.py

from datetime import datetime
from typing import Optional

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

app = FastAPI(title="Generative-Agent API")

# ---------- LLM & embeddings ----------
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
emb = OpenAIEmbeddings()

# ---------- Create an empty FAISS store ----------
# Probe embedding dimension with one call
_embedding_probe = emb.embed_query("probe vector dimension")
embedding_size = len(_embedding_probe)

index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(
    embedding_function=emb,
    index=index,
    docstore=InMemoryDocstore({}),
    index_to_docstore_id={},
)

retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    k=15,
    decay_rate=0.01,
)

memory = GenerativeAgentMemory(
    llm=llm,
    memory_retriever=retriever,
    reflection_threshold=8,
)

agent = GenerativeAgent(
    name="Ava",
    age=28,
    traits="curious, friendly, loves classical music",
    status="waiting to greet visitors on the web",
    memory=memory,
    llm=llm,
)

# ---------- Pydantic models ----------
class TalkReq(BaseModel):
    prompt: str

class ObserveReq(BaseModel):
    observation: str

# ---------- Endpoints ----------------
@app.get("/")
async def root():
    return {"message": "Your Generative-Agent is alive!"}

@app.post("/talk")
async def talk(req: TalkReq):
    text = req.prompt.strip()
    if not text:
        raise HTTPException(400, "Prompt may not be empty")

    # NOTE: generate_dialogue_response returns (should_write_memory: bool, response: str)
    should_write, response = agent.generate_dialogue_response(text, datetime.now())

    # Optionally add the user’s prompt to memory if the agent thinks it’s important
    if should_write:
        agent.memory.add_memory(text)

    return {"agent": agent.name, "response": response}

@app.post("/observe")
async def observe(req: ObserveReq):
    obs = req.observation.strip()
    if not obs:
        raise HTTPException(400, "Observation may not be empty")
    agent.memory.add_memory(obs)
    return {"stored": obs}

@app.get("/summary")
async def summary():
    return {"summary": agent.get_summary(force_refresh=True)}

# (Optional) preserve your existing demo endpoint
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
