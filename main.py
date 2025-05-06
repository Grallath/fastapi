from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# NEW import paths (no more deprecation warnings)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)
from langchain.retrievers import TimeWeightedVectorStoreRetriever

app = FastAPI(title="Generative‑Agent API")

# ---------- LLM & embeddings ----------
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
emb = OpenAIEmbeddings()

# ---------- Vector store --------------
# ① Start with an empty store; we’ll add memories on the fly
vectorstore = FAISS.from_texts([], emb)        # ✅ no attribute error
retriever   = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore, k=15, decay_rate=0.01
)

memory = GenerativeAgentMemory(
    llm=llm,
    memory_retriever=retriever,
    reflection_threshold=8
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
    return {"message": "Your Generative‑Agent is alive!"}

@app.post("/talk")
async def talk(req: TalkReq):
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt may not be empty")
    reply = agent.generate_dialog_response(req.prompt, datetime.now())
    return {"agent": agent.name, "response": reply}

@app.post("/observe")
async def observe(req: ObserveReq):
    if not req.observation.strip():
        raise HTTPException(400, "Observation may not be empty")
    agent.memory.add_memory(req.observation)
    return {"stored": req.observation}

@app.get("/summary")
async def summary():
    return {"summary": agent.get_summary(force_refresh=True)}

# (optional) keep your demo endpoint
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
