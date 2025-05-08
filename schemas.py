# File: schemas.py
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class CreateAgentReq(BaseModel):
    name: str
    age: int
    traits: str
    status: str
    agent_id: Optional[str] = None
    summary_refresh_seconds: int = Field(default=3600, ge=0)
    reflection_threshold: int = Field(default=0, ge=0)
    verbose: bool = False
    model_name: Optional[str] = None
    embedding_model_name: Optional[str] = None

class GenerateResponseReq(BaseModel):
    prompt: str
    k: Optional[int] = Field(default=None, gt=0)

class GenerateReactionResponse(BaseModel):
    agent_name: str
    reaction_type: str # SAY, THINK, DO, IGNORE, UNKNOWN
    content: str
    observation_was_important: bool

class AddMemoryReq(BaseModel):
    text_to_memorize: str

class FetchMemoriesReq(BaseModel):
    observation: str
    k: Optional[int] = Field(default=None, gt=0)

class UpdateStatusReq(BaseModel):
    new_status: str

class AgentDetail(BaseModel):
    agent_id: str
    name: str
    status: str
    llm_model: str
    embedding_model: str

class AgentListResponse(BaseModel):
    agents: List[AgentDetail]

class AgentCreationResponse(BaseModel):
    agent_id: str
    name: str
    llm_model_used: str
    embedding_model_used: str

class SimpleStatusResponse(BaseModel):
    status: str
    message: Optional[str] = None

class AddedMemoryResponse(BaseModel):
    status: str
    added_memory: str

class FetchedMemoriesDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]
    relevance_score: float

class FetchedMemoriesResponse(BaseModel):
    memories: List[FetchedMemoriesDocument]

class AgentSummaryResponse(BaseModel):
    agent_id: str
    summary: str

class DeletedAgentResponse(BaseModel):
    deleted_agent_id: str
    status: str
