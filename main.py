# File: main.py
from fastapi import FastAPI

from utils import BColors # Import from new utils
from config import OPENAI_API_KEY_SET # Import from new config
from routers import agent_router # Import the new router
# The global agents_db is now in routers.agent_router

app = FastAPI(title="Autonomous Generative-Agent API v2") # Updated title

print(f"{BColors.OKGREEN}DEBUG: FastAPI application starting up... (Using Autonomous Agents - Refactored){BColors.ENDC}", flush=True)

# --- API Key Check ---
if not OPENAI_API_KEY_SET:
    print(f"{BColors.FAIL}CRITICAL_WARNING: OPENAI_API_KEY environment variable is NOT SET. OpenAI calls likely fail.{BColors.ENDC}", flush=True)
else:
    print(f"{BColors.OKGREEN}DEBUG: OPENAI_API_KEY environment variable is detected.{BColors.ENDC}", flush=True)


# --- Health Check ---
@app.get("/")
async def health_check():
    print(f"{BColors.OKGREEN}DEBUG: Health check '/' endpoint hit.{BColors.ENDC}", flush=True)
    return {"status": "ok", "message": "Autonomous Agent API is running."}

# --- Include Routers ---
app.include_router(agent_router.router) # Add the agent routes

# --- Global Agent Storage (managed within agent_router.py now) ---
# The `agents_db` dictionary is now managed within `routers/agent_router.py`
# This keeps it closer to the operations that use it.
# If you need to access it from `main.py` for other reasons (e.g. startup/shutdown events),
# you could import it: `from routers.agent_router import agents_db`
# Or pass `app.state.agents_db` around if you prefer that pattern.

print(f"{BColors.OKGREEN}DEBUG: FastAPI application finished loading. (Using Autonomous Agents - Refactored){BColors.ENDC}", flush=True)

# To run: uvicorn main:app --reload
# Ensure Python's import system can find your modules (e.g., routers.agent_router)
# If running from the project root, and `routers` is a subdirectory, it should work.
