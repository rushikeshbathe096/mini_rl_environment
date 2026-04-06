import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from environment import HallucinationEnvironment
from models import HallucinationAction

app = FastAPI()
env = HallucinationEnvironment()

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"

class StepRequest(BaseModel):
    action: HallucinationAction

@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    obs = env.reset(task_id=body.task_id)
    return {"observation": obs.model_dump(), "reward": None, "done": False}

@app.post("/step")
def step(body: StepRequest):
    obs = env.step(body.action)
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}

@app.get("/state")
def state():
    return env.state.model_dump()

@app.get("/health")
def health():
    return {"status": "ok"}

def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=False
    )

if __name__ == "__main__":
    main()