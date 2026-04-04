# server/app.py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core.env_server import create_web_interface_app
from server.environment import HallucinationEnvironment
from models import HallucinationAction, HallucinationObservation

app = create_web_interface_app(
    env=HallucinationEnvironment,          # class, not instance
    action_cls=HallucinationAction,
    observation_cls=HallucinationObservation,
    env_name="Hallucination Detector RL Environment",
    max_concurrent_envs=16
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=False
    )