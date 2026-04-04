# client.py
from typing import Any
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import HallucinationAction, HallucinationObservation, HallucinationState


class HallucinationEnvClient(EnvClient):

    def _step_payload(self, action: HallucinationAction) -> dict:
        """Convert action to dict for sending to server."""
        return {
            "has_hallucination": action.has_hallucination,
            "hallucinated_claim": action.hallucinated_claim,
            "correct_fact": action.correct_fact,
            "confidence": action.confidence
        }

    def _parse_result(self, payload: dict) -> StepResult:
        """Convert server response dict to StepResult with typed observation."""
        obs_data = payload.get("observation", payload)
        obs = HallucinationObservation(
            done=obs_data.get("done", False),
            reward=obs_data.get("reward"),
            task_id=obs_data.get("task_id", ""),
            sample_index=obs_data.get("sample_index", 0),
            total_samples=obs_data.get("total_samples", 0),
            reference_document=obs_data.get("reference_document", ""),
            llm_response=obs_data.get("llm_response", ""),
            feedback=obs_data.get("feedback"),
            score=obs_data.get("score", 0.0),
            steps_taken=obs_data.get("steps_taken", 0),
            max_steps=obs_data.get("max_steps", 10),
            metadata=obs_data.get("metadata", {})
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.reward),
            done=payload.get("done", obs.done)
        )

    def _parse_state(self, payload: dict) -> HallucinationState:
        """Convert server state dict to typed HallucinationState."""
        return HallucinationState(
            episode_id=payload.get("episode_id"),
            task_id=payload.get("task_id", ""),
            sample_index=payload.get("sample_index", 0),
            total_samples=payload.get("total_samples", 0),
            episode_score=payload.get("episode_score", 0.0),
            steps_taken=payload.get("steps_taken", 0),
            is_done=payload.get("is_done", False)
        )