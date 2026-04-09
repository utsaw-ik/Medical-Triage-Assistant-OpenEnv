"""
Medical Triage Assistant — FastAPI Server
==========================================
Exposes the OpenEnv-compliant REST API:
  POST /reset         → initialize / restart episode
  POST /step          → take one action
  GET  /state         → current internal state
  GET  /tasks         → list available tasks
  GET  /health        → liveness probe
  GET  /              → environment info
"""

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from medical_triage_env import (
    MedicalTriageEnv,
    TriageAction,
    StepResult,
    TaskType,
)

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Medical Triage Assistant — OpenEnv",
    description=(
        "A non-diagnostic medical triage environment for evaluating AI agents "
        "on real-world clinical triage workflows. Implements the OpenEnv spec: "
        "step() / reset() / state()."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global environment instance (stateful per server) ────────────────────────

_env: Optional[MedicalTriageEnv] = None


def get_env() -> MedicalTriageEnv:
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call POST /reset first.")
    return _env


# ─── Request / Response Schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_type: str = TaskType.ESI_ASSIGNMENT
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action_type: str
    content: str
    target_patient_id: Optional[str] = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Medical Triage Assistant",
        "version": "1.0.0",
        "description": "OpenEnv-compliant non-diagnostic medical triage environment",
        "tasks": [t.value for t in TaskType],
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "tasks": "GET /tasks",
            "health": "GET /health",
        },
        "spec": "openenv v1",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": TaskType.ESI_ASSIGNMENT,
                "difficulty": "easy",
                "description": (
                    "Assign the correct Emergency Severity Index (ESI) level 1–5 "
                    "to a patient based on chief complaint and vitals."
                ),
                "max_steps": MedicalTriageEnv.MAX_STEPS[TaskType.ESI_ASSIGNMENT],
                "reward_range": [0.0, 1.0],
            },
            {
                "name": TaskType.INTAKE_INTERVIEW,
                "difficulty": "medium",
                "description": (
                    "Conduct a structured patient intake interview, collecting all "
                    "required fields efficiently using ask_field actions."
                ),
                "max_steps": MedicalTriageEnv.MAX_STEPS[TaskType.INTAKE_INTERVIEW],
                "reward_range": [0.0, 1.0],
            },
            {
                "name": TaskType.QUEUE_MANAGEMENT,
                "difficulty": "hard",
                "description": (
                    "Manage a dynamic queue of multiple patients, reprioritizing "
                    "when patient conditions change mid-episode."
                ),
                "max_steps": MedicalTriageEnv.MAX_STEPS[TaskType.QUEUE_MANAGEMENT],
                "reward_range": [0.0, 1.0],
            },
        ]
    }


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    global _env
    try:
        task = TaskType(request.task_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_type '{request.task_type}'. "
                   f"Valid options: {[t.value for t in TaskType]}",
        )
    _env = MedicalTriageEnv(task_type=task, seed=request.seed)
    result = _env.reset()
    return result.model_dump()


@app.post("/step")
def step(request: StepRequest):
    env = get_env()
    action = TriageAction(
        action_type=request.action_type,
        content=request.content,
        target_patient_id=request.target_patient_id,
    )
    result = env.step(action)
    return result.model_dump()


@app.get("/state")
def state():
    env = get_env()
    raw_state = env.state()

    # Serialize Pydantic models within state for JSON response
    serialized: Dict[str, Any] = {}
    for k, v in raw_state.items():
        if hasattr(v, "model_dump"):
            serialized[k] = v.model_dump()
        elif isinstance(v, list):
            serialized[k] = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in v
            ]
        else:
            serialized[k] = v
    return serialized


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
