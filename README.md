# 🏥 Medical Triage Assistant — OpenEnv 

> A non-diagnostic medical triage environment for evaluating AI agents on
> real-world clinical triage workflows.

[![OpenEnv](https://img.shields.io/badge/openenv-v1-blue)](https://github.com/huggingface/openenv)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview & Motivation

Emergency Department (ED) triage is one of the most time-critical tasks in
healthcare. Triage nurses must:

1. Rapidly assess patient acuity and assign an **Emergency Severity Index (ESI)** level
2. Collect structured **patient intake information** efficiently
3. **Manage a dynamic queue** of waiting patients, reprioritizing as conditions change

All three workflows are cognitive, language-heavy, and well-suited to evaluation
with language model agents — yet no existing OpenEnv environment covers this domain.

**Important:** This environment is *non-diagnostic*. The agent never diagnoses
medical conditions. It performs the administrative and routing tasks that triage
staff actually do: assigning urgency levels, collecting intake fields, and sorting
a queue by priority.

---

## Environment Design

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     FastAPI Server (:7860)                    │
│                                                              │
│   POST /reset  →  MedicalTriageEnv.reset()                  │
│   POST /step   →  MedicalTriageEnv.step(action)             │
│   GET  /state  →  MedicalTriageEnv.state()                  │
│   GET  /tasks  →  task catalogue                             │
└──────────────────────────────────────────────────────────────┘
```

### Observation Space

Each observation is a `TriageObservation` Pydantic model:

| Field | Type | Description |
|-------|------|-------------|
| `task_type` | `str` | Active task name |
| `patient` | `Patient \| null` | Single patient (ESI/intake tasks) |
| `queue` | `List[Patient] \| null` | All patients (queue task) |
| `conversation_history` | `List[{role, content}]` | Intake dialogue so far |
| `available_actions` | `List[str]` | Valid actions for this step |
| `step_number` | `int` | Current step within the episode |
| `message` | `str` | Narrative description of current state |
| `done` | `bool` | Whether the episode has ended |

**Patient fields:**

| Field | Type | Description |
|-------|------|-------------|
| `patient_id` | `str` | Unique patient identifier |
| `age` | `int` | Patient age |
| `gender` | `str` | Patient gender |
| `chief_complaint` | `str` | Presenting complaint in natural language |
| `vitals` | `PatientVitals` | HR, BP, RR, O2 sat, temp, pain scale |
| `arrival_minutes_ago` | `int` | Time waiting (queue task) |
| `status` | `str` | waiting / in_triage / critical / treated |

### Action Space

Each action is a `TriageAction` Pydantic model:

```json
{
  "action_type": "assign_level",
  "content": "2",
  "target_patient_id": null
}
```

---

## Tasks

### Task 1 — ESI Level Assignment `[EASY]`

**Objective:** Assign the correct Emergency Severity Index (ESI) triage level
to a patient given their chief complaint and vital signs.

**ESI Levels:**
| Level | Name | Example |
|-------|------|---------|
| 1 | Immediate | Cardiac arrest, respiratory failure |
| 2 | Emergent | STEMI, stroke, severe anaphylaxis |
| 3 | Urgent | Fever with complex symptoms, moderate pain |
| 4 | Less Urgent | Ankle sprain, minor laceration |
| 5 | Non-Urgent | Prescription refill, mild cold |

**Actions:**
- `assign_level:<1-5>` — submit the ESI assignment (terminates episode)
- `request_vitals` — request detailed vital signs (no cost)
- `request_history` — request additional patient history (no cost)

**Grading:**
```
Exact match      → 1.0
One level off    → 0.5
Two levels off   → 0.2
Three+ levels    → 0.0
Invalid action   → -0.05
```

**Max steps:** 3 | **Difficulty:** Easy

---

### Task 2 — Patient Intake Interview `[MEDIUM]`

**Objective:** Conduct a structured patient intake interview, collecting all
9 required fields by asking targeted questions. Complete the intake efficiently
(fewer steps = higher efficiency bonus).

**Required Fields:**
`pain_scale` · `onset` · `duration` · `location` · `associated_symptoms` ·
`medical_history` · `current_medications` · `allergies` · `last_meal`

**Actions:**
- `ask_field:<field_name>` — ask the patient about a specific field
- `complete_intake` — finalize intake and receive graded score

**Grading:**
```
Score = 0.7 × completeness + 0.3 × efficiency

completeness = fields_collected / 9
efficiency:
  steps ≤ 9:       1.0
  9 < steps ≤ 18:  linear decay from 1.0 → 0.5
  steps > 18:      0.1

Intermediate reward: +0.07 per new field collected
Penalty:            -0.05 for repeated or invalid questions
```

**Max steps:** 20 | **Difficulty:** Medium

---

### Task 3 — Dynamic Queue Management `[HARD]`

**Objective:** Manage a queue of 5 waiting patients with different urgency
levels. Submit a prioritized ordering (most urgent first), then respond
correctly when a patient deteriorates mid-episode.

**Mid-Episode Event (Step 2):** Patient Q003 deteriorates — abdominal pain
worsens with rebound tenderness and fever, escalating from ESI 3 → ESI 2.
The agent must resubmit an updated queue.

**Actions:**
- `submit_queue:<Q001,Q002,...>` — submit prioritized patient order
- `get_details:<patient_id>` — inspect a patient's full vitals
- `escalate:<patient_id>` — flag a deteriorating patient (+0.05 if appropriate)

**Grading:**
```
Score = 0.6 × kendall_tau + 0.2 × critical_placement + 0.2 × deterioration_response

kendall_tau:          agreement with optimal ESI-ordered ranking, normalized [0,1]
critical_placement:   fraction of ESI 1/2 patients in correct top positions
deterioration_response: Q003 placement after deterioration event
```

**Max steps:** 8 | **Difficulty:** Hard

---

## Reward Function

All rewards are **dense** — emitted at every step to provide learning signal
throughout the trajectory (not just at episode end).

| Task | Per-step reward | Episode-end reward |
|------|-----------------|--------------------|
| ESI Assignment | ±0.05 for info requests/errors | Final graded score (0–1) |
| Intake Interview | +0.07/field, −0.05/error | Full grade on complete_intake |
| Queue Management | +0.3 × partial_score per submit | Full grade at episode end |

**Penalty behaviors:** invalid action types, repeated questions, unnecessary escalation.

---

## Setup & Usage

### Local (Python)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
python server.py
# Server available at http://localhost:7860

# 3. Test the API
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_type": "esi_assignment", "seed": 42}'

# 4. Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "assign_level", "content": "2"}'
```

### Docker

```bash
# Build
docker build -t medical-triage-env .

# Run
docker run -p 7860:7860 medical-triage-env

# Validate (requires openenv-core)
pip install openenv-core
openenv validate
```

### Run Baseline Inference

```bash
export HF_TOKEN="hf_your_token_here"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

### HuggingFace Spaces

Deploy to HF Spaces using the Docker SDK. Set space hardware to CPU Basic
(vcpu=2, memory=8GB). Tag the Space with `openenv`.

Required Space secrets: `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`

---

## API Reference

| Endpoint | Method | Body | Returns |
|----------|--------|------|---------|
| `/reset` | POST | `{task_type, seed?}` | `StepResult` |
| `/step` | POST | `{action_type, content, target_patient_id?}` | `StepResult` |
| `/state` | GET | — | Current internal state dict |
| `/tasks` | GET | — | Task catalogue |
| `/health` | GET | — | `{status: ok}` |

**StepResult schema:**
```json
{
  "observation": { ... TriageObservation ... },
  "reward": 0.0,
  "done": false,
  "info": {}
}
```

---

## Baseline Performance Scores

Scores obtained with `Qwen/Qwen2.5-72B-Instruct` via HF Inference Router,
seed=42, temperature=0.2:

| Task | Score | Notes |
|------|-------|-------|
| ESI Assignment (Easy) | 0.85 | Strong pattern recognition on vitals |
| Intake Interview (Medium) | 0.72 | Misses some fields without explicit prompting |
| Queue Management (Hard) | 0.61 | Struggles with dynamic reprioritization event |
| **Aggregate** | **0.73** | Average across all 3 tasks |

---

## Project Structure

```
medical-triage-env/
├── Dockerfile               # Container build for HF Spaces
├── README.md                # This file
├── openenv.yaml             # OpenEnv spec metadata
├── requirements.txt         # Python dependencies
├── medical_triage_env.py    # Core environment (models, tasks, graders)
├── server.py                # FastAPI REST server
└── inference.py             # Baseline inference script
```

---

## Design Decisions & Real-World Grounding

- **ESI framework** is the standard used in 70%+ of US emergency departments
- **Intake fields** match the SOAPIE nursing documentation format
- **Kendall tau** ranking metric is standard for evaluating ranked-list quality
- **Dense rewards** (not sparse) enable RL training without reward hacking at episode end
- **Deterioration event** at step 2 of the queue task tests dynamic adaptation — a key skill for real triage

---

## License

MIT — see [LICENSE](LICENSE)
