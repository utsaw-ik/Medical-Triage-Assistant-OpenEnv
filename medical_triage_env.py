"""
Medical Triage Assistant Environment
=====================================
A non-diagnostic medical triage simulation environment implementing the OpenEnv spec.

Tasks:
  1. ESI Level Assignment (Easy)   – Assign the correct Emergency Severity Index level
  2. Patient Intake Interview (Medium) – Conduct structured patient intake
  3. Dynamic Queue Management (Hard)  – Prioritize a dynamic multi-patient queue

All tasks simulate real clinical workflows performed by triage nurses and ED clerks.
No medical diagnosis is made; the agent performs administrative/routing triage only.
"""

from __future__ import annotations

import copy
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# ─── Enums ──────────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    ESI_ASSIGNMENT = "esi_assignment"
    INTAKE_INTERVIEW = "intake_interview"
    QUEUE_MANAGEMENT = "queue_management"


# ─── Pydantic Data Models ────────────────────────────────────────────────────

class PatientVitals(BaseModel):
    heart_rate: Optional[int] = None            # bpm
    bp_systolic: Optional[int] = None           # mmHg
    bp_diastolic: Optional[int] = None          # mmHg
    respiratory_rate: Optional[int] = None      # breaths/min
    oxygen_saturation: Optional[float] = None   # %
    temperature_f: Optional[float] = None       # °F
    pain_scale: Optional[int] = None            # 0–10


class Patient(BaseModel):
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    vitals: PatientVitals
    arrival_minutes_ago: int = 0
    status: str = "waiting"                     # waiting | in_triage | critical | treated
    correct_esi_level: Optional[int] = None     # ground truth 1–5
    collected_fields: Dict[str, Any] = Field(default_factory=dict)
    deteriorated: bool = False


class TriageObservation(BaseModel):
    task_type: str
    patient: Optional[Patient] = None
    queue: Optional[List[Patient]] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)
    step_number: int = 0
    message: str = ""
    done: bool = False


class TriageAction(BaseModel):
    action_type: str        # assign_level | ask_field | complete_intake | submit_queue | escalate
    content: str            # level number, field name, ordered patient IDs (comma-sep), etc.
    target_patient_id: Optional[str] = None


class StepResult(BaseModel):
    observation: TriageObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ─── Patient Case Library ────────────────────────────────────────────────────

ESI_CASES: List[Dict[str, Any]] = [
    # ESI 1 – Immediate (life-threatening, requires immediate intervention)
    {
        "patient_id": "P001",
        "age": 58,
        "gender": "Male",
        "chief_complaint": "Unresponsive, not breathing, bystander started CPR",
        "vitals": PatientVitals(heart_rate=0, bp_systolic=None, bp_diastolic=None,
                                respiratory_rate=0, oxygen_saturation=None, temperature_f=97.2),
        "correct_esi_level": 1,
    },
    {
        "patient_id": "P002",
        "age": 34,
        "gender": "Female",
        "chief_complaint": "Severe anaphylaxis after bee sting — throat swelling, cannot swallow",
        "vitals": PatientVitals(heart_rate=128, bp_systolic=72, bp_diastolic=40,
                                respiratory_rate=28, oxygen_saturation=88.0, pain_scale=9),
        "correct_esi_level": 1,
    },
    # ESI 2 – Emergent (high risk, confused, or severe pain)
    {
        "patient_id": "P003",
        "age": 62,
        "gender": "Male",
        "chief_complaint": "Crushing chest pain radiating to left arm for 30 minutes",
        "vitals": PatientVitals(heart_rate=110, bp_systolic=165, bp_diastolic=98,
                                respiratory_rate=22, oxygen_saturation=94.0, pain_scale=9),
        "correct_esi_level": 2,
    },
    {
        "patient_id": "P004",
        "age": 45,
        "gender": "Female",
        "chief_complaint": "Sudden severe headache — worst headache of my life, started 1 hour ago",
        "vitals": PatientVitals(heart_rate=92, bp_systolic=178, bp_diastolic=105,
                                respiratory_rate=18, oxygen_saturation=97.0, pain_scale=10),
        "correct_esi_level": 2,
    },
    {
        "patient_id": "P005",
        "age": 71,
        "gender": "Male",
        "chief_complaint": "Right-sided facial droop and arm weakness, onset 45 minutes ago",
        "vitals": PatientVitals(heart_rate=88, bp_systolic=190, bp_diastolic=110,
                                respiratory_rate=20, oxygen_saturation=96.0, pain_scale=3),
        "correct_esi_level": 2,
    },
    # ESI 3 – Urgent (stable vitals, needs multiple resources)
    {
        "patient_id": "P006",
        "age": 28,
        "gender": "Female",
        "chief_complaint": "Fever 102.8°F, severe sore throat, difficulty swallowing for 2 days",
        "vitals": PatientVitals(heart_rate=98, bp_systolic=118, bp_diastolic=76,
                                respiratory_rate=18, oxygen_saturation=98.0,
                                temperature_f=102.8, pain_scale=6),
        "correct_esi_level": 3,
    },
    {
        "patient_id": "P007",
        "age": 52,
        "gender": "Male",
        "chief_complaint": "Lower back pain after lifting heavy boxes, radiating to right leg",
        "vitals": PatientVitals(heart_rate=82, bp_systolic=130, bp_diastolic=84,
                                respiratory_rate=16, oxygen_saturation=99.0, pain_scale=7),
        "correct_esi_level": 3,
    },
    # ESI 4 – Less Urgent (one resource needed, stable)
    {
        "patient_id": "P008",
        "age": 22,
        "gender": "Female",
        "chief_complaint": "Rolled ankle while running, moderate swelling, can bear weight",
        "vitals": PatientVitals(heart_rate=80, bp_systolic=118, bp_diastolic=74,
                                respiratory_rate=16, oxygen_saturation=99.0, pain_scale=4),
        "correct_esi_level": 4,
    },
    {
        "patient_id": "P009",
        "age": 35,
        "gender": "Male",
        "chief_complaint": "Minor laceration on right palm from kitchen knife, bleeding controlled",
        "vitals": PatientVitals(heart_rate=76, bp_systolic=122, bp_diastolic=80,
                                respiratory_rate=15, oxygen_saturation=99.0, pain_scale=3),
        "correct_esi_level": 4,
    },
    # ESI 5 – Non-Urgent (single resource or none, stable, chronic)
    {
        "patient_id": "P010",
        "age": 48,
        "gender": "Female",
        "chief_complaint": "Requesting prescription refill for hypertension medication",
        "vitals": PatientVitals(heart_rate=70, bp_systolic=128, bp_diastolic=82,
                                respiratory_rate=14, oxygen_saturation=99.0, pain_scale=0),
        "correct_esi_level": 5,
    },
    {
        "patient_id": "P011",
        "age": 19,
        "gender": "Male",
        "chief_complaint": "Mild cold symptoms — runny nose, mild sore throat for 3 days",
        "vitals": PatientVitals(heart_rate=72, bp_systolic=115, bp_diastolic=72,
                                respiratory_rate=15, oxygen_saturation=99.0,
                                temperature_f=99.1, pain_scale=2),
        "correct_esi_level": 5,
    },
]

INTAKE_REQUIRED_FIELDS = [
    "pain_scale",
    "onset",
    "duration",
    "location",
    "associated_symptoms",
    "medical_history",
    "current_medications",
    "allergies",
    "last_meal",
]

INTAKE_FIELD_DESCRIPTIONS = {
    "pain_scale": "Pain level on a scale of 0 to 10",
    "onset": "When did the symptoms start?",
    "duration": "How long have you had these symptoms?",
    "location": "Where exactly is the pain/discomfort located?",
    "associated_symptoms": "Any other symptoms accompanying the main complaint?",
    "medical_history": "Any relevant past medical conditions or surgeries?",
    "current_medications": "What medications are you currently taking?",
    "allergies": "Any known drug or food allergies?",
    "last_meal": "When did you last eat or drink?",
}

PATIENT_INTAKE_ANSWERS: Dict[str, Dict[str, str]] = {
    "P006": {
        "pain_scale": "6 out of 10",
        "onset": "2 days ago, started suddenly after a meal",
        "duration": "Continuous for 2 days, getting worse",
        "location": "Throat and neck area, worse on the right side",
        "associated_symptoms": "Mild fever, fatigue, difficulty swallowing liquids",
        "medical_history": "Seasonal allergies, no prior throat issues",
        "current_medications": "Cetirizine 10mg daily for allergies",
        "allergies": "Penicillin — rash",
        "last_meal": "Yesterday evening, mostly liquids",
    },
    "P007": {
        "pain_scale": "7 out of 10",
        "onset": "This morning while lifting boxes at work",
        "duration": "About 4 hours",
        "location": "Lower back, right side, with radiation down the right leg",
        "associated_symptoms": "Tingling and numbness in right foot",
        "medical_history": "Two prior episodes of back pain, resolved without surgery",
        "current_medications": "Ibuprofen as needed",
        "allergies": "No known drug allergies",
        "last_meal": "Lunch 2 hours ago",
    },
}

QUEUE_SCENARIO: List[Dict[str, Any]] = [
    {
        "patient_id": "Q001",
        "age": 67,
        "gender": "Male",
        "chief_complaint": "Mild shortness of breath on exertion, stable at rest",
        "vitals": PatientVitals(heart_rate=88, bp_systolic=140, bp_diastolic=90,
                                respiratory_rate=18, oxygen_saturation=96.0, pain_scale=2),
        "correct_esi_level": 3,
        "arrival_minutes_ago": 15,
    },
    {
        "patient_id": "Q002",
        "age": 19,
        "gender": "Female",
        "chief_complaint": "Wrist pain after falling off bicycle, possible fracture",
        "vitals": PatientVitals(heart_rate=82, bp_systolic=112, bp_diastolic=72,
                                respiratory_rate=16, oxygen_saturation=99.0, pain_scale=5),
        "correct_esi_level": 4,
        "arrival_minutes_ago": 25,
    },
    {
        "patient_id": "Q003",
        "age": 44,
        "gender": "Male",
        "chief_complaint": "Abdominal pain, right lower quadrant, nausea for 6 hours",
        "vitals": PatientVitals(heart_rate=96, bp_systolic=122, bp_diastolic=78,
                                respiratory_rate=17, oxygen_saturation=98.0, pain_scale=6),
        "correct_esi_level": 3,
        "arrival_minutes_ago": 10,
    },
    {
        "patient_id": "Q004",
        "age": 31,
        "gender": "Female",
        "chief_complaint": "Prescription refill for birth control",
        "vitals": PatientVitals(heart_rate=68, bp_systolic=116, bp_diastolic=70,
                                respiratory_rate=14, oxygen_saturation=99.0, pain_scale=0),
        "correct_esi_level": 5,
        "arrival_minutes_ago": 30,
    },
    {
        "patient_id": "Q005",
        "age": 78,
        "gender": "Female",
        "chief_complaint": "Confusion and slurred speech noticed by family 1 hour ago",
        "vitals": PatientVitals(heart_rate=102, bp_systolic=182, bp_diastolic=108,
                                respiratory_rate=20, oxygen_saturation=95.0, pain_scale=None),
        "correct_esi_level": 2,
        "arrival_minutes_ago": 5,
    },
]


# ─── Graders ─────────────────────────────────────────────────────────────────

def grade_esi_assignment(assigned: int, correct: int) -> float:
    """Score 1.0 for exact match, decreasing for each level off."""
    diff = abs(assigned - correct)
    if diff == 0:
        score = 1.0
    elif diff == 1:
        score = 0.5
    elif diff == 2:
        score = 0.2
    else:
        score = 0.0
    # Clamp to (0.001, 0.999) to satisfy validator requirement
    return max(0.001, min(0.999, score))


def clamp_score(score: float) -> float:
    """
    Clamp any score to be strictly between 0 and 1 (exclusive).
    Validator requires scores in range (0, 1) not [0, 1].
    """
    return max(0.001, min(0.999, score))


def grade_intake_interview(
    collected_fields: Dict[str, Any],
    required_fields: List[str],
    conversation_len: int,
) -> float:
    """
    Score based on:
    - 70%: completeness (fields collected / total required)
    - 30%: efficiency (fewer steps is better; penalize excessive questioning)
    """
    collected = [f for f in required_fields if f in collected_fields]
    completeness = len(collected) / len(required_fields)

    # Efficiency: ideal is ~1 step per field. Penalize >2x the required steps.
    ideal_steps = len(required_fields)
    if conversation_len <= ideal_steps:
        efficiency = 1.0
    elif conversation_len <= ideal_steps * 2:
        efficiency = 1.0 - 0.5 * (conversation_len - ideal_steps) / ideal_steps
    else:
        efficiency = 0.1

    score = round(0.7 * completeness + 0.3 * efficiency, 4)
    # Clamp to (0.001, 0.999) to satisfy validator requirement
    return max(0.001, min(0.999, score))


def kendall_tau_score(predicted: List[str], optimal: List[str]) -> float:
    """
    Compute a normalized Kendall tau distance score in [0, 1].
    1.0 = perfect agreement, 0.0 = complete reversal.
    """
    n = len(optimal)
    if n <= 1:
        score = 1.0
        return max(0.001, min(0.999, score))

    rank_map = {pid: i for i, pid in enumerate(optimal)}
    pred_filtered = [p for p in predicted if p in rank_map]

    concordant = 0
    discordant = 0
    pairs = 0
    for i in range(len(pred_filtered)):
        for j in range(i + 1, len(pred_filtered)):
            pi, pj = pred_filtered[i], pred_filtered[j]
            opt_i, opt_j = rank_map[pi], rank_map[pj]
            pairs += 1
            if (opt_i < opt_j):
                concordant += 1
            else:
                discordant += 1

    if pairs == 0:
        score = 0.0
    else:
        score = (concordant - discordant) / pairs * 0.5 + 0.5
    # Clamp to (0.001, 0.999) to satisfy validator requirement
    return max(0.001, min(0.999, score))


def grade_queue_management(
    submitted_order: List[str],
    queue: List[Patient],
    event_history: List[str],
) -> float:
    """
    Score based on:
    - 60%: Kendall tau agreement with optimal ESI-level ordering
    - 20%: Critical patient (ESI 1/2) in top positions
    - 20%: Appropriate response to deterioration event (if any)
    """
    esi_map = {p.patient_id: p.correct_esi_level or 5 for p in queue}

    # Build optimal order (lower ESI = higher priority; tie-break by arrival time)
    arrival_map = {p.patient_id: p.arrival_minutes_ago for p in queue}
    optimal = sorted(
        [p.patient_id for p in queue],
        key=lambda pid: (esi_map[pid], arrival_map[pid]),
    )

    tau = kendall_tau_score(submitted_order, optimal)

    # Check critical patients in top 2 slots
    critical_ids = [pid for pid, lvl in esi_map.items() if lvl <= 2]
    top_slots = submitted_order[:len(critical_ids)] if submitted_order else []
    critical_score = (
        sum(1 for c in critical_ids if c in top_slots) / max(len(critical_ids), 1)
    )

    # Deterioration response: if a patient was escalated, reward for moving them up
    deterioration_score = 1.0
    if "Q003_deteriorated" in event_history:
        idx = submitted_order.index("Q003") if "Q003" in submitted_order else 999
        deterioration_score = max(0.0, 1.0 - idx * 0.2)

    score = round(0.6 * tau + 0.2 * critical_score + 0.2 * deterioration_score, 4)
    # Clamp to (0.001, 0.999) to satisfy validator requirement
    return max(0.001, min(0.999, score))


# ─── Environment Class ────────────────────────────────────────────────────────

class MedicalTriageEnv:
    """
    OpenEnv-compliant Medical Triage Assistant environment.
    Implements step() / reset() / state() for three distinct tasks.
    """

    MAX_STEPS = {
        TaskType.ESI_ASSIGNMENT: 3,
        TaskType.INTAKE_INTERVIEW: 20,
        TaskType.QUEUE_MANAGEMENT: 8,
    }

    def __init__(self, task_type: str = TaskType.ESI_ASSIGNMENT, seed: Optional[int] = None):
        self.task_type = TaskType(task_type)
        self.seed = seed
        self._rng = random.Random(seed)
        self._state: Dict[str, Any] = {}
        self._obs: Optional[TriageObservation] = None

    # ── reset ──────────────────────────────────────────────────────────────

    def reset(self) -> StepResult:
        self._rng = random.Random(self.seed)

        if self.task_type == TaskType.ESI_ASSIGNMENT:
            return self._reset_esi()
        elif self.task_type == TaskType.INTAKE_INTERVIEW:
            return self._reset_intake()
        else:
            return self._reset_queue()

    def _reset_esi(self) -> StepResult:
        case_data = self._rng.choice(ESI_CASES)
        patient = Patient(**case_data)
        self._state = {
            "patient": patient,
            "step": 0,
            "done": False,
            "rewards": [],
            "assigned_level": None,
        }
        obs = TriageObservation(
            task_type=self.task_type,
            patient=patient,
            step_number=0,
            available_actions=[
                "assign_level:1", "assign_level:2", "assign_level:3",
                "assign_level:4", "assign_level:5",
                "request_vitals", "request_history",
            ],
            message=(
                f"NEW PATIENT — {patient.gender}, {patient.age}y/o\n"
                f"Chief Complaint: {patient.chief_complaint}\n"
                f"Vitals: HR={patient.vitals.heart_rate}, "
                f"BP={patient.vitals.bp_systolic}/{patient.vitals.bp_diastolic}, "
                f"RR={patient.vitals.respiratory_rate}, "
                f"O2={patient.vitals.oxygen_saturation}%, "
                f"Pain={patient.vitals.pain_scale}/10\n\n"
                "Assign the ESI triage level (1=Immediate, 2=Emergent, 3=Urgent, "
                "4=Less Urgent, 5=Non-Urgent)."
            ),
        )
        self._obs = obs
        return StepResult(observation=obs, reward=0.0, done=False, info={})

    def _reset_intake(self) -> StepResult:
        intake_pid = self._rng.choice(list(PATIENT_INTAKE_ANSWERS.keys()))
        case_data = next(c for c in ESI_CASES if c["patient_id"] == intake_pid)
        patient = Patient(**case_data)
        self._state = {
            "patient": patient,
            "step": 0,
            "done": False,
            "rewards": [],
            "collected_fields": {},
            "asked_fields": [],
            "conversation": [],
        }
        obs = TriageObservation(
            task_type=self.task_type,
            patient=patient,
            conversation_history=[],
            step_number=0,
            available_actions=[f"ask_field:{f}" for f in INTAKE_REQUIRED_FIELDS]
            + ["complete_intake"],
            message=(
                f"BEGIN INTAKE — {patient.gender}, {patient.age}y/o\n"
                f"Chief Complaint: {patient.chief_complaint}\n\n"
                "Conduct a structured intake interview by asking the patient relevant "
                "questions. Use ask_field:<field_name> to ask about a specific field, "
                "or complete_intake when finished.\n"
                f"Required fields: {', '.join(INTAKE_REQUIRED_FIELDS)}"
            ),
        )
        self._obs = obs
        return StepResult(observation=obs, reward=0.0, done=False, info={})

    def _reset_queue(self) -> StepResult:
        queue = [Patient(**{**c}) for c in QUEUE_SCENARIO]
        self._state = {
            "queue": queue,
            "step": 0,
            "done": False,
            "rewards": [],
            "event_history": [],
            "submitted_orders": [],
            "phase": "initial",  # initial | event_arrived | final
        }
        queue_summary = self._format_queue(queue)
        obs = TriageObservation(
            task_type=self.task_type,
            queue=queue,
            step_number=0,
            available_actions=[
                "submit_queue:<comma-sep-patient-ids>",
                "escalate:<patient_id>",
                "get_details:<patient_id>",
            ],
            message=(
                "TRIAGE QUEUE — 5 patients waiting\n\n"
                + queue_summary
                + "\n\nSubmit a prioritized ordering using "
                "submit_queue:Q005,Q003,Q001,Q002,Q004 (example).\n"
                "Highest priority first. Use patient IDs: Q001–Q005."
            ),
        )
        self._obs = obs
        return StepResult(observation=obs, reward=0.0, done=False, info={})

    # ── step ───────────────────────────────────────────────────────────────

    def step(self, action: TriageAction) -> StepResult:
        if self._state.get("done"):
            return StepResult(
                observation=self._obs,
                reward=0.0,
                done=True,
                info={"error": "Episode already done. Call reset() to start a new episode."},
            )

        self._state["step"] += 1
        max_steps = self.MAX_STEPS[self.task_type]
        timed_out = self._state["step"] >= max_steps

        if self.task_type == TaskType.ESI_ASSIGNMENT:
            return self._step_esi(action, timed_out)
        elif self.task_type == TaskType.INTAKE_INTERVIEW:
            return self._step_intake(action, timed_out)
        else:
            return self._step_queue(action, timed_out)

    # ── ESI task steps ─────────────────────────────────────────────────────

    def _step_esi(self, action: TriageAction, timed_out: bool) -> StepResult:
        patient: Patient = self._state["patient"]
        step = self._state["step"]

        if action.action_type == "assign_level":
            try:
                level = int(action.content.strip())
                assert 1 <= level <= 5
            except (ValueError, AssertionError):
                reward = -0.05
                self._state["rewards"].append(reward)
                obs = self._build_esi_obs(
                    patient, step,
                    message=f"Invalid level '{action.content}'. Please assign a level 1–5.",
                )
                return StepResult(observation=obs, reward=reward, done=False, info={})

            score = grade_esi_assignment(level, patient.correct_esi_level)
            self._state["assigned_level"] = level
            self._state["done"] = True
            reward = clamp_score(score)  # Ensure clamped
            self._state["rewards"].append(reward)

            if score >= 0.99:  # Near-perfect score
                msg = f"✓ Correct! ESI Level {level} assigned. Score: {score:.2f}"
            elif score >= 0.5:
                msg = (f"Close — you assigned ESI {level}, correct is ESI "
                       f"{patient.correct_esi_level}. Score: {score:.2f}")
            else:
                msg = (f"Incorrect — you assigned ESI {level}, correct is ESI "
                       f"{patient.correct_esi_level}. Score: {score:.2f}")

            obs = self._build_esi_obs(patient, step, message=msg, done=True)
            return StepResult(
                observation=obs, reward=reward, done=True,
                info={"score": clamp_score(score), "assigned": level, "correct": patient.correct_esi_level},
            )

        elif action.action_type == "request_vitals":
            reward = 0.0
            msg = (
                "Detailed Vitals:\n"
                f"  Heart Rate: {patient.vitals.heart_rate} bpm\n"
                f"  Blood Pressure: {patient.vitals.bp_systolic}/{patient.vitals.bp_diastolic} mmHg\n"
                f"  Respiratory Rate: {patient.vitals.respiratory_rate} breaths/min\n"
                f"  O2 Saturation: {patient.vitals.oxygen_saturation}%\n"
                f"  Temperature: {patient.vitals.temperature_f}°F\n"
                f"  Pain Scale: {patient.vitals.pain_scale}/10"
            )
            obs = self._build_esi_obs(patient, step, message=msg)
            self._state["rewards"].append(reward)
            return StepResult(observation=obs, reward=reward, done=timed_out, info={})

        elif action.action_type == "request_history":
            reward = 0.0
            msg = (
                "Additional context: Patient arrived via ambulance. "
                "Family present and able to provide history. "
                "No prior ED visits on record in the past 6 months."
            )
            obs = self._build_esi_obs(patient, step, message=msg)
            self._state["rewards"].append(reward)
            return StepResult(observation=obs, reward=reward, done=timed_out, info={})

        else:
            # Unrecognized action
            reward = -0.05
            self._state["rewards"].append(reward)
            obs = self._build_esi_obs(
                patient, step,
                message=f"Unknown action '{action.action_type}'. Use assign_level, request_vitals, or request_history.",
            )
            if timed_out:
                self._state["done"] = True
            return StepResult(observation=obs, reward=reward, done=timed_out, info={})

    def _build_esi_obs(
        self, patient: Patient, step: int, message: str, done: bool = False
    ) -> TriageObservation:
        obs = TriageObservation(
            task_type=self.task_type,
            patient=patient,
            step_number=step,
            done=done,
            available_actions=(
                []
                if done
                else ["assign_level:1", "assign_level:2", "assign_level:3",
                      "assign_level:4", "assign_level:5",
                      "request_vitals", "request_history"]
            ),
            message=message,
        )
        self._obs = obs
        return obs

    # ── Intake task steps ──────────────────────────────────────────────────

    def _step_intake(self, action: TriageAction, timed_out: bool) -> StepResult:
        patient: Patient = self._state["patient"]
        step = self._state["step"]
        collected = self._state["collected_fields"]
        asked = self._state["asked_fields"]
        conversation = self._state["conversation"]
        answers = PATIENT_INTAKE_ANSWERS.get(patient.patient_id, {})

        if action.action_type == "ask_field":
            field = action.content.strip()
            if field not in INTAKE_REQUIRED_FIELDS:
                reward = -0.05
                self._state["rewards"].append(reward)
                msg = (f"Unknown field '{field}'. Available fields: "
                       f"{', '.join(INTAKE_REQUIRED_FIELDS)}")
                obs = self._build_intake_obs(patient, step, conversation, msg)
                return StepResult(observation=obs, reward=reward, done=False, info={})

            if field in asked:
                reward = -0.05
                self._state["rewards"].append(reward)
                msg = f"Already asked about '{field}'. Answer: {answers.get(field, 'N/A')}"
                obs = self._build_intake_obs(patient, step, conversation, msg)
                return StepResult(observation=obs, reward=reward, done=timed_out, info={})

            # New field — reward partial progress
            asked.append(field)
            answer = answers.get(field, "Patient did not respond clearly.")
            collected[field] = answer
            self._state["collected_fields"] = collected

            conversation.append({"role": "nurse", "content": INTAKE_FIELD_DESCRIPTIONS[field]})
            conversation.append({"role": "patient", "content": answer})

            reward = 0.07  # partial credit per field
            self._state["rewards"].append(reward)
            msg = f"[{field.upper()}] Patient says: \"{answer}\""
            obs = self._build_intake_obs(patient, step, conversation, msg)

            if timed_out:
                self._state["done"] = True
                final_score = grade_intake_interview(collected, INTAKE_REQUIRED_FIELDS, step)
                obs.done = True
                obs.message = msg + f"\n\n⏱ Time limit reached. Final intake score: {final_score:.2f}"
                return StepResult(
                    observation=obs, reward=clamp_score(final_score), done=True,
                    info={"final_score": clamp_score(final_score), "collected": list(collected.keys())},
                )

            return StepResult(observation=obs, reward=reward, done=False, info={})

        elif action.action_type == "complete_intake":
            self._state["done"] = True
            final_score = grade_intake_interview(collected, INTAKE_REQUIRED_FIELDS, step)
            missing = [f for f in INTAKE_REQUIRED_FIELDS if f not in collected]
            reward = clamp_score(final_score)  # Ensure clamped
            self._state["rewards"].append(reward)
            msg = (
                f"Intake completed at step {step}.\n"
                f"Fields collected: {len(collected)}/{len(INTAKE_REQUIRED_FIELDS)}\n"
                f"Missing: {', '.join(missing) if missing else 'None'}\n"
                f"Final score: {final_score:.2f}"
            )
            obs = self._build_intake_obs(patient, step, conversation, msg, done=True)
            return StepResult(
                observation=obs, reward=reward, done=True,
                info={"final_score": clamp_score(final_score), "collected": list(collected.keys()), "missing": missing},
            )
        else:
            reward = -0.05
            self._state["rewards"].append(reward)
            msg = f"Unknown action '{action.action_type}'. Use ask_field:<field> or complete_intake."
            obs = self._build_intake_obs(patient, step, conversation, msg)
            return StepResult(observation=obs, reward=reward, done=timed_out, info={})

    def _build_intake_obs(
        self, patient: Patient, step: int, conversation: List, message: str, done: bool = False
    ) -> TriageObservation:
        collected = self._state["collected_fields"]
        remaining = [f for f in INTAKE_REQUIRED_FIELDS if f not in collected]
        obs = TriageObservation(
            task_type=self.task_type,
            patient=patient,
            conversation_history=list(conversation),
            step_number=step,
            done=done,
            available_actions=(
                []
                if done
                else [f"ask_field:{f}" for f in remaining] + ["complete_intake"]
            ),
            message=message,
        )
        self._obs = obs
        return obs

    # ── Queue task steps ───────────────────────────────────────────────────

    def _step_queue(self, action: TriageAction, timed_out: bool) -> StepResult:
        queue: List[Patient] = self._state["queue"]
        step = self._state["step"]
        event_history: List[str] = self._state["event_history"]
        phase = self._state["phase"]

        # Trigger mid-episode event at step 2
        if step == 2 and phase == "initial":
            self._state["phase"] = "event_arrived"
            # Q003 deteriorates (appendicitis worsening) → ESI escalation
            for p in queue:
                if p.patient_id == "Q003":
                    p.status = "critical"
                    p.correct_esi_level = 2
                    p.deteriorated = True
            event_history.append("Q003_deteriorated")
            self._state["event_history"] = event_history
            msg = (
                "⚠ ALERT: Patient Q003 has deteriorated!\n"
                "Q003 (44y/o Male, abdominal pain) now reports rebound tenderness, "
                "guarding, and fever 101.2°F. ESI level escalated to 2.\n\n"
                + self._format_queue(queue)
                + "\n\nResubmit the updated priority queue."
            )
            obs = self._build_queue_obs(queue, step, msg)
            self._state["rewards"].append(0.0)
            return StepResult(observation=obs, reward=0.0, done=False, info={"event": "Q003_deteriorated"})

        if action.action_type == "submit_queue":
            submitted_ids = [s.strip() for s in action.content.split(",")]
            valid_ids = {p.patient_id for p in queue}
            filtered = [pid for pid in submitted_ids if pid in valid_ids]

            if len(filtered) == 0:
                reward = -0.1
                self._state["rewards"].append(reward)
                msg = "Invalid queue submission. Use patient IDs: " + ", ".join(valid_ids)
                obs = self._build_queue_obs(queue, step, msg)
                return StepResult(observation=obs, reward=reward, done=False, info={})

            self._state["submitted_orders"] = filtered

            # Partial reward per submission
            partial_score = grade_queue_management(filtered, queue, event_history)
            reward = partial_score * 0.3  # partial weight; full reward at end
            self._state["rewards"].append(reward)

            if timed_out or phase == "event_arrived":
                self._state["done"] = True
                final_score = grade_queue_management(filtered, queue, event_history)
                esi_map = {p.patient_id: p.correct_esi_level for p in queue}
                optimal = sorted(valid_ids, key=lambda pid: (esi_map[pid] or 5))
                msg = (
                    f"Queue submitted: {' → '.join(filtered)}\n"
                    f"Optimal order:   {' → '.join(optimal)}\n"
                    f"Final score: {final_score:.2f}"
                )
                obs = self._build_queue_obs(queue, step, msg, done=True)
                return StepResult(
                    observation=obs, reward=clamp_score(final_score), done=True,
                    info={"final_score": clamp_score(final_score), "optimal": optimal, "submitted": filtered},
                )

            msg = (
                f"Queue recorded: {' → '.join(filtered)}\n"
                f"Current partial score: {partial_score:.2f}\n"
                "Continue monitoring. New events may require reprioritization."
            )
            obs = self._build_queue_obs(queue, step, msg)
            return StepResult(observation=obs, reward=reward, done=False, info={"partial_score": partial_score})

        elif action.action_type == "get_details":
            pid = action.content.strip()
            patient = next((p for p in queue if p.patient_id == pid), None)
            if not patient:
                reward = -0.02
                self._state["rewards"].append(reward)
                msg = f"Patient {pid} not found in queue."
                obs = self._build_queue_obs(queue, step, msg)
                return StepResult(observation=obs, reward=reward, done=False, info={})

            reward = 0.0
            msg = (
                f"Details — {patient.patient_id} ({patient.gender}, {patient.age}y/o)\n"
                f"Complaint: {patient.chief_complaint}\n"
                f"Vitals: HR={patient.vitals.heart_rate}, "
                f"BP={patient.vitals.bp_systolic}/{patient.vitals.bp_diastolic}, "
                f"O2={patient.vitals.oxygen_saturation}%, "
                f"Pain={patient.vitals.pain_scale}/10\n"
                f"Status: {patient.status} | Waiting: {patient.arrival_minutes_ago} min"
                + (" | ⚠ DETERIORATED" if patient.deteriorated else "")
            )
            self._state["rewards"].append(reward)
            obs = self._build_queue_obs(queue, step, msg)
            return StepResult(observation=obs, reward=reward, done=timed_out, info={})

        elif action.action_type == "escalate":
            pid = action.content.strip()
            patient = next((p for p in queue if p.patient_id == pid), None)
            if not patient:
                reward = -0.02
                self._state["rewards"].append(reward)
                msg = f"Patient {pid} not found."
                obs = self._build_queue_obs(queue, step, msg)
                return StepResult(observation=obs, reward=reward, done=False, info={})

            reward = 0.05 if patient.deteriorated else -0.02
            self._state["rewards"].append(reward)
            msg = (
                f"{'✓ Appropriate escalation' if patient.deteriorated else '⚠ Unnecessary escalation'} "
                f"for {pid}."
            )
            obs = self._build_queue_obs(queue, step, msg)
            return StepResult(observation=obs, reward=reward, done=timed_out, info={})

        else:
            reward = -0.05
            self._state["rewards"].append(reward)
            msg = f"Unknown action '{action.action_type}'."
            obs = self._build_queue_obs(queue, step, msg)
            return StepResult(observation=obs, reward=reward, done=timed_out, info={})

    def _format_queue(self, queue: List[Patient]) -> str:
        lines = []
        for p in sorted(queue, key=lambda x: x.arrival_minutes_ago, reverse=True):
            flag = " ⚠ DETERIORATED" if p.deteriorated else ""
            lines.append(
                f"  [{p.patient_id}] {p.gender} {p.age}y/o — {p.chief_complaint[:55]}... "
                f"| Wait: {p.arrival_minutes_ago}min | Status: {p.status}{flag}"
            )
        return "\n".join(lines)

    def _build_queue_obs(
        self, queue: List[Patient], step: int, message: str, done: bool = False
    ) -> TriageObservation:
        obs = TriageObservation(
            task_type=self.task_type,
            queue=queue,
            step_number=step,
            done=done,
            available_actions=(
                []
                if done
                else [
                    "submit_queue:<comma-sep-patient-ids>",
                    "escalate:<patient_id>",
                    "get_details:<patient_id>",
                ]
            ),
            message=message,
        )
        self._obs = obs
        return obs

    # ── state ──────────────────────────────────────────────────────────────

    def state(self) -> Dict[str, Any]:
        return copy.deepcopy(self._state)

    # ── helpers ────────────────────────────────────────────────────────────

    @property
    def current_observation(self) -> Optional[TriageObservation]:
        return self._obs
