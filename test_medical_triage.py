"""
Comprehensive Test Suite for Medical Triage Environment
========================================================
Tests all functional requirements for OpenEnv hackathon submission:
  ✓ OpenEnv spec compliance
  ✓ Three tasks with graders (easy/medium/hard)
  ✓ Meaningful reward functions
  ✓ Reproducibility and determinism
"""

import json
import copy
import pytest
from typing import Dict, Any
from medical_triage_env import (
    MedicalTriageEnv,
    TriageAction,
    TaskType,
    grade_esi_assignment,
    grade_intake_interview,
    grade_queue_management,
    Patient,
    PatientVitals,
)


# ─── Test Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def esi_env():
    """ESI assignment environment with fixed seed"""
    return MedicalTriageEnv(task_type=TaskType.ESI_ASSIGNMENT, seed=42)


@pytest.fixture
def intake_env():
    """Intake interview environment with fixed seed"""
    return MedicalTriageEnv(task_type=TaskType.INTAKE_INTERVIEW, seed=42)


@pytest.fixture
def queue_env():
    """Queue management environment with fixed seed"""
    return MedicalTriageEnv(task_type=TaskType.QUEUE_MANAGEMENT, seed=42)


# ═══ OPENENV SPEC COMPLIANCE TESTS ═══════════════════════════════════════════

class TestOpenEnvCompliance:
    """Verify complete OpenEnv interface implementation"""

    def test_env_has_required_methods(self, esi_env):
        """Environment must implement step(), reset(), and state()"""
        assert hasattr(esi_env, 'step')
        assert hasattr(esi_env, 'reset')
        assert hasattr(esi_env, 'state')
        assert callable(esi_env.step)
        assert callable(esi_env.reset)
        assert callable(esi_env.state)

    def test_reset_returns_observation(self, esi_env):
        """reset() must return a valid initial observation"""
        result = esi_env.reset()
        
        assert result is not None
        assert hasattr(result, 'observation')
        
        obs = result.observation
        assert hasattr(obs, 'task_type')
        assert hasattr(obs, 'message')
        assert hasattr(obs, 'step_number')
        assert hasattr(obs, 'done')
        assert hasattr(obs, 'available_actions')
        
        assert obs.step_number == 0
        assert obs.done == False
        assert len(obs.available_actions) > 0

    def test_step_returns_correct_tuple(self, esi_env):
        """step() must return (observation, reward, done, info)"""
        esi_env.reset()
        action = TriageAction(action_type="assign_level", content="3")
        result = esi_env.step(action)
        
        assert hasattr(result, 'observation')
        assert hasattr(result, 'reward')
        assert hasattr(result, 'done')
        assert hasattr(result, 'info')
        
        assert isinstance(result.reward, (int, float))
        assert isinstance(result.done, bool)
        assert isinstance(result.info, dict)

    def test_state_returns_current_state(self, esi_env):
        """state() must return the complete environment state"""
        esi_env.reset()
        state = esi_env.state()
        
        assert isinstance(state, dict)
        assert 'step' in state
        assert 'done' in state

    def test_pydantic_models_are_typed(self):
        """All models must use Pydantic with typed fields"""
        from pydantic import BaseModel
        from medical_triage_env import TriageObservation, TriageAction, Patient
        
        assert issubclass(TriageObservation, BaseModel)
        assert issubclass(TriageAction, BaseModel)
        assert issubclass(Patient, BaseModel)

    def test_reproducibility_with_seed(self):
        """Same seed must produce identical episodes"""
        env1 = MedicalTriageEnv(task_type=TaskType.ESI_ASSIGNMENT, seed=123)
        env2 = MedicalTriageEnv(task_type=TaskType.ESI_ASSIGNMENT, seed=123)
        
        result1 = env1.reset()
        result2 = env2.reset()
        
        obs1 = result1.observation
        obs2 = result2.observation
        
        assert obs1.patient.patient_id == obs2.patient.patient_id
        assert obs1.patient.chief_complaint == obs2.patient.chief_complaint


# ═══ TASK 1: ESI ASSIGNMENT (EASY) ═══════════════════════════════════════════

class TestESIAssignment:
    """Test Easy task: ESI level assignment"""

    def test_task_has_clear_objective(self, esi_env):
        """Task must have well-defined objective"""
        result = esi_env.reset()
        obs = result.observation
        assert "ESI" in obs.message or "level" in obs.message.lower()
        assert obs.patient is not None
        assert obs.patient.vitals is not None

    def test_action_space_is_valid(self, esi_env):
        """Available actions must be appropriate for task"""
        result = esi_env.reset()
        obs = result.observation
        assert len(obs.available_actions) > 0
        
        # Should allow level assignment
        assert any("assign_level" in a for a in obs.available_actions)

    def test_grader_returns_score_0_to_1(self):
        """Grader must return normalized score [0.0, 1.0]"""
        # Test perfect match
        assert grade_esi_assignment(2, 2) == 1.0
        
        # Test off by one
        assert 0.0 < grade_esi_assignment(2, 3) < 1.0
        
        # Test completely wrong
        assert grade_esi_assignment(1, 5) >= 0.0

    def test_grader_is_deterministic(self):
        """Grader must produce identical scores for identical inputs"""
        score1 = grade_esi_assignment(3, 2)
        score2 = grade_esi_assignment(3, 2)
        assert score1 == score2

    def test_complete_episode_workflow(self, esi_env):
        """Full episode must complete successfully"""
        obs = esi_env.reset()
        assert not obs.done
        
        # Make assignment
        action = TriageAction(action_type="assign_level", content="3")
        result = esi_env.step(action)
        
        assert result.reward is not None
        assert isinstance(result.reward, (int, float))
        assert result.reward >= 0.0
        assert result.reward <= 1.0

    def test_invalid_action_handling(self, esi_env):
        """Environment must handle invalid actions gracefully"""
        esi_env.reset()
        
        # Invalid level (out of range)
        action = TriageAction(action_type="assign_level", content="10")
        result = esi_env.step(action)
        
        # Should penalize but not crash
        assert result.reward <= 0.0
        assert result.observation is not None


# ═══ TASK 2: INTAKE INTERVIEW (MEDIUM) ═══════════════════════════════════════

class TestIntakeInterview:
    """Test Medium task: Patient intake interview"""

    def test_task_requires_field_collection(self, intake_env):
        """Task must require collecting multiple fields"""
        result = intake_env.reset()
        obs = result.observation
        
        # Should have multiple fields to collect
        assert len(obs.available_actions) >= 5
        
        # Should allow asking about different fields
        assert any("ask_field" in a for a in obs.available_actions)

    def test_grader_rewards_completeness(self):
        """Grader must reward collecting required fields"""
        required = ["pain_scale", "onset", "duration", "location"]
        
        # All fields collected
        collected_all = {f: "value" for f in required}
        score_all = grade_intake_interview(collected_all, required, len(required))
        
        # Half fields collected
        collected_half = {f: "value" for f in required[:2]}
        score_half = grade_intake_interview(collected_half, required, len(required))
        
        assert score_all > score_half
        assert score_all >= 0.7  # High completeness

    def test_grader_rewards_efficiency(self):
        """Grader must penalize excessive steps"""
        required = ["pain_scale", "onset"]
        collected = {f: "value" for f in required}
        
        # Efficient (2 steps)
        score_efficient = grade_intake_interview(collected, required, 2)
        
        # Inefficient (20 steps)
        score_inefficient = grade_intake_interview(collected, required, 20)
        
        assert score_efficient > score_inefficient

    def test_conversation_history_tracking(self, intake_env):
        """Environment must track conversation history"""
        obs = intake_env.reset()
        
        action = TriageAction(action_type="ask_field", content="pain_scale")
        result = intake_env.step(action)
        
        assert len(result.observation.conversation_history) > 0

    def test_complete_intake_workflow(self, intake_env):
        """Must be able to complete full intake"""
        result = intake_env.reset()
        obs = result.observation
        
        # Collect some fields
        for _ in range(3):
            if obs.done:
                break
            
            actions = [a for a in obs.available_actions if "ask_field" in a]
            if not actions:
                break
                
            field = actions[0].split(":")[-1]
            action = TriageAction(action_type="ask_field", content=field)
            result = intake_env.step(action)
            obs = result.observation
        
        # Complete intake
        action = TriageAction(action_type="complete_intake", content="")
        result = intake_env.step(action)
        
        assert result.done == True
        assert "final_score" in result.info or "score" in result.info


# ═══ TASK 3: QUEUE MANAGEMENT (HARD) ═══════════════════════════════════════

class TestQueueManagement:
    """Test Hard task: Dynamic queue prioritization"""

    def test_task_has_multiple_patients(self, queue_env):
        """Queue task must manage multiple patients"""
        result = queue_env.reset()
        obs = result.observation
        
        assert obs.queue is not None
        assert len(obs.queue) >= 3  # Minimum for meaningful prioritization

    def test_grader_rewards_correct_ordering(self):
        """Grader must reward ESI-based prioritization"""
        # Create test queue
        patients = [
            Patient(
                patient_id=f"P{i}",
                age=30,
                gender="M",
                chief_complaint="test",
                vitals=PatientVitals(heart_rate=80),
                correct_esi_level=esi,
                arrival_minutes_ago=i*10
            )
            for i, esi in enumerate([3, 1, 5, 2, 4])  # ESI levels
        ]
        
        # Optimal order: P1 (ESI 1), P3 (ESI 2), P0 (ESI 3), P4 (ESI 4), P2 (ESI 5)
        optimal = ["P1", "P3", "P0", "P4", "P2"]
        
        # Perfect ordering
        score_perfect = grade_queue_management(optimal, patients, [])
        
        # Random ordering
        score_random = grade_queue_management(["P2", "P4", "P0", "P3", "P1"], patients, [])
        
        assert score_perfect > score_random
        assert score_perfect >= 0.8

    def test_dynamic_event_handling(self, queue_env):
        """Queue must handle mid-episode deterioration events"""
        result = queue_env.reset()
        obs = result.observation
        
        # Submit initial queue
        patient_ids = [p.patient_id for p in obs.queue]
        action = TriageAction(
            action_type="submit_queue",
            content=",".join(patient_ids)
        )
        result = queue_env.step(action)
        
        # Step forward to trigger event
        if not result.done:
            result = queue_env.step(action)
            
            # Check for event notification
            if "deteriorated" in result.observation.message.lower() or "alert" in result.observation.message.lower():
                assert "event" in result.info or "deteriorated" in str(result.observation.message).lower()

    def test_grader_rewards_deterioration_response(self):
        """Grader must reward appropriate response to deterioration"""
        patients = [
            Patient(
                patient_id="P1",
                age=30,
                gender="M",
                chief_complaint="test",
                vitals=PatientVitals(heart_rate=80),
                correct_esi_level=3,
                arrival_minutes_ago=10
            ),
            Patient(
                patient_id="P2",
                age=40,
                gender="F",
                chief_complaint="test2",
                vitals=PatientVitals(heart_rate=90),
                correct_esi_level=2,
                arrival_minutes_ago=5,
                deteriorated=True  # This patient deteriorated
            ),
        ]
        
        # P2 moved to front (appropriate)
        good_response = ["P2", "P1"]
        score_good = grade_queue_management(good_response, patients, ["Q003_deteriorated"])
        
        # P2 at back (inappropriate)
        bad_response = ["P1", "P2"]
        score_bad = grade_queue_management(bad_response, patients, ["Q003_deteriorated"])
        
        assert score_good > score_bad


# ═══ REWARD FUNCTION TESTS ═══════════════════════════════════════════════════

class TestRewardFunctions:
    """Test meaningful reward shaping throughout episodes"""

    def test_incremental_rewards_esi(self, esi_env):
        """ESI task should provide feedback on each action"""
        esi_env.reset()
        
        # Request vitals (should give small reward/penalty)
        action = TriageAction(action_type="request_vitals", content="")
        result = esi_env.step(action)
        assert result.reward != 0.0 or not result.done

    def test_incremental_rewards_intake(self, intake_env):
        """Intake task should reward each field collected"""
        result = intake_env.reset()
        obs = result.observation
        state_before = intake_env.state()
        
        # Collect a field
        actions = [a for a in obs.available_actions if "ask_field" in a]
        if actions:
            field = actions[0].split(":")[-1]
            action = TriageAction(action_type="ask_field", content=field)
            result = intake_env.step(action)
            
            state_after = intake_env.state()
            # Rewards should accumulate
            assert len(state_after["rewards"]) > len(state_before["rewards"])

    def test_penalty_for_invalid_actions(self, esi_env):
        """Invalid actions should receive negative reward"""
        esi_env.reset()
        
        # Invalid action
        action = TriageAction(action_type="invalid_action_type", content="")
        result = esi_env.step(action)
        
        assert result.reward < 0.0

    def test_max_steps_timeout(self, intake_env):
        """Episodes should timeout after max steps"""
        intake_env.reset()
        
        # Exhaust max steps with repeated actions
        for i in range(50):  # Well beyond max
            action = TriageAction(action_type="complete_intake", content="")
            result = intake_env.step(action)
            if result.done:
                break
        
        assert result.done == True


# ═══ REPRODUCIBILITY AND DETERMINISM ═══════════════════════════════════════

class TestReproducibility:
    """Verify grading is reproducible and deterministic"""

    def test_same_seed_same_trajectory(self):
        """Identical seeds must produce identical trajectories"""
        trajectories = []
        
        for _ in range(2):
            env = MedicalTriageEnv(task_type=TaskType.ESI_ASSIGNMENT, seed=999)
            result = env.reset()
            obs = result.observation
            
            action = TriageAction(action_type="assign_level", content="2")
            result = env.step(action)
            
            trajectories.append({
                'patient_id': obs.patient.patient_id,
                'complaint': obs.patient.chief_complaint,
                'reward': result.reward
            })
        
        assert trajectories[0] == trajectories[1]

    def test_different_seeds_different_cases(self):
        """Different seeds should produce different cases"""
        env1 = MedicalTriageEnv(task_type=TaskType.ESI_ASSIGNMENT, seed=1)
        env2 = MedicalTriageEnv(task_type=TaskType.ESI_ASSIGNMENT, seed=2)
        
        result1 = env1.reset()
        result2 = env2.reset()
        
        obs1 = result1.observation
        obs2 = result2.observation
        
        # May get different patients (not guaranteed, but likely)
        # At minimum, should handle different seeds gracefully
        assert obs1.patient is not None
        assert obs2.patient is not None

    def test_grader_determinism(self):
        """Graders must be pure functions"""
        # ESI grader
        for _ in range(10):
            assert grade_esi_assignment(2, 3) == grade_esi_assignment(2, 3)
        
        # Intake grader
        collected = {"pain_scale": "8", "onset": "2 hours ago"}
        required = ["pain_scale", "onset", "duration"]
        for _ in range(10):
            score1 = grade_intake_interview(collected, required, 5)
            score2 = grade_intake_interview(collected, required, 5)
            assert score1 == score2


# ═══ INTEGRATION TESTS ═══════════════════════════════════════════════════════

class TestEndToEnd:
    """Full workflow integration tests"""

    def test_all_three_tasks_complete(self):
        """All three tasks must be runnable"""
        tasks = [
            (TaskType.ESI_ASSIGNMENT, "assign_level", "3"),
            (TaskType.INTAKE_INTERVIEW, "complete_intake", ""),
            (TaskType.QUEUE_MANAGEMENT, "submit_queue", "Q001,Q002,Q003,Q004,Q005"),
        ]
        
        for task_type, action_type, content in tasks:
            env = MedicalTriageEnv(task_type=task_type, seed=42)
            obs = env.reset()
            assert not obs.done
            
            # Execute one action
            action = TriageAction(action_type=action_type, content=content)
            result = env.step(action)
            
            # Should complete or continue
            assert result.observation is not None
            assert isinstance(result.reward, (int, float))

    def test_state_serialization(self):
        """Environment state must be serializable"""
        env = MedicalTriageEnv(task_type=TaskType.ESI_ASSIGNMENT, seed=42)
        env.reset()
        
        state = env.state()
        
        # Should be JSON serializable
        try:
            json_str = json.dumps(state, default=str)
            assert len(json_str) > 0
        except Exception as e:
            pytest.fail(f"State not serializable: {e}")


# ═══ SCORE VALIDATION ═══════════════════════════════════════════════════════

class TestScoreValidation:
    """Verify all scores are in valid range [0, 1]"""

    def test_all_grader_scores_normalized(self):
        """All graders must return scores in [0.0, 1.0]"""
        # Test ESI grader
        for assigned in range(1, 6):
            for correct in range(1, 6):
                score = grade_esi_assignment(assigned, correct)
                assert 0.0 <= score <= 1.0, f"ESI score out of range: {score}"
        
        # Test intake grader
        required = ["field1", "field2", "field3"]
        for num_collected in range(len(required) + 1):
            collected = {f"field{i+1}": "value" for i in range(num_collected)}
            for steps in [1, 5, 10, 20]:
                score = grade_intake_interview(collected, required, steps)
                assert 0.0 <= score <= 1.0, f"Intake score out of range: {score}"

    def test_rewards_are_bounded(self, esi_env):
        """All step rewards should be reasonable"""
        esi_env.reset()
        
        action = TriageAction(action_type="assign_level", content="3")
        result = esi_env.step(action)
        
        # Rewards should be normalized or at least bounded
        assert -1.0 <= result.reward <= 1.0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])