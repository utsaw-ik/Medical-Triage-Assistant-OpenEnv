#!/usr/bin/env python3
"""
OpenEnv Validation Test
=======================
Validates that the environment meets all OpenEnv specification requirements.
This script mimics what `openenv validate` would check.
"""

import sys
import os
import yaml
from pathlib import Path


def validate_openenv_yaml():
    """Validate openenv.yaml exists and has required fields"""
    print("🔍 Checking openenv.yaml...")
    
    yaml_path = Path("openenv.yaml")
    if not yaml_path.exists():
        print("❌ FAIL: openenv.yaml not found")
        return False
    
    try:
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ FAIL: Could not parse openenv.yaml: {e}")
        return False
    
    required_fields = [
        "name",
        "version",
        "description",
        "tasks",
        "observation_space",
        "action_space"
    ]
    
    missing = [field for field in required_fields if field not in config]
    if missing:
        print(f"❌ FAIL: Missing required fields: {missing}")
        return False
    
    # Validate tasks structure
    if not isinstance(config.get("tasks"), list) or len(config["tasks"]) < 3:
        print(f"❌ FAIL: Must have at least 3 tasks, found {len(config.get('tasks', []))}")
        return False
    
    for task in config["tasks"]:
        if "name" not in task or "difficulty" not in task:
            print(f"❌ FAIL: Task missing name or difficulty: {task}")
            return False
    
    print("✅ PASS: openenv.yaml is valid")
    return True


def validate_environment_interface():
    """Validate environment implements required interface"""
    print("\n🔍 Checking environment interface...")
    
    try:
        from medical_triage_env import MedicalTriageEnv, TriageAction, TaskType
    except ImportError as e:
        print(f"❌ FAIL: Could not import environment: {e}")
        return False
    
    # Check required methods exist
    env = MedicalTriageEnv(task_type=TaskType.ESI_ASSIGNMENT, seed=42)
    
    required_methods = ["reset", "step", "state"]
    for method in required_methods:
        if not hasattr(env, method) or not callable(getattr(env, method)):
            print(f"❌ FAIL: Missing or non-callable method: {method}")
            return False
    
    # Test reset returns observation
    try:
        result = env.reset()
        if result is None:
            print("❌ FAIL: reset() returned None")
            return False
        
        # Check if result is StepResult or direct observation
        if hasattr(result, 'observation'):
            obs = result.observation
        else:
            obs = result
        
        required_obs_fields = ["task_type", "message", "done", "step_number", "available_actions"]
        for field in required_obs_fields:
            if not hasattr(obs, field):
                print(f"❌ FAIL: Observation missing field: {field}")
                return False
    except Exception as e:
        print(f"❌ FAIL: reset() raised exception: {e}")
        return False
    
    # Test step returns correct structure
    try:
        action = TriageAction(action_type="assign_level", content="3")
        result = env.step(action)
        
        required_result_fields = ["observation", "reward", "done", "info"]
        for field in required_result_fields:
            if not hasattr(result, field):
                print(f"❌ FAIL: Step result missing field: {field}")
                return False
        
        if not isinstance(result.reward, (int, float)):
            print(f"❌ FAIL: reward must be numeric, got {type(result.reward)}")
            return False
        
        if not isinstance(result.done, bool):
            print(f"❌ FAIL: done must be boolean, got {type(result.done)}")
            return False
        
        if not isinstance(result.info, dict):
            print(f"❌ FAIL: info must be dict, got {type(result.info)}")
            return False
    except Exception as e:
        print(f"❌ FAIL: step() raised exception: {e}")
        return False
    
    # Test state returns dict
    try:
        state = env.state()
        if not isinstance(state, dict):
            print(f"❌ FAIL: state() must return dict, got {type(state)}")
            return False
    except Exception as e:
        print(f"❌ FAIL: state() raised exception: {e}")
        return False
    
    print("✅ PASS: Environment interface is valid")
    return True


def validate_pydantic_models():
    """Validate that models use Pydantic typing"""
    print("\n🔍 Checking Pydantic models...")
    
    try:
        from pydantic import BaseModel
        from medical_triage_env import (
            TriageObservation,
            TriageAction,
            Patient,
            PatientVitals,
            StepResult
        )
    except ImportError as e:
        print(f"❌ FAIL: Could not import models: {e}")
        return False
    
    models_to_check = [
        ("TriageObservation", TriageObservation),
        ("TriageAction", TriageAction),
        ("Patient", Patient),
        ("PatientVitals", PatientVitals),
        ("StepResult", StepResult),
    ]
    
    for name, model in models_to_check:
        if not issubclass(model, BaseModel):
            print(f"❌ FAIL: {name} must inherit from Pydantic BaseModel")
            return False
    
    print("✅ PASS: All models use Pydantic")
    return True


def validate_tasks_and_graders():
    """Validate three tasks with working graders"""
    print("\n🔍 Checking tasks and graders...")
    
    try:
        from medical_triage_env import (
            MedicalTriageEnv,
            TaskType,
            TriageAction,
            grade_esi_assignment,
            grade_intake_interview,
            grade_queue_management,
        )
    except ImportError as e:
        print(f"❌ FAIL: Could not import graders: {e}")
        return False
    
    # Test all three task types
    task_configs = [
        (TaskType.ESI_ASSIGNMENT, "assign_level", "3", "ESI/Easy"),
        (TaskType.INTAKE_INTERVIEW, "complete_intake", "", "Intake/Medium"),
        (TaskType.QUEUE_MANAGEMENT, "submit_queue", "Q001,Q002,Q003", "Queue/Hard"),
    ]
    
    for task_type, action_type, content, label in task_configs:
        try:
            env = MedicalTriageEnv(task_type=task_type, seed=42)
            obs = env.reset()
            
            if obs.done:
                print(f"❌ FAIL: {label} - Episode done immediately after reset")
                return False
            
            action = TriageAction(action_type=action_type, content=content)
            result = env.step(action)
            
            if not isinstance(result.reward, (int, float)):
                print(f"❌ FAIL: {label} - Invalid reward type")
                return False
            
            if not (0.0 <= result.reward <= 1.0 or -1.0 <= result.reward <= 0.0):
                print(f"⚠️  WARNING: {label} - Reward {result.reward} outside typical range [-1, 1]")
        
        except Exception as e:
            print(f"❌ FAIL: {label} raised exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test graders return scores in [0, 1]
    grader_tests = [
        ("ESI", lambda: grade_esi_assignment(2, 3)),
        ("Intake", lambda: grade_intake_interview({"field1": "val"}, ["field1", "field2"], 5)),
    ]
    
    for name, grader_fn in grader_tests:
        try:
            score = grader_fn()
            if not isinstance(score, (int, float)):
                print(f"❌ FAIL: {name} grader must return numeric score")
                return False
            if not (0.0 <= score <= 1.0):
                print(f"❌ FAIL: {name} grader score {score} not in [0, 1]")
                return False
        except Exception as e:
            print(f"❌ FAIL: {name} grader raised exception: {e}")
            return False
    
    print("✅ PASS: All tasks and graders working")
    return True


def validate_reproducibility():
    """Validate that seeded environments are reproducible"""
    print("\n🔍 Checking reproducibility...")
    
    try:
        from medical_triage_env import MedicalTriageEnv, TaskType, TriageAction
    except ImportError as e:
        print(f"❌ FAIL: Could not import environment: {e}")
        return False
    
    # Run same episode twice with same seed
    results = []
    for _ in range(2):
        env = MedicalTriageEnv(task_type=TaskType.ESI_ASSIGNMENT, seed=12345)
        result = env.reset()
        
        # Handle both StepResult and direct observation
        if hasattr(result, 'observation'):
            obs = result.observation
        else:
            obs = result
        
        action = TriageAction(action_type="assign_level", content="2")
        step_result = env.step(action)
        
        results.append({
            'patient_id': obs.patient.patient_id,
            'complaint': obs.patient.chief_complaint,
            'reward': step_result.reward
        })
    
    if results[0] != results[1]:
        print(f"❌ FAIL: Episodes not reproducible with same seed")
        print(f"  Run 1: {results[0]}")
        print(f"  Run 2: {results[1]}")
        return False
    
    print("✅ PASS: Environment is reproducible")
    return True


def validate_dockerfile():
    """Check that Dockerfile exists and looks valid"""
    print("\n🔍 Checking Dockerfile...")
    
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        print("❌ FAIL: Dockerfile not found")
        return False
    
    with open(dockerfile_path) as f:
        content = f.read()
    
    required_sections = ["FROM", "COPY", "CMD"]
    missing = [section for section in required_sections if section not in content]
    
    if missing:
        print(f"⚠️  WARNING: Dockerfile may be missing: {missing}")
    
    print("✅ PASS: Dockerfile exists")
    return True


def validate_inference_script():
    """Check inference.py meets hackathon requirements"""
    print("\n🔍 Checking inference.py...")
    
    inference_path = Path("inference.py")
    if not inference_path.exists():
        print("❌ FAIL: inference.py not found in root directory")
        return False
    
    with open(inference_path) as f:
        content = f.read()
    
    # Check for required environment variable reads
    required_env_vars = [
        ("HF_TOKEN", True),  # (var_name, must_have_default)
        ("API_BASE_URL", False),
        ("MODEL_NAME", False),
    ]
    
    for var_name, needs_default in required_env_vars:
        if f'os.getenv("{var_name}"' not in content and f"os.environ['{var_name}']" not in content:
            print(f"❌ FAIL: inference.py must read {var_name} from environment")
            return False
        
        if not needs_default and f'os.getenv("{var_name}", ' not in content:
            print(f"⚠️  WARNING: {var_name} should have a default value")
    
    # Check for OpenAI client usage
    if "OpenAI" not in content or "from openai import" not in content:
        print("❌ FAIL: inference.py must use OpenAI client")
        return False
    
    # Check for required output format
    output_markers = ["[START]", "[STEP]", "[END]"]
    missing_markers = [m for m in output_markers if m not in content]
    
    if missing_markers:
        print(f"⚠️  WARNING: inference.py may be missing output markers: {missing_markers}")
    
    print("✅ PASS: inference.py meets requirements")
    return True


def main():
    """Run all validation checks"""
    print("=" * 70)
    print("OpenEnv Hackathon Validation")
    print("=" * 70)
    
    checks = [
        ("openenv.yaml", validate_openenv_yaml),
        ("Environment Interface", validate_environment_interface),
        ("Pydantic Models", validate_pydantic_models),
        ("Tasks & Graders", validate_tasks_and_graders),
        ("Reproducibility", validate_reproducibility),
        ("Dockerfile", validate_dockerfile),
        ("inference.py", validate_inference_script),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"❌ FAIL: {name} - Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All validation checks passed! Ready for submission.")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix issues before submitting.")
        return 1


if __name__ == "__main__":
    sys.exit(main())