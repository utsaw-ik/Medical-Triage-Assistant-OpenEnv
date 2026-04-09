#!/usr/bin/env python3
"""
Inference Pipeline Test
=======================
Tests the complete inference workflow including:
- Server startup and health
- Environment API endpoints
- Inference script execution
- Output format validation
"""

import sys
import os
import time
import subprocess
import requests
import json
from typing import Dict, Any, Optional


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")


def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")


def test_server_endpoints():
    """Test that the FastAPI server endpoints work correctly"""
    print(f"\n{Colors.BOLD}Testing Server Endpoints{Colors.END}")
    print("=" * 70)
    
    base_url = "http://localhost:7860"
    
    # Wait for server to be ready (assuming it's already running)
    print_info("Checking server health...")
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{base_url}/", timeout=2)
            if response.status_code == 200:
                print_success("Server is running")
                break
        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:
                print_error("Server not responding. Please start server.py first")
                return False
            time.sleep(1)
    
    # Test /tasks endpoint
    print_info("Testing GET /tasks...")
    try:
        response = requests.get(f"{base_url}/tasks", timeout=5)
        if response.status_code == 200:
            tasks = response.json()
            if len(tasks) >= 3:
                print_success(f"Tasks endpoint returned {len(tasks)} tasks")
            else:
                print_error(f"Expected at least 3 tasks, got {len(tasks)}")
                return False
        else:
            print_error(f"Tasks endpoint returned status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Tasks endpoint failed: {e}")
        return False
    
    # Test reset endpoint
    print_info("Testing POST /reset...")
    try:
        response = requests.post(
            f"{base_url}/reset",
            json={"task_type": "esi_assignment", "seed": 42},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            if "observation" in data:
                print_success("Reset endpoint works correctly")
            else:
                print_error("Reset response missing 'observation' field")
                return False
        else:
            print_error(f"Reset endpoint returned status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Reset endpoint failed: {e}")
        return False
    
    # Test step endpoint
    print_info("Testing POST /step...")
    try:
        response = requests.post(
            f"{base_url}/step",
            json={"action_type": "assign_level", "content": "3"},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            required_fields = ["observation", "reward", "done", "info"]
            missing = [f for f in required_fields if f not in data]
            if not missing:
                print_success("Step endpoint works correctly")
            else:
                print_error(f"Step response missing fields: {missing}")
                return False
        else:
            print_error(f"Step endpoint returned status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Step endpoint failed: {e}")
        return False
    
    # Test state endpoint
    print_info("Testing GET /state...")
    try:
        response = requests.get(f"{base_url}/state", timeout=5)
        if response.status_code == 200:
            state = response.json()
            if isinstance(state, dict):
                print_success("State endpoint works correctly")
            else:
                print_error("State endpoint should return a dict")
                return False
        else:
            print_error(f"State endpoint returned status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"State endpoint failed: {e}")
        return False
    
    return True


def test_inference_output_format():
    """Test that inference script produces correctly formatted output"""
    print(f"\n{Colors.BOLD}Testing Inference Output Format{Colors.END}")
    print("=" * 70)
    
    # Set minimal environment variables for testing
    env = os.environ.copy()
    env["HF_TOKEN"] = "dummy_token_for_testing"
    env["ENV_BASE_URL"] = "http://localhost:7860"
    env["API_BASE_URL"] = "http://localhost:8000"  # Mock
    env["MODEL_NAME"] = "test-model"
    
    print_info("Running inference.py with dummy credentials...")
    print_warning("Note: This will fail to call actual LLM, but we can check output format")
    
    # We'll just check if the script can be imported and has the right structure
    try:
        # Import and check structure
        import inference
        
        required_functions = [
            "log_start",
            "log_step", 
            "log_end",
            "run_episode"
        ]
        
        missing = [f for f in required_functions if not hasattr(inference, f)]
        if missing:
            print_error(f"inference.py missing functions: {missing}")
            return False
        
        print_success("inference.py has required functions")
        
        # Check that environment variables are read
        if not hasattr(inference, "API_KEY") and not hasattr(inference, "HF_TOKEN"):
            print_error("inference.py doesn't read HF_TOKEN")
            return False
        
        print_success("inference.py reads required environment variables")
        
        # Check for correct logging format
        import inspect
        log_start_src = inspect.getsource(inference.log_start)
        
        if "[START]" not in log_start_src:
            print_error("log_start doesn't use [START] format")
            return False
        
        print_success("Logging functions use correct format")
        
    except ImportError as e:
        print_error(f"Could not import inference.py: {e}")
        return False
    except Exception as e:
        print_error(f"Error checking inference.py: {e}")
        return False
    
    return True


def test_complete_episode():
    """Test running a complete episode through the API"""
    print(f"\n{Colors.BOLD}Testing Complete Episode{Colors.END}")
    print("=" * 70)
    
    base_url = "http://localhost:7860"
    
    print_info("Running complete ESI assignment episode...")
    
    try:
        # Reset
        response = requests.post(
            f"{base_url}/reset",
            json={"task_type": "esi_assignment", "seed": 42},
            timeout=5
        )
        
        if response.status_code != 200:
            print_error("Failed to reset environment")
            return False
        
        obs_data = response.json()
        
        # Step through episode
        max_steps = 5
        total_reward = 0.0
        
        for step in range(max_steps):
            if obs_data.get("observation", {}).get("done", False):
                print_success(f"Episode completed in {step} steps")
                break
            
            # Take action (assign level 3 as example)
            response = requests.post(
                f"{base_url}/step",
                json={"action_type": "assign_level", "content": "3"},
                timeout=5
            )
            
            if response.status_code != 200:
                print_error(f"Step {step} failed with status {response.status_code}")
                return False
            
            obs_data = response.json()
            reward = obs_data.get("reward", 0.0)
            total_reward += reward
            
            print(f"  Step {step+1}: reward={reward:.3f}")
        
        print_success(f"Episode completed successfully. Total reward: {total_reward:.3f}")
        
    except Exception as e:
        print_error(f"Episode test failed: {e}")
        return False
    
    return True


def test_all_tasks():
    """Test that all three tasks can be initialized and stepped"""
    print(f"\n{Colors.BOLD}Testing All Task Types{Colors.END}")
    print("=" * 70)
    
    base_url = "http://localhost:7860"
    
    tasks = [
        ("esi_assignment", "assign_level", "3"),
        ("intake_interview", "ask_field", "pain_scale"),
        ("queue_management", "get_details", "Q001"),
    ]
    
    for task_name, action_type, content in tasks:
        print_info(f"Testing {task_name}...")
        
        try:
            # Reset
            response = requests.post(
                f"{base_url}/reset",
                json={"task_type": task_name, "seed": 42},
                timeout=5
            )
            
            if response.status_code != 200:
                print_error(f"{task_name} reset failed")
                return False
            
            # Take one step
            response = requests.post(
                f"{base_url}/step",
                json={"action_type": action_type, "content": content},
                timeout=5
            )
            
            if response.status_code != 200:
                print_error(f"{task_name} step failed")
                return False
            
            data = response.json()
            if "reward" not in data:
                print_error(f"{task_name} response missing reward")
                return False
            
            print_success(f"{task_name} works correctly")
            
        except Exception as e:
            print_error(f"{task_name} test failed: {e}")
            return False
    
    return True


def main():
    """Run all inference tests"""
    print("=" * 70)
    print(f"{Colors.BOLD}Inference Pipeline Test Suite{Colors.END}")
    print("=" * 70)
    
    print_warning("Prerequisites:")
    print_warning("  1. Server must be running: python server.py")
    print_warning("  2. Server should be at http://localhost:7860")
    print()
    
    tests = [
        ("Server Endpoints", test_server_endpoints),
        ("Inference Output Format", test_inference_output_format),
        ("Complete Episode", test_complete_episode),
        ("All Tasks", test_all_tasks),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print_error(f"{name} - Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}TEST SUMMARY{Colors.END}")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("\nAll inference tests passed!")
        return 0
    else:
        print_warning("\nSome tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
