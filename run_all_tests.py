#!/usr/bin/env python3
"""
Master Test Runner
==================
Runs all tests in the correct sequence and provides a comprehensive report.
"""

import sys
import subprocess
import os
import time
from pathlib import Path


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.END}\n")


def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")


def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")


def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")


def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")


def run_command(cmd, env=None, cwd=None, shell=False):
    """Run a command and return success status"""
    try:
        if shell:
            result = subprocess.run(
                cmd,
                shell=True,
                env=env or os.environ,
                cwd=cwd,
                capture_output=True,
                text=True
            )
        else:
            result = subprocess.run(
                cmd,
                env=env or os.environ,
                cwd=cwd,
                capture_output=True,
                text=True
            )
        
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def check_prerequisites():
    """Check that all required files and dependencies exist"""
    print_header("Checking Prerequisites")
    
    required_files = [
        "medical_triage_env.py",
        "server.py",
        "inference.py",
        "openenv.yaml",
        "Dockerfile",
        "requirements.txt",
        "test_medical_triage.py",
        "test_openenv_validation.py",
        "test_inference_pipeline.py",
    ]
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            print_success(f"Found {file}")
        else:
            print_error(f"Missing {file}")
            missing.append(file)
    
    if missing:
        print_error(f"Missing required files: {missing}")
        return False
    
    # Check Python dependencies
    print_info("Checking Python dependencies...")
    deps = ["pytest", "pydantic", "requests", "pyyaml", "fastapi"]
    
    for dep in deps:
        success, _, _ = run_command([sys.executable, "-c", f"import {dep}"])
        if success:
            print_success(f"Installed: {dep}")
        else:
            print_warning(f"Missing: {dep} (will try to install)")
    
    return True


def run_unit_tests():
    """Run pytest unit tests"""
    print_header("Running Unit Tests")
    
    print_info("Executing: pytest test_medical_triage.py -v --tb=short")
    
    success, stdout, stderr = run_command([
        sys.executable, "-m", "pytest",
        "test_medical_triage.py",
        "-v",
        "--tb=short"
    ])
    
    if success:
        # Count passed tests
        passed = stdout.count(" PASSED")
        print_success(f"Unit tests passed ({passed} tests)")
        return True
    else:
        print_error("Unit tests failed")
        print("\nOutput:")
        print(stdout)
        print(stderr)
        return False


def run_openenv_validation():
    """Run OpenEnv specification validation"""
    print_header("Running OpenEnv Validation")
    
    print_info("Executing: python test_openenv_validation.py")
    
    success, stdout, stderr = run_command([
        sys.executable,
        "test_openenv_validation.py"
    ])
    
    print(stdout)
    
    if success:
        print_success("OpenEnv validation passed")
        return True
    else:
        print_error("OpenEnv validation failed")
        return False


def run_inference_tests():
    """Run inference pipeline tests"""
    print_header("Running Inference Pipeline Tests")
    
    # Check if server is already running
    print_info("Checking if server is running...")
    import requests
    
    server_running = False
    try:
        response = requests.get("http://localhost:7860/", timeout=2)
        if response.status_code == 200:
            server_running = True
            print_success("Server is already running")
    except:
        print_warning("Server not running, will start it")
    
    # Start server if needed
    server_process = None
    if not server_running:
        print_info("Starting server...")
        server_process = subprocess.Popen(
            [sys.executable, "server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(3)
        
        # Check if started
        try:
            response = requests.get("http://localhost:7860/", timeout=2)
            if response.status_code == 200:
                print_success("Server started successfully")
            else:
                print_error("Server failed to start properly")
                if server_process:
                    server_process.terminate()
                return False
        except:
            print_error("Could not connect to server")
            if server_process:
                server_process.terminate()
            return False
    
    # Run tests
    print_info("Executing: python test_inference_pipeline.py")
    
    success, stdout, stderr = run_command([
        sys.executable,
        "test_inference_pipeline.py"
    ])
    
    print(stdout)
    
    # Stop server if we started it
    if server_process:
        print_info("Stopping server...")
        server_process.terminate()
        server_process.wait(timeout=5)
    
    if success:
        print_success("Inference pipeline tests passed")
        return True
    else:
        print_error("Inference pipeline tests failed")
        return False


def run_docker_tests():
    """Run Docker build and deployment tests"""
    print_header("Running Docker Tests")
    
    # Check if Docker is available
    docker_available, _, _ = run_command(["docker", "--version"])
    
    if not docker_available:
        print_warning("Docker not available, skipping Docker tests")
        print_info("Install Docker to run these tests")
        return None  # None means skipped
    
    print_info("Executing: ./test_docker.sh")
    
    # Make script executable
    os.chmod("test_docker.sh", 0o755)
    
    success, stdout, stderr = run_command(
        "./test_docker.sh",
        shell=True
    )
    
    print(stdout)
    
    if success:
        print_success("Docker tests passed")
        return True
    else:
        print_error("Docker tests failed")
        print(stderr)
        return False


def generate_report(results):
    """Generate final test report"""
    print_header("Test Summary Report")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    for test_name, result in results.items():
        if result is True:
            print_success(f"{test_name}")
        elif result is False:
            print_error(f"{test_name}")
        else:
            print_warning(f"{test_name} (skipped)")
    
    print(f"\n{Colors.BOLD}Results:{Colors.END}")
    print(f"  Total:   {total}")
    print(f"  {Colors.GREEN}Passed:  {passed}{Colors.END}")
    print(f"  {Colors.RED}Failed:  {failed}{Colors.END}")
    print(f"  {Colors.YELLOW}Skipped: {skipped}{Colors.END}")
    
    percentage = (passed / (total - skipped) * 100) if (total - skipped) > 0 else 0
    print(f"\n  {Colors.BOLD}Pass rate: {percentage:.1f}%{Colors.END}")
    
    if failed == 0 and passed > 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All tests passed! Ready for submission!{Colors.END}")
        return 0
    elif failed > 0:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  Some tests failed. Please fix issues before submitting.{Colors.END}")
        return 1
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  No tests were run.{Colors.END}")
        return 1


def main():
    """Main test runner"""
    print(f"{Colors.BOLD}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}Medical Triage Environment - Master Test Suite{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.END}")
    
    # Check prerequisites
    if not check_prerequisites():
        print_error("Prerequisites check failed")
        return 1
    
    # Run all tests
    results = {}
    
    # 1. Unit tests
    results["Unit Tests"] = run_unit_tests()
    
    # 2. OpenEnv validation
    results["OpenEnv Validation"] = run_openenv_validation()
    
    # 3. Inference pipeline (requires server)
    results["Inference Pipeline"] = run_inference_tests()
    
    # 4. Docker tests (optional if Docker not available)
    results["Docker Build & Deployment"] = run_docker_tests()
    
    # Generate report
    return generate_report(results)


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
